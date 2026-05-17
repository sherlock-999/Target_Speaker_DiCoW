#!/usr/bin/env python3
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torchaudio
from librosa import load as libr_load

PROJECT_ROOT = Path(__file__).resolve().parent

import sys
sys.path.insert(0, str(PROJECT_ROOT / 'DiariZen'))
sys.path.insert(0, str(PROJECT_ROOT))

from streaming_utils import (  # noqa: E402
    decode_window,
    load_dicow,
    normalize_words,
    real_streaming_stno,
)
from diarizen.pipelines.target_conditioned import TargetConditionedSTNOPipeline  # noqa: E402


@dataclass
class RollingChunkResult:
    chunk_index: int
    start_sec: float
    end_sec: float
    committed_text: str
    dominant_stno: Optional[str] = None
    tentative_text: str = ''


@dataclass
class WindowWord:
    start: float
    end: float
    display: str
    match: str


STNO_NAMES = ('S', 'T', 'N', 'O')
SPK_HEADER_RE = re.compile(r'\s*Speaker\s*\d+\s*:', re.I)


def clean_display_text(text: str) -> str:
    text = SPK_HEADER_RE.sub(' ', str(text))
    text = re.sub(r'<\|[^>]+?\|>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def display_words(text: str) -> List[str]:
    return clean_display_text(text).split()


def join_transcript_parts(parts: Sequence[str], overlap_tail_words: int = 6, overlap_head_words: int = 6) -> str:
    merged: List[str] = []
    for raw_part in parts:
        part = clean_display_text(raw_part)
        if not part:
            continue
        if not merged:
            merged.append(part)
            continue

        prev_display_words = merged[-1].split()
        curr_display_words = part.split()
        prev_match_words = normalize_words(merged[-1]).split()
        curr_match_words = normalize_words(part).split()
        max_overlap = min(len(prev_match_words), len(curr_match_words), overlap_tail_words, overlap_head_words)

        overlap = 0
        for size in range(max_overlap, 0, -1):
            if prev_match_words[-size:] == curr_match_words[:size]:
                overlap = size
                break

        if overlap > 0:
            merged[-1] = ' '.join(prev_display_words + curr_display_words[overlap:]).strip()
        else:
            merged.append(part)

    return ' '.join(merged).strip()


def finalize_transcript_text(parts: Sequence[str]) -> str:
    text = join_transcript_parts(parts)
    return re.sub(r'\s+', ' ', text).strip()


def _window_words_from_text(text: str, start_s: float, end_s: float) -> List[WindowWord]:
    words = display_words(text)
    if not words:
        return []
    duration = max(0.0, end_s - start_s)
    out: List[WindowWord] = []
    for index, word in enumerate(words):
        match = normalize_words(word)
        if not match:
            continue
        word_start = start_s + duration * (index / len(words))
        word_end = start_s + duration * ((index + 1) / len(words))
        out.append(WindowWord(word_start, word_end, word, match))
    return out


def segments_to_window_words(
    segments: Sequence[object],
    fallback_text: str,
    window_start_s: float,
    window_end_s: float,
) -> List[WindowWord]:
    words: List[WindowWord] = []
    for segment in segments:
        start = float(getattr(segment, 'start', window_start_s))
        end = float(getattr(segment, 'end', start))
        text = str(getattr(segment, 'text', ''))
        if end > start and text:
            words.extend(_window_words_from_text(text, start, end))
    if words:
        return words
    return _window_words_from_text(fallback_text, window_start_s, window_end_s)


class LocalAgreementBuffer:
    def __init__(
        self,
        min_match_words: int = 4,
        max_current_match_start: int = 20,
        boundary_tolerance_s: float = 0.1,
        max_dedup_ngram: int = 5,
    ):
        self.min_match_words = min_match_words
        self.max_current_match_start = max_current_match_start
        self.boundary_tolerance_s = boundary_tolerance_s
        self.max_dedup_ngram = max_dedup_ngram
        self.committed_in_buffer: List[WindowWord] = []
        self.buffer: List[WindowWord] = []
        self.new: List[WindowWord] = []
        self.last_committed_time = 0.0

    @staticmethod
    def _matches(words: Sequence[WindowWord]) -> List[str]:
        return [word.match for word in words]

    def insert(self, words: Sequence[WindowWord]) -> None:
        self.new = [
            word for word in words
            if word.start > self.last_committed_time - self.boundary_tolerance_s
        ]
        if not self.new or not self.committed_in_buffer:
            return
        if abs(self.new[0].start - self.last_committed_time) >= 1.0:
            return

        max_ngram = min(len(self.committed_in_buffer), len(self.new), self.max_dedup_ngram)
        for size in range(1, max_ngram + 1):
            committed_tail = self._matches(self.committed_in_buffer[-size:])
            new_head = self._matches(self.new[:size])
            if committed_tail == new_head:
                del self.new[:size]
                break

    def flush(self, final: bool = False) -> List[WindowWord]:
        committed: List[WindowWord] = []
        if self.buffer and self.new:
            matcher = difflib.SequenceMatcher(None, self._matches(self.buffer), self._matches(self.new), autojunk=False)
            blocks = [block for block in matcher.get_matching_blocks() if block.size >= self.min_match_words]
            if blocks:
                plausible = [block for block in blocks if block.b <= self.max_current_match_start]
                if not plausible:
                    plausible = blocks
                block = min(plausible, key=lambda item: (item.b, item.a, -item.size))
                committed = self.buffer[:block.a]

        if committed:
            self.last_committed_time = committed[-1].end
            self.committed_in_buffer.extend(committed)

        if final:
            tail = self.new or self.buffer
            if tail:
                committed.extend(tail)
                self.last_committed_time = tail[-1].end
                self.committed_in_buffer.extend(tail)
            self.buffer = []
            self.new = []
        else:
            self.buffer = self.new
            self.new = []

        return committed


def streaming_update_times(*, duration_s: float, chunk_s: float) -> List[float]:
    if duration_s <= 0:
        return []
    times: List[float] = []
    current = min(duration_s, max(0.0, float(chunk_s)))
    while current < duration_s:
        times.append(round(current, 3))
        current += chunk_s
    if not times or times[-1] < duration_s:
        times.append(duration_s)
    return times


def dominant_state(stno: np.ndarray, end_s: float, chunk_s: float, frame_hz: float = 50.0) -> str:
    end_frame = min(stno.shape[-1], max(1, int(round(end_s * frame_hz))))
    start_frame = max(0, int(round((end_s - chunk_s) * frame_hz)))
    if end_frame <= start_frame:
        return 'S'
    mean = stno[:, start_frame:end_frame].mean(axis=1)
    return STNO_NAMES[int(np.argmax(mean))]


class RollingOnlineTargetDiCoW:
    def __init__(
        self,
        pipeline,
        *,
        dicow_model_name: str,
        stno_config: Path | None = None,
        stno_checkpoint: Path | None = None,
        stno_model_id: str = "Adnan256/streaming-target-stno-wavlm-base",
        chunk_s: float = 15.0,
        lookback_s: float = 20.0,
        decode_window_s: float = 30.0,
        commit_min_match_words: int = 4,
        commit_max_current_match_start: int = 20,
    ):
        self.pipeline = pipeline
        self.device = pipeline.device
        self.chunk_s = chunk_s
        self.lookback_s = lookback_s
        self.decode_window_s = decode_window_s
        self.commit_min_match_words = commit_min_match_words
        self.commit_max_current_match_start = commit_max_current_match_start
        if stno_config is not None and stno_checkpoint is not None:
            self.stno_pipeline = TargetConditionedSTNOPipeline.from_experiment(
                config_path=stno_config,
                checkpoint_path=stno_checkpoint,
                device=self.device,
            )
        else:
            self.stno_pipeline = TargetConditionedSTNOPipeline.from_pretrained(
                stno_model_id,
                device=self.device,
            )
        self.model, self.feature_extractor, self.tokenizer = load_dicow(
            dicow_model_name,
            device=self.device,
            local_files_only=False,
        )

    def _load_audio(self, path: str) -> tuple[np.ndarray, int]:
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000
        return wav.numpy().astype(np.float32), sr

    def transcribe(self, audio_path: str, enrollment_audio: str, presence_threshold: Optional[float] = None) -> Dict[str, object]:
        del presence_threshold
        audio, sr = self._load_audio(audio_path)
        stno = real_streaming_stno(
            mixed_audio=audio,
            sample_rate=sr,
            enrollment_path=Path(enrollment_audio),
            pipeline=self.stno_pipeline,
            chunk_s=self.chunk_s,
            lookback_s=self.lookback_s,
        )
        committer = LocalAgreementBuffer(
            min_match_words=self.commit_min_match_words,
            max_current_match_start=self.commit_max_current_match_start,
        )
        duration = audio.shape[0] / float(sr)
        update_times = streaming_update_times(duration_s=duration, chunk_s=self.chunk_s)

        chunks: List[RollingChunkResult] = []
        transcript_parts: List[str] = []
        for chunk_index, current_time in enumerate(update_times):
            window_start = max(0.0, current_time - self.decode_window_s)
            segments, tentative = decode_window(
                model=self.model,
                feature_extractor=self.feature_extractor,
                tokenizer=self.tokenizer,
                audio=audio,
                sample_rate=sr,
                stno=stno,
                window_start_s=window_start,
                window_end_s=current_time,
                decode_window_s=self.decode_window_s,
                device=self.device,
            )
            words = segments_to_window_words(
                segments,
                tentative,
                window_start,
                current_time,
            )
            committer.insert(words)
            committed = committer.flush(final=chunk_index == len(update_times) - 1)
            committed_text = ' '.join(word.display for word in committed).strip()
            if committed_text:
                transcript_parts.append(committed_text)
            chunks.append(
                RollingChunkResult(
                    chunk_index=chunk_index,
                    start_sec=window_start,
                    end_sec=current_time,
                    committed_text=committed_text,
                    tentative_text=clean_display_text(tentative),
                    dominant_stno=dominant_state(stno, current_time, self.chunk_s),
                )
            )

        rolling_text = finalize_transcript_text(transcript_parts)

        return {
            'text': rolling_text,
            'rolling_text': rolling_text,
            'stable_committed_text': rolling_text,
            'chunks': chunks,
            'num_vad_speech_segments': len(chunks),
        }

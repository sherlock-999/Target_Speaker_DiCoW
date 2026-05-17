#!/usr/bin/env python3
import math
import re
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

try:
    import gradio  # noqa: F401
except Exception:
    sys.modules["gradio"] = types.SimpleNamespace(
        Info=lambda *a, **k: None,
        Warning=lambda *a, **k: None,
    )

import numpy as np
import torch
import torchaudio
from librosa import load as libr_load

from pipeline import DiCoWPipeline, change_state_of_sbs


TAG_RE = re.compile(r"<[^>]+>")
SPK_HEADER_RE = re.compile(r"\s*Speaker\s*\d+\s*:", re.I)


@dataclass(frozen=True)
class SpeechChunk:
    start_sample: int
    end_sample: int
    cut_reason: str

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample

    def start_sec(self, sample_rate: int) -> float:
        return self.start_sample / float(sample_rate)

    def end_sec(self, sample_rate: int) -> float:
        return self.end_sample / float(sample_rate)


@dataclass
class OnlineChunkResult:
    chunk_index: int
    start_sec: float
    end_sec: float
    cut_reason: str
    accepted: bool
    best_score: Optional[float]
    best_speaker_index: Optional[int]
    transcript: str
    skip_reason: Optional[str] = None


def clean_transcript_text(text: str) -> str:
    text = SPK_HEADER_RE.sub(" ", str(text).replace("🗣️", " "))
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def word_error_rate(reference: str, hypothesis: str) -> Dict[str, float]:
    ref_words = clean_transcript_text(reference).split()
    hyp_words = clean_transcript_text(hypothesis).split()
    n = len(ref_words)
    if n == 0:
        return {"wer": 0.0 if not hyp_words else 1.0, "errors": float(len(hyp_words)), "ref_words": 0.0}

    prev = list(range(len(hyp_words) + 1))
    for i, ref_word in enumerate(ref_words, start=1):
        cur = [i]
        for j, hyp_word in enumerate(hyp_words, start=1):
            sub_cost = 0 if ref_word == hyp_word else 1
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + sub_cost,
                )
            )
        prev = cur
    errors = float(prev[-1])
    return {"wer": errors / float(n), "errors": errors, "ref_words": float(n)}


def _normalise_speech_timestamps(speech_timestamps: Iterable[Dict[str, int]]) -> List[Dict[str, int]]:
    normalised = []
    for item in speech_timestamps:
        start = int(item["start"])
        end = int(item["end"])
        if end > start:
            normalised.append({"start": start, "end": end})
    normalised.sort(key=lambda x: (x["start"], x["end"]))
    return normalised


def build_chunks_from_speech_timestamps(
    speech_timestamps: Sequence[Dict[str, int]],
    total_samples: int,
    sample_rate: int = 16000,
    min_chunk_sec: float = 3.0,
    max_chunk_sec: float = 8.0,
) -> List[SpeechChunk]:
    if min_chunk_sec <= 0:
        raise ValueError("min_chunk_sec must be positive")
    if max_chunk_sec <= min_chunk_sec:
        raise ValueError("max_chunk_sec must be greater than min_chunk_sec")

    speech = _normalise_speech_timestamps(speech_timestamps)
    if not speech:
        return []

    min_samples = int(round(min_chunk_sec * sample_rate))
    max_samples = int(round(max_chunk_sec * sample_rate))
    total_samples = int(total_samples)
    chunks: List[SpeechChunk] = []
    i = 0

    while i < len(speech):
        start = max(0, speech[i]["start"])
        if start >= total_samples:
            break

        min_cut = min(total_samples, start + min_samples)
        max_cut = min(total_samples, start + max_samples)

        active_at_min = None
        next_after_min = None
        last_speech_end = start
        j = i
        while j < len(speech) and speech[j]["start"] < max_cut:
            seg = speech[j]
            if seg["end"] <= start:
                j += 1
                continue
            last_speech_end = max(last_speech_end, min(seg["end"], total_samples))
            if seg["start"] <= min_cut < seg["end"]:
                active_at_min = seg
                break
            if seg["start"] > min_cut:
                next_after_min = seg
                break
            j += 1

        if active_at_min is None:
            if last_speech_end <= min_cut and i == len(speech) - 1:
                cut = min(last_speech_end, max_cut)
                reason = "eof"
            elif next_after_min is not None or last_speech_end <= min_cut:
                cut = min_cut
                reason = "silence_after_min"
            else:
                cut = min(max(last_speech_end, min_cut), max_cut)
                reason = "silence"
        else:
            cut = min(active_at_min["end"], max_cut)
            reason = "max_duration" if active_at_min["end"] > max_cut else "silence"

        if cut <= start:
            break

        chunks.append(SpeechChunk(start_sample=start, end_sample=cut, cut_reason=reason))

        while i < len(speech) and speech[i]["end"] <= cut:
            i += 1
        if i < len(speech) and speech[i]["start"] < cut < speech[i]["end"]:
            speech[i] = {"start": cut, "end": speech[i]["end"]}

    return [chunk for chunk in chunks if chunk.end_sample > chunk.start_sample]


class OnlineTargetDiCoW:
    def __init__(
        self,
        pipeline: DiCoWPipeline,
        sample_rate: int = 16000,
        min_chunk_sec: float = 3.0,
        max_chunk_sec: float = 8.0,
        vad_threshold: float = 0.5,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ):
        self.pipeline = pipeline
        self.sample_rate = sample_rate
        self.min_chunk_sec = min_chunk_sec
        self.max_chunk_sec = max_chunk_sec
        self.vad_threshold = vad_threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self._vad_model = None

    def _load_vad_model(self):
        if self._vad_model is None:
            from silero_vad import load_silero_vad

            self._vad_model = load_silero_vad()
        return self._vad_model

    def get_speech_timestamps(self, waveform: torch.Tensor) -> List[Dict[str, int]]:
        from silero_vad import get_speech_timestamps

        if waveform.ndim > 1:
            waveform = waveform.squeeze(0)
        return get_speech_timestamps(
            waveform,
            self._load_vad_model(),
            threshold=self.vad_threshold,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=250,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=False,
        )

    def build_chunks(self, audio_path: str) -> List[SpeechChunk]:
        wav, sr = torchaudio.load(audio_path)
        wav = wav.mean(dim=0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        speech_timestamps = self.get_speech_timestamps(wav)
        return build_chunks_from_speech_timestamps(
            speech_timestamps,
            total_samples=int(wav.numel()),
            sample_rate=self.sample_rate,
            min_chunk_sec=self.min_chunk_sec,
            max_chunk_sec=self.max_chunk_sec,
        )

    def _write_chunk(self, waveform: torch.Tensor, chunk: SpeechChunk, path: Path):
        piece = waveform[chunk.start_sample : chunk.end_sample].unsqueeze(0)
        torchaudio.save(str(path), piece.cpu(), self.sample_rate)

    def _best_target_speaker(
        self,
        chunk_path: str,
        diarization_output,
        enrollment_embedding: np.ndarray,
    ) -> Dict[str, Optional[float]]:
        labels = list(diarization_output.labels())
        best_index = None
        best_score = -float("inf")
        for idx, speaker in enumerate(labels):
            timeline = diarization_output.label_timeline(speaker)
            speaker_embedding = self.pipeline._compute_embedding_for_timeline(chunk_path, timeline)
            if speaker_embedding is None:
                continue
            score = self.pipeline._cosine_similarity(enrollment_embedding, speaker_embedding)
            if math.isnan(score):
                continue
            if score > best_score:
                best_score = score
                best_index = idx
        return {
            "index": best_index,
            "score": None if best_index is None else float(best_score),
            "num_speakers": len(labels),
        }

    def _transcribe_selected_speaker(self, chunk_path: str, diarization_output, target_speaker_index: int) -> str:
        inp_aud, _ = libr_load(chunk_path, sr=self.sample_rate, mono=True)
        generator = super(DiCoWPipeline, self.pipeline).preprocess(
            inp_aud,
            chunk_length_s=0,
            stride_length_s=None,
        )
        samples = next(generator)

        per_speaker_samples = [diarization_output.label_timeline(speaker) for speaker in diarization_output.labels()]
        diarization_mask = self.pipeline.get_diarization_mask(
            per_speaker_samples,
            samples["input_features"].shape[-1] // 2,
        )
        stno_mask = self.pipeline.get_stno_mask(diarization_mask, target_speaker_index)

        samples["stno_mask"] = stno_mask.unsqueeze(0).to(
            samples["input_features"].device,
            dtype=samples["input_features"].dtype,
        )
        samples["input_features"] = samples["input_features"].repeat(1, 1, 1)
        samples["attention_mask"] = torch.ones(
            samples["input_features"].shape[0],
            samples["input_features"].shape[2],
            dtype=torch.bool,
            device=samples["input_features"].device,
        )
        if "num_frames" in samples:
            del samples["num_frames"]

        if hasattr(self.pipeline.model.config, "uses_enrollments") and self.pipeline.model.config.uses_enrollments:
            change_state_of_sbs(self.pipeline.model.model, False)

        samples = self.pipeline._ensure_tensor_on_device(samples, device=self.pipeline.device)
        with torch.no_grad():
            model_output = self.pipeline._forward(samples, return_timestamps=True)
        decoded = self.pipeline.postprocess([model_output], return_timestamps=True)
        per_spk = decoded.get("per_spk_outputs", []) if isinstance(decoded, dict) else []
        if per_spk:
            return clean_transcript_text(per_spk[0])
        return clean_transcript_text(decoded.get("text", "") if isinstance(decoded, dict) else str(decoded))

    def transcribe(
        self,
        audio_path: str,
        enrollment_audio: str,
        presence_threshold: Optional[float] = None,
    ) -> Dict[str, object]:
        threshold = -float("inf") if presence_threshold is None else float(presence_threshold)
        enrollment_embedding = self.pipeline._compute_embedding_for_file(enrollment_audio)
        if enrollment_embedding is None:
            raise RuntimeError(f"Could not compute enrollment embedding for {enrollment_audio}")

        wav, sr = torchaudio.load(audio_path)
        wav = wav.mean(dim=0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        speech_timestamps = self.get_speech_timestamps(wav)
        chunks = build_chunks_from_speech_timestamps(
            speech_timestamps,
            total_samples=int(wav.numel()),
            sample_rate=self.sample_rate,
            min_chunk_sec=self.min_chunk_sec,
            max_chunk_sec=self.max_chunk_sec,
        )

        results: List[OnlineChunkResult] = []
        transcript_parts: List[str] = []

        with tempfile.TemporaryDirectory(prefix="dicow_online_chunks_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            for idx, chunk in enumerate(chunks):
                chunk_path = tmpdir_path / f"chunk_{idx:04d}.wav"
                self._write_chunk(wav, chunk, chunk_path)

                try:
                    diarization_output = self.pipeline.diarization_pipeline(str(chunk_path))
                    best = self._best_target_speaker(str(chunk_path), diarization_output, enrollment_embedding)
                except Exception as exc:
                    results.append(
                        OnlineChunkResult(
                            chunk_index=idx,
                            start_sec=chunk.start_sec(self.sample_rate),
                            end_sec=chunk.end_sec(self.sample_rate),
                            cut_reason=chunk.cut_reason,
                            accepted=False,
                            best_score=None,
                            best_speaker_index=None,
                            transcript="",
                            skip_reason=f"diarization_or_embedding_failed:{type(exc).__name__}:{exc}",
                        )
                    )
                    continue

                best_index = best["index"]
                best_score = best["score"]
                if best_index is None or best_score is None:
                    results.append(
                        OnlineChunkResult(
                            chunk_index=idx,
                            start_sec=chunk.start_sec(self.sample_rate),
                            end_sec=chunk.end_sec(self.sample_rate),
                            cut_reason=chunk.cut_reason,
                            accepted=False,
                            best_score=None,
                            best_speaker_index=None,
                            transcript="",
                            skip_reason="no_matching_speaker_embedding",
                        )
                    )
                    continue

                if best_score < threshold:
                    results.append(
                        OnlineChunkResult(
                            chunk_index=idx,
                            start_sec=chunk.start_sec(self.sample_rate),
                            end_sec=chunk.end_sec(self.sample_rate),
                            cut_reason=chunk.cut_reason,
                            accepted=False,
                            best_score=best_score,
                            best_speaker_index=int(best_index),
                            transcript="",
                            skip_reason="below_threshold",
                        )
                    )
                    continue

                try:
                    text = self._transcribe_selected_speaker(str(chunk_path), diarization_output, int(best_index))
                except Exception as exc:
                    results.append(
                        OnlineChunkResult(
                            chunk_index=idx,
                            start_sec=chunk.start_sec(self.sample_rate),
                            end_sec=chunk.end_sec(self.sample_rate),
                            cut_reason=chunk.cut_reason,
                            accepted=False,
                            best_score=best_score,
                            best_speaker_index=int(best_index),
                            transcript="",
                            skip_reason=f"asr_failed:{type(exc).__name__}:{exc}",
                        )
                    )
                    continue

                if text:
                    transcript_parts.append(text)
                results.append(
                    OnlineChunkResult(
                        chunk_index=idx,
                        start_sec=chunk.start_sec(self.sample_rate),
                        end_sec=chunk.end_sec(self.sample_rate),
                        cut_reason=chunk.cut_reason,
                        accepted=True,
                        best_score=best_score,
                        best_speaker_index=int(best_index),
                        transcript=text,
                    )
                )

        return {
            "text": clean_transcript_text(" ".join(transcript_parts)),
            "chunks": results,
            "num_vad_speech_segments": len(speech_timestamps),
        }


def transcript_from_chunks(chunks: Sequence[OnlineChunkResult], threshold: float) -> str:
    return clean_transcript_text(" ".join(c.transcript for c in chunks if c.best_score is not None and c.best_score >= threshold))

#!/usr/bin/env python3
"""Inference-only utilities for streaming STNO + DiCoW decoding."""
from __future__ import annotations

import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent
FRAME_HZ = 50
STNO_NAMES = ("S", "T", "N", "O")
WORD_RE = re.compile(r"[^a-z0-9']+")
TIMESTAMP_SEGMENT_RE = re.compile(r"<\|([0-9]+(?:\.[0-9]+)?)\|>(.*?)<\|([0-9]+(?:\.[0-9]+)?)\|>", re.S)


@dataclass
class Segment:
    start: float
    end: float
    text: str

    def to_json(self) -> dict[str, Any]:
        return {"start_time": self.start, "end_time": self.end, "text": self.text}


def normalize_words(text: str) -> str:
    lowered = str(text).strip().lower().replace('""', '"')
    lowered = WORD_RE.sub(" ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


# ---------------------------------------------------------------------------
# STNO post-processing helpers
# ---------------------------------------------------------------------------

def remove_short_positive_runs(mask: np.ndarray, min_frames: int) -> np.ndarray:
    if min_frames <= 1:
        return mask.astype(np.float32)
    out = mask.astype(bool).copy()
    starts = np.flatnonzero(np.diff(np.r_[False, out]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[out, False]) == -1)
    for start, end in zip(starts, ends):
        if end - start < min_frames:
            out[start:end] = False
    return out.astype(np.float32)


def fill_short_negative_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    if max_gap_frames <= 0:
        return mask.astype(np.float32)
    out = mask.astype(bool).copy()
    inverse = ~out
    starts = np.flatnonzero(np.diff(np.r_[False, inverse]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[inverse, False]) == -1)
    for start, end in zip(starts, ends):
        if 0 < start and end < len(out) and end - start <= max_gap_frames:
            out[start:end] = True
    return out.astype(np.float32)


def derive_stno_from_activity(target: np.ndarray, other: np.ndarray) -> np.ndarray:
    target = target.astype(np.float32)
    other = other.astype(np.float32)
    silence = (1.0 - target) * (1.0 - other)
    target_only = target * (1.0 - other)
    other_only = (1.0 - target) * other
    overlap = target * other
    return np.stack([silence, target_only, other_only, overlap], axis=0).astype(np.float32)


def threshold_stno_probs(
    probs: np.ndarray,
    *,
    target_head: int,
    other_head: int,
    target_threshold: float,
    other_threshold: float,
    min_target_on_s: float,
    fill_target_gap_s: float,
    suppress_other_when_target: bool,
) -> np.ndarray:
    target = (probs[:, target_head] >= target_threshold).astype(np.float32)
    other = (probs[:, other_head] >= other_threshold).astype(np.float32)
    target = fill_short_negative_gaps(target, int(round(fill_target_gap_s * FRAME_HZ)))
    target = remove_short_positive_runs(target, int(round(min_target_on_s * FRAME_HZ)))
    if suppress_other_when_target:
        other = other * (1.0 - target)
    return derive_stno_from_activity(target, other)


# ---------------------------------------------------------------------------
# Streaming STNO inference
# ---------------------------------------------------------------------------

def real_streaming_stno(
    *,
    mixed_audio: np.ndarray,
    sample_rate: int,
    enrollment_path: Path,
    pipeline: Any,
    chunk_s: float,
    lookback_s: float,
    target_head: int = 0,
    other_head: int = 1,
    target_threshold: float = 0.5,
    other_threshold: float = 0.5,
    min_target_on_s: float = 0.0,
    fill_target_gap_s: float = 0.0,
    suppress_other_when_target: bool = False,
) -> np.ndarray:
    result = pipeline.infer_streaming(
        mixed_audio=torch.from_numpy(mixed_audio).unsqueeze(0),
        reference_audio=enrollment_path.as_posix(),
        mixed_sample_rate=sample_rate,
        chunk_s=chunk_s,
        lookback_s=lookback_s,
    )
    probs = result["probs"].detach().cpu().numpy().astype(np.float32)
    stno = threshold_stno_probs(
        probs,
        target_head=target_head,
        other_head=other_head,
        target_threshold=target_threshold,
        other_threshold=other_threshold,
        min_target_on_s=min_target_on_s,
        fill_target_gap_s=fill_target_gap_s,
        suppress_other_when_target=suppress_other_when_target,
    )
    if stno.shape[0] != 4 and stno.shape[-1] == 4:
        stno = stno.T
    if stno.shape[0] != 4:
        raise ValueError(f"Expected real STNO shape [4, frames], got {stno.shape}")
    return stno


# ---------------------------------------------------------------------------
# DiCoW model loader
# ---------------------------------------------------------------------------

def load_dicow(model_id: str, device: torch.device, local_files_only: bool):
    sys.path.insert(0, REPO_ROOT.as_posix())
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_warning()
    from model.DiCoW.modeling_dicow import DiCoWForConditionalGeneration
    from transformers import GenerationConfig, WhisperFeatureExtractor, WhisperTokenizerFast
    from transformers.generation.utils import GenerationMixin

    hf_logging.set_verbosity_warning()
    try:
        model = DiCoWForConditionalGeneration.from_pretrained(model_id, local_files_only=local_files_only)
    except Exception:
        if local_files_only:
            model = DiCoWForConditionalGeneration.from_pretrained(model_id, local_files_only=False)
        else:
            raise
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo", local_files_only=False)
    tokenizer = WhisperTokenizerFast.from_pretrained(
        "openai/whisper-large-v3-turbo",
        predict_timestamps=True,
        local_files_only=False,
    )
    tokenizer.set_prefix_tokens(predict_timestamps=True, task="transcribe", language="en")
    model.set_tokenizer(tokenizer)
    model.generation_config = GenerationConfig.from_pretrained(
        "openai/whisper-large-v3-turbo",
        local_files_only=local_files_only,
    )
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.generation_config.return_timestamps = True
    model.generation_config._pad_token_tensor = torch.tensor(model.generation_config.pad_token_id, device=device)
    model.generation_config._eos_token_tensor = torch.tensor([model.generation_config.eos_token_id], device=device)
    model._sample = types.MethodType(GenerationMixin._sample, model)
    model.eval().to(device)
    return model, feature_extractor, tokenizer


# ---------------------------------------------------------------------------
# Windowed decode helpers
# ---------------------------------------------------------------------------

def stno_slice_for_window(stno: np.ndarray, start_s: float, end_s: float, expected_frames: int = 1500) -> torch.Tensor:
    start_frame = max(0, int(round(start_s * FRAME_HZ)))
    end_frame = max(start_frame, int(round(end_s * FRAME_HZ)))
    piece = stno[:, start_frame:end_frame]
    if piece.shape[-1] < expected_frames:
        pad = np.zeros((4, expected_frames - piece.shape[-1]), dtype=np.float32)
        pad[0, :] = 1.0
        piece = np.concatenate([piece, pad], axis=-1)
    else:
        piece = piece[:, :expected_frames]
    return torch.from_numpy(piece.astype(np.float32)).unsqueeze(0)


def audio_window(audio: np.ndarray, sample_rate: int, start_s: float, end_s: float, decode_window_s: float) -> np.ndarray:
    start_sample = max(0, int(round(start_s * sample_rate)))
    end_sample = max(start_sample, int(round(end_s * sample_rate)))
    window = audio[start_sample:end_sample]
    max_samples = int(round(decode_window_s * sample_rate))
    if window.shape[0] > max_samples:
        window = window[-max_samples:]
    return window.astype(np.float32)


def parse_generated_segments(output: Any, tokenizer: Any, window_start_s: float) -> tuple[list[Segment], str]:
    text = ""
    if isinstance(output, dict):
        if "segments" in output and output["segments"]:
            segments = []
            for seg in output["segments"][0] if isinstance(output["segments"], list) and output["segments"] and isinstance(output["segments"][0], list) else output["segments"]:
                if isinstance(seg, dict):
                    start = float(seg.get("start", seg.get("start_time", 0.0))) + window_start_s
                    end = float(seg.get("end", seg.get("end_time", start))) + window_start_s
                    tokens = seg.get("tokens")
                    body = tokenizer.decode(tokens, skip_special_tokens=True).strip() if tokens is not None else str(seg.get("text", "")).strip()
                    if body and end > start:
                        segments.append(Segment(start, end, body))
            if segments:
                return segments, " ".join(seg.text for seg in segments).strip()
        sequences = output.get("sequences")
        if sequences is not None:
            text = tokenizer.decode(sequences[0], skip_special_tokens=False)
    elif torch.is_tensor(output):
        token_ids = output[0].tolist()
        pieces: list[str] = []
        previous_had_text = False
        for token_id in token_ids:
            piece = tokenizer.decode([token_id], skip_special_tokens=False)
            if not piece:
                if previous_had_text:
                    pieces.append(' ')
                continue
            if piece.startswith('<|') and piece.endswith('|>'):
                pieces.append(piece)
                previous_had_text = False
                continue
            pieces.append(piece)
            previous_had_text = True
        text = ''.join(pieces)
    else:
        text = str(output)

    segments = []
    for match in TIMESTAMP_SEGMENT_RE.finditer(text):
        start = float(match.group(1)) + window_start_s
        end = float(match.group(3)) + window_start_s
        body = normalize_generated_text(match.group(2))
        if body and end > start:
            segments.append(Segment(start, end, body))
    if segments:
        return segments, " ".join(seg.text for seg in segments).strip()
    return [], normalize_generated_text(text)


def normalize_generated_text(text: str) -> str:
    text = re.sub(r"(\w)(<\|[^>]+?\|>)(\w)", r"\1 \3", str(text))
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    return " ".join(text.split()).strip()


def has_interval_overlap(candidate: Segment, committed: list[Segment], tolerance_s: float = 0.05) -> bool:
    for segment in committed:
        overlap = min(candidate.end, segment.end) - max(candidate.start, segment.start)
        if overlap > tolerance_s:
            return True
    return False


def decode_window(
    *,
    model: Any,
    feature_extractor: Any,
    tokenizer: Any,
    audio: np.ndarray,
    sample_rate: int,
    stno: np.ndarray,
    window_start_s: float,
    window_end_s: float,
    decode_window_s: float,
    device: torch.device,
) -> tuple[list[Segment], str]:
    samples = audio_window(audio, sample_rate, window_start_s, window_end_s, decode_window_s)
    batch = feature_extractor(
        samples,
        sampling_rate=sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
        padding="max_length",
        truncation=True,
        max_length=int(round(decode_window_s * sample_rate)),
    )
    input_features = batch["input_features"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        hop_length = int(getattr(feature_extractor, "hop_length", 160))
        num_frames = int(input_features.shape[-1])
        valid_frames = min(num_frames, int(np.ceil(samples.shape[0] / max(1, hop_length))))
        attention_mask = torch.zeros((1, num_frames), dtype=torch.long)
        attention_mask[:, :valid_frames] = 1
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    stno_mask = stno_slice_for_window(stno, window_start_s, window_end_s).to(device)
    with torch.inference_mode():
        output = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            stno_mask=stno_mask,
            return_timestamps=True,
            max_new_tokens=128,
            num_beams=1,
            do_sample=False,
        )
    return parse_generated_segments(output, tokenizer, window_start_s)

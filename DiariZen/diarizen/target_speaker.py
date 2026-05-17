from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

from diarizen.stno_utils import derive_stno_from_binary, estimate_pos_weight


Interval = Tuple[float, float]


@dataclass(frozen=True)
class SegmentRecord:
    session: str
    speaker: str
    start: float
    end: float


def load_scp(scp_file: str) -> Dict[str, str]:
    lines = [line.strip().split(None, 1) for line in open(scp_file)]
    return {item[0]: item[1] for item in lines}


def load_uem(uem_file: str) -> Dict[str, Tuple[float, float]]:
    lines = [line.strip().split() for line in open(uem_file)]
    return {item[0]: (float(item[-2]), float(item[-1])) for item in lines}


def load_rttm_segments(rttm_file: str) -> Dict[str, Dict[str, List[Interval]]]:
    sessions: Dict[str, Dict[str, List[Interval]]] = {}
    with open(rttm_file, "r") as handle:
        for raw_line in handle:
            fields = raw_line.strip().split()
            if len(fields) < 8:
                continue
            session = fields[1]
            start = float(fields[3])
            end = start + float(fields[4])
            speaker = fields[-2] if fields[-2] != "<NA>" else fields[-3]
            sessions.setdefault(session, {}).setdefault(speaker, []).append((start, end))

    for speakers in sessions.values():
        for speaker, intervals in speakers.items():
            speakers[speaker] = merge_intervals(intervals)
    return sessions


def merge_intervals(intervals: Sequence[Interval], min_duration: float = 0.0) -> List[Interval]:
    valid = sorted((float(start), float(end)) for start, end in intervals if end - start > 0.0)
    if not valid:
        return []

    merged: List[List[float]] = [[valid[0][0], valid[0][1]]]
    for start, end in valid[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged if end - start >= min_duration]


def intersect_intervals(intervals: Sequence[Interval], region: Interval) -> List[Interval]:
    region_start, region_end = region
    output = []
    for start, end in intervals:
        clipped_start = max(start, region_start)
        clipped_end = min(end, region_end)
        if clipped_end > clipped_start:
            output.append((clipped_start, clipped_end))
    return output


def subtract_intervals(intervals: Sequence[Interval], subtractors: Sequence[Interval]) -> List[Interval]:
    if not intervals:
        return []
    if not subtractors:
        return list(intervals)

    subtractors = merge_intervals(subtractors)
    output: List[Interval] = []
    for start, end in intervals:
        cursor = start
        for sub_start, sub_end in subtractors:
            if sub_end <= cursor:
                continue
            if sub_start >= end:
                break
            if sub_start > cursor:
                output.append((cursor, min(sub_start, end)))
            cursor = max(cursor, sub_end)
            if cursor >= end:
                break
        if cursor < end:
            output.append((cursor, end))
    return merge_intervals(output)


def intervals_duration(intervals: Sequence[Interval]) -> float:
    return float(sum(end - start for start, end in intervals))


def trim_intervals(intervals: Sequence[Interval], max_duration: Optional[float]) -> List[Interval]:
    if max_duration is None:
        return list(intervals)
    remaining = float(max_duration)
    if remaining <= 0.0:
        return []

    trimmed = []
    for start, end in intervals:
        if remaining <= 0.0:
            break
        duration = end - start
        kept = min(duration, remaining)
        trimmed.append((start, start + kept))
        remaining -= kept
    return trimmed


def filter_intervals_by_min_duration(
    intervals: Sequence[Interval],
    segment_min_duration: float = 0.0,
) -> List[Interval]:
    threshold = float(segment_min_duration)
    return [(start, end) for start, end in intervals if end - start >= threshold]


def reorder_intervals(
    intervals: Sequence[Interval],
    selection_strategy: str = "all",
    max_segments: Optional[int] = None,
) -> List[Interval]:
    ordered = list(intervals)
    if selection_strategy == "all":
        pass
    elif selection_strategy in {"longest_first", "topk_longest"}:
        ordered = sorted(ordered, key=lambda item: (-(item[1] - item[0]), item[0], item[1]))
    else:
        raise ValueError(f"Unsupported enrollment selection strategy: {selection_strategy}")

    if selection_strategy == "topk_longest" and max_segments is None:
        raise ValueError("`max_segments` must be set when `selection_strategy='topk_longest'`.")
    if max_segments is not None:
        ordered = ordered[:max_segments]
    return ordered


def finalize_enrollment_intervals(
    candidate_intervals: Sequence[Interval],
    min_duration: float,
    max_duration: Optional[float] = None,
    selection_strategy: str = "all",
    segment_min_duration: float = 0.0,
    max_segments: Optional[int] = None,
) -> List[Interval]:
    selected = merge_intervals(candidate_intervals)
    selected = filter_intervals_by_min_duration(selected, segment_min_duration=segment_min_duration)
    selected = reorder_intervals(
        selected,
        selection_strategy=selection_strategy,
        max_segments=max_segments,
    )
    selected = trim_intervals(selected, max_duration)
    if intervals_duration(selected) < min_duration:
        return []
    return selected


def build_target_only_intervals(
    session_segments: Dict[str, List[Interval]],
    target_speaker: str,
) -> List[Interval]:
    target = session_segments.get(target_speaker, [])
    others = []
    for speaker, intervals in session_segments.items():
        if speaker == target_speaker:
            continue
        others.extend(intervals)
    return subtract_intervals(target, others)


def select_enrollment_intervals(
    target_only_intervals: Sequence[Interval],
    chunk_region: Interval,
    min_duration: float,
    max_duration: Optional[float] = None,
    selection_strategy: str = "all",
    segment_min_duration: float = 0.0,
    max_segments: Optional[int] = None,
) -> List[Interval]:
    outside = subtract_intervals(target_only_intervals, [chunk_region])
    selected = list(outside)

    if intervals_duration(selected) < min_duration:
        inside = intersect_intervals(target_only_intervals, chunk_region)
        selected.extend(inside)

    return finalize_enrollment_intervals(
        selected,
        min_duration=min_duration,
        max_duration=max_duration,
        selection_strategy=selection_strategy,
        segment_min_duration=segment_min_duration,
        max_segments=max_segments,
    )


def select_full_session_enrollment_intervals(
    target_only_intervals: Sequence[Interval],
    min_duration: float,
    max_duration: Optional[float] = None,
    selection_strategy: str = "all",
    segment_min_duration: float = 0.0,
    max_segments: Optional[int] = None,
) -> List[Interval]:
    return finalize_enrollment_intervals(
        target_only_intervals,
        min_duration=min_duration,
        max_duration=max_duration,
        selection_strategy=selection_strategy,
        segment_min_duration=segment_min_duration,
        max_segments=max_segments,
    )


def build_binary_frame_labels(
    session_segments: Dict[str, List[Interval]],
    target_speaker: str,
    chunk_region: Interval,
    num_frames: int,
    frame_duration: float,
    frame_step: float,
) -> torch.Tensor:
    chunk_start, chunk_end = chunk_region
    half = 0.5 * frame_duration

    target = np.zeros(num_frames, dtype=np.float32)
    other = np.zeros(num_frames, dtype=np.float32)

    def _paint(intervals: Sequence[Interval], output: np.ndarray) -> None:
        chunked = intersect_intervals(intervals, chunk_region)
        for start, end in chunked:
            start_idx = max(0, int(np.round((start - chunk_start - half) / frame_step)))
            end_idx = int(np.round((end - chunk_start - half) / frame_step))
            end_idx = min(num_frames - 1, end_idx)
            if end_idx >= start_idx:
                output[start_idx : end_idx + 1] = 1.0

    _paint(session_segments.get(target_speaker, []), target)
    for speaker, intervals in session_segments.items():
        if speaker == target_speaker:
            continue
        _paint(intervals, other)

    return torch.from_numpy(np.stack([target, other], axis=-1))


def derive_stno_labels(binary_targets: torch.Tensor) -> torch.Tensor:
    return derive_stno_from_binary(binary_targets[..., 0], binary_targets[..., 1]).float()


def make_cache_key(session_id: str, speaker_id: str) -> str:
    safe_session = session_id.replace("/", "__")
    safe_speaker = speaker_id.replace("/", "__")
    return f"{safe_session}__{safe_speaker}"


def cache_entry_path(cache_root: str | Path, split: str, session_id: str, speaker_id: str) -> Path:
    return Path(cache_root) / split / session_id / f"{speaker_id}.pt"


def save_cached_enrollment(
    cache_root: str | Path,
    split: str,
    session_id: str,
    speaker_id: str,
    embedding: torch.Tensor,
    enrollment_duration: float,
    intervals: Sequence[Interval],
    embedding_model: Optional[str] = None,
    sample_rate: Optional[int] = None,
    enrollment_options: Optional[Dict[str, object]] = None,
) -> Path:
    path = cache_entry_path(cache_root, split, session_id, speaker_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": split,
        "session_id": session_id,
        "speaker_id": speaker_id,
        "embedding": embedding.detach().cpu().float(),
        "enrollment_duration": float(enrollment_duration),
        "intervals": [(float(start), float(end)) for start, end in intervals],
        "embedding_model": embedding_model,
        "sample_rate": sample_rate,
        "enrollment_options": dict(enrollment_options or {}),
    }
    torch.save(payload, path)
    return path


def load_cached_enrollment(
    cache_root: str | Path,
    split: str,
    session_id: str,
    speaker_id: str,
) -> Dict[str, object]:
    path = cache_entry_path(cache_root, split, session_id, speaker_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing enrollment cache entry: {path}")
    payload = torch.load(path, map_location="cpu")
    if "embedding" not in payload:
        raise ValueError(f"Invalid enrollment cache payload: {path}")
    return payload


def estimate_head_pos_weights_from_examples(
    examples: Sequence[dict],
    sessions: Dict[str, Dict[str, List[Interval]]],
    num_frames: int,
    frame_duration: float,
    frame_step: float,
) -> Dict[str, float]:
    target_sum = 0
    other_sum = 0
    total = 0

    for example in examples:
        labels = build_binary_frame_labels(
            sessions[example["session"]],
            example["target_speaker"],
            (example["chunk_start"], example["chunk_end"]),
            num_frames=num_frames,
            frame_duration=frame_duration,
            frame_step=frame_step,
        )
        target_sum += int(labels[:, 0].sum().item())
        other_sum += int(labels[:, 1].sum().item())
        total += int(labels.shape[0])

    return {
        "target": estimate_pos_weight(target_sum, total),
        "other": estimate_pos_weight(other_sum, total),
    }


class SpeakerEmbeddingExtractor:
    def __init__(
        self,
        embedding_model: str,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        self.extractor = PretrainedSpeakerEmbedding(embedding_model, device=self.device)

    @property
    def dimension(self) -> int:
        return int(self.extractor.dimension)

    @property
    def sample_rate(self) -> int:
        return int(self.extractor.sample_rate)

    @staticmethod
    def normalize_embedding(embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(embedding.float(), dim=0)

    def embed_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform[:1].float()
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        embedding = self.extractor(waveform.unsqueeze(0), masks=None)
        if np.isnan(embedding).any():
            raise ValueError("Enrollment embedding extraction returned NaN.")
        return torch.from_numpy(embedding[0]).float()

    def embed_segments(
        self,
        wav_path: str,
        intervals: Sequence[Interval],
        sample_rate: int,
        aggregation: str = "concat",
        l2_normalize: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        if not intervals:
            raise ValueError("No intervals were provided for enrollment extraction.")
        chunks = []
        for start, end in intervals:
            start_sample = int(round(start * sample_rate))
            end_sample = int(round(end * sample_rate))
            waveform, loaded_sr = sf.read(wav_path, start=start_sample, stop=end_sample, always_2d=True)
            if loaded_sr != sample_rate:
                raise ValueError(f"Unexpected sample rate: {loaded_sr} != {sample_rate}")
            chunk = torch.from_numpy(waveform.T).float()
            chunks.append(chunk[:1])
        duration = sum(chunk.shape[-1] for chunk in chunks) / float(sample_rate)

        if aggregation == "concat":
            waveform = torch.cat(chunks, dim=-1)
            embedding = self.embed_waveform(waveform, sample_rate)
        elif aggregation == "mean":
            embeddings = [self.normalize_embedding(self.embed_waveform(chunk, sample_rate)) for chunk in chunks]
            embedding = torch.stack(embeddings, dim=0).mean(dim=0)
        else:
            raise ValueError(f"Unsupported embedding aggregation mode: {aggregation}")

        if l2_normalize:
            embedding = self.normalize_embedding(embedding)
        return embedding, duration

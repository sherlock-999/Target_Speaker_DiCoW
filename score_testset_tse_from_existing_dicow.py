#!/usr/bin/env python3
import json
import re
from collections import defaultdict
from pathlib import Path

from evaluate_testset_tse_dicow_offline import (
    compute_der,
    parse_textgrid,
    prepare_metrics,
    prepare_normalized_metrics,
    read_jsonl_rows,
    write_json,
    write_jsonl,
)


def load_hypothesis_map(path: Path):
    by_session = defaultdict(list)
    for row in read_jsonl_rows(path):
        by_session[row["session_id"]].append(row)
    return by_session


def map_ami_session(stem: str):
    m = re.match(r"^sdm_([A-Z]{2}\d+[a-z])-\d+$", stem)
    return m.group(1) if m else None


def map_nsf_session(stem: str):
    m = re.match(r"^sdm_(MTG_\d+)_(sc_[a-z]+)_([0-9]+)-[0-9]+$", stem)
    if not m:
        return None
    return f"{m.group(1)}/{m.group(2)}_{m.group(3)}"


def map_l2m_session(stem: str):
    return stem


def build_dataset(
    name: str,
    root: Path,
    hyp_map: dict,
    map_fn,
    out_dir: Path,
    collar: int = 5,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_multi = []
    ref_wer = []
    hyp_multi = []
    hyp_wer = []
    missing = []

    wav_files = sorted((root / "audio").glob("*.wav"))
    for wav in wav_files:
        stem = wav.stem
        mapped = map_fn(stem)
        if not mapped:
            missing.append({"file": stem, "reason": "mapping_failed"})
            continue

        tg = root / "textgrid" / f"{stem}.TextGrid"
        if not tg.exists():
            missing.append({"file": stem, "session": mapped, "reason": "textgrid_missing"})
            continue

        ref_segments = parse_textgrid(tg)
        if not ref_segments:
            missing.append({"file": stem, "session": mapped, "reason": "empty_reference"})
            continue

        hyp_segments = hyp_map.get(mapped, [])
        if not hyp_segments:
            missing.append({"file": stem, "session": mapped, "reason": "hypothesis_missing"})
            continue

        for i, seg in enumerate(ref_segments):
            ref_multi.append(
                {
                    "session_id": mapped,
                    "speaker": seg["speaker"],
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "words": seg["words"],
                    "segment_index": i,
                }
            )
            ref_wer.append(
                {
                    "session_id": mapped,
                    "speaker": "single",
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "words": seg["words"],
                    "segment_index": i,
                }
            )

        hyp_segs_sorted = sorted(hyp_segments, key=lambda x: (x.get("start_time", 0.0), x.get("end_time", 0.0)))
        for j, row in enumerate(hyp_segs_sorted):
            item = dict(row)
            item["session_id"] = mapped
            item["segment_index"] = j
            hyp_multi.append(item)
            hyp_wer.append(
                {
                    "session_id": mapped,
                    "speaker": "single",
                    "start_time": item.get("start_time", 0.0),
                    "end_time": item.get("end_time", 0.0),
                    "words": item.get("words", ""),
                    "segment_index": j,
                }
            )

    write_jsonl(out_dir / "reference_multi.jsonl", ref_multi)
    write_jsonl(out_dir / "reference_wer.jsonl", ref_wer)
    write_jsonl(out_dir / "hypothesis_multi.jsonl", hyp_multi)
    write_jsonl(out_dir / "hypothesis_wer.jsonl", hyp_wer)
    write_json(out_dir / "missing_sessions.json", missing)

    prepare_metrics(out_dir, collar=collar)
    norm_dir = out_dir / "normalized_eval"
    prepare_normalized_metrics(out_dir, norm_dir, collar=collar)
    der = compute_der(ref_multi, hyp_multi)
    write_json(out_dir / "der_summary.json", der)

    metrics = {
        "wer": json.loads((norm_dir / "wer_average.norm.json").read_text())["error_rate"],
        "cpwer": json.loads((norm_dir / "cpwer_average.norm.json").read_text())["error_rate"],
        "tcpwer": json.loads((norm_dir / "tcpwer_average.norm.json").read_text())["error_rate"],
        "tcorcwer": json.loads((norm_dir / "tcorcwer_average.norm.json").read_text())["error_rate"],
        "DER": der["rates"]["DER"],
    }
    summary = {
        "dataset": name,
        "scored_files": len({r["session_id"] for r in ref_multi}),
        "missing_files": len(missing),
        "normalized_metrics": metrics,
    }
    write_json(out_dir / "run_summary.json", summary)
    return summary


def main():
    testset_root = Path("/home3/adnan/DICOW/mt-asr-data-prep/testset_tse")
    out_root = Path("output/testset_tse_dicow_from_existing")
    out_root.mkdir(parents=True, exist_ok=True)

    ami_hyp = load_hypothesis_map(Path("output/ami_dicow_offline/hypothesis_multi.jsonl"))
    nsf_hyp = load_hypothesis_map(Path("output/notsofar/mtg_sc160_dicow_offline/hypothesis_multi.jsonl"))

    l2m_hyp = defaultdict(list)
    for shard in sorted(Path("output/libri2mix_dicow_sharded").glob("shard_*/hypothesis_multi.jsonl")):
        for row in read_jsonl_rows(shard):
            l2m_hyp[row["session_id"]].append(row)

    all_summaries = {
        "ami": build_dataset("ami", testset_root / "ami", ami_hyp, map_ami_session, out_root / "ami"),
        "nsf": build_dataset("nsf", testset_root / "nsf", nsf_hyp, map_nsf_session, out_root / "nsf"),
        "l2m": build_dataset("l2m", testset_root / "l2m", l2m_hyp, map_l2m_session, out_root / "l2m"),
    }
    write_json(out_root / "metric_summary_normalized.json", all_summaries)
    print(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from evaluate_testset_tse_dicow_offline import (
    EnglishTextNormalizer,
    aggregate_rows_by_session_speaker,
    compute_der,
    normalize_rows,
    parse_textgrid,
    read_jsonl_rows,
    write_json,
    write_jsonl,
    write_seglst_json,
)


def run_metric(cmd):
    cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if cp.stdout.strip():
        print(cp.stdout.strip())
    if cp.stderr.strip():
        print(cp.stderr.strip(), file=sys.stderr)


def prepare_metrics_no_tcorc(raw_dir: Path, collar: int):
    py = sys.executable
    ref_multi_rows = read_jsonl_rows(raw_dir / "reference_multi.jsonl")
    hyp_multi_rows = read_jsonl_rows(raw_dir / "hypothesis_multi.jsonl")
    ref_wer_rows = read_jsonl_rows(raw_dir / "reference_wer.jsonl")
    hyp_wer_rows = read_jsonl_rows(raw_dir / "hypothesis_wer.jsonl")

    ref_multi_seglst = raw_dir / "reference_multi.seglst.json"
    hyp_multi_seglst = raw_dir / "hypothesis_multi.seglst.json"
    ref_wer_agg_seglst = raw_dir / "reference_wer_agg.seglst.json"
    hyp_wer_agg_seglst = raw_dir / "hypothesis_wer_agg.seglst.json"
    ref_multi_agg_seglst = raw_dir / "reference_multi_agg.seglst.json"
    hyp_multi_agg_seglst = raw_dir / "hypothesis_multi_agg.seglst.json"

    write_seglst_json(ref_multi_seglst, ref_multi_rows)
    write_seglst_json(hyp_multi_seglst, hyp_multi_rows)
    write_seglst_json(ref_wer_agg_seglst, aggregate_rows_by_session_speaker(ref_wer_rows))
    write_seglst_json(hyp_wer_agg_seglst, aggregate_rows_by_session_speaker(hyp_wer_rows))
    write_seglst_json(ref_multi_agg_seglst, aggregate_rows_by_session_speaker(ref_multi_rows))
    write_seglst_json(hyp_multi_agg_seglst, aggregate_rows_by_session_speaker(hyp_multi_rows))

    run_metric(
        [
            py,
            "-m",
            "meeteval.wer",
            "wer",
            "-r",
            str(ref_wer_agg_seglst),
            "-h",
            str(hyp_wer_agg_seglst),
            "--average-out",
            str(raw_dir / "wer_average.json"),
            "--per-reco-out",
            str(raw_dir / "wer_per_reco.json"),
        ]
    )
    run_metric(
        [
            py,
            "-m",
            "meeteval.wer",
            "cpwer",
            "-r",
            str(ref_multi_agg_seglst),
            "-h",
            str(hyp_multi_agg_seglst),
            "--average-out",
            str(raw_dir / "cpwer_average.json"),
            "--per-reco-out",
            str(raw_dir / "cpwer_per_reco.json"),
        ]
    )
    run_metric(
        [
            py,
            "-m",
            "meeteval.wer",
            "tcpwer",
            "-r",
            str(ref_multi_seglst),
            "-h",
            str(hyp_multi_seglst),
            "--collar",
            str(collar),
            "--average-out",
            str(raw_dir / "tcpwer_average.json"),
            "--per-reco-out",
            str(raw_dir / "tcpwer_per_reco.json"),
        ]
    )


def prepare_normalized_metrics_no_tcorc(raw_dir: Path, norm_dir: Path, collar: int):
    normalizer = EnglishTextNormalizer(
        standardize_numbers=False,
        standardize_numbers_rev=True,
        remove_fillers=True,
    )

    ref_multi_rows = read_jsonl_rows(raw_dir / "reference_multi.jsonl")
    hyp_multi_rows = read_jsonl_rows(raw_dir / "hypothesis_multi.jsonl")
    ref_wer_rows = read_jsonl_rows(raw_dir / "reference_wer.jsonl")
    hyp_wer_rows = read_jsonl_rows(raw_dir / "hypothesis_wer.jsonl")

    ref_multi_norm = normalize_rows(ref_multi_rows, normalizer)
    hyp_multi_norm = normalize_rows(hyp_multi_rows, normalizer)
    ref_wer_norm = normalize_rows(ref_wer_rows, normalizer)
    hyp_wer_norm = normalize_rows(hyp_wer_rows, normalizer)

    norm_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(norm_dir / "reference_multi.norm.jsonl", ref_multi_norm)
    write_jsonl(norm_dir / "hypothesis_multi.norm.jsonl", hyp_multi_norm)
    write_jsonl(norm_dir / "reference_wer.norm.jsonl", ref_wer_norm)
    write_jsonl(norm_dir / "hypothesis_wer.norm.jsonl", hyp_wer_norm)

    py = sys.executable
    ref_multi_seglst = norm_dir / "reference_multi.norm.seglst.json"
    hyp_multi_seglst = norm_dir / "hypothesis_multi.norm.seglst.json"
    ref_wer_agg_seglst = norm_dir / "reference_wer_agg.norm.seglst.json"
    hyp_wer_agg_seglst = norm_dir / "hypothesis_wer_agg.norm.seglst.json"
    ref_multi_agg_seglst = norm_dir / "reference_multi_agg.norm.seglst.json"
    hyp_multi_agg_seglst = norm_dir / "hypothesis_multi_agg.norm.seglst.json"

    write_seglst_json(ref_multi_seglst, ref_multi_norm)
    write_seglst_json(hyp_multi_seglst, hyp_multi_norm)
    write_seglst_json(ref_wer_agg_seglst, aggregate_rows_by_session_speaker(ref_wer_norm))
    write_seglst_json(hyp_wer_agg_seglst, aggregate_rows_by_session_speaker(hyp_wer_norm))
    write_seglst_json(ref_multi_agg_seglst, aggregate_rows_by_session_speaker(ref_multi_norm))
    write_seglst_json(hyp_multi_agg_seglst, aggregate_rows_by_session_speaker(hyp_multi_norm))

    run_metric(
        [
            py,
            "-m",
            "meeteval.wer",
            "wer",
            "-r",
            str(ref_wer_agg_seglst),
            "-h",
            str(hyp_wer_agg_seglst),
            "--average-out",
            str(norm_dir / "wer_average.norm.json"),
            "--per-reco-out",
            str(norm_dir / "wer_per_reco.norm.json"),
        ]
    )
    run_metric(
        [
            py,
            "-m",
            "meeteval.wer",
            "cpwer",
            "-r",
            str(ref_multi_agg_seglst),
            "-h",
            str(hyp_multi_agg_seglst),
            "--average-out",
            str(norm_dir / "cpwer_average.norm.json"),
            "--per-reco-out",
            str(norm_dir / "cpwer_per_reco.norm.json"),
        ]
    )
    run_metric(
        [
            py,
            "-m",
            "meeteval.wer",
            "tcpwer",
            "-r",
            str(ref_multi_seglst),
            "-h",
            str(hyp_multi_seglst),
            "--collar",
            str(collar),
            "--average-out",
            str(norm_dir / "tcpwer_average.norm.json"),
            "--per-reco-out",
            str(norm_dir / "tcpwer_per_reco.norm.json"),
        ]
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
    ref_multi, ref_wer, hyp_multi, hyp_wer = [], [], [], []
    missing = []

    wav_files = sorted((root / "audio").glob("*.wav"))
    for wav in wav_files:
        stem = wav.stem
        session_id = map_fn(stem)
        if not session_id:
            missing.append({"file": stem, "reason": "mapping_failed"})
            continue
        tg = root / "textgrid" / f"{stem}.TextGrid"
        if not tg.exists():
            missing.append({"file": stem, "session": session_id, "reason": "textgrid_missing"})
            continue
        ref_segments = parse_textgrid(tg)
        if not ref_segments:
            missing.append({"file": stem, "session": session_id, "reason": "empty_reference"})
            continue
        hyp_segments = hyp_map.get(session_id, [])
        if not hyp_segments:
            missing.append({"file": stem, "session": session_id, "reason": "hypothesis_missing"})
            continue

        for i, seg in enumerate(ref_segments):
            ref_multi.append(
                {
                    "session_id": session_id,
                    "speaker": seg["speaker"],
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "words": seg["words"],
                    "segment_index": i,
                }
            )
            ref_wer.append(
                {
                    "session_id": session_id,
                    "speaker": "single",
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "words": seg["words"],
                    "segment_index": i,
                }
            )

        hyp_sorted = sorted(hyp_segments, key=lambda x: (x.get("start_time", 0.0), x.get("end_time", 0.0)))
        for j, row in enumerate(hyp_sorted):
            item = dict(row)
            item["session_id"] = session_id
            item["segment_index"] = j
            hyp_multi.append(item)
            hyp_wer.append(
                {
                    "session_id": session_id,
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

    prepare_metrics_no_tcorc(out_dir, collar=collar)
    norm_dir = out_dir / "normalized_eval"
    prepare_normalized_metrics_no_tcorc(out_dir, norm_dir, collar=collar)

    der = compute_der(ref_multi, hyp_multi)
    write_json(out_dir / "der_summary.json", der)
    summary = {
        "dataset": name,
        "scored_files": len({r["session_id"] for r in ref_multi}),
        "missing_files": len(missing),
        "normalized_metrics": {
            "wer": json.loads((norm_dir / "wer_average.norm.json").read_text())["error_rate"],
            "cpwer": json.loads((norm_dir / "cpwer_average.norm.json").read_text())["error_rate"],
            "tcpwer": json.loads((norm_dir / "tcpwer_average.norm.json").read_text())["error_rate"],
            "DER": der["rates"]["DER"],
        },
    }
    write_json(out_dir / "run_summary.json", summary)
    return summary


def main():
    testset_root = Path("/home3/adnan/DICOW/mt-asr-data-prep/testset_tse")
    out_root = Path("output/testset_tse_dicow_no_tcorc")
    out_root.mkdir(parents=True, exist_ok=True)

    ami_hyp = load_hypothesis_map(Path("output/ami_dicow_offline/hypothesis_multi.jsonl"))
    nsf_hyp = load_hypothesis_map(Path("output/notsofar/mtg_sc160_dicow_offline/hypothesis_multi.jsonl"))
    l2m_hyp = defaultdict(list)
    for shard in sorted(Path("output/libri2mix_dicow_sharded").glob("shard_*/hypothesis_multi.jsonl")):
        for row in read_jsonl_rows(shard):
            l2m_hyp[row["session_id"]].append(row)

    summaries = {
        "ami": build_dataset("ami", testset_root / "ami", ami_hyp, map_ami_session, out_root / "ami"),
        "nsf": build_dataset("nsf", testset_root / "nsf", nsf_hyp, map_nsf_session, out_root / "nsf"),
        "l2m": build_dataset("l2m", testset_root / "l2m", l2m_hyp, map_l2m_session, out_root / "l2m"),
    }
    write_json(out_root / "metric_summary_normalized.json", summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

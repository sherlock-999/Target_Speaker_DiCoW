#!/usr/bin/env python3
import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

from run_vibevoice_notsofar_eval import prepare_metrics, prepare_normalized_metrics


NON_SPEECH_RE = re.compile(r"^\s*\[[^\]]+\]\s*$")


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def is_nonspeech(text: str) -> bool:
    return bool(NON_SPEECH_RE.match((text or "").strip()))


def clean_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_dialogue_blob(text: str):
    text = str(text or "")
    # Supports lowercase/uppercase keys and slight formatting errors near speaker key.
    pattern = re.compile(
        r'\{\s*"(?:start|Start)"\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*"(?:end|End)"\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:,\s*"?(?:speaker|Speaker)"?\s*:\s*([^,}\"]+|"[^\"]*")\s*)?,\s*"(?:content|Content)"\s*:\s*"((?:[^"\\]|\\.)*)"',
        re.S,
    )
    out = []
    for m in pattern.finditer(text):
        st = float(m.group(1))
        et = float(m.group(2))
        spk_raw = (m.group(3) or "0").strip()
        if spk_raw.startswith('"') and spk_raw.endswith('"') and len(spk_raw) >= 2:
            spk_raw = spk_raw[1:-1]
        content_raw = m.group(4)
        try:
            content = json.loads(f'"{content_raw}"')
        except Exception:
            content = content_raw.replace('\\"', '"')
        out.append(
            {
                "start_time": st,
                "end_time": et,
                "speaker": f"hyp_spk{spk_raw}",
                "words": clean_text(content),
            }
        )
    return out


def ensure_strict_duration(start_time: float, end_time: float, eps: float = 0.01):
    st = float(start_time)
    et = float(end_time)
    if et <= st:
        et = st + eps
    return st, et


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("output/vibevoice_notsofar_offline"))
    parser.add_argument("--collar", type=int, default=5)
    args = parser.parse_args()

    out_dir = args.output_dir
    hyp_multi_path = out_dir / "hypothesis_multi.jsonl"
    hyp_wer_path = out_dir / "hypothesis_wer.jsonl"

    if not hyp_multi_path.exists() or not hyp_wer_path.exists():
        raise FileNotFoundError("hypothesis files are missing")

    # Backup originals once.
    for p in [hyp_multi_path, hyp_wer_path]:
        b = p.with_suffix(p.suffix + ".orig")
        if not b.exists():
            shutil.copy2(p, b)

    src_multi = hyp_multi_path.with_suffix(hyp_multi_path.suffix + ".orig")
    rows = read_jsonl(src_multi if src_multi.exists() else hyp_multi_path)

    cleaned_multi = []
    dropped_nonspeech = 0
    expanded_blob_rows = 0
    blob_segments_kept = 0

    for row in rows:
        base = {
            "session_id": row["session_id"],
            "speaker": row.get("speaker", "hyp_spk0"),
            "start_time": float(row.get("start_time", 0.0)),
            "end_time": float(row.get("end_time", 0.0)),
            "words": clean_text(row.get("words", "")),
        }

        if "assistant [{" in base["words"] or base["words"].startswith('[{"start"') or base["words"].startswith("[{\"start\""):
            segs = parse_dialogue_blob(base["words"])
            expanded_blob_rows += 1
            for seg in segs:
                if not seg["words"] or is_nonspeech(seg["words"]):
                    dropped_nonspeech += 1
                    continue
                st, et = ensure_strict_duration(seg["start_time"], seg["end_time"])
                cleaned_multi.append(
                    {
                        "session_id": base["session_id"],
                        "speaker": seg["speaker"],
                        "start_time": st,
                        "end_time": et,
                        "words": seg["words"],
                    }
                )
                blob_segments_kept += 1
            continue

        if not base["words"] or is_nonspeech(base["words"]):
            dropped_nonspeech += 1
            continue
        st, et = ensure_strict_duration(base["start_time"], base["end_time"])
        base["start_time"] = st
        base["end_time"] = et
        cleaned_multi.append(base)

    # Re-index segment_index after sort for determinism.
    cleaned_multi.sort(key=lambda r: (r["session_id"], r["start_time"], r["end_time"], r["speaker"]))
    out_multi = []
    c = defaultdict(int)
    for r in cleaned_multi:
        key = r["session_id"]
        out_multi.append(
            {
                "session_id": r["session_id"],
                "speaker": r["speaker"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "words": r["words"],
                "segment_index": c[key],
            }
        )
        c[key] += 1

    # Build single-speaker rows from cleaned_multi.
    by_session = defaultdict(list)
    for r in out_multi:
        by_session[r["session_id"]].append(r)

    out_wer = []
    for session_id, segs in by_session.items():
        segs.sort(key=lambda x: (x["start_time"], x["end_time"]))
        for i, s in enumerate(segs):
            out_wer.append(
                {
                    "session_id": session_id,
                    "speaker": "single",
                    "start_time": s["start_time"],
                    "end_time": s["end_time"],
                    "words": s["words"],
                    "segment_index": i,
                }
            )

    write_jsonl(hyp_multi_path, out_multi)
    write_jsonl(hyp_wer_path, out_wer)

    # Regenerate metrics from cleaned hypotheses.
    prepare_metrics(out_dir, collar=args.collar)
    norm_dir = out_dir / "normalized_eval"
    prepare_normalized_metrics(out_dir, norm_dir, collar=args.collar)

    # Refresh summaries.
    raw = {
        "wer": json.loads((out_dir / "wer_average.json").read_text(encoding="utf-8")),
        "cpwer": json.loads((out_dir / "cpwer_average.json").read_text(encoding="utf-8")),
        "tcpwer": json.loads((out_dir / "tcpwer_average.json").read_text(encoding="utf-8")),
        "tcorcwer": json.loads((out_dir / "tcorcwer_average.json").read_text(encoding="utf-8")),
    }
    norm = {
        "wer": json.loads((norm_dir / "wer_average.norm.json").read_text(encoding="utf-8")),
        "cpwer": json.loads((norm_dir / "cpwer_average.norm.json").read_text(encoding="utf-8")),
        "tcpwer": json.loads((norm_dir / "tcpwer_average.norm.json").read_text(encoding="utf-8")),
        "tcorcwer": json.loads((norm_dir / "tcorcwer_average.norm.json").read_text(encoding="utf-8")),
    }
    metrics_summary = {"raw": raw, "normalized": norm}
    (out_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    run_summary_path = out_dir / "run_summary.json"
    run_summary = {}
    if run_summary_path.exists():
        run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    run_summary.update(
        {
            "files_attempted": len(by_session),
            "files_completed": len(by_session),
            "files_failed": 0,
            "collar_seconds": args.collar,
            "chunking_mode": "full-audio-single-pass (use_streaming=False)",
            "raw_metrics": {k: v.get("error_rate") for k, v in raw.items()},
            "normalized_metrics": {k: v.get("error_rate") for k, v in norm.items()},
            "cleaning": {
                "dropped_nonspeech_rows": dropped_nonspeech,
                "expanded_blob_rows": expanded_blob_rows,
                "blob_segments_kept": blob_segments_kept,
                "original_hypothesis_multi_backup": str(hyp_multi_path.with_suffix(hyp_multi_path.suffix + ".orig")),
                "original_hypothesis_wer_backup": str(hyp_wer_path.with_suffix(hyp_wer_path.suffix + ".orig")),
            },
        }
    )
    run_summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print(json.dumps({
        "rows_before": len(rows),
        "rows_after_multi": len(out_multi),
        "rows_after_wer": len(out_wer),
        "dropped_nonspeech_rows": dropped_nonspeech,
        "expanded_blob_rows": expanded_blob_rows,
        "blob_segments_kept": blob_segments_kept,
        "raw_wer": raw["wer"]["error_rate"],
        "raw_cpwer": raw["cpwer"]["error_rate"],
        "raw_tcpwer": raw["tcpwer"]["error_rate"],
        "raw_tcorcwer": raw["tcorcwer"]["error_rate"],
        "norm_wer": norm["wer"]["error_rate"],
        "norm_cpwer": norm["cpwer"]["error_rate"],
        "norm_tcpwer": norm["tcpwer"]["error_rate"],
        "norm_tcorcwer": norm["tcorcwer"]["error_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()

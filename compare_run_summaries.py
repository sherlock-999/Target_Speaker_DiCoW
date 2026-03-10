#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def load(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(x):
    if x is None:
        return "N/A"
    return f"{x:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-summary", type=Path, required=True)
    ap.add_argument("--candidate-summary", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--baseline-name", type=str, default="baseline")
    ap.add_argument("--candidate-name", type=str, default="candidate")
    args = ap.parse_args()

    b = load(args.baseline_summary)
    c = load(args.candidate_summary)

    bm = b.get("metrics", {})
    cm = c.get("metrics", {})
    keys = ["WER", "cpWER", "tcpWER", "DER"]
    deltas = {}
    for k in keys:
        bv = bm.get(k)
        cv = cm.get(k)
        abs_delta = None if bv is None or cv is None else cv - bv
        rel_delta = None if bv in (None, 0) or cv is None else (cv - bv) / bv
        deltas[k] = {"baseline": bv, "candidate": cv, "abs_delta": abs_delta, "rel_delta": rel_delta}

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "baseline_summary": str(args.baseline_summary),
        "candidate_summary": str(args.candidate_summary),
        "baseline_sessions_scored": b.get("num_sessions_scored"),
        "candidate_sessions_scored": c.get("num_sessions_scored"),
        "deltas": deltas,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Metric Comparison")
    lines.append("")
    lines.append(f"- Baseline: `{args.baseline_name}`")
    lines.append(f"- Candidate: `{args.candidate_name}`")
    lines.append(f"- Baseline summary: `{args.baseline_summary}`")
    lines.append(f"- Candidate summary: `{args.candidate_summary}`")
    lines.append("")
    lines.append("| Metric | Baseline | Candidate | Abs Delta | Rel Delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for k in keys:
        d = deltas[k]
        rd = "N/A" if d["rel_delta"] is None else f"{d['rel_delta'] * 100:.2f}%"
        lines.append(f"| {k} | {fmt(d['baseline'])} | {fmt(d['candidate'])} | {fmt(d['abs_delta'])} | {rd} |")
    lines.append("")
    lines.append("## Session Coverage")
    lines.append(f"- Baseline scored sessions: `{out['baseline_sessions_scored']}`")
    lines.append(f"- Candidate scored sessions: `{out['candidate_sessions_scored']}`")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

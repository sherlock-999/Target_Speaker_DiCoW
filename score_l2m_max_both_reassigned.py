#!/usr/bin/env python3
import argparse
import itertools
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from evaluate_testset_tse_dicow_offline import (
    EnglishTextNormalizer,
    compute_der,
    parse_textgrid,
    read_jsonl_rows,
    write_json,
    write_jsonl,
)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text


def aggregate_rows_by_session_speaker(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["session_id"], row["speaker"])].append(row)
    out = []
    for (session_id, speaker), segs in grouped.items():
        segs.sort(key=lambda x: x.get("start_time", 0.0))
        words = " ".join(s.get("words", "") for s in segs).strip()
        out.append(
            {
                "session_id": session_id,
                "speaker": speaker,
                "start_time": 0,
                "end_time": 0,
                "words": words,
            }
        )
    return out


def run_metric(cmd):
    cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if cp.stdout.strip():
        print(cp.stdout.strip())
    if cp.stderr.strip():
        print(cp.stderr.strip(), file=sys.stderr)


def write_seglst_json(path: Path, rows):
    path.write_text(json.dumps(rows, ensure_ascii=True), encoding="utf-8")


def prepare_metrics(raw_dir: Path, collar: int):
    py = sys.executable
    ref_multi_rows = read_jsonl_rows(raw_dir / "reference_multi.jsonl")
    hyp_multi_rows = read_jsonl_rows(raw_dir / "hypothesis_multi_reassigned.jsonl")
    ref_wer_rows = read_jsonl_rows(raw_dir / "reference_wer.jsonl")
    hyp_wer_rows = read_jsonl_rows(raw_dir / "hypothesis_wer_reassigned.jsonl")

    ref_multi_seglst = raw_dir / "reference_multi.seglst.json"
    hyp_multi_seglst = raw_dir / "hypothesis_multi_reassigned.seglst.json"
    ref_wer_agg_seglst = raw_dir / "reference_wer_agg.seglst.json"
    hyp_wer_agg_seglst = raw_dir / "hypothesis_wer_reassigned_agg.seglst.json"
    ref_multi_agg_seglst = raw_dir / "reference_multi_agg.seglst.json"
    hyp_multi_agg_seglst = raw_dir / "hypothesis_multi_reassigned_agg.seglst.json"

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
            str(raw_dir / "wer_average.norm.json"),
            "--per-reco-out",
            str(raw_dir / "wer_per_reco.norm.json"),
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
            str(raw_dir / "cpwer_average.norm.json"),
            "--per-reco-out",
            str(raw_dir / "cpwer_per_reco.norm.json"),
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
            str(raw_dir / "tcpwer_average.norm.json"),
            "--per-reco-out",
            str(raw_dir / "tcpwer_per_reco.norm.json"),
        ]
    )


def edit_distance_words(a: str, b: str):
    a_t = a.split() if a else []
    b_t = b.split() if b else []
    n, m = len(a_t), len(b_t)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ai = a_t[i - 1]
        row = dp[i]
        prev = dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b_t[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)
    return dp[n][m]


def load_hypotheses(hyp_root: Path):
    by_session = defaultdict(list)
    for shard_hyp in sorted(hyp_root.glob("shard_*/hypothesis_multi.jsonl")):
        for row in read_jsonl_rows(shard_hyp):
            sid = str(row.get("session_id", "")).strip()
            if sid:
                by_session[sid].append(row)
    return by_session


def build_reference_rows(audio_root: Path, textgrid_root: Path):
    ref_multi = []
    ref_wer = []
    sessions = []
    missing_textgrid = []
    for wav in sorted(audio_root.glob("*.wav")):
        sid = wav.stem
        sessions.append(sid)
        tg = textgrid_root / f"{sid}.TextGrid"
        if not tg.exists():
            missing_textgrid.append(sid)
            continue
        segs = parse_textgrid(tg)
        for i, seg in enumerate(segs):
            ref_multi.append(
                {
                    "session_id": sid,
                    "speaker": seg["speaker"],
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "words": seg["words"],
                    "segment_index": i,
                }
            )
            ref_wer.append(
                {
                    "session_id": sid,
                    "speaker": "single",
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "words": seg["words"],
                    "segment_index": i,
                }
            )
    return sessions, ref_multi, ref_wer, missing_textgrid


def normalize_rows(rows, normalizer):
    out = []
    for r in rows:
        rr = dict(r)
        rr["words"] = clean_text(normalizer(rr.get("words", "")))
        out.append(rr)
    return out


def reassign_one_session(sid, ref_rows_sess, hyp_rows_sess):
    ref_speakers = sorted({r["speaker"] for r in ref_rows_sess})
    hyp_speakers = sorted({h["speaker"] for h in hyp_rows_sess})

    ref_concat = {}
    for spk in ref_speakers:
        segs = sorted((r for r in ref_rows_sess if r["speaker"] == spk), key=lambda x: x.get("start_time", 0.0))
        ref_concat[spk] = clean_text(" ".join(s["words"] for s in segs))

    hyp_concat = {}
    for spk in hyp_speakers:
        segs = sorted((h for h in hyp_rows_sess if h["speaker"] == spk), key=lambda x: (x.get("start_time", 0.0), x.get("end_time", 0.0)))
        hyp_concat[spk] = clean_text(" ".join(s["words"] for s in segs))

    pair_cost = {
        ref_spk: {hyp_spk: edit_distance_words(ref_concat[ref_spk], hyp_concat[hyp_spk]) for hyp_spk in hyp_speakers}
        for ref_spk in ref_speakers
    }

    mapping = {}
    rule = "none"
    if not hyp_speakers:
        rule = "no_hypothesis"
    elif len(hyp_speakers) >= len(ref_speakers):
        best = None
        for perm in itertools.permutations(hyp_speakers, len(ref_speakers)):
            total = sum(pair_cost[ref_speakers[i]][perm[i]] for i in range(len(ref_speakers)))
            if best is None or total < best[0]:
                best = (total, perm)
        for i, ref_spk in enumerate(ref_speakers):
            mapping[best[1][i]] = ref_spk
        rule = "min_cost_distinct_permutation"
    else:
        remaining_h = set(hyp_speakers)
        for ref_spk in ref_speakers:
            if not remaining_h:
                break
            best_h = min(remaining_h, key=lambda h: pair_cost[ref_spk][h])
            mapping[best_h] = ref_spk
            remaining_h.remove(best_h)
        rule = "greedy_undercomplete_hyp"

    reassigned = []
    for row in hyp_rows_sess:
        old_spk = row["speaker"]
        if old_spk in mapping:
            new_row = dict(row)
            new_row["speaker"] = mapping[old_spk]
            reassigned.append(new_row)

    debug = {
        "session_id": sid,
        "reference_speakers": ref_speakers,
        "hypothesis_speakers": hyp_speakers,
        "pairwise_cost": pair_cost,
        "assignment_rule": rule,
        "mapping_hyp_to_ref": mapping,
    }
    return reassigned, debug


def main():
    parser = argparse.ArgumentParser(description="Re-score old libri2mix max_both DiCoW with transcript-closest speaker reassignment")
    parser.add_argument("--hyp-root", type=Path, default=Path("output/libri2mix_max_both_dicow_sharded"))
    parser.add_argument("--textgrid-root", type=Path, default=Path("/home3/adnan/DICOW/mt-asr-data-prep/testset_tse/l2m/textgrid"))
    parser.add_argument("--audio-root", type=Path, default=Path("/home3/adnan/DICOW/mt-asr-data-prep/testset_tse/l2m/audio"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/libri2mix_max_both_reassigned_eval"))
    parser.add_argument("--collar", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    normalizer = EnglishTextNormalizer(
        standardize_numbers=False,
        standardize_numbers_rev=True,
        remove_fillers=True,
    )

    sessions_all, ref_multi_raw, ref_wer_raw, missing_tg = build_reference_rows(args.audio_root, args.textgrid_root)
    hyp_by_session = load_hypotheses(args.hyp_root)

    ref_multi = normalize_rows(ref_multi_raw, normalizer)
    ref_wer = normalize_rows(ref_wer_raw, normalizer)

    by_ref = defaultdict(list)
    for r in ref_multi:
        by_ref[r["session_id"]].append(r)

    by_hyp = defaultdict(list)
    for sid, rows in hyp_by_session.items():
        norm_rows = normalize_rows(rows, normalizer)
        by_hyp[sid].extend(norm_rows)

    reassigned_hyp_multi = []
    debug_all = {}
    missing_hyp = []
    for sid in sorted(set(sessions_all)):
        ref_rows_sess = by_ref.get(sid, [])
        if not ref_rows_sess:
            continue
        hyp_rows_sess = by_hyp.get(sid, [])
        if not hyp_rows_sess:
            missing_hyp.append(sid)
            debug_all[sid] = {
                "session_id": sid,
                "assignment_rule": "no_hypothesis",
                "mapping_hyp_to_ref": {},
                "reference_speakers": sorted({r["speaker"] for r in ref_rows_sess}),
                "hypothesis_speakers": [],
                "pairwise_cost": {},
            }
            continue
        reassigned, dbg = reassign_one_session(sid, ref_rows_sess, hyp_rows_sess)
        reassigned_hyp_multi.extend(reassigned)
        debug_all[sid] = dbg

    # Ensure every referenced session has a hypothesis entry so meeteval doesn't fail on missing recordings.
    ref_sessions = sorted({r["session_id"] for r in ref_multi})
    hyp_sessions = {h["session_id"] for h in reassigned_hyp_multi}
    for sid in ref_sessions:
        if sid not in hyp_sessions:
            reassigned_hyp_multi.append(
                {
                    "session_id": sid,
                    "speaker": "__empty__",
                    "start_time": 0.0,
                    "end_time": 0.0,
                    "words": "",
                    "segment_index": 0,
                }
            )

    reassigned_hyp_multi.sort(key=lambda x: (x["session_id"], x.get("start_time", 0.0), x.get("end_time", 0.0), x.get("segment_index", 0)))

    hyp_wer = []
    by_sid = defaultdict(list)
    for r in reassigned_hyp_multi:
        by_sid[r["session_id"]].append(r)
    for sid, segs in by_sid.items():
        segs = sorted(segs, key=lambda x: (x.get("start_time", 0.0), x.get("end_time", 0.0), x.get("segment_index", 0)))
        for i, s in enumerate(segs):
            hyp_wer.append(
                {
                    "session_id": sid,
                    "speaker": "single",
                    "start_time": s.get("start_time", 0.0),
                    "end_time": s.get("end_time", 0.0),
                    "words": s.get("words", ""),
                    "segment_index": i,
                }
            )

    write_jsonl(args.out_dir / "reference_multi.jsonl", ref_multi)
    write_jsonl(args.out_dir / "reference_wer.jsonl", ref_wer)
    write_jsonl(args.out_dir / "hypothesis_multi_reassigned.jsonl", reassigned_hyp_multi)
    write_jsonl(args.out_dir / "hypothesis_wer_reassigned.jsonl", hyp_wer)
    write_json(args.out_dir / "speaker_assignment_debug.json", debug_all)
    write_json(
        args.out_dir / "missing_sessions.json",
        {
            "missing_textgrid": sorted(missing_tg),
            "missing_hypothesis": sorted(missing_hyp),
        },
    )

    prepare_metrics(args.out_dir, collar=args.collar)
    der = compute_der(ref_multi, reassigned_hyp_multi)
    write_json(args.out_dir / "der_summary.json", der)

    wer = json.loads((args.out_dir / "wer_average.norm.json").read_text(encoding="utf-8"))
    cpwer = json.loads((args.out_dir / "cpwer_average.norm.json").read_text(encoding="utf-8"))
    tcpwer = json.loads((args.out_dir / "tcpwer_average.norm.json").read_text(encoding="utf-8"))

    scored_sessions = sorted(set(r["session_id"] for r in ref_multi) & set(r["session_id"] for r in reassigned_hyp_multi))
    summary = {
        "dataset": "l2m (max_both old run, reassigned)",
        "num_sessions_total": len(set(sessions_all)),
        "num_sessions_scored": len(scored_sessions),
        "num_sessions_missing_hypothesis": len(set(missing_hyp)),
        "num_sessions_missing_textgrid": len(set(missing_tg)),
        "metrics": {
            "WER": wer.get("error_rate"),
            "cpWER": cpwer.get("error_rate"),
            "tcpWER": tcpwer.get("error_rate"),
            "DER": der["rates"]["DER"],
        },
        "artifacts": {
            "reference_multi": str(args.out_dir / "reference_multi.jsonl"),
            "hypothesis_multi_reassigned": str(args.out_dir / "hypothesis_multi_reassigned.jsonl"),
            "speaker_assignment_debug": str(args.out_dir / "speaker_assignment_debug.json"),
            "wer_average": str(args.out_dir / "wer_average.norm.json"),
            "cpwer_average": str(args.out_dir / "cpwer_average.norm.json"),
            "tcpwer_average": str(args.out_dir / "tcpwer_average.norm.json"),
            "der_summary": str(args.out_dir / "der_summary.json"),
        },
        "scoring": {
            "normalization": "EnglishTextNormalizer(standardize_numbers=False, standardize_numbers_rev=True, remove_fillers=True)",
            "tcpwer_collar_seconds": args.collar,
            "der_collar_seconds": 0.0,
            "der_skip_overlap": False,
        },
    }
    write_json(args.out_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

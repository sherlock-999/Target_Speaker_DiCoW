#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


TAG_RE = re.compile(r"<[^>]+>")
SPK_HEADER_RE = re.compile(r"🗣️\s*Speaker\s*\d+\s*:", re.I)


def clean_text(text: str) -> str:
    text = SPK_HEADER_RE.sub(" ", str(text))
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize(text: str):
    text = clean_text(text)
    if not text:
        return []
    return text.split()


def edit_counts(ref_tokens, hyp_tokens):
    n = len(ref_tokens)
    m = len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "I"

    for i in range(1, n + 1):
        r = ref_tokens[i - 1]
        for j in range(1, m + 1):
            h = hyp_tokens[j - 1]
            sub_cost = 0 if r == h else 1
            candidates = [
                (dp[i - 1][j] + 1, "D"),
                (dp[i][j - 1] + 1, "I"),
                (dp[i - 1][j - 1] + sub_cost, "C" if sub_cost == 0 else "S"),
            ]
            best_cost, best_op = min(candidates, key=lambda x: x[0])
            dp[i][j] = best_cost
            bt[i][j] = best_op

    i, j = n, m
    s = d = ins = 0
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == "C":
            i -= 1
            j -= 1
        elif op == "S":
            s += 1
            i -= 1
            j -= 1
        elif op == "D":
            d += 1
            i -= 1
        elif op == "I":
            ins += 1
            j -= 1
        else:
            break
    return {"errors": s + d + ins, "substitutions": s, "deletions": d, "insertions": ins}


def load_reference_rows(csv_path: Path):
    refs = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mix_id = str(row.get("mix_id", "")).strip()
            if not mix_id:
                continue
            refs[mix_id] = {
                "u1": {
                    "speaker_id": str(row.get("speaker1_id", "speaker1")).strip() or "speaker1",
                    "text": clean_text(row.get("speaker1_text", "")),
                },
                "u2": {
                    "speaker_id": str(row.get("speaker2_id", "speaker2")).strip() or "speaker2",
                    "text": clean_text(row.get("speaker2_text", "")),
                },
            }
    return refs


def load_hypothesis_speakers(hyp_multi_jsonl: Path):
    grouped = {}
    with hyp_multi_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row.get("session_id")
            spk = str(row.get("speaker", "hyp_spk0"))
            if not sid:
                continue
            grouped.setdefault(sid, {}).setdefault(spk, []).append(row)

    hyp_by_session = {}
    for sid, spk_rows in grouped.items():
        hyp_by_session[sid] = {}
        for spk, rows in spk_rows.items():
            rows.sort(key=lambda x: (x.get("start_time", 0.0), x.get("end_time", 0.0), x.get("segment_index", 0)))
            words = " ".join(clean_text(r.get("words", "")) for r in rows).strip()
            hyp_by_session[sid][spk] = words
    return hyp_by_session


def choose_best_two_speakers(ref_u1_tokens, ref_u2_tokens, hyp_texts_by_spk):
    speakers = list(hyp_texts_by_spk.keys())
    if not speakers:
        speakers = ["hyp_spk0"]
        hyp_texts_by_spk = {"hyp_spk0": ""}

    per_utt_cost = {}
    for spk, text in hyp_texts_by_spk.items():
        hyp_tokens = tokenize(text)
        per_utt_cost[("u1", spk)] = edit_counts(ref_u1_tokens, hyp_tokens)
        per_utt_cost[("u2", spk)] = edit_counts(ref_u2_tokens, hyp_tokens)

    best = None
    # If we have >=2 speakers, force two distinct selected speakers.
    if len(speakers) >= 2:
        for spk1 in speakers:
            for spk2 in speakers:
                if spk1 == spk2:
                    continue
                c1 = per_utt_cost[("u1", spk1)]
                c2 = per_utt_cost[("u2", spk2)]
                total = c1["errors"] + c2["errors"]
                cand = (total, c1, c2, spk1, spk2)
                if best is None or cand[0] < best[0]:
                    best = cand
    else:
        spk = speakers[0]
        c1 = per_utt_cost[("u1", spk)]
        c2 = per_utt_cost[("u2", spk)]
        best = (c1["errors"] + c2["errors"], c1, c2, spk, spk)

    _, c1, c2, spk1, spk2 = best
    return c1, c2, spk1, spk2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path(
            "/home3/adnan/data/TS_ASR/libri2mix/LibriMix/downloaded_data/Libri2Mix/wav16k/min/test/whisper_base_clean_speakers_transcripts.csv"
        ),
    )
    parser.add_argument(
        "--hypothesis-multi",
        type=Path,
        default=Path("output/libri2mix_dicow_offline/hypothesis_multi.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/libri2mix_dicow_offline"))
    args = parser.parse_args()

    if not args.reference_csv.exists():
        raise FileNotFoundError(f"Reference CSV not found: {args.reference_csv}")
    if not args.hypothesis_multi.exists():
        raise FileNotFoundError(f"Hypothesis jsonl not found: {args.hypothesis_multi}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    references = load_reference_rows(args.reference_csv)
    hypotheses = load_hypothesis_speakers(args.hypothesis_multi)

    per_reco = {}
    total_errors = total_length = total_ins = total_del = total_sub = 0
    scored_utts = 0

    for session_id, ref in references.items():
        if session_id not in hypotheses:
            # If missing hypothesis for a session, treat as empty hypothesis stream.
            hyp_texts_by_spk = {"hyp_spk0": ""}
        else:
            hyp_texts_by_spk = hypotheses[session_id]

        ref_u1_tokens = tokenize(ref["u1"]["text"])
        ref_u2_tokens = tokenize(ref["u2"]["text"])
        u1_len = len(ref_u1_tokens)
        u2_len = len(ref_u2_tokens)

        c1, c2, spk1, spk2 = choose_best_two_speakers(ref_u1_tokens, ref_u2_tokens, hyp_texts_by_spk)

        u1_errors = c1["errors"] if u1_len > 0 else 0
        u2_errors = c2["errors"] if u2_len > 0 else 0

        if u1_len > 0:
            total_errors += u1_errors
            total_ins += c1["insertions"]
            total_del += c1["deletions"]
            total_sub += c1["substitutions"]
            total_length += u1_len
            scored_utts += 1
        if u2_len > 0:
            total_errors += u2_errors
            total_ins += c2["insertions"]
            total_del += c2["deletions"]
            total_sub += c2["substitutions"]
            total_length += u2_len
            scored_utts += 1

        per_reco[session_id] = {
            "u1": {
                "reference_speaker": ref["u1"]["speaker_id"],
                "reference_length": u1_len,
                "chosen_hyp_speaker": spk1,
                "errors": u1_errors,
                "insertions": c1["insertions"] if u1_len > 0 else 0,
                "deletions": c1["deletions"] if u1_len > 0 else 0,
                "substitutions": c1["substitutions"] if u1_len > 0 else 0,
            },
            "u2": {
                "reference_speaker": ref["u2"]["speaker_id"],
                "reference_length": u2_len,
                "chosen_hyp_speaker": spk2,
                "errors": u2_errors,
                "insertions": c2["insertions"] if u2_len > 0 else 0,
                "deletions": c2["deletions"] if u2_len > 0 else 0,
                "substitutions": c2["substitutions"] if u2_len > 0 else 0,
            },
            "totals": {
                "errors": u1_errors + u2_errors,
                "length": u1_len + u2_len,
                "insertions": (c1["insertions"] if u1_len > 0 else 0) + (c2["insertions"] if u2_len > 0 else 0),
                "deletions": (c1["deletions"] if u1_len > 0 else 0) + (c2["deletions"] if u2_len > 0 else 0),
                "substitutions": (c1["substitutions"] if u1_len > 0 else 0) + (c2["substitutions"] if u2_len > 0 else 0),
            },
        }

    average = {
        "error_rate": (total_errors / total_length) if total_length > 0 else 0.0,
        "errors": total_errors,
        "length": total_length,
        "insertions": total_ins,
        "deletions": total_del,
        "substitutions": total_sub,
    }

    notes = {
        "metric_name": "oracle utterance matched WER",
        "matching_mode": "best-of-two per utterance, choose best 2 distinct hypothesis speakers when available",
        "aggregation": "micro",
        "normalization": "clean_text: remove speaker headers + tags, collapse whitespace, lowercase",
        "inputs": {
            "reference_csv": str(args.reference_csv),
            "hypothesis_multi": str(args.hypothesis_multi),
        },
        "summary": {
            "num_sessions_in_reference": len(references),
            "num_sessions_in_hypothesis": len(hypotheses),
            "num_scored_reference_utterances": scored_utts,
        },
    }

    (args.output_dir / "oracle_utt_match_per_reco.json").write_text(
        json.dumps(per_reco, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (args.output_dir / "oracle_utt_match_average.json").write_text(
        json.dumps(average, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (args.output_dir / "oracle_utt_match_notes.json").write_text(
        json.dumps(notes, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(json.dumps(average, indent=2))


if __name__ == "__main__":
    main()

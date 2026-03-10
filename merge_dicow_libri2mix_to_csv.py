#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import defaultdict
from itertools import permutations
from pathlib import Path

from jiwer import process_words


TAG_RE = re.compile(r"<[^>]+>")
SPK_HEADER_RE = re.compile(r"🗣️\s*Speaker\s*\d+\s*:", re.I)


def clean_text(text: str) -> str:
    text = SPK_HEADER_RE.sub(" ", str(text))
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def word_counts(ref: str, hyp: str):
    out = process_words(ref, hyp)
    return {
        "wer": float(out.wer),
        "errors": int(out.substitutions + out.deletions + out.insertions),
        "substitutions": int(out.substitutions),
        "deletions": int(out.deletions),
        "insertions": int(out.insertions),
        "reference_words": int(out.substitutions + out.deletions + out.hits),
    }


def load_references(reference_csv: Path):
    refs = {}
    with reference_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mix_id = str(row.get("mix_id", "")).strip()
            if not mix_id:
                continue
            refs[mix_id] = row
    return refs


def load_hypothesis_concat(shards_root: Path):
    by_mix_spk = defaultdict(list)
    for shard_dir in sorted(shards_root.glob("shard_*")):
        hyp_path = shard_dir / "hypothesis_multi.jsonl"
        if not hyp_path.exists():
            continue
        with hyp_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                mix_id = row.get("session_id")
                spk = str(row.get("speaker", "hyp_spk0"))
                if not mix_id:
                    continue
                by_mix_spk[(mix_id, spk)].append(row)

    mix_to_spk_text = defaultdict(dict)
    for (mix_id, spk), rows in by_mix_spk.items():
        rows.sort(key=lambda x: (x.get("start_time", 0.0), x.get("end_time", 0.0), x.get("segment_index", 0)))
        text = " ".join(clean_text(r.get("words", "")) for r in rows).strip()
        mix_to_spk_text[mix_id][spk] = text
    return mix_to_spk_text


def choose_assignment(ref_s1, ref_s2, hyp_text_by_spk):
    speakers = sorted(hyp_text_by_spk.keys())
    pair_scores = {}
    for spk in speakers:
        pair_scores[("s1", spk)] = word_counts(ref_s1, hyp_text_by_spk[spk])
        pair_scores[("s2", spk)] = word_counts(ref_s2, hyp_text_by_spk[spk])

    debug = {
        "candidate_hyp_speakers": speakers,
        "pairwise": {
            "s1": {spk: pair_scores[("s1", spk)] for spk in speakers},
            "s2": {spk: pair_scores[("s2", spk)] for spk in speakers},
        },
    }

    if len(speakers) >= 2:
        best = None
        for spk1, spk2 in permutations(speakers, 2):
            total = pair_scores[("s1", spk1)]["errors"] + pair_scores[("s2", spk2)]["errors"]
            candidate = (total, spk1, spk2)
            if best is None or candidate[0] < best[0]:
                best = candidate
        _, s1_spk, s2_spk = best
        debug["assignment_rule"] = "1-to-1 global best (distinct speakers)"
        debug["selected"] = {
            "s1": {"hyp_speaker": s1_spk, "score": pair_scores[("s1", s1_spk)]},
            "s2": {"hyp_speaker": s2_spk, "score": pair_scores[("s2", s2_spk)]},
        }
        return s1_spk, s2_spk, debug

    if len(speakers) == 1:
        spk = speakers[0]
        s1_e = pair_scores[("s1", spk)]["errors"]
        s2_e = pair_scores[("s2", spk)]["errors"]
        if s1_e <= s2_e:
            s1_spk, s2_spk = spk, None
            fallback = "single_hyp_to_s1"
        else:
            s1_spk, s2_spk = None, spk
            fallback = "single_hyp_to_s2"
        debug["assignment_rule"] = "single-speaker fallback"
        debug["fallback_reason"] = fallback
        debug["selected"] = {
            "s1": {"hyp_speaker": s1_spk, "score": pair_scores[("s1", spk)]},
            "s2": {"hyp_speaker": s2_spk, "score": pair_scores[("s2", spk)]},
        }
        return s1_spk, s2_spk, debug

    debug["assignment_rule"] = "no-hypothesis"
    debug["fallback_reason"] = "missing_mix_in_hypothesis"
    debug["selected"] = {"s1": {"hyp_speaker": None}, "s2": {"hyp_speaker": None}}
    return None, None, debug


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
        "--mix-clean-dir",
        type=Path,
        default=Path(
            "/home3/adnan/data/TS_ASR/libri2mix/LibriMix/downloaded_data/Libri2Mix/wav16k/min/test/mix_clean"
        ),
    )
    parser.add_argument("--shards-root", type=Path, default=Path("output/libri2mix_dicow_sharded"))
    parser.add_argument("--out-csv", type=Path, default=Path("output/libri2mix_dicow_sharded/dicow_predictions.csv"))
    parser.add_argument(
        "--debug-json",
        type=Path,
        default=Path("output/libri2mix_dicow_sharded/dicow_matching_debug.json"),
    )
    args = parser.parse_args()

    refs = load_references(args.reference_csv)
    hyps = load_hypothesis_concat(args.shards_root)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.debug_json.parent.mkdir(parents=True, exist_ok=True)

    csv_fields = ["mix_id", "target_speaker_id", "mode", "hyp_text", "ref_text", "enroll_wav", "mix_wav", "target_side"]
    debug = {}
    rows = []

    for mix_id in sorted(refs.keys()):
        row = refs[mix_id]
        ref_s1 = str(row.get("speaker1_text", "")).strip()
        ref_s2 = str(row.get("speaker2_text", "")).strip()
        spk1_id = str(row.get("speaker1_id", "")).strip()
        spk2_id = str(row.get("speaker2_id", "")).strip()
        mix_wav = str(args.mix_clean_dir / f"{mix_id}.wav")

        hyp_text_by_spk = hyps.get(mix_id, {})
        s1_hyp_spk, s2_hyp_spk, dbg = choose_assignment(clean_text(ref_s1), clean_text(ref_s2), hyp_text_by_spk)
        debug[mix_id] = dbg

        s1_hyp = hyp_text_by_spk.get(s1_hyp_spk, "") if s1_hyp_spk else ""
        s2_hyp = hyp_text_by_spk.get(s2_hyp_spk, "") if s2_hyp_spk else ""

        rows.append(
            {
                "mix_id": mix_id,
                "target_speaker_id": spk1_id,
                "mode": "dicow_offline",
                "hyp_text": s1_hyp,
                "ref_text": ref_s1,
                "enroll_wav": "",
                "mix_wav": mix_wav,
                "target_side": "s1",
            }
        )
        rows.append(
            {
                "mix_id": mix_id,
                "target_speaker_id": spk2_id,
                "mode": "dicow_offline",
                "hyp_text": s2_hyp,
                "ref_text": ref_s2,
                "enroll_wav": "",
                "mix_wav": mix_wav,
                "target_side": "s2",
            }
        )

    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)

    args.debug_json.write_text(json.dumps(debug, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote CSV rows: {len(rows)} to {args.out_csv}")
    print(f"Wrote debug JSON: {args.debug_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from subprocess import run


try:
    import gradio  # noqa: F401
except Exception:
    sys.modules["gradio"] = types.SimpleNamespace(
        Info=lambda *a, **k: None,
        Warning=lambda *a, **k: None,
    )

import torch
from transformers import AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "DiariZen"))
sys.path.insert(0, str(PROJECT_ROOT))

from diarizen.pipelines.inference import DiariZenPipeline
from inference import create_lower_uppercase_mapping
from pipeline import DiCoWPipeline


TS_SEGMENT_RE = re.compile(r"<\|([0-9]+(?:\.[0-9]+)?)\|>(.*?)<\|([0-9]+(?:\.[0-9]+)?)\|>", re.S)
TAG_RE = re.compile(r"<[^>]+>")
SPK_HEADER_RE = re.compile(r"🗣️\s*Speaker\s*\d+\s*:", re.I)


def clean_text(text: str) -> str:
    text = SPK_HEADER_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def parse_timestamped_segments(text: str):
    segments = []
    for m in TS_SEGMENT_RE.finditer(text):
        start = float(m.group(1))
        body = clean_text(m.group(2))
        end = float(m.group(3))
        if body:
            segments.append((start, end, body))
    return segments


def append_jsonl(path: Path, rows):
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl_rows(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_seglst_json(path: Path, rows):
    path.write_text(json.dumps(rows, ensure_ascii=True), encoding="utf-8")


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


def read_session_ids(path: Path):
    session_ids = set()
    if not path.exists():
        return session_ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                session_ids.add(json.loads(line)["session_id"])
            except Exception:
                continue
    return session_ids


def run_metric(cmd):
    completed = run(cmd, check=True, capture_output=True, text=True)
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home3/adnan/data/TS_ASR/benchmark-datasets/eval_set/240629.1_eval_small_with_GT/MTG"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/mtg_sc160_dicow_offline"))
    parser.add_argument("--dicow-model", type=str, default="BUT-FIT/DiCoW_v3_2")
    parser.add_argument("--diarization-model", type=str, default="BUT-FIT/diarizen-wavlm-large-s80-md")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--collar", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Resume from existing output files")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    ref_multi = args.output_dir / "reference_multi.jsonl"
    hyp_multi = args.output_dir / "hypothesis_multi.jsonl"
    ref_wer = args.output_dir / "reference_wer.jsonl"
    hyp_wer = args.output_dir / "hypothesis_wer.jsonl"
    manifest = args.output_dir / "manifest_sc160.txt"
    failures = args.output_dir / "failures.txt"

    if not args.resume:
        for p in [ref_multi, hyp_multi, ref_wer, hyp_wer, manifest, failures]:
            if p.exists():
                p.unlink()

    audio_files = sorted(args.dataset_root.glob("MTG_*/sc_*/ch0.wav"))
    if len(audio_files) != 160:
        print(f"Warning: expected 160 sc files, found {len(audio_files)}", file=sys.stderr)

    with manifest.open("w", encoding="utf-8") as f:
        for wav in audio_files:
            f.write(str(wav) + "\n")

    completed_hyp_sessions = read_session_ids(hyp_wer)
    existing_ref_sessions = read_session_ids(ref_wer)
    if args.resume:
        print(
            f"Resume mode: completed sessions in hypothesis={len(completed_hyp_sessions)}, "
            f"sessions in reference={len(existing_ref_sessions)}"
        )

    device = torch.device(args.device)
    print(f"Loading models on {device} ...")
    dicow = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.dicow_model,
        trust_remote_code=True,
        cache_dir=str(models_dir),
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.dicow_model, cache_dir=str(models_dir))
    tokenizer = AutoTokenizer.from_pretrained(args.dicow_model, cache_dir=str(models_dir))
    create_lower_uppercase_mapping(tokenizer)
    dicow.set_tokenizer(tokenizer)
    diar_pipeline = DiariZenPipeline.from_pretrained(args.diarization_model, cache_dir=str(models_dir)).to(device)
    pipeline = DiCoWPipeline(
        dicow,
        diarization_pipeline=diar_pipeline,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        device=device,
    )

    start_time = time.time()
    done = len(completed_hyp_sessions)
    done_this_run = 0
    for idx, wav_path in enumerate(audio_files, start=1):
        session_id = f"{wav_path.parents[1].name}/{wav_path.parent.name}"
        if session_id in completed_hyp_sessions:
            continue

        gt_path = wav_path.parents[1] / "gt_transcription.json"
        gt_entries = json.loads(gt_path.read_text(encoding="utf-8"))

        ref_multi_rows = []
        ref_single_rows = []
        for seg_i, item in enumerate(gt_entries):
            text = clean_text(item.get("text", ""))
            if not text:
                continue
            st = float(item["start_time"])
            et = float(item["end_time"])
            spk = str(item["speaker_id"])
            ref_multi_rows.append(
                {
                    "session_id": session_id,
                    "speaker": spk,
                    "start_time": st,
                    "end_time": et,
                    "words": text,
                    "segment_index": seg_i,
                }
            )
            ref_single_rows.append(
                {
                    "session_id": session_id,
                    "speaker": "single",
                    "start_time": st,
                    "end_time": et,
                    "words": text,
                    "segment_index": seg_i,
                }
            )

        if session_id not in existing_ref_sessions:
            append_jsonl(ref_multi, ref_multi_rows)
            append_jsonl(ref_wer, ref_single_rows)
            existing_ref_sessions.add(session_id)

        try:
            out = pipeline(str(wav_path), return_timestamps=True)
        except Exception as e:
            with failures.open("a", encoding="utf-8") as f:
                f.write(f"{session_id}\t{type(e).__name__}\t{e}\n")
            continue

        per_spk = out.get("per_spk_outputs", []) if isinstance(out, dict) else []
        hyp_multi_rows = []
        all_hyp_segments = []
        segment_counter = 0
        for spk_idx, spk_text in enumerate(per_spk):
            segments = parse_timestamped_segments(str(spk_text))
            for st, et, words in segments:
                hyp_multi_rows.append(
                    {
                        "session_id": session_id,
                        "speaker": f"hyp_spk{spk_idx}",
                        "start_time": st,
                        "end_time": et,
                        "words": words,
                        "segment_index": segment_counter,
                    }
                )
                all_hyp_segments.append((st, et, words))
                segment_counter += 1

        if not all_hyp_segments:
            flat_text = clean_text(out.get("text", "") if isinstance(out, dict) else str(out))
            if flat_text:
                hyp_multi_rows.append(
                    {
                        "session_id": session_id,
                        "speaker": "hyp_spk0",
                        "start_time": 0.0,
                        "end_time": 0.0,
                        "words": flat_text,
                        "segment_index": 0,
                    }
                )
                all_hyp_segments.append((0.0, 0.0, flat_text))

        append_jsonl(hyp_multi, hyp_multi_rows)

        hyp_single_rows = []
        for seg_i, (st, et, words) in enumerate(sorted(all_hyp_segments, key=lambda x: x[0])):
            hyp_single_rows.append(
                {
                    "session_id": session_id,
                    "speaker": "single",
                    "start_time": st,
                    "end_time": et,
                    "words": words,
                    "segment_index": seg_i,
                }
            )
        append_jsonl(hyp_wer, hyp_single_rows)

        done += 1
        done_this_run += 1
        completed_hyp_sessions.add(session_id)
        if idx % 5 == 0 or idx == len(audio_files):
            elapsed = time.time() - start_time
            print(
                f"[{idx}/{len(audio_files)}] attempted, completed_sessions={len(completed_hyp_sessions)}, "
                f"successful_in_this_run={done_this_run}, elapsed={elapsed:.1f}s"
            )

    print("Computing meeteval metrics ...")
    py = sys.executable

    # Convert jsonl intermediates to meeteval-readable SegLST JSON.
    ref_multi_rows = read_jsonl_rows(ref_multi)
    hyp_multi_rows = read_jsonl_rows(hyp_multi)
    ref_wer_rows = read_jsonl_rows(ref_wer)
    hyp_wer_rows = read_jsonl_rows(hyp_wer)

    ref_multi_seglst = args.output_dir / "reference_multi.seglst.json"
    hyp_multi_seglst = args.output_dir / "hypothesis_multi.seglst.json"
    ref_wer_agg_seglst = args.output_dir / "reference_wer_agg.seglst.json"
    hyp_wer_agg_seglst = args.output_dir / "hypothesis_wer_agg.seglst.json"
    ref_multi_agg_seglst = args.output_dir / "reference_multi_agg.seglst.json"
    hyp_multi_agg_seglst = args.output_dir / "hypothesis_multi_agg.seglst.json"

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
            str(args.output_dir / "wer_average.json"),
            "--per-reco-out",
            str(args.output_dir / "wer_per_reco.json"),
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
            str(args.output_dir / "cpwer_average.json"),
            "--per-reco-out",
            str(args.output_dir / "cpwer_per_reco.json"),
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
            str(args.collar),
            "--average-out",
            str(args.output_dir / "tcpwer_average.json"),
            "--per-reco-out",
            str(args.output_dir / "tcpwer_per_reco.json"),
        ]
    )
    run_metric(
        [
            py,
            "-m",
            "meeteval.wer",
            "tcorcwer",
            "-r",
            str(ref_multi_seglst),
            "-h",
            str(hyp_multi_seglst),
            "--collar",
            str(args.collar),
            "--average-out",
            str(args.output_dir / "tcorcwer_average.json"),
            "--per-reco-out",
            str(args.output_dir / "tcorcwer_per_reco.json"),
        ]
    )

    elapsed = time.time() - start_time
    summary = {
        "num_audio_files": len(audio_files),
        "successful_inference_files": done,
        "failed_files": 0 if not failures.exists() else sum(1 for _ in failures.open("r", encoding="utf-8")),
        "collar_seconds": args.collar,
        "runtime_seconds": elapsed,
        "artifacts": {
            "reference_multi": str(ref_multi),
            "hypothesis_multi": str(hyp_multi),
            "reference_wer": str(ref_wer),
            "hypothesis_wer": str(hyp_wer),
            "wer_average": str(args.output_dir / "wer_average.json"),
            "cpwer_average": str(args.output_dir / "cpwer_average.json"),
            "tcpwer_average": str(args.output_dir / "tcpwer_average.json"),
            "tcorcwer_average": str(args.output_dir / "tcorcwer_average.json"),
        },
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

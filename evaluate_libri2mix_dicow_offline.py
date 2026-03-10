#!/usr/bin/env python3
import argparse
import csv
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
    text = SPK_HEADER_RE.sub(" ", str(text))
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


def load_reference_csv(csv_path: Path):
    by_mix_id = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mix_id = str(row.get("mix_id", "")).strip()
            if mix_id:
                by_mix_id[mix_id] = row
    return by_mix_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mix-clean-dir",
        type=Path,
        default=Path(
            "/home3/adnan/data/TS_ASR/libri2mix/LibriMix/downloaded_data/Libri2Mix/wav16k/min/test/mix_clean"
        ),
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path(
            "/home3/adnan/data/TS_ASR/libri2mix/LibriMix/downloaded_data/Libri2Mix/wav16k/min/test/whisper_base_clean_speakers_transcripts.csv"
        ),
    )
    parser.add_argument(
        "--mix-id-list",
        type=Path,
        default=None,
        help="Optional file with one mix_id per line to restrict processed sessions.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/libri2mix_dicow_offline"))
    parser.add_argument("--dicow-model", type=str, default="BUT-FIT/DiCoW_v3_2")
    parser.add_argument("--diarization-model", type=str, default="BUT-FIT/diarizen-wavlm-large-s80-md")
    parser.add_argument("--diarization-min-speakers", type=int, default=1)
    parser.add_argument("--diarization-max-speakers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--collar", type=int, default=5)
    parser.add_argument("--max-files", type=int, default=0, help="0 means all files")
    parser.add_argument("--compute-metrics", action="store_true", help="If set, compute meeteval metrics at the end")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output files")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    ref_multi = args.output_dir / "reference_multi.jsonl"
    hyp_multi = args.output_dir / "hypothesis_multi.jsonl"
    ref_wer = args.output_dir / "reference_wer.jsonl"
    hyp_wer = args.output_dir / "hypothesis_wer.jsonl"
    manifest = args.output_dir / "manifest_mix_clean.txt"
    failures = args.output_dir / "failures.txt"

    if not args.resume:
        for p in [ref_multi, hyp_multi, ref_wer, hyp_wer, manifest, failures]:
            if p.exists():
                p.unlink()

    if not args.mix_clean_dir.exists():
        raise FileNotFoundError(f"mix_clean directory not found: {args.mix_clean_dir}")
    if not args.reference_csv.exists():
        raise FileNotFoundError(f"reference csv not found: {args.reference_csv}")
    selected_mix_ids = None
    if args.mix_id_list is not None:
        if not args.mix_id_list.exists():
            raise FileNotFoundError(f"mix id list not found: {args.mix_id_list}")
        selected_mix_ids = set()
        with args.mix_id_list.open("r", encoding="utf-8") as f:
            for line in f:
                mix_id = line.strip()
                if mix_id:
                    selected_mix_ids.add(mix_id)
        print(f"Loaded {len(selected_mix_ids)} mix IDs from {args.mix_id_list}")

    # Exclude pipeline scratch artifacts that may live in the same folder.
    audio_files = sorted(p for p in args.mix_clean_dir.glob("*.wav") if p.name != "resampled.wav")
    if selected_mix_ids is not None:
        audio_files = [p for p in audio_files if p.stem in selected_mix_ids]
    if args.max_files > 0:
        audio_files = audio_files[: args.max_files]
    if len(audio_files) == 0:
        raise RuntimeError(f"No wav files found under: {args.mix_clean_dir}")

    with manifest.open("w", encoding="utf-8") as f:
        for wav in audio_files:
            f.write(str(wav) + "\n")

    ref_by_mix_id = load_reference_csv(args.reference_csv)
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
    diar_pipeline.min_speakers = int(args.diarization_min_speakers)
    diar_pipeline.max_speakers = int(args.diarization_max_speakers)
    print(
        "DiariZen runtime speaker bounds set to "
        f"min_speakers={diar_pipeline.min_speakers}, max_speakers={diar_pipeline.max_speakers}"
    )
    if diar_pipeline.max_speakers != int(args.diarization_max_speakers):
        raise RuntimeError("Failed to apply diarization_max_speakers override to DiariZen pipeline.")

    original_diar_call = DiariZenPipeline.__call__
    diar_verify_state = {"printed": False}

    def verified_diar_call(self, *call_args, **call_kwargs):
        if not diar_verify_state["printed"]:
            print(
                "[VERIFY] First DiariZen call sees "
                f"min_speakers={diar_pipeline.min_speakers}, max_speakers={diar_pipeline.max_speakers}"
            )
            diar_verify_state["printed"] = True
        return original_diar_call(self, *call_args, **call_kwargs)

    DiariZenPipeline.__call__ = verified_diar_call

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
        session_id = wav_path.stem
        if session_id in completed_hyp_sessions:
            continue

        row = ref_by_mix_id.get(session_id)
        if row is None:
            with failures.open("a", encoding="utf-8") as f:
                f.write(f"{session_id}\tMissingReference\tNo CSV row for mix_id\n")
            continue

        s1_text = clean_text(row.get("speaker1_text", ""))
        s2_text = clean_text(row.get("speaker2_text", ""))
        s1_id = str(row.get("speaker1_id", "ref_spk1")).strip() or "ref_spk1"
        s2_id = str(row.get("speaker2_id", "ref_spk2")).strip() or "ref_spk2"

        ref_multi_rows = []
        if s1_text:
            ref_multi_rows.append(
                {
                    "session_id": session_id,
                    "speaker": s1_id,
                    "start_time": 0.0,
                    "end_time": 0.0,
                    "words": s1_text,
                    "segment_index": 0,
                }
            )
        if s2_text:
            ref_multi_rows.append(
                {
                    "session_id": session_id,
                    "speaker": s2_id,
                    "start_time": 0.0,
                    "end_time": 0.0,
                    "words": s2_text,
                    "segment_index": 1,
                }
            )

        ref_single_rows = []
        if s1_text:
            ref_single_rows.append(
                {
                    "session_id": session_id,
                    "speaker": "single",
                    "start_time": 0.0,
                    "end_time": 0.0,
                    "words": s1_text,
                    "segment_index": 0,
                }
            )
        if s2_text:
            ref_single_rows.append(
                {
                    "session_id": session_id,
                    "speaker": "single",
                    "start_time": 0.0,
                    "end_time": 0.0,
                    "words": s2_text,
                    "segment_index": 1,
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
        if idx % 10 == 0 or idx == len(audio_files):
            elapsed = time.time() - start_time
            print(
                f"[{idx}/{len(audio_files)}] attempted, completed_sessions={len(completed_hyp_sessions)}, "
                f"successful_in_this_run={done_this_run}, elapsed={elapsed:.1f}s"
            )

    elapsed = time.time() - start_time
    summary = {
        "num_audio_files": len(audio_files),
        "successful_inference_files": done,
        "failed_files": 0 if not failures.exists() else sum(1 for _ in failures.open("r", encoding="utf-8")),
        "collar_seconds": args.collar,
        "compute_metrics": bool(args.compute_metrics),
        "diarization_runtime": {
            "min_speakers": int(diar_pipeline.min_speakers),
            "max_speakers": int(diar_pipeline.max_speakers),
            "first_call_verified": bool(diar_verify_state["printed"]),
        },
        "runtime_seconds": elapsed,
        "artifacts": {
            "reference_multi": str(ref_multi),
            "hypothesis_multi": str(hyp_multi),
            "reference_wer": str(ref_wer),
            "hypothesis_wer": str(hyp_wer),
        },
    }

    if args.compute_metrics:
        print("Computing meeteval metrics ...")
        py = sys.executable
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
        summary["artifacts"].update(
            {
                "wer_average": str(args.output_dir / "wer_average.json"),
                "cpwer_average": str(args.output_dir / "cpwer_average.json"),
                "tcpwer_average": str(args.output_dir / "tcpwer_average.json"),
                "tcorcwer_average": str(args.output_dir / "tcorcwer_average.json"),
            }
        )

    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

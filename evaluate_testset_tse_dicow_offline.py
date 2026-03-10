#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
import time
import types
from collections import defaultdict
from pathlib import Path


try:
    import gradio  # noqa: F401
except Exception:
    sys.modules["gradio"] = types.SimpleNamespace(
        Info=lambda *a, **k: None,
        Warning=lambda *a, **k: None,
    )

import torch
import torchaudio
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from transformers import AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "DiariZen"))
sys.path.insert(0, str(PROJECT_ROOT))

from diarizen.pipelines.inference import DiariZenPipeline
from inference import create_lower_uppercase_mapping
from local_text_norm.english import EnglishTextNormalizer
from pipeline import DiCoWPipeline

TS_SEGMENT_RE = re.compile(r"<\|([0-9]+(?:\.[0-9]+)?)\|>(.*?)<\|([0-9]+(?:\.[0-9]+)?)\|>", re.S)
TAG_RE = re.compile(r"<[^>]+>")
SPK_HEADER_RE = re.compile(r"Speaker\s*\d+\s*:", re.I)
TIER_RE = re.compile(
    r'item\s*\[\d+\]:\s*class\s*=\s*"IntervalTier"\s*name\s*=\s*"(.*?)"\s*'
    r"xmin\s*=\s*[-+0-9.eE]+\s*xmax\s*=\s*[-+0-9.eE]+\s*"
    r"intervals:\s*size\s*=\s*\d+\s*(.*?)(?=\n\s*item\s*\[\d+\]:|\Z)",
    re.S,
)
INTERVAL_RE = re.compile(
    r"intervals\s*\[\d+\]:\s*xmin\s*=\s*([-+0-9.eE]+)\s*xmax\s*=\s*([-+0-9.eE]+)\s*text\s*=\s*\"(.*?)\"",
    re.S,
)


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
        if body and end > start:
            segments.append((start, end, body))
    return segments


def parse_textgrid(path: Path):
    content = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    seg_i = 0
    for tier_name, tier_body in TIER_RE.findall(content):
        speaker = clean_text(tier_name) or "spk"
        for start, end, text in INTERVAL_RE.findall(tier_body):
            words = clean_text(text.replace('""', '"'))
            st = float(start)
            et = float(end)
            if not words or et <= st:
                continue
            rows.append(
                {
                    "speaker": speaker,
                    "start_time": st,
                    "end_time": et,
                    "words": words,
                    "segment_index": seg_i,
                }
            )
            seg_i += 1
    rows.sort(key=lambda x: (x["start_time"], x["end_time"], x["speaker"]))
    return rows


def append_jsonl(path: Path, rows):
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
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


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


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


def run_metric(cmd):
    cp = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if cp.stdout.strip():
        print(cp.stdout.strip())
    if cp.stderr.strip():
        print(cp.stderr.strip(), file=sys.stderr)


def prepare_metrics(raw_dir: Path, collar: int):
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
            str(collar),
            "--average-out",
            str(raw_dir / "tcorcwer_average.json"),
            "--per-reco-out",
            str(raw_dir / "tcorcwer_per_reco.json"),
        ]
    )


def normalize_rows(rows, normalizer):
    out = []
    for row in rows:
        nrow = dict(row)
        words = normalizer(nrow.get("words", ""))
        words = re.sub(r"\s+", " ", words).strip()
        nrow["words"] = words
        out.append(nrow)
    return out


def prepare_normalized_metrics(raw_dir: Path, norm_dir: Path, collar: int):
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
            str(collar),
            "--average-out",
            str(norm_dir / "tcorcwer_average.norm.json"),
            "--per-reco-out",
            str(norm_dir / "tcorcwer_per_reco.norm.json"),
        ]
    )


def rows_to_annotation(rows):
    ann = Annotation()
    for idx, row in enumerate(rows):
        st = float(row.get("start_time", 0.0))
        et = float(row.get("end_time", st))
        if et <= st:
            continue
        ann[Segment(st, et), f"{idx}"] = str(row.get("speaker", "spk"))
    return ann


def compute_der(reference_multi_rows, hypothesis_multi_rows):
    by_ref = defaultdict(list)
    by_hyp = defaultdict(list)
    for r in reference_multi_rows:
        by_ref[r["session_id"]].append(r)
    for r in hypothesis_multi_rows:
        by_hyp[r["session_id"]].append(r)

    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    scored = 0
    for session_id in sorted(by_ref.keys()):
        ref_ann = rows_to_annotation(by_ref[session_id])
        hyp_ann = rows_to_annotation(by_hyp.get(session_id, []))
        metric(ref_ann, hyp_ann)
        scored += 1

    details = metric[:]
    total = float(details.get("total", 0.0))
    fa = float(details.get("false alarm", 0.0))
    ms = float(details.get("missed detection", 0.0))
    sc = float(details.get("confusion", 0.0))
    der = float(abs(metric))
    return {
        "sessions_scored": scored,
        "total": total,
        "correct": float(details.get("correct", 0.0)),
        "false_alarm": fa,
        "missed_speech": ms,
        "speaker_confusion": sc,
        "rates": {
            "DER": der,
            "FA": 0.0 if total <= 0 else fa / total,
            "MS": 0.0 if total <= 0 else ms / total,
            "SC": 0.0 if total <= 0 else sc / total,
        },
    }


def load_metric(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def session_id_from_wav(wav_path: Path):
    return wav_path.stem


def build_references(dataset_dir: Path, output_dir: Path):
    ref_multi = output_dir / "reference_multi.jsonl"
    ref_wer = output_dir / "reference_wer.jsonl"
    manifest = output_dir / "manifest.txt"
    failures = output_dir / "failures.txt"

    wav_files = sorted((dataset_dir / "audio").glob("*.wav"))
    rows_multi = []
    rows_wer = []
    for wav in wav_files:
        session_id = session_id_from_wav(wav)
        tg = dataset_dir / "textgrid" / f"{session_id}.TextGrid"
        if not tg.exists():
            with failures.open("a", encoding="utf-8") as f:
                f.write(f"{session_id}\tMissingTextGrid\t{tg}\n")
            continue
        segs = parse_textgrid(tg)
        if not segs:
            with failures.open("a", encoding="utf-8") as f:
                f.write(f"{session_id}\tEmptyReference\t{tg}\n")
            continue
        for i, seg in enumerate(segs):
            rows_multi.append(
                {
                    "session_id": session_id,
                    "speaker": seg["speaker"],
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "words": seg["words"],
                    "segment_index": i,
                }
            )
            rows_wer.append(
                {
                    "session_id": session_id,
                    "speaker": "single",
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "words": seg["words"],
                    "segment_index": i,
                }
            )

    write_jsonl(ref_multi, rows_multi)
    write_jsonl(ref_wer, rows_wer)
    manifest.write_text("\n".join(str(p) for p in wav_files) + "\n", encoding="utf-8")
    return wav_files


def run_dataset(
    dataset_name: str,
    dataset_dir: Path,
    output_root: Path,
    pipeline: DiCoWPipeline,
    collar: int,
    resume: bool,
):
    t0 = time.time()
    out_dir = output_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = out_dir / "failures.txt"
    hyp_multi = out_dir / "hypothesis_multi.jsonl"
    hyp_wer = out_dir / "hypothesis_wer.jsonl"

    if not resume:
        for p in [failures, hyp_multi, hyp_wer]:
            if p.exists():
                p.unlink()

    wav_files = build_references(dataset_dir, out_dir)

    completed = set()
    if resume and hyp_wer.exists():
        for row in read_jsonl_rows(hyp_wer):
            completed.add(row["session_id"])

    for idx, wav in enumerate(wav_files, start=1):
        session_id = session_id_from_wav(wav)
        if session_id in completed:
            continue
        try:
            info = torchaudio.info(str(wav))
            duration = float(info.num_frames) / float(info.sample_rate)
            out = pipeline(str(wav), return_timestamps=True)
            per_spk = out.get("per_spk_outputs", []) if isinstance(out, dict) else []

            hyp_multi_rows = []
            all_segments = []
            seg_i = 0
            for spk_idx, spk_text in enumerate(per_spk):
                for st, et, words in parse_timestamped_segments(str(spk_text)):
                    st = max(0.0, min(st, duration))
                    et = max(0.0, min(et, duration))
                    if et <= st:
                        continue
                    hyp_multi_rows.append(
                        {
                            "session_id": session_id,
                            "speaker": f"hyp_spk{spk_idx}",
                            "start_time": st,
                            "end_time": et,
                            "words": words,
                            "segment_index": seg_i,
                        }
                    )
                    all_segments.append((st, et, words))
                    seg_i += 1

            if not all_segments:
                flat = clean_text(out.get("text", "") if isinstance(out, dict) else str(out))
                if flat:
                    hyp_multi_rows.append(
                        {
                            "session_id": session_id,
                            "speaker": "hyp_spk0",
                            "start_time": 0.0,
                            "end_time": duration,
                            "words": flat,
                            "segment_index": 0,
                        }
                    )
                    all_segments.append((0.0, duration, flat))

            append_jsonl(hyp_multi, hyp_multi_rows)
            hyp_wer_rows = []
            for i, (st, et, words) in enumerate(sorted(all_segments, key=lambda x: (x[0], x[1]))):
                hyp_wer_rows.append(
                    {
                        "session_id": session_id,
                        "speaker": "single",
                        "start_time": st,
                        "end_time": et,
                        "words": words,
                        "segment_index": i,
                    }
                )
            append_jsonl(hyp_wer, hyp_wer_rows)
            completed.add(session_id)
        except Exception as e:
            with failures.open("a", encoding="utf-8") as f:
                f.write(f"{session_id}\t{type(e).__name__}\t{e}\n")

        if idx % 10 == 0 or idx == len(wav_files):
            print(f"{dataset_name}: [{idx}/{len(wav_files)}] completed={len(completed)}")

    prepare_metrics(out_dir, collar=collar)
    norm_dir = out_dir / "normalized_eval"
    prepare_normalized_metrics(out_dir, norm_dir, collar=collar)

    ref_multi_rows = read_jsonl_rows(out_dir / "reference_multi.jsonl")
    hyp_multi_rows = read_jsonl_rows(out_dir / "hypothesis_multi.jsonl")
    der = compute_der(ref_multi_rows, hyp_multi_rows)
    write_json(out_dir / "der_summary.json", der)

    metrics = {
        "wer": load_metric(norm_dir / "wer_average.norm.json").get("error_rate"),
        "cpwer": load_metric(norm_dir / "cpwer_average.norm.json").get("error_rate"),
        "tcpwer": load_metric(norm_dir / "tcpwer_average.norm.json").get("error_rate"),
        "tcorcwer": load_metric(norm_dir / "tcorcwer_average.norm.json").get("error_rate"),
        "DER": der["rates"]["DER"],
    }
    summary = {
        "dataset": dataset_name,
        "files_total": len(wav_files),
        "files_completed": len(completed),
        "files_failed": 0 if not failures.exists() else sum(1 for _ in failures.open("r", encoding="utf-8")),
        "collar_seconds": collar,
        "english_normalization": {
            "standardize_numbers": False,
            "standardize_numbers_rev": True,
            "remove_fillers": True,
        },
        "runtime_seconds": time.time() - t0,
        "normalized_metrics": metrics,
        "der_details": der,
    }
    write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--testset-root",
        type=Path,
        default=Path("/home3/adnan/DICOW/mt-asr-data-prep/testset_tse"),
    )
    parser.add_argument("--datasets", nargs="+", default=["ami", "nsf", "l2m"])
    parser.add_argument("--output-root", type=Path, default=Path("output/testset_tse_dicow"))
    parser.add_argument("--dicow-model", type=str, default="BUT-FIT/DiCoW_v3_2")
    parser.add_argument("--diarization-model", type=str, default="BUT-FIT/diarizen-wavlm-large-s80-md")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--collar", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

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

    all_summaries = {}
    for dataset in args.datasets:
        dataset_dir = args.testset_root / dataset
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_dir}")
        print(f"=== Running dataset: {dataset} ===")
        all_summaries[dataset] = run_dataset(
            dataset_name=dataset,
            dataset_dir=dataset_dir,
            output_root=args.output_root,
            pipeline=pipeline,
            collar=args.collar,
            resume=args.resume,
        )

    combined = {
        d: s["normalized_metrics"] for d, s in all_summaries.items()
    }
    write_json(args.output_root / "metric_summary_normalized.json", combined)
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()

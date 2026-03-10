#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


SPK_HEADER_RE = re.compile(r"🗣️\s*Speaker\s*\d+\s*:", re.I)
TAG_RE = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    text = SPK_HEADER_RE.sub(" ", str(text))
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def append_jsonl(path: Path, rows):
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl_rows(path: Path):
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


def parse_session_id(wav_path: Path) -> str:
    return f"{wav_path.parents[1].name}/{wav_path.parent.name}"


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def extract_json_payload(text: str):
    if not text:
        return None
    s = str(text)
    if "```json" in s:
        start = s.find("```json") + 7
        end = s.find("```", start)
        if end > start:
            s = s[start:end].strip()
    else:
        start = min([i for i in [s.find("["), s.find("{")] if i != -1], default=-1)
        if start != -1:
            depth = 0
            end = -1
            for i, ch in enumerate(s[start:], start=start):
                if ch in "[{":
                    depth += 1
                elif ch in "]}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end != -1:
                s = s[start:end]
    try:
        return json.loads(s)
    except Exception:
        pass

    # Recover truncated JSON arrays by cutting to last complete object.
    lb = s.find("[")
    if lb != -1:
        t = s[lb:]
        rb = t.rfind("}")
        if rb != -1:
            t = t[: rb + 1]
            if not t.endswith("]"):
                t = t + "]"
            try:
                return json.loads(t)
            except Exception:
                pass
    return None


def robust_parse_segments(raw_text: str, processor):
    parsed = processor.post_process_transcription(raw_text)
    if parsed:
        return parsed

    payload = extract_json_payload(raw_text)
    if payload is None:
        return []
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []

    out = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "start_time": item.get("start_time", item.get("start", item.get("Start time", item.get("Start", 0.0)))),
                "end_time": item.get("end_time", item.get("end", item.get("End time", item.get("End", 0.0)))),
                "speaker_id": item.get("speaker_id", item.get("speaker", item.get("Speaker ID", item.get("Speaker", "0")))),
                "text": item.get("text", item.get("content", item.get("Content", ""))),
            }
        )
    if out:
        return out

    # Last-resort parser for truncated/non-closed JSON arrays with repeated objects.
    text = str(raw_text)
    pattern = re.compile(
        r'\{"start"\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*"end"\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*(?:"speaker"\s*:\s*([^,}]+)\s*,\s*)?"content"\s*:\s*"((?:[^"\\\\]|\\\\.)*)"',
        re.S,
    )
    for m in pattern.finditer(text):
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
                "start_time": m.group(1),
                "end_time": m.group(2),
                "speaker_id": spk_raw,
                "text": content,
            }
        )
    return out


def prepare_metrics(raw_dir: Path, collar: int):
    py = sys.executable

    ref_multi = raw_dir / "reference_multi.jsonl"
    hyp_multi = raw_dir / "hypothesis_multi.jsonl"
    ref_wer = raw_dir / "reference_wer.jsonl"
    hyp_wer = raw_dir / "hypothesis_wer.jsonl"

    ref_multi_rows = read_jsonl_rows(ref_multi)
    hyp_multi_rows = read_jsonl_rows(hyp_multi)
    ref_wer_rows = read_jsonl_rows(ref_wer)
    hyp_wer_rows = read_jsonl_rows(hyp_wer)

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
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from local_text_norm.english import EnglishTextNormalizer

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

    ref_multi_norm_path = norm_dir / "reference_multi.norm.jsonl"
    hyp_multi_norm_path = norm_dir / "hypothesis_multi.norm.jsonl"
    ref_wer_norm_path = norm_dir / "reference_wer.norm.jsonl"
    hyp_wer_norm_path = norm_dir / "hypothesis_wer.norm.jsonl"
    write_jsonl(ref_multi_norm_path, ref_multi_norm)
    write_jsonl(hyp_multi_norm_path, hyp_multi_norm)
    write_jsonl(ref_wer_norm_path, ref_wer_norm)
    write_jsonl(hyp_wer_norm_path, hyp_wer_norm)

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


def load_metric(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("output/notsofar/mtg_sc160_dicow_offline/manifest_sc160.txt"),
    )
    parser.add_argument(
        "--reference-multi",
        type=Path,
        default=Path("output/notsofar/mtg_sc160_dicow_offline/reference_multi.jsonl"),
    )
    parser.add_argument(
        "--reference-wer",
        type=Path,
        default=Path("output/notsofar/mtg_sc160_dicow_offline/reference_wer.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/vibevoice_notsofar_offline"),
    )
    parser.add_argument(
        "--vibevoice-repo",
        type=Path,
        default=Path("/home3/adnan/repos/vvasr/VibeVoice"),
    )
    parser.add_argument("--model-path", type=str, default="microsoft/VibeVoice-ASR")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--collar", type=int, default=5)
    parser.add_argument("--max-files", type=int, default=0, help="0 means all files")
    parser.add_argument("--random-seed", type=int, default=1337)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.reference_multi.exists() or not args.reference_wer.exists() or not args.manifest.exists():
        raise FileNotFoundError("Required reference/manifest files are missing.")

    manifest_lines = [
        Path(x.strip())
        for x in args.manifest.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if args.max_files > 0:
        rnd = random.Random(args.random_seed)
        if args.max_files > len(manifest_lines):
            raise ValueError(f"max-files={args.max_files} > available files={len(manifest_lines)}")
        manifest_lines = rnd.sample(manifest_lines, args.max_files)

    # Keep reference data aligned to attempted sessions only.
    attempted_sessions = {parse_session_id(p) for p in manifest_lines}
    ref_multi_rows = [r for r in read_jsonl_rows(args.reference_multi) if r.get("session_id") in attempted_sessions]
    ref_wer_rows = [r for r in read_jsonl_rows(args.reference_wer) if r.get("session_id") in attempted_sessions]

    raw_ref_multi = out_dir / "reference_multi.jsonl"
    raw_ref_wer = out_dir / "reference_wer.jsonl"
    raw_hyp_multi = out_dir / "hypothesis_multi.jsonl"
    raw_hyp_wer = out_dir / "hypothesis_wer.jsonl"
    raw_manifest = out_dir / "manifest_sc160.txt"
    failures = out_dir / "failures.txt"
    raw_results_jsonl = out_dir / "vibevoice_raw_outputs.jsonl"

    write_jsonl(raw_ref_multi, ref_multi_rows)
    write_jsonl(raw_ref_wer, ref_wer_rows)
    raw_manifest.write_text("\n".join(str(p) for p in manifest_lines) + "\n", encoding="utf-8")

    if not args.resume:
        for p in [raw_hyp_multi, raw_hyp_wer, failures, raw_results_jsonl]:
            if p.exists():
                p.unlink()

    done_sessions = set()
    if args.resume and raw_hyp_wer.exists():
        for row in read_jsonl_rows(raw_hyp_wer):
            done_sessions.add(row.get("session_id"))

    sys.path.insert(0, str(args.vibevoice_repo))
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

    if args.device.startswith("cuda") and torch.cuda.is_available():
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    processor = VibeVoiceASRProcessor.from_pretrained(
        args.model_path,
        language_model_pretrained_name="Qwen/Qwen2.5-7B",
    )
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=model_dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model = model.to(args.device)
    model.eval()
    eos_ids = [processor.tokenizer.eos_token_id]
    im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
        eos_ids.append(im_end_id)

    attempted = len(manifest_lines)
    completed = 0
    failed = 0

    for idx, wav in enumerate(manifest_lines, start=1):
        session_id = parse_session_id(wav)
        if session_id in done_sessions:
            completed += 1
            continue

        try:
            print(f"[{idx}/{attempted}] decoding {session_id}")
            inputs = processor(
                audio=str(wav),
                sampling_rate=None,
                return_tensors="pt",
                padding=True,
                add_generation_prompt=True,
                use_streaming=False,
            )
            inputs = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=processor.pad_id,
                    eos_token_id=eos_ids,
                    do_sample=False,
                    num_beams=1,
                )

            prompt_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0, prompt_len:]
            eos_positions = (generated_ids == processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                generated_ids = generated_ids[: eos_positions[0] + 1]
            raw_text = processor.decode(generated_ids, skip_special_tokens=True)
            segments = robust_parse_segments(raw_text, processor)

            hyp_multi_rows = []
            if segments:
                for seg_idx, seg in enumerate(segments):
                    words = clean_text(seg.get("text", ""))
                    if not words:
                        continue
                    spk = str(seg.get("speaker_id", "0")).strip().replace(" ", "_")
                    st = safe_float(seg.get("start_time", 0.0), 0.0)
                    et = safe_float(seg.get("end_time", st), st)
                    hyp_multi_rows.append(
                        {
                            "session_id": session_id,
                            "speaker": f"hyp_spk{spk}",
                            "start_time": st,
                            "end_time": et,
                            "words": words,
                            "segment_index": seg_idx,
                        }
                    )

            if not hyp_multi_rows:
                fallback = clean_text(raw_text)
                hyp_multi_rows = [
                    {
                        "session_id": session_id,
                        "speaker": "hyp_spk0",
                        "start_time": 0.0,
                        "end_time": 0.0,
                        "words": fallback,
                        "segment_index": 0,
                    }
                ]

            hyp_multi_rows.sort(key=lambda x: (x.get("start_time", 0.0), x.get("end_time", 0.0)))
            hyp_wer_rows = []
            for i, seg in enumerate(hyp_multi_rows):
                hyp_wer_rows.append(
                    {
                        "session_id": session_id,
                        "speaker": "single",
                        "start_time": seg["start_time"],
                        "end_time": seg["end_time"],
                        "words": seg["words"],
                        "segment_index": i,
                    }
                )

            append_jsonl(raw_hyp_multi, hyp_multi_rows)
            append_jsonl(raw_hyp_wer, hyp_wer_rows)
            append_jsonl(raw_results_jsonl, [{"session_id": session_id, "file": str(wav), "raw_text": raw_text, "segments": segments}])

            done_sessions.add(session_id)
            completed += 1
        except Exception as e:
            failed += 1
            with failures.open("a", encoding="utf-8") as f:
                f.write(f"{session_id}\t{type(e).__name__}\t{e}\n")

        if idx % 2 == 0 or idx == attempted:
            print(f"[{idx}/{attempted}] completed={completed} failed={failed}")

    prepare_metrics(out_dir, collar=args.collar)
    norm_dir = out_dir / "normalized_eval"
    prepare_normalized_metrics(out_dir, norm_dir, collar=args.collar)

    raw_metrics = {
        "wer": load_metric(out_dir / "wer_average.json"),
        "cpwer": load_metric(out_dir / "cpwer_average.json"),
        "tcpwer": load_metric(out_dir / "tcpwer_average.json"),
        "tcorcwer": load_metric(out_dir / "tcorcwer_average.json"),
    }
    norm_metrics = {
        "wer": load_metric(norm_dir / "wer_average.norm.json"),
        "cpwer": load_metric(norm_dir / "cpwer_average.norm.json"),
        "tcpwer": load_metric(norm_dir / "tcpwer_average.norm.json"),
        "tcorcwer": load_metric(norm_dir / "tcorcwer_average.norm.json"),
    }

    elapsed = time.time() - t0
    summary = {
        "files_attempted": attempted,
        "files_completed": completed,
        "files_failed": failed,
        "runtime_seconds": elapsed,
        "collar_seconds": args.collar,
        "chunking_mode": "full-audio-single-pass (use_streaming=False)",
        "model_path": args.model_path,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "raw_metrics": {k: v.get("error_rate") for k, v in raw_metrics.items()},
        "normalized_metrics": {k: v.get("error_rate") for k, v in norm_metrics.items()},
        "artifacts": {
            "manifest": str(raw_manifest),
            "reference_multi": str(raw_ref_multi),
            "reference_wer": str(raw_ref_wer),
            "hypothesis_multi": str(raw_hyp_multi),
            "hypothesis_wer": str(raw_hyp_wer),
            "raw_outputs": str(raw_results_jsonl),
            "failures": str(failures),
            "normalized_dir": str(norm_dir),
        },
    }

    metrics_summary = {
        "raw": raw_metrics,
        "normalized": norm_metrics,
    }

    write_json(out_dir / "run_summary.json", summary)
    write_json(out_dir / "metrics_summary.json", metrics_summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

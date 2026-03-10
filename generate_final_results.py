#!/usr/bin/env python3
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "output" / "final_results.json"
OUT_MD = ROOT / "output" / "final_results.md"


SOURCES = {
    "dicow_metric_summary": ROOT / "output" / "testset_tse_dicow_no_tcorc" / "metric_summary_normalized.json",
    "dicow_run_summary_ami": ROOT / "output" / "testset_tse_dicow_no_tcorc" / "ami" / "run_summary.json",
    "dicow_run_summary_nsf": ROOT / "output" / "testset_tse_dicow_no_tcorc" / "nsf" / "run_summary.json",
    "dicow_run_summary_l2m": ROOT / "output" / "testset_tse_dicow_no_tcorc" / "l2m" / "run_summary.json",
    "dicow_der_ami": ROOT / "output" / "testset_tse_dicow_no_tcorc" / "ami" / "der_summary.json",
    "dicow_der_nsf": ROOT / "output" / "testset_tse_dicow_no_tcorc" / "nsf" / "der_summary.json",
    "dicow_der_l2m": ROOT / "output" / "testset_tse_dicow_no_tcorc" / "l2m" / "der_summary.json",
    "dicow_reassigned_run_summary": ROOT / "output" / "libri2mix_max_both_reassigned_eval" / "run_summary.json",
    "solospeech_summary": ROOT / "output" / "libri2mix_solospeech_whisper" / "meeteval_norm" / "summary.json",
    "method_script_dicow_score": ROOT / "score_testset_tse_from_existing_dicow_no_tcorc.py",
    "method_script_dicow_eval": ROOT / "evaluate_testset_tse_dicow_offline.py",
    "method_script_solospeech": ROOT / "tmp_solospeech_meeteval.py",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_sources():
    missing = [str(path) for path in SOURCES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required source files:\n" + "\n".join(missing))


def build_methodology():
    return {
        "dicow_testset_tse_no_tcorc_norm": {
            "normalization": "EnglishTextNormalizer(standardize_numbers=False, standardize_numbers_rev=True, remove_fillers=True)",
            "wer": "Computed with `python -m meeteval.wer wer` on aggregated single-speaker seglst.",
            "cpwer": "Computed with `python -m meeteval.wer cpwer` on aggregated multi-speaker seglst.",
            "tcpwer": "Computed with `python -m meeteval.wer tcpwer --collar 5` on timestamped multi-speaker seglst.",
            "der": "Computed with pyannote `DiarizationErrorRate(collar=0.0, skip_overlap=False)`.",
            "speaker_handling": "Original hypothesis speaker labels were kept (no speaker reassignment).",
            "source_scripts": [
                "score_testset_tse_from_existing_dicow_no_tcorc.py",
                "evaluate_testset_tse_dicow_offline.py",
            ],
        },
        "dicow_l2m_max_both_reassigned_norm": {
            "normalization": "EnglishTextNormalizer(standardize_numbers=False, standardize_numbers_rev=True, remove_fillers=True)",
            "wer": "Computed with `python -m meeteval.wer wer` on aggregated single-speaker seglst.",
            "cpwer": "Computed with `python -m meeteval.wer cpwer` on aggregated multi-speaker seglst.",
            "tcpwer": "Computed with `python -m meeteval.wer tcpwer --collar 5` on timestamped multi-speaker seglst.",
            "der": "Computed with pyannote `DiarizationErrorRate(collar=0.0, skip_overlap=False)`.",
            "speaker_handling": "Hypothesis speakers were reassigned to transcript-closest reference speakers before scoring.",
            "source_scripts": [
                "score_l2m_max_both_reassigned.py",
                "evaluate_testset_tse_dicow_offline.py",
            ],
        },
        "solospeech_whisper_meeteval_norm": {
            "normalization": "EnglishTextNormalizer(standardize_numbers=False, standardize_numbers_rev=True, remove_fillers=True)",
            "wer": "Computed with `python -m meeteval.wer wer` on aggregated single-speaker seglst.",
            "cpwer": "Computed with `python -m meeteval.wer cpwer` on aggregated multi-speaker seglst.",
            "tcpwer": "Not computed for this run.",
            "der": "Not computed for this run.",
            "speaker_handling": "`target_side` from predictions CSV was used as the speaker key during meeteval preparation.",
            "source_scripts": [
                "tmp_solospeech_meeteval.py",
            ],
        },
    }


def build_systems():
    dicow_summary = load_json(SOURCES["dicow_metric_summary"])
    reassigned_summary = load_json(SOURCES["dicow_reassigned_run_summary"])
    solospeech_summary = load_json(SOURCES["solospeech_summary"])

    systems = []
    for dataset in ("ami", "nsf"):
        item = dicow_summary[dataset]
        metrics = item["normalized_metrics"]
        systems.append(
            {
                "system": "dicow",
                "dataset": dataset,
                "run_variant": "testset_tse_no_tcorc_norm",
                "metrics": {
                    "wer": metrics["wer"],
                    "cpwer": metrics["cpwer"],
                    "tcpwer": metrics["tcpwer"],
                    "der": metrics["DER"],
                },
                "coverage": {
                    "scored_files": item["scored_files"],
                    "missing_files": item["missing_files"],
                    "num_sessions_total": None,
                    "num_sessions_scored": None,
                    "num_sessions_missing_hypothesis": None,
                    "num_sessions_missing_textgrid": None,
                },
                "methodology_ref": "dicow_testset_tse_no_tcorc_norm",
                "source_files": [
                    "output/testset_tse_dicow_no_tcorc/metric_summary_normalized.json",
                    f"output/testset_tse_dicow_no_tcorc/{dataset}/run_summary.json",
                    f"output/testset_tse_dicow_no_tcorc/{dataset}/der_summary.json",
                ],
            }
        )

    reassigned_metrics = reassigned_summary["metrics"]
    systems.append(
        {
            "system": "dicow",
            "dataset": reassigned_summary["dataset"],
            "run_variant": "l2m_max_both_old_reassigned_norm",
            "metrics": {
                "wer": reassigned_metrics["WER"],
                "cpwer": reassigned_metrics["cpWER"],
                "tcpwer": reassigned_metrics["tcpWER"],
                "der": reassigned_metrics["DER"],
            },
            "coverage": {
                "scored_files": None,
                "missing_files": None,
                "num_sessions_total": reassigned_summary["num_sessions_total"],
                "num_sessions_scored": reassigned_summary["num_sessions_scored"],
                "num_sessions_missing_hypothesis": reassigned_summary["num_sessions_missing_hypothesis"],
                "num_sessions_missing_textgrid": reassigned_summary["num_sessions_missing_textgrid"],
            },
            "methodology_ref": "dicow_l2m_max_both_reassigned_norm",
            "source_files": [
                "output/libri2mix_max_both_reassigned_eval/run_summary.json",
            ],
        }
    )

    for mode in ("non_streaming", "streaming"):
        entry = solospeech_summary[mode]
        systems.append(
            {
                "system": "solospeech_whisper",
                "dataset": "libri2mix",
                "run_variant": mode,
                "metrics": {
                    "wer": entry["wer"],
                    "cpwer": entry["cpwer"],
                    "tcpwer": None,
                    "der": None,
                },
                "coverage": {
                    "scored_files": None,
                    "missing_files": None,
                    "num_sessions_total": None,
                    "num_sessions_scored": None,
                    "num_sessions_missing_hypothesis": None,
                    "num_sessions_missing_textgrid": None,
                },
                "methodology_ref": "solospeech_whisper_meeteval_norm",
                "source_files": [
                    "output/libri2mix_solospeech_whisper/meeteval_norm/summary.json",
                    f"output/libri2mix_solospeech_whisper/meeteval_norm/{mode}/wer_average.norm.json",
                    f"output/libri2mix_solospeech_whisper/meeteval_norm/{mode}/cpwer_average.norm.json",
                ],
            }
        )

    return systems


def render_number(value):
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def render_markdown(report):
    lines = []
    lines.append("# Final Results: DiCoW + SoloSpeech/Whisper")
    lines.append("")
    lines.append("## A) Consolidated Metrics")
    lines.append("")
    lines.append("| System | Dataset | Run Variant | WER | cpWER | tcpWER | DER |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in report["systems"]:
        lines.append(
            "| {system} | {dataset} | {variant} | {wer} | {cpwer} | {tcpwer} | {der} |".format(
                system=row["system"],
                dataset=row["dataset"],
                variant=row["run_variant"],
                wer=render_number(row["metrics"]["wer"]),
                cpwer=render_number(row["metrics"]["cpwer"]),
                tcpwer=render_number(row["metrics"]["tcpwer"]),
                der=render_number(row["metrics"]["der"]),
            )
        )
    lines.append("")
    lines.append("## B) Methodology")
    lines.append("")
    for key, details in report["methodology"].items():
        lines.append(f"### `{key}`")
        lines.append(f"- Normalization: `{details['normalization']}`")
        lines.append(f"- WER: {details['wer']}")
        lines.append(f"- cpWER: {details['cpwer']}")
        lines.append(f"- tcpWER: {details['tcpwer']}")
        lines.append(f"- DER: {details['der']}")
        lines.append(f"- Speaker handling: {details['speaker_handling']}")
        lines.append("- Source scripts:")
        for script in details["source_scripts"]:
            lines.append(f"  - `{script}`")
        lines.append("")

    lines.append("## C) Provenance (Source Artifacts)")
    lines.append("")
    for row in report["systems"]:
        lines.append(f"### `{row['system']} | {row['dataset']} | {row['run_variant']}`")
        for src in row["source_files"]:
            lines.append(f"- `{src}`")
        lines.append("")

    lines.append("## D) Metric Availability Notes")
    lines.append("")
    lines.append("- `solospeech_whisper` rows have `tcpWER` and `DER` as `N/A` because those metrics were not computed in that evaluation pipeline.")
    lines.append("")
    lines.append(f"_Generated at: {report['generated_at']}_")
    lines.append("")
    return "\n".join(lines)


def validate_report(report):
    required = {
        ("dicow", "ami"),
        ("dicow", "nsf"),
        ("dicow", "l2m (max_both old run, reassigned)"),
        ("solospeech_whisper", "libri2mix"),
    }
    present = {(r["system"], r["dataset"]) for r in report["systems"]}
    if not required.issubset(present):
        missing = sorted(required - present)
        raise ValueError(f"Missing required rows: {missing}")

    run_variants = {(r["system"], r["dataset"], r["run_variant"]) for r in report["systems"]}
    for required_variant in [
        ("solospeech_whisper", "libri2mix", "streaming"),
        ("solospeech_whisper", "libri2mix", "non_streaming"),
    ]:
        if required_variant not in run_variants:
            raise ValueError(f"Missing required run variant row: {required_variant}")


def main():
    validate_sources()
    systems = build_systems()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": build_methodology(),
        "systems": systems,
    }
    validate_report(report)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_markdown(report), encoding="utf-8")

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()

# Metrics Configuration Reference

This file documents the exact metric protocols used in this repository so results are reproducible across ASR systems.

## 1) Core Metrics

The following ASR metrics are computed with `meeteval.wer`:

- `wer` (single-speaker aggregated WER)
- `cpwer` (concatenated minimum-permutation WER)
- `tcpwer` (time-constrained cpWER)
- `tcorcwer` (time-constrained ORC-WER)

## 2) Time Collar

- For `tcpwer` and `tcorcwer`: `--collar 5`
- For plain `wer` and `cpwer`: no collar argument.

## 3) Input/Output Schema for Scoring

Metrics are computed from JSON/JSONL rows with fields:

- `session_id` (string)
- `speaker` (string)
- `start_time` (float seconds)
- `end_time` (float seconds)
- `words` (string transcript)

Intermediate files:

- `reference_multi.jsonl`
- `hypothesis_multi.jsonl`
- `reference_wer.jsonl`
- `hypothesis_wer.jsonl`

Converted to SegLST JSON for `meeteval`:

- `reference_multi*.seglst.json`
- `hypothesis_multi*.seglst.json`
- `reference_wer_agg*.seglst.json`
- `hypothesis_wer_agg*.seglst.json`

Aggregation rule for `wer`/`cpwer`:

- Rows are aggregated by `(session_id, speaker)` and sorted by `start_time`, then `words` are concatenated with spaces.

## 4) Text Normalization Protocols

Two protocols are used in this repo.

### A. Legacy/Raw Protocol (older run artifacts)

- Preprocessing in evaluation scripts applies lowercasing and whitespace cleanup (`clean_text`).
- `meeteval` normalizer is left at default (`None`).
- AMI reference excludes XML tokens with `punc=true`.

Used by:

- `metrics.md`
- `output/*/wer_average.json`, `cpwer_average.json`, `tcpwer_average.json`, `tcorcwer_average.json`

### B. Whisper-like Normalized Protocol (current comparison protocol)

- Uses NOTSOFAR Whisper-like normalizer:
  - Source: `https://github.com/microsoft/NOTSOFAR1-Challenge/blob/main/utils/text_norm_whisper_like/english.py#L542`
  - Local copy: `local_text_norm/english.py`, `local_text_norm/basic.py`, `local_text_norm/english.json`, `local_text_norm/pre_english.json`
- Config:
  - `EnglishTextNormalizer(standardize_numbers=False, standardize_numbers_rev=True, remove_fillers=True)`
- Applied to both reference and hypothesis `words` before metric computation.
- `tcpwer`/`tcorcwer` still use `--collar 5`.

Used by:

- `metrics_new.md`
- `output/notsofar/mtg_sc160_dicow_offline/normalized_eval/*`
- `output/ami_dicow_offline/normalized_eval/*`
- `output/normalized_metric_summary.json`

## 5) Diarization Metrics (DER/MS/FA/SC)

Diarization metrics are computed with `pyannote.metrics`:

- Tool: `pyannote.metrics.diarization.DiarizationErrorRate`
- Parameters:
  - `collar=0.0`
  - `skip_overlap=False`
- UEM:
  - Not explicitly provided; pyannote uses union of reference/hypothesis extents.

Reported components:

- `DER`
- `MS` (missed speech)
- `FA` (false alarm)
- `SC` (speaker confusion)

Artifact:

- `output/der_summary.json`

## 6) Dataset Definitions Used in Existing Results

### Notsofar (MTG SC160)

- Root: `/home3/adnan/data/TS_ASR/benchmark-datasets/eval_set/240629.1_eval_small_with_GT/MTG`
- Manifest: `output/notsofar/mtg_sc160_dicow_offline/manifest_sc160.txt`
- Session ID convention: `MTG_xxxxx/sc_xxx` (derived from path)

### AMI

- Audio root: `/home3/adnan/data/TS_ASR/AMI/eval_ami_full`
- Annotation root: `/home3/adnan/data/TS_ASR/AMI/ami_manual_1.6.1`
- Manifest: `output/ami_dicow_offline/manifest_sc160.txt`
- Session ID convention: meeting ID (e.g., `EN2002a`)

## 7) Canonical CLI Commands (meeteval)

Example command shapes used:

```bash
python -m meeteval.wer wer \
  -r reference_wer_agg.seglst.json \
  -h hypothesis_wer_agg.seglst.json \
  --average-out wer_average.json \
  --per-reco-out wer_per_reco.json

python -m meeteval.wer cpwer \
  -r reference_multi_agg.seglst.json \
  -h hypothesis_multi_agg.seglst.json \
  --average-out cpwer_average.json \
  --per-reco-out cpwer_per_reco.json

python -m meeteval.wer tcpwer \
  -r reference_multi.seglst.json \
  -h hypothesis_multi.seglst.json \
  --collar 5 \
  --average-out tcpwer_average.json \
  --per-reco-out tcpwer_per_reco.json

python -m meeteval.wer tcorcwer \
  -r reference_multi.seglst.json \
  -h hypothesis_multi.seglst.json \
  --collar 5 \
  --average-out tcorcwer_average.json \
  --per-reco-out tcorcwer_per_reco.json
```

## 8) Notes for Cross-System Comparison

To compare models fairly:

- Use identical session set and reference files per dataset.
- Use the same normalization protocol (recommended: Whisper-like normalized protocol above).
- Use the same collar (`5s`) for time-constrained metrics.
- Keep DER settings fixed (`collar=0`, `skip_overlap=False`).

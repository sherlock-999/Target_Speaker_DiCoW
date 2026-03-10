# DiCoW Offline Evaluation Metrics (Normalized)

## Test Conditions

- Model: `BUT-FIT/DiCoW_v3_2`
- Diarization model: `BUT-FIT/diarizen-wavlm-large-s80-md`
- Evaluation scripts:
  - Notsofar: `evaluate_mtg_sc160_dicow_offline.py`
  - AMI: `evaluate_ami_dicow_offline.py`
- Metric toolkit: `meeteval.wer`
- Computed metrics: `wer`, `cpwer`, `tcpwer`, `tcorcwer`
- Collar for time-constrained metrics (`tcpwer`, `tcorcwer`): `5s`

### Text Normalization Used for This File

- Normalizer: NOTSOFAR Whisper-like English normalizer
- Source: `https://github.com/microsoft/NOTSOFAR1-Challenge/blob/main/utils/text_norm_whisper_like/english.py#L542`
- Local files used:
  - `local_text_norm/english.py`
  - `local_text_norm/basic.py`
  - `local_text_norm/english.json`
  - `local_text_norm/pre_english.json`
- Config: `EnglishTextNormalizer(standardize_numbers=False, standardize_numbers_rev=True, remove_fillers=True)`

## Test Sets Used

| Dataset | Data Root Used | Annotation / GT Source | Manifest Entries | Scored Files | Scored Audio Hours |
|---|---|---|---:|---:|---:|
| Notsofar (MTG SC160) | `/home3/adnan/data/TS_ASR/benchmark-datasets/eval_set/240629.1_eval_small_with_GT/MTG` | From dataset GT loaded by `evaluate_mtg_sc160_dicow_offline.py` | 160 | 160 | 16.667 h |
| AMI | `/home3/adnan/data/TS_ASR/AMI/eval_ami_full` | `/home3/adnan/data/TS_ASR/AMI/ami_manual_1.6.1` (`words/*.words.xml`) | 19 | 16 | 9.062 h |

Notes:
- AMI manifest has 19 entries, but scoring is on 16 unique meeting sessions.

## Speaker Statistics (From Ground Truth)

| Dataset | Total Speaker-Session Pairs | Speakers/Recording (Min-Max) | Speakers/Recording (Avg) |
|---|---:|---:|---:|
| Notsofar (MTG SC160) | 758 | 3-7 | 4.738 |
| AMI | 63 | 3-4 | 3.938 |

## Overlap Ratio Metrics (From Ground Truth)

Definition:
- `Overlap ratio = (time with >1 active speakers) / (total speech-active union time)`.

| Dataset | Dataset Overlap Ratio | Overlap Hours | Speech-Union Hours | Per-Session Overlap Ratio (Min-Max) |
|---|---:|---:|---:|---:|
| Notsofar (MTG SC160) | 0.2921 (29.21%) | 4.524 h | 15.484 h | 0.0000 - 0.6945 |
| AMI | 0.1464 (14.64%) | 1.069 h | 7.299 h | 0.0432 - 0.3011 |

## Diarization Metrics (DER / MS / FA / SC)

Settings used:
- Tool: `pyannote.metrics.diarization.DiarizationErrorRate`
- Collar: `0.0s`
- `skip_overlap=False`
- UEM: union of reference/hypothesis extents

| Dataset | DER | MS | FA | SC |
|---|---:|---:|---:|---:|
| Notsofar (MTG SC160) | 0.1948 (19.48%) | 0.1077 (10.77%) | 0.0263 (2.63%) | 0.0609 (6.09%) |
| AMI | 0.1485 (14.85%) | 0.0850 (8.50%) | 0.0374 (3.74%) | 0.0261 (2.61%) |

## Recognition Metrics (Normalized + Collar 5s)

### Notsofar (MTG SC160)

| Metric | Error Rate | Errors | Ref Length | Insertions | Deletions | Substitutions |
|---|---:|---:|---:|---:|---:|---:|
| WER | 0.2578 | 60781 | 235740 | 13653 | 24945 | 22183 |
| cpWER | 0.2671 | 62964 | 235740 | 15106 | 26398 | 21460 |
| tcpWER (collar=5s) | 0.2804 | 66103 | 235740 | 18219 | 29511 | 18373 |
| tcORCWER (collar=5s) | 0.2010 | 47380 | 235740 | 7787 | 19079 | 20514 |

### AMI

| Metric | Error Rate | Errors | Ref Length | Insertions | Deletions | Substitutions |
|---|---:|---:|---:|---:|---:|---:|
| WER | 0.2146 | 19285 | 89877 | 5272 | 6428 | 7585 |
| cpWER | 0.2016 | 18117 | 89877 | 4938 | 6094 | 7085 |
| tcpWER (collar=5s) | 0.2138 | 19220 | 89877 | 5751 | 6907 | 6562 |
| tcORCWER (collar=5s) | 0.1851 | 16636 | 89877 | 4212 | 5368 | 7056 |

## Artifact Paths

- Notsofar original output dir: `output/notsofar/mtg_sc160_dicow_offline`
- AMI original output dir: `output/ami_dicow_offline`
- Notsofar normalized metrics dir: `output/notsofar/mtg_sc160_dicow_offline/normalized_eval`
- AMI normalized metrics dir: `output/ami_dicow_offline/normalized_eval`
- Normalized metric summary: `output/normalized_metric_summary.json`
- DER summary: `output/der_summary.json`

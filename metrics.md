# DiCoW Offline Evaluation Metrics

## Test Conditions

- Model: `BUT-FIT/DiCoW_v3_2`
- Diarization model: `BUT-FIT/diarizen-wavlm-large-s80-md`
- Evaluation scripts:
  - Notsofar: `evaluate_mtg_sc160_dicow_offline.py`
  - AMI: `evaluate_ami_dicow_offline.py`
- Metric toolkit: `meeteval.wer`
- Computed metrics: `wer`, `cpwer`, `tcpwer`, `tcorcwer`
- Collar for time-constrained metrics (`tcpwer`, `tcorcwer`): `5s`
- Text normalization in this pipeline:
  - Text is lowercased in preprocessing (`clean_text`)
  - XML punctuation tokens with `punc=true` are removed on AMI reference side
  - `meeteval` normalizer left at default (`None`)

## Test Sets Used

| Dataset | Data Root Used | Annotation / GT Source | Manifest Entries | Scored Files | Scored Audio Hours |
|---|---|---|---:|---:|---:|
| Notsofar (MTG SC160) | `/home3/adnan/data/TS_ASR/benchmark-datasets/eval_set/240629.1_eval_small_with_GT/MTG` | From dataset GT loaded by `evaluate_mtg_sc160_dicow_offline.py` | 160 | 160 | 16.667 h |
| AMI | `/home3/adnan/data/TS_ASR/AMI/eval_ami_full` | `/home3/adnan/data/TS_ASR/AMI/ami_manual_1.6.1` (`words/*.words.xml`) | 19 | 16 | 9.062 h |

Notes:
- AMI manifest has 19 entries, but scoring is on 16 unique meeting sessions (`run_summary: successful_inference_files=16`).

## Speaker Statistics (From Ground Truth)

Definition:
- `Speakers/recording` is computed from GT speaker labels per scored recording/session.

| Dataset | Total Speaker-Session Pairs | Speakers/Recording (Min-Max) | Speakers/Recording (Avg) |
|---|---:|---:|---:|
| Notsofar (MTG SC160) | 758 | 3-7 | 4.738 |
| AMI | 63 | 3-4 | 3.938 |

This gives the requested speaker range over the dataset (per recording/session).

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
- UEM: not explicitly provided (pyannote uses union of reference/hypothesis extents)
- Input files:
  - Notsofar: `output/notsofar/mtg_sc160_dicow_offline/reference_multi.jsonl` vs `output/notsofar/mtg_sc160_dicow_offline/hypothesis_multi.jsonl`
  - AMI: `output/ami_dicow_offline/reference_multi.jsonl` vs `output/ami_dicow_offline/hypothesis_multi.jsonl`

Definitions:
- `DER = MS + FA + SC` (all normalized by total reference speaker time).
- `MS`: missed speech
- `FA`: false alarm speech
- `SC`: speaker confusion

| Dataset | DER | MS | FA | SC |
|---|---:|---:|---:|---:|
| Notsofar (MTG SC160) | 0.1948 (19.48%) | 0.1077 (10.77%) | 0.0263 (2.63%) | 0.0609 (6.09%) |
| AMI | 0.1485 (14.85%) | 0.0850 (8.50%) | 0.0374 (3.74%) | 0.0261 (2.61%) |

Component totals (speaker-time units):

| Dataset | Total | Correct | Missed Speech | False Alarm | Speaker Confusion |
|---|---:|---:|---:|---:|---:|
| Notsofar (MTG SC160) | 75734.470 | 62968.733 | 8156.093 | 1990.423 | 4609.644 |
| AMI | 30771.434 | 27351.758 | 2615.084 | 1150.270 | 804.592 |

## Recognition Metrics

### Notsofar (MTG SC160)

Source files:
- `output/notsofar/mtg_sc160_dicow_offline/wer_average.json`
- `output/notsofar/mtg_sc160_dicow_offline/cpwer_average.json`
- `output/notsofar/mtg_sc160_dicow_offline/tcpwer_average.json`
- `output/notsofar/mtg_sc160_dicow_offline/tcorcwer_average.json`

| Metric | Error Rate | Errors | Ref Length | Insertions | Deletions | Substitutions |
|---|---:|---:|---:|---:|---:|---:|
| WER | 0.4561 | 107075 | 234752 | 17917 | 28254 | 60904 |
| cpWER | 0.4687 | 110027 | 234752 | 20624 | 30961 | 58442 |
| tcpWER (collar=5s) | 0.4849 | 113835 | 234752 | 24728 | 35065 | 54042 |
| tcORCWER (collar=5s) | 0.4157 | 97581 | 234752 | 13224 | 23561 | 60796 |

### AMI

Source files:
- `output/ami_dicow_offline/wer_average.json`
- `output/ami_dicow_offline/cpwer_average.json`
- `output/ami_dicow_offline/tcpwer_average.json`
- `output/ami_dicow_offline/tcorcwer_average.json`

| Metric | Error Rate | Errors | Ref Length | Insertions | Deletions | Substitutions |
|---|---:|---:|---:|---:|---:|---:|
| WER | 0.3447 | 30616 | 88828 | 8152 | 8264 | 14200 |
| cpWER | 0.3340 | 29669 | 88828 | 8095 | 8207 | 13367 |
| tcpWER (collar=5s) | 0.3478 | 30895 | 88828 | 9109 | 9221 | 12565 |
| tcORCWER (collar=5s) | 0.3201 | 28431 | 88828 | 7467 | 7579 | 13385 |

## Artifact Paths

- Notsofar output dir: `output/notsofar/mtg_sc160_dicow_offline`
- AMI output dir: `output/ami_dicow_offline`
- DER summary (computed): `output/der_summary.json`

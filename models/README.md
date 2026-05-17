# Model Cache

This directory is the HuggingFace cache for all models used by the pipeline.
Models are auto-downloaded on first run — you only need to pre-populate this
folder if you plan to run strictly offline.

## Pre-download all models (optional)

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

cache = Path.cwd() / "models"
cache.mkdir(parents=True, exist_ok=True)

for repo in (
    "BUT-FIT/DiCoW_v3_2",
    "BUT-FIT/diarizen-wavlm-large-s80-md",
    "Adnan256/streaming-target-stno-wavlm-base",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
):
    snapshot_download(repo_id=repo, cache_dir=cache)
    print(f"OK: {repo}")
PY
```

## When you need this

- **Online / normal use** — skip this. HF Hub downloads models on first run.
- **Offline / air-gapped machines** — run the script once on a connected machine,
  then copy the `models/` directory to the target server.

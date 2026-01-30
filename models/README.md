# Models Directory

This folder is the dedicated cache location for all DiCoW and DiariZen models
used by the pipeline. Keep model downloads here so runs are reproducible and
do not depend on the global Hugging Face cache.

## Populate this folder (required before offline use)

Run these commands from the repository root:

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

models_dir = Path.cwd() / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# DiCoW model
snapshot_download(repo_id="BUT-FIT/DiCoW_v3_2", cache_dir=models_dir)

# DiariZen diarization model
snapshot_download(repo_id="BUT-FIT/diarizen-wavlm-large-s80-md", cache_dir=models_dir)

# DiariZen embedding model dependency
hf_hub_download(
    repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
    filename="pytorch_model.bin",
    cache_dir=models_dir
)
PY
```

Notes:
- The pipeline scripts (`inference.py`, `app.py`, `local_run.py`) are configured
  to use this folder as the cache directory.
- If you want strict offline mode, run the above once, then keep this directory
  intact for future runs.

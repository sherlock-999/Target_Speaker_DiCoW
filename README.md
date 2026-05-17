# DiCoW: Diarization-Conditioned Whisper for Target Speaker Automatic Speech Recognition

DiCoW (Diarization-Conditioned Whisper) enhances OpenAI's Whisper ASR model by integrating **speaker diarization** for multi-speaker transcription. Supports offline, chunked-online, and streaming target-speaker ASR, powered by **BUT-FIT/DiariZen** speaker segmentation and **Adnan256/streaming-target-stno-wavlm-base** for streaming activity detection.


## Features

- **Multi-Speaker ASR**: Handles multi-speaker audio using diarization-aware transcription.
- **Target Speaker ASR (Online & Streaming)**: Chunked VAD-based online transcription and streaming rolling-window decoding with STNO conditioning.
- **Flexible Input Sources**:
  - **Microphone**: Record and transcribe live audio.
  - **Audio File Upload**: Upload pre-recorded audio files for transcription.
  - **Reference Speaker Enrollment**: Provide a reference audio sample to isolate a specific target speaker.
  - **Folder Batch Processing**: Process multiple `.wav` files from a directory via the command line.
- **Three Inference Modes**:

  | Mode | Route | Description |
  |------|-------|-------------|
  | **Offline DiCoW** | `/gradio-demo` | Full-utterance diarization + speaker-attributed ASR. Optional enrollment for target-speaker mode. |
  | **Chunked Online** | `/chunked-demo` | VAD-based speech chunks → per-chunk diarization → target speaker matched by embedding similarity → DiCoW ASR. |
  | **Streaming STNO** | `/target-demo` | Streaming STNO model tracks target speaker activity → rolling-window DiCoW decodes with committed transcript. |
- **Diarization**: Powered by `BUT-FIT/diarizen-wavlm-large-s80-md` for accurate speaker segmentation.
- **Built with 🤗 Transformers**: Uses Whisper checkpoints adapted for diarization conditioning via `BUT-FIT/DiCoW_v3_2`.
- **FastAPI + Gradio Web Interface**: Modern web UI with three dedicated demo routes.

## Models

| Model | Description | License |
|-------|-------------|---------|
| [`BUT-FIT/DiCoW_v3_2`](https://huggingface.co/BUT-FIT/DiCoW_v3_2) | Latest DiCoW checkpoint for diarization-conditioned Whisper | CC BY 4.0 |
| [`BUT-FIT/SE_DiCoW`](https://huggingface.co/BUT-FIT/SE_DiCoW) | Speaker-Embedding conditioned DiCoW variant | CC BY 4.0 |
| [`BUT-FIT/diarizen-wavlm-large-s80-md`](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md) | DiariZen speaker segmentation model | CC BY-NC 4.0 |
| [`Adnan256/streaming-target-stno-wavlm-base`](https://huggingface.co/Adnan256/streaming-target-stno-wavlm-base) | Streaming STNO target-speaker activity detection | CC BY-NC 4.0 |

## Installation

### Requirements

- **Conda** (Miniconda or Anaconda)
- **Python 3.11**
- **FFmpeg**: Required for audio processing.
- **NVIDIA GPU with CUDA 12.1+**: Recommended for inference (CPU fallback supported).

### Setup (Verified)

#### 1. Create conda environment

```bash
conda create -n dicow python=3.11 -y
conda activate dicow
```

#### 2. Install PyTorch with CUDA support

```bash
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

#### 3. Install core Python dependencies

```bash
pip install transformers==4.42.0
pip install pyannote.core==5.0.0 pyannote.database==5.1.3 pyannote.metrics==3.2.1 pyannote.pipeline==3.0.1
pip install librosa soundfile fastapi uvicorn gradio==3.50.2
pip install accelerate onnxruntime toml silero-vad coloredlogs
```

> **Note:** `requirements.txt` contains a full `pip freeze` snapshot with many development tools (jupyter, pytest, etc.). The commands above install the minimal set needed for inference.

#### 4. Clone the repository

```bash
git clone https://github.com/sherlock-999/Target_Speaker_DiCoW.git
cd Target_Speaker_DiCoW
git checkout enroll_DiCoW_adnan
```

> **Note:** DiariZen is vendored (included directly in this repo). No git submodule commands needed.

#### 5. Install pyannote.audio from vendored DiariZen

```bash
pip install "setuptools<70" flit_core
cd DiariZen/pyannote-audio
pip install --no-build-isolation -e .
cd ../..
```

#### 6. Install DiariZen

```bash
cd DiariZen
pip install --no-build-isolation -e .
cd ..
```

#### 7. Verify the installation

```bash
python -c "
import sys; sys.path.insert(0, 'DiariZen')
from pipeline import DiCoWPipeline
from diarizen.pipelines.inference import DiariZenPipeline
print('All imports OK — installation successful!')
"
```

#### 8. (Optional) Pre-download models for offline use

```bash
python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

models_dir = Path.cwd() / "models"
models_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(repo_id="BUT-FIT/DiCoW_v3_2", cache_dir=models_dir)
snapshot_download(repo_id="BUT-FIT/diarizen-wavlm-large-s80-md", cache_dir=models_dir)
PY
```

## Usage

### Web Interface (FastAPI + Gradio)

Run the application locally:
```bash
python app.py
```

This starts a FastAPI server with three Gradio-mounted UIs:
- **Offline DiCoW UI**: `http://localhost:7860/gradio-demo`
- **Chunked Online UI**: `http://localhost:7860/chunked-demo`
- **Streaming STNO UI**: `http://localhost:7860/target-demo`

For a shared Gradio link (no FastAPI routing):
```bash
python app.py --share
```

### Command-Line Batch Processing

Process a folder of audio files:
```bash
python inference.py --input-folder /path/to/wav/files --output-folder ./output
```

With target-speaker reference audio:
```bash
python inference.py --input-folder /path/to/wav/files \
    --reference-audio /path/to/speaker_enrollment.wav \
    --output-folder ./output
```

Additional CLI options:
```bash
python inference.py \
    --input-folder /path/to/wav/files \
    --output-folder ./output \
    --dicow-model BUT-FIT/DiCoW_v3_2 \
    --diarization-model BUT-FIT/diarizen-wavlm-large-s80-md \
    --reference-audio /path/to/reference.wav \
    --file-pattern "*.wav" \
    --device cuda \
    --verbose
```

## Pipeline Architecture

### Offline DiCoW (`DiCoWPipeline`)

Extends HuggingFace's `AutomaticSpeechRecognitionPipeline` with diarization awareness:

1. **Speaker Diarization**: Runs DiariZen to segment speakers.
2. **STNO Mask Generation**: Produces a 4-class mask (silence, target speaker, non-target speaker, overlap).
3. **Conditioned Decoding**: The mask is fed into Whisper's encoder via custom FDDT (Frame-Dependent Delay-Time) layers to focus on the target speaker.
4. **Transcription**: Generates speaker-aware transcripts with optional timestamps.

Supports **target speaker selection** via reference audio embedding similarity (cosine similarity).

### Online Target Speaker Chunked (`OnlineTargetDiCoW`)

VAD-based pseudo-streaming variant:

1. **Silero VAD**: Splits audio into speech chunks (3–8s).
2. **Per-Chunk Diarization**: Runs full diarization on each chunk.
3. **Speaker Matching**: Computes embedding cosine similarity against enrollment.
4. **Target-Speaker DiCoW**: Conditions DiCoW on the matched speaker.

### Online Target Speaker Streaming (`RollingOnlineTargetDiCoW`)

Streaming variant for low-latency target-speaker ASR:

1. **Chunked Processing**: Processes audio in overlapping windows.
2. **STNO Conditioning**: Real-time streaming speaker-timeline conditioning.
3. **Rolling Decode Window**: Maintains a sliding decode window with word-level commitment.
4. **Commit Strategy**: Commits transcribed words once they are stably outside the lookback window.

## Modes

1. **Offline DiCoW (Microphone)**: Record live audio for diarization-conditioned transcription.
2. **Offline DiCoW (File Upload)**: Upload pre-recorded audio files for transcription.
3. **Chunked Online Target**: VAD-chunked per-chunk diarization + target speaker matching.
4. **Streaming STNO Target**: Streaming STNO activity tracking + rolling-window DiCoW decoding.
5. **Folder Batch Processing**: Process multiple WAV files from command line.

## Contributing

We welcome contributions! If you'd like to add features or improve the pipeline, please open an issue or submit a pull request.

## License

This project combines multiple components, each with its own license:

* **DiCoW** (this repository): Licensed under the [Apache License 2.0](LICENSE).
* **DiCoW Model Weights** (`BUT-FIT/DiCoW_v3_2`, `BUT-FIT/SE_DiCoW`): Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) – attribution required.
* **DiariZen** (`BUT-FIT/diarizen-wavlm-large-s80-md`): Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) – free for research and non-commercial use only.

Please ensure compliance with the respective licenses when using, modifying, or redistributing these components.

## Citation

If you use our model or code, please cite:

```bibtex
@article{POLOK2026101841,
    title = {DiCoW: Diarization-conditioned Whisper for target speaker automatic speech recognition},
    journal = {Computer Speech & Language},
    volume = {95},
    pages = {101841},
    year = {2026},
    issn = {0885-2308},
    doi = {https://doi.org/10.1016/j.csl.2025.101841},
    url = {https://www.sciencedirect.com/science/article/pii/S088523082500066X},
    author = {Alexander Polok and Dominik Klement and Martin Kocour and Jiangyu Han and Federico Landini and Bolaji Yusuf and Matthew Wiesner and Sanjeev Khudanpur and Jan Černocký and Lukáš Burget},
    keywords = {Diarization-conditioned Whisper, Target-speaker ASR, Speaker diarization, Long-form ASR, Whisper adaptation},
}

@INPROCEEDINGS{10887683,
  author={Polok, Alexander and Klement, Dominik and Wiesner, Matthew and Khudanpur, Sanjeev and Černocký, Jan and Burget, Lukáš},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Target Speaker ASR with Whisper}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Transforms;Signal processing;Transformers;Acoustics;Speech processing;target-speaker ASR;diarization conditioning;multi-speaker ASR;Whisper},
  doi={10.1109/ICASSP49660.2025.10887683}
}
```

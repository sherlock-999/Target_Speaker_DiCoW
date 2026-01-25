import os
import sys
import torch
from pathlib import Path

# --------------------------------------------------
# Make DiCoW a proper package import
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "model"))

from transformers import AutoTokenizer, AutoFeatureExtractor

from DiCoW.modeling_dicow import DiCoWForConditionalGeneration

from pipeline import DiCoWPipeline
from DiariZen.diarizen.pipelines.inference import DiariZenPipeline


# -----------------------------
# Hardcoded paths
# -----------------------------
INPUT_DIR = "input"
OUTPUT_DIR = "output"
DIAR_MODEL_PATH = "BUT-FIT/diarizen-wavlm-large-s80-md"
DICOW_MODEL_PATH = Path("model/DiCoW").resolve()
sys.path.insert(0, str(DICOW_MODEL_PATH))


# -----------------------------
# Helper
# -----------------------------
def create_lower_uppercase_mapping(tokenizer):
    tokenizer.upper_cased_tokens = {}
    vocab = tokenizer.get_vocab()
    for token, index in vocab.items():
        if len(token) < 1:
            continue
        if token[0] == 'Ġ' and len(token) > 1:
            lower = token[0] + token[1].lower() + (token[2:] if len(token) > 2 else '')
        else:
            lower = token[0].lower() + token[1:]
        if lower != token and lower in vocab:
            tokenizer.upper_cased_tokens[vocab[lower]] = index


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wav_files = sorted(Path(INPUT_DIR).glob("*.wav"))
    if not wav_files:
        raise RuntimeError("No .wav file found in input/")

    audio_path = str(wav_files[0])
    print(f"Processing: {audio_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------------------
    # Load DiCoW model (LOCAL + MODIFIABLE)
    # --------------------------------------------------
    dicow = DiCoWForConditionalGeneration.from_pretrained(
        DICOW_MODEL_PATH,
        local_files_only=True
    ).to(device)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        DICOW_MODEL_PATH,
        local_files_only=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        DICOW_MODEL_PATH,
        local_files_only=True
    )

    create_lower_uppercase_mapping(tokenizer)
    dicow.set_tokenizer(tokenizer)

    # --------------------------------------------------
    # Load DiariZen
    # --------------------------------------------------
    diar_pipeline = DiariZenPipeline.from_pretrained(DIAR_MODEL_PATH).to(device)
    diar_pipeline.embedding_batch_size = 16
    diar_pipeline.segmentation_batch_size = 16

    # --------------------------------------------------
    # Build pipeline
    # --------------------------------------------------
    pipeline = DiCoWPipeline(
        dicow,
        diarization_pipeline=diar_pipeline,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        device=device
    )

    # --------------------------------------------------
    # Run inference
    # --------------------------------------------------
    result = pipeline(audio_path, return_timestamps=True)

    out_path = Path(OUTPUT_DIR) / (Path(audio_path).stem + ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print("\n===== TRANSCRIPTION =====\n")
    print(result["text"])
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()

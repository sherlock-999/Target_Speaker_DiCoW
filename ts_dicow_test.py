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

from ts_pipeline import TS_DiCoW_Pipeline
from DiariZen.diarizen.pipelines.inference import DiariZenPipeline

from pyannote.audio import Model

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

    mixed_audio_path = INPUT_DIR + "/mixed.wav"
    enrollment_audio_path = INPUT_DIR + "/enrollment.wav"

    print(f"Processing: {mixed_audio_path}")

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
    # Load Speaker Verification Model
    # --------------------------------------------------
    speaker_verification_model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")

    # --------------------------------------------------
    # Build pipeline
    # --------------------------------------------------
    pipeline = TS_DiCoW_Pipeline(
        dicow,
        diarization_pipeline=diar_pipeline,
        speaker_embedding_model=speaker_verification_model,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        device=device
    )

    # --------------------------------------------------
    # Run inference
    # --------------------------------------------------
    inputs = {
        "mixed_audio_path": mixed_audio_path,
        "enrollment_audio_path": enrollment_audio_path
    }
    result = pipeline(inputs, return_timestamps=True)


    print("\n===== TRANSCRIPTION =====\n")
    target_speaker_transcription = result["per_spk_outputs"][0]
    print(target_speaker_transcription)
    print()


    out_path = Path(OUTPUT_DIR + "/transcription.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(target_speaker_transcription)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()

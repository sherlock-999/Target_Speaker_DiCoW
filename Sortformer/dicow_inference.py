import os
import sys
import torch
from pathlib import Path

from transformers import AutoTokenizer, AutoFeatureExtractor

from DiCoW.modeling_dicow import DiCoWForConditionalGeneration
from dicow_pipeline import TS_DiCoW_Pipeline

import argparse
import json

# -----------------------------
# Model paths
# -----------------------------
DICOW_MODEL_PATH = Path("/home/users/ntu/d230009/scratch/Target_Speaker_DiCoW/model/DiCoW")

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


def main():
    parser = argparse.ArgumentParser(description="DiCoW batch inference from manifest")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest JSONL")
    parser.add_argument("--diar_mask_dir", type=str, required=True, help="Directory containing diarization masks (.pt)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save transcriptions")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------
    # Load DiCoW model + tokenizer + feature extractor
    # -------------------------------
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

    # -------------------------------
    # Speaker verification model (optional)
    # -------------------------------
    speaker_verification_model = None  # Set to None if using precomputed diarization masks

    # -------------------------------
    # Initialize pipeline
    # -------------------------------
    pipeline = TS_DiCoW_Pipeline(
        dicow,
        speaker_embedding_model=speaker_verification_model,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        device=device
    )

    # -------------------------------
    # Batch inference from manifest
    # -------------------------------
    with open(args.manifest, "r") as f:
        manifest_items = [json.loads(line) for line in f.readlines()]

    for item in manifest_items:

        mixed_audio_path = item["mixed_filepath"]
        enrollment_audio_path = item.get("enrollment_filepath", None)

        # Load corresponding diarization mask
        mixed_audio_name = os.path.basename(mixed_audio_path).replace(".wav", "")
        diar_mask_path = os.path.join(args.diar_mask_dir, f"{mixed_audio_name}_mask.pt")
        diarization_mask = torch.load(diar_mask_path)
        pipeline.diarization_mask = diarization_mask

        # Run inference
        inputs = {
            "mixed_audio_path": mixed_audio_path,
            "enrollment_audio_path": enrollment_audio_path
        }
        result = pipeline(inputs, return_timestamps=True)

        target_speaker_transcription = result["per_spk_outputs"][0]

        out_path = Path(args.output_dir) / f"{mixed_audio_name}_transcription.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(target_speaker_transcription)

        print(f"Saved transcription for {mixed_audio_name} -> {out_path}")


if __name__ == "__main__":
    main()
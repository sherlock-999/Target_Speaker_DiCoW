#!/usr/bin/env python3
"""Quick local test: transcribe the first .wav in input/ using offline DiCoW."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "DiariZen"))

import torch
from transformers import AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoTokenizer

from pipeline import DiCoWPipeline
from diarizen.pipelines.inference import DiariZenPipeline

INPUT_DIR = "input"
OUTPUT_DIR = "output"
MODEL_NAME = "BUT-FIT/DiCoW_v3_2"
DIAR_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md"
MODELS_CACHE = PROJECT_ROOT / "models"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wav_files = sorted(Path(INPUT_DIR).glob("*.wav"))
    if not wav_files:
        print("Put a .wav file in input/ and re-run.")
        return

    audio_path = str(wav_files[0])
    print(f"Processing: {audio_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dicow = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_NAME, trust_remote_code=True, cache_dir=str(MODELS_CACHE),
    ).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_NAME, cache_dir=str(MODELS_CACHE),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=str(MODELS_CACHE),
    )
    dicow.set_tokenizer(tokenizer)

    diar = DiariZenPipeline.from_pretrained(
        DIAR_MODEL, cache_dir=str(MODELS_CACHE),
    ).to(device)

    pipeline = DiCoWPipeline(
        dicow, diarization_pipeline=diar,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer, device=device,
    )

    result = pipeline(audio_path, return_timestamps=True)
    out_path = Path(OUTPUT_DIR) / (Path(audio_path).stem + ".txt")
    out_path.write_text(result["text"])
    print(result["text"])
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()

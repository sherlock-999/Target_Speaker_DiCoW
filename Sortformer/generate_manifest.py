# generate_manifest.py
# Copyright (c) 2022, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0.

import argparse
import json
import librosa
import random
from pathlib import Path

random.seed(42)


def main(
    mixed_path_file,
    enrollment_path_file=None,
    manifest_filepath=None,
    add_duration=False
):
    mixed_path_file = Path(mixed_path_file)
    if not mixed_path_file.is_file():
        raise FileNotFoundError(f"Mixed audio list file not found: {mixed_path_file}")

    # Read mixed audio files
    with open(mixed_path_file, "r") as f:
        wav_files = [line.strip() for line in f.readlines() if line.strip()]
    if len(wav_files) == 0:
        raise ValueError("No mixed audio files found in the input file.")

    # Read enrollment files
    if enrollment_path_file is not None:
        enrollment_path_file = Path(enrollment_path_file)
        if not enrollment_path_file.is_file():
            raise FileNotFoundError(f"Enrollment audio list file not found: {enrollment_path_file}")

        with open(enrollment_path_file, "r") as f:
            enroll_files = [line.strip() for line in f.readlines() if line.strip()]

        if len(enroll_files) != len(wav_files):
            raise ValueError(
                "Number of enrollment files must be 1 or match the number of mixed audio files"
            )
    else:
        enroll_files = [None] * len(wav_files)

    # Create manifest entries
    manifest_entries = []
    for i, wav in enumerate(wav_files):
        entry = {
            "mixed_filepath": wav,
            "enrollment_filepath": enroll_files[i],
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": None,
            "num_speakers": None,
            "rttm_filepath": None,
            "uem_filepath": None,
            "ctm_filepath": None
        }
        if add_duration:
            y, sr = librosa.load(wav, sr=None)
            entry["duration"] = float(len(y) / sr)
        manifest_entries.append(entry)

    # Write manifest file
    manifest_path = Path(manifest_filepath)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths2mixed_files",
        help="Path to text file containing list of mixed audio files",
        type=str,
        required=True
    )
    parser.add_argument(
        "--paths2enrollment_files",
        help="Path to text file containing enrollment audio files",
        type=str,
        default=None
    )
    parser.add_argument(
        "--manifest_filepath",
        help="Path to output manifest file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--add_duration",
        help="Add duration of audio files to output manifest",
        action='store_true'
    )

    args = parser.parse_args()

    main(
        mixed_path_file=args.paths2mixed_files,  # ✅ fixed
        enrollment_path_file=args.paths2enrollment_files,
        manifest_filepath=args.manifest_filepath,
        add_duration=args.add_duration
    )
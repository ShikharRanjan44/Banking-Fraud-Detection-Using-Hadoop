"""
run_augmentation.py
-------------------
Audio augmentation script for the Digital Stethoscope ML pipeline.

Reads cleaned .wav files from data/cleaned/{heart,lung}/
Applies augmentation (noise, pitch shift, time stretch) and
saves results to data/augmented/{heart,lung}/.

Usage:
    python run_augmentation.py [--organ heart|lung|all] [--n_aug N]

All paths are relative to this script's location (ml_pipeline/).
"""

import os
import sys
import argparse
import logging
import random

import numpy as np

# ---- configure logging (ASCII only, no emoji) --------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---- paths (all relative to this file) --------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
AUGMENTED_DIR = os.path.join(BASE_DIR, "data", "augmented")

ORGANS = ["heart", "lung"]


# ---- augmentation helpers ---------------------------------------------------

def add_white_noise(audio, noise_factor=0.005):
    """Add Gaussian white noise to an audio signal."""
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise


def time_stretch(audio, rate=1.1):
    """Simple time-stretch via resampling (no external dependency)."""
    indices = np.round(np.arange(0, len(audio), rate)).astype(int)
    indices = indices[indices < len(audio)]
    return audio[indices]


def pitch_shift_simple(audio, sr, semitones=2):
    """
    Approximate pitch shift by resampling at a shifted rate.
    For a proper pitch shift use librosa.effects.pitch_shift when available.
    """
    factor = 2 ** (semitones / 12.0)
    indices = np.round(np.arange(0, len(audio), factor)).astype(int)
    indices = indices[indices < len(audio)]
    return audio[indices]


def augment_audio(audio, sr):
    """
    Apply a random subset of augmentations to an audio signal.
    Returns a list of (suffix, augmented_audio) tuples.
    """
    augmented = []

    # noise
    augmented.append(("noise", add_white_noise(audio)))

    # time stretch (slow down)
    augmented.append(("stretch_slow", time_stretch(audio, rate=0.9)))

    # time stretch (speed up)
    augmented.append(("stretch_fast", time_stretch(audio, rate=1.1)))

    # pitch shift up
    augmented.append(("pitch_up", pitch_shift_simple(audio, sr, semitones=2)))

    # pitch shift down
    augmented.append(("pitch_down", pitch_shift_simple(audio, sr, semitones=-2)))

    return augmented


# ---- main processing --------------------------------------------------------

def process_organ(organ, n_aug):
    """Augment all .wav files for a given organ."""
    src_dir = os.path.join(CLEANED_DIR, organ)
    dst_dir = os.path.join(AUGMENTED_DIR, organ)
    os.makedirs(dst_dir, exist_ok=True)

    try:
        import soundfile as sf
    except ImportError:
        log.error("soundfile is not installed. Run: pip install soundfile")
        sys.exit(1)

    wav_files = [f for f in os.listdir(src_dir) if f.lower().endswith(".wav")]
    if not wav_files:
        log.warning("No .wav files found in: %s", src_dir)
        return

    log.info("Processing organ='%s': %d files found", organ, len(wav_files))

    generated = 0
    for fname in wav_files:
        src_path = os.path.join(src_dir, fname)
        try:
            audio, sr = sf.read(src_path)
        except Exception as exc:
            log.warning("Could not read %s: %s", src_path, exc)
            continue

        # Convert stereo to mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        stem = os.path.splitext(fname)[0]
        augmented_versions = augment_audio(audio, sr)

        # Limit to n_aug versions per file
        random.shuffle(augmented_versions)
        for suffix, aug_audio in augmented_versions[:n_aug]:
            out_name = f"{stem}_{suffix}.wav"
            out_path = os.path.join(dst_dir, out_name)
            try:
                sf.write(out_path, aug_audio, sr)
                generated += 1
            except Exception as exc:
                log.warning("Could not write %s: %s", out_path, exc)

    log.info("Augmentation complete for '%s': %d files generated -> %s", organ, generated, dst_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Audio augmentation for Digital Stethoscope ML pipeline"
    )
    parser.add_argument(
        "--organ",
        choices=["heart", "lung", "all"],
        default="all",
        help="Which organ data to augment (default: all)",
    )
    parser.add_argument(
        "--n_aug",
        type=int,
        default=3,
        help="Number of augmented versions to generate per file (default: 3)",
    )
    args = parser.parse_args()

    organs_to_process = ORGANS if args.organ == "all" else [args.organ]
    log.info("Starting augmentation pipeline (organ=%s, n_aug=%d)", args.organ, args.n_aug)

    for organ in organs_to_process:
        process_organ(organ, args.n_aug)

    log.info("All augmentation tasks completed.")


if __name__ == "__main__":
    main()

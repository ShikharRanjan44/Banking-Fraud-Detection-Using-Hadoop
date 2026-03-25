"""
run_mfcc.py
-----------
MFCC (Mel-Frequency Cepstral Coefficients) feature extraction for the
Digital Stethoscope ML pipeline.

Reads .wav files from data/augmented/{heart,lung}/ (and optionally
data/cleaned/{heart,lung}/) and saves extracted features as .npy files
in data/features/.

Output filenames are: <stem>_<organ>_mfcc.npy

Usage:
    python run_mfcc.py [--organ heart|lung|all] [--n_mfcc N]
                       [--include_cleaned]

All paths are relative to this script's location (ml_pipeline/).
"""

import os
import sys
import argparse
import logging

import numpy as np

# ---- configure logging -------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---- paths -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUGMENTED_DIR = os.path.join(BASE_DIR, "data", "augmented")
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")

ORGANS = ["heart", "lung"]

# ---- MFCC helpers ------------------------------------------------------------

def compute_mfcc(audio, sr, n_mfcc=40, n_fft=2048, hop_length=512, n_mels=128):
    """
    Compute MFCC features from an audio signal.

    Falls back to a lightweight numpy-based implementation when librosa
    is not available, so the script can run in minimal environments.
    """
    try:
        import librosa
        mfcc = librosa.feature.mfcc(
            y=audio.astype(np.float32),
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        return mfcc  # shape: (n_mfcc, T)
    except ImportError:
        log.warning("librosa not installed; using numpy-only MFCC approximation.")
        return _numpy_mfcc(audio, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)


def _numpy_mfcc(audio, sr, n_mfcc=40, n_fft=2048, hop_length=512):
    """
    Approximate MFCC using numpy only (no librosa dependency).
    Produces a (n_mfcc, T) array of DCT-II cepstral coefficients
    over mel-scaled log-power spectrogram frames.
    """
    audio = audio.astype(np.float64)

    # Framing
    frames = []
    for start in range(0, len(audio) - n_fft, hop_length):
        frame = audio[start: start + n_fft]
        frame = frame * np.hanning(n_fft)
        frames.append(frame)
    if not frames:
        return np.zeros((n_mfcc, 1))

    frames = np.array(frames)  # (T, n_fft)

    # Power spectrum
    power = (np.abs(np.fft.rfft(frames, n=n_fft)) ** 2)  # (T, n_fft//2+1)

    # Mel filterbank (triangular)
    n_mels = 40
    f_min, f_max = 0.0, sr / 2.0
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m - f_m_minus > 0:
                filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus - f_m > 0:
                filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    mel_energy = np.dot(power, filterbank.T)  # (T, n_mels)
    log_mel = np.log(mel_energy + 1e-10)

    # DCT-II
    mfcc = _dct2(log_mel, n_mfcc)  # (T, n_mfcc)
    return mfcc.T  # (n_mfcc, T)


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _dct2(x, n_mfcc):
    """Compute the first n_mfcc DCT-II coefficients row-wise."""
    T, M = x.shape
    n = np.arange(M)
    result = np.zeros((T, n_mfcc))
    for k in range(n_mfcc):
        result[:, k] = np.sum(x * np.cos(np.pi * k * (2 * n + 1) / (2 * M)), axis=1)
    return result


# ---- processing --------------------------------------------------------------

def process_organ(organ, n_mfcc, include_cleaned):
    """Extract MFCC features for all .wav files of a given organ."""
    os.makedirs(FEATURES_DIR, exist_ok=True)

    dirs_to_scan = [os.path.join(AUGMENTED_DIR, organ)]
    if include_cleaned:
        dirs_to_scan.append(os.path.join(CLEANED_DIR, organ))

    try:
        import soundfile as sf
    except ImportError:
        log.error("soundfile is not installed. Run: pip install soundfile")
        sys.exit(1)

    total = 0
    for src_dir in dirs_to_scan:
        if not os.path.isdir(src_dir):
            log.warning("Directory not found, skipping: %s", src_dir)
            continue

        wav_files = [f for f in os.listdir(src_dir) if f.lower().endswith(".wav")]
        log.info("Extracting MFCCs from: %s (%d files)", src_dir, len(wav_files))

        for fname in wav_files:
            src_path = os.path.join(src_dir, fname)
            try:
                audio, sr = sf.read(src_path)
            except Exception as exc:
                log.warning("Could not read %s: %s", src_path, exc)
                continue

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            mfcc = compute_mfcc(audio, sr, n_mfcc=n_mfcc)

            stem = os.path.splitext(fname)[0]
            out_name = f"{stem}_{organ}_mfcc.npy"
            out_path = os.path.join(FEATURES_DIR, out_name)
            np.save(out_path, mfcc)
            total += 1

    log.info("MFCC extraction complete for '%s': %d feature files -> %s", organ, total, FEATURES_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="MFCC feature extraction for Digital Stethoscope ML pipeline"
    )
    parser.add_argument(
        "--organ",
        choices=["heart", "lung", "all"],
        default="all",
        help="Which organ data to process (default: all)",
    )
    parser.add_argument(
        "--n_mfcc",
        type=int,
        default=40,
        help="Number of MFCC coefficients to extract (default: 40)",
    )
    parser.add_argument(
        "--include_cleaned",
        action="store_true",
        help="Also extract MFCCs from the cleaned (non-augmented) data",
    )
    args = parser.parse_args()

    organs_to_process = ORGANS if args.organ == "all" else [args.organ]
    log.info(
        "Starting MFCC extraction (organ=%s, n_mfcc=%d, include_cleaned=%s)",
        args.organ, args.n_mfcc, args.include_cleaned,
    )

    for organ in organs_to_process:
        process_organ(organ, args.n_mfcc, args.include_cleaned)

    log.info("All MFCC extraction tasks completed.")


if __name__ == "__main__":
    main()

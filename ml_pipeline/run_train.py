"""
run_train.py
------------
Model training script for the Digital Stethoscope ML pipeline.

Loads MFCC features from data/features/ and trains a classifier
(CNN or simple baseline) to distinguish normal vs. abnormal heart/lung sounds.

Saves the trained model to models/model_<organ>_<timestamp>.pkl (sklearn) or
models/model_<organ>_<timestamp>.h5 (Keras/TF).

Usage:
    python run_train.py [--organ heart|lung|all] [--model cnn|baseline]
                        [--epochs N] [--batch_size N] [--test_split F]

All paths are relative to this script's location (ml_pipeline/).
"""

import os
import sys
import argparse
import logging
import datetime
import pickle

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
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")

ORGANS = ["heart", "lung"]

# Label convention: filenames containing "normal" -> 0, "abnormal" -> 1
# Files without either keyword are skipped with a warning.


# ---- data loading ------------------------------------------------------------

def load_features(organ):
    """
    Load all MFCC .npy files for the given organ and derive labels from filenames.

    Returns:
        X : np.ndarray of shape (N, n_mfcc * T_mean) -- mean-pooled over time
        y : np.ndarray of shape (N,) with integer labels 0/1
        fnames : list of source filenames
    """
    if not os.path.isdir(FEATURES_DIR):
        log.error("Features directory not found: %s", FEATURES_DIR)
        log.error("Run run_mfcc.py first.")
        sys.exit(1)

    pattern = f"_{organ}_mfcc.npy"
    npy_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith(pattern)]

    if not npy_files:
        log.warning("No feature files found for organ='%s' in %s", organ, FEATURES_DIR)
        return None, None, []

    X_list, y_list, fnames = [], [], []
    skipped = 0

    for fname in sorted(npy_files):
        lower = fname.lower()
        if "normal" in lower and "abnormal" not in lower:
            label = 0
        elif "abnormal" in lower:
            label = 1
        else:
            log.warning("Cannot determine label for: %s -- skipping", fname)
            skipped += 1
            continue

        mfcc = np.load(os.path.join(FEATURES_DIR, fname))  # (n_mfcc, T)
        # Mean-pool over time to get a fixed-length feature vector
        feat = mfcc.mean(axis=1)
        X_list.append(feat)
        y_list.append(label)
        fnames.append(fname)

    if skipped:
        log.warning("Skipped %d files without clear normal/abnormal label.", skipped)

    if not X_list:
        return None, None, []

    X = np.array(X_list)
    y = np.array(y_list, dtype=np.int32)
    log.info(
        "Loaded %d samples for organ='%s' (normal=%d, abnormal=%d)",
        len(y), organ, (y == 0).sum(), (y == 1).sum(),
    )
    return X, y, fnames


# ---- model builders ----------------------------------------------------------

def build_baseline_model():
    """Return a scikit-learn RandomForest pipeline."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        log.error("scikit-learn is not installed. Run: pip install scikit-learn")
        sys.exit(1)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ])


def build_cnn_model(input_dim, n_mfcc=40):
    """Return a small Keras CNN for 1-D feature sequences."""
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        log.error("TensorFlow is not installed. Run: pip install tensorflow")
        sys.exit(1)

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Reshape((input_dim, 1)),
        keras.layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ---- training ----------------------------------------------------------------

def train_organ(organ, model_type, epochs, batch_size, test_split):
    """Load features, train, evaluate on hold-out split, and save the model."""
    X, y, fnames = load_features(organ)
    if X is None or len(X) < 4:
        log.warning("Not enough data to train for organ='%s'. Skipping.", organ)
        return

    # Train/test split
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        log.error("scikit-learn is not installed. Run: pip install scikit-learn")
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    log.info("Train=%d  Test=%d", len(X_train), len(X_test))

    os.makedirs(MODELS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if model_type == "baseline":
        model = build_baseline_model()
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        log.info("Baseline model test accuracy: %.4f", acc)

        model_path = os.path.join(MODELS_DIR, f"model_{organ}_{timestamp}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        log.info("Model saved -> %s", model_path)

    elif model_type == "cnn":
        model = build_cnn_model(input_dim=X_train.shape[1])
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        log.info("CNN model test accuracy: %.4f  loss: %.4f", acc, loss)

        model_path = os.path.join(MODELS_DIR, f"model_{organ}_{timestamp}.h5")
        model.save(model_path)
        log.info("Model saved -> %s", model_path)

    else:
        log.error("Unknown model type: %s", model_type)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Model training for Digital Stethoscope ML pipeline"
    )
    parser.add_argument(
        "--organ", choices=["heart", "lung", "all"], default="all",
        help="Which organ to train on (default: all)",
    )
    parser.add_argument(
        "--model", choices=["cnn", "baseline"], default="baseline",
        help="Model type: 'baseline' (RandomForest) or 'cnn' (TF/Keras, default: baseline)",
    )
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs for CNN (default: 30)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for CNN training (default: 32)")
    parser.add_argument("--test_split", type=float, default=0.2,
                        help="Fraction of data for test set (default: 0.2)")
    args = parser.parse_args()

    organs_to_process = ORGANS if args.organ == "all" else [args.organ]
    log.info(
        "Starting training (organ=%s, model=%s, epochs=%d)",
        args.organ, args.model, args.epochs,
    )

    for organ in organs_to_process:
        train_organ(organ, args.model, args.epochs, args.batch_size, args.test_split)

    log.info("All training tasks completed.")


if __name__ == "__main__":
    main()

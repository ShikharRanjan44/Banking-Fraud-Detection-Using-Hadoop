"""
evaluation/evaluate_model.py
-----------------------------
Standalone evaluation utilities for the Digital Stethoscope ML pipeline.

Provides:
    - load_model(path)          -> load a sklearn (.pkl) or Keras (.h5) model
    - predict(model, X)         -> run inference, return (y_pred, y_proba)
    - full_report(y_true, y_pred, y_proba, labels, output_path)
         -> write classification report, confusion matrix, ROC-AUC to a file

Can also be run directly:
    python evaluation/evaluate_model.py \\
        --model_path models/<file> \\
        --features_dir data/features \\
        --organ heart|lung|all \\
        [--output_dir evaluation]

All paths should be relative to the ml_pipeline/ root or absolute.
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

# ---- default paths (relative to ml_pipeline/) --------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_SCRIPT_DIR)  # one level up from evaluation/
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EVAL_DIR = _SCRIPT_DIR  # output next to this script by default

ORGANS = ["heart", "lung"]
LABEL_NAMES = ["normal", "abnormal"]


# ---- model loading -----------------------------------------------------------

def load_model(model_path):
    """
    Load a saved model from disk.

    Supports:
        - .pkl  -> sklearn Pipeline (or any pickle-able estimator)
        - .h5 / .keras -> Keras/TensorFlow model

    Returns (model, model_type) where model_type is 'sklearn' or 'keras'.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    ext = os.path.splitext(model_path)[1].lower()

    if ext == ".pkl":
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        log.info("Loaded sklearn model: %s", model_path)
        return model, "sklearn"

    elif ext in (".h5", ".keras"):
        try:
            from tensorflow import keras
        except ImportError as e:
            raise ImportError(
                "TensorFlow is required for .h5/.keras models. "
                "Install with: pip install tensorflow"
            ) from e
        model = keras.models.load_model(model_path)
        log.info("Loaded Keras model: %s", model_path)
        return model, "keras"

    else:
        raise ValueError(f"Unsupported model extension '{ext}'. Expected .pkl or .h5")


# ---- prediction --------------------------------------------------------------

def predict(model, X, model_type):
    """
    Run model inference.

    Returns:
        y_pred  : np.ndarray of int  (0/1 class predictions)
        y_proba : np.ndarray of float (probability of class 1), or None
    """
    if model_type == "sklearn":
        y_pred = model.predict(X)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
        else:
            y_proba = None
        return y_pred.astype(np.int32), y_proba

    elif model_type == "keras":
        probs = model.predict(X, verbose=0).squeeze()
        y_pred = (probs >= 0.5).astype(np.int32)
        return y_pred, probs

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---- reporting ---------------------------------------------------------------

def full_report(y_true, y_pred, y_proba=None, labels=None, output_path=None):
    """
    Compute and optionally save:
        - Accuracy
        - Classification report (precision, recall, F1)
        - Confusion matrix
        - ROC-AUC (if y_proba provided)

    Args:
        y_true      : ground-truth integer labels
        y_pred      : predicted integer labels
        y_proba     : predicted probabilities for class 1 (optional)
        labels      : list of class name strings (default: ['normal', 'abnormal'])
        output_path : file path to write the report; if None, prints only to log

    Returns a dict with keys: accuracy, report, confusion_matrix, roc_auc
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            roc_auc_score,
        )
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required. Install with: pip install scikit-learn"
        ) from e

    if labels is None:
        labels = LABEL_NAMES

    acc = accuracy_score(y_true, y_pred)
    report_str = classification_report(
        y_true, y_pred, target_names=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    roc_auc = None
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
        except Exception:
            pass

    log.info("Accuracy: %.4f", acc)
    log.info("\nClassification Report:\n%s", report_str)
    log.info("Confusion Matrix:\n%s", str(cm))
    if roc_auc is not None:
        log.info("ROC-AUC: %.4f", roc_auc)

    result = {
        "accuracy": acc,
        "report": report_str,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Digital Stethoscope ML Pipeline - Evaluation Report\n")
            f.write("=" * 55 + "\n\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report_str + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n")
            if roc_auc is not None:
                f.write(f"\nROC-AUC: {roc_auc:.4f}\n")
        log.info("Report saved -> %s", output_path)

    return result


# ---- data loading (mirrors run_train.py) ------------------------------------

def load_features(features_dir, organ):
    """Load mean-pooled MFCC features and labels for the given organ."""
    pattern = f"_{organ}_mfcc.npy"
    npy_files = [f for f in os.listdir(features_dir) if f.endswith(pattern)]

    X_list, y_list = [], []
    for fname in sorted(npy_files):
        lower = fname.lower()
        if "normal" in lower and "abnormal" not in lower:
            label = 0
        elif "abnormal" in lower:
            label = 1
        else:
            continue
        mfcc = np.load(os.path.join(features_dir, fname))
        X_list.append(mfcc.mean(axis=1))
        y_list.append(label)

    if not X_list:
        return None, None
    return np.array(X_list), np.array(y_list, dtype=np.int32)


# ---- CLI entry point ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Digital Stethoscope model"
    )
    parser.add_argument(
        "--model_path", required=True,
        help="Path to the model file (.pkl or .h5), relative to ml_pipeline/ or absolute",
    )
    parser.add_argument(
        "--features_dir", default=FEATURES_DIR,
        help="Directory containing .npy MFCC feature files",
    )
    parser.add_argument(
        "--organ", choices=["heart", "lung", "all"], default="all",
    )
    parser.add_argument(
        "--output_dir", default=EVAL_DIR,
        help="Directory to save evaluation reports",
    )
    args = parser.parse_args()

    # Resolve paths
    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(BASE_DIR, model_path)
    features_dir = args.features_dir
    if not os.path.isabs(features_dir):
        features_dir = os.path.join(BASE_DIR, features_dir)

    model, model_type = load_model(model_path)

    organs = ORGANS if args.organ == "all" else [args.organ]
    for organ in organs:
        if not os.path.isdir(features_dir):
            log.error("Features directory not found: %s", features_dir)
            sys.exit(1)

        X, y = load_features(features_dir, organ)
        if X is None:
            log.warning("No data for organ='%s'. Skipping.", organ)
            continue

        log.info("Evaluating organ='%s' (%d samples)", organ, len(y))
        y_pred, y_proba = predict(model, X, model_type)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, f"report_{organ}_{timestamp}.txt")
        full_report(y, y_pred, y_proba=y_proba, output_path=output_path)


if __name__ == "__main__":
    main()

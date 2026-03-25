"""
run_eval.py
-----------
Model evaluation script for the Digital Stethoscope ML pipeline.

Loads a saved model from models/ and evaluates it against feature files
in data/features/. Writes a classification report and confusion matrix
to evaluation/.

Usage:
    python run_eval.py --model_path models/<filename>
                       [--organ heart|lung|all]
                       [--output_dir evaluation]

All paths are relative to this script's location (ml_pipeline/).
"""

import os
import sys
import argparse
import logging
import datetime

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
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")

ORGANS = ["heart", "lung"]


# ---- data loading (same as run_train.py) ------------------------------------

def load_features(organ):
    """Load mean-pooled MFCC feature vectors and labels for the given organ."""
    if not os.path.isdir(FEATURES_DIR):
        log.error("Features directory not found: %s", FEATURES_DIR)
        sys.exit(1)

    pattern = f"_{organ}_mfcc.npy"
    npy_files = [f for f in os.listdir(FEATURES_DIR) if f.endswith(pattern)]

    if not npy_files:
        log.warning("No feature files found for organ='%s'.", organ)
        return None, None

    X_list, y_list = [], []
    for fname in sorted(npy_files):
        lower = fname.lower()
        if "normal" in lower and "abnormal" not in lower:
            label = 0
        elif "abnormal" in lower:
            label = 1
        else:
            continue

        mfcc = np.load(os.path.join(FEATURES_DIR, fname))
        X_list.append(mfcc.mean(axis=1))
        y_list.append(label)

    if not X_list:
        return None, None

    return np.array(X_list), np.array(y_list, dtype=np.int32)


# ---- model loading -----------------------------------------------------------

def load_model(model_path):
    """Auto-detect model type by extension and load it."""
    if not os.path.isfile(model_path):
        log.error("Model file not found: %s", model_path)
        sys.exit(1)

    ext = os.path.splitext(model_path)[1].lower()

    if ext == ".pkl":
        import pickle
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        log.info("Loaded sklearn model from: %s", model_path)
        return model, "sklearn"

    elif ext in (".h5", ".keras"):
        try:
            from tensorflow import keras
        except ImportError:
            log.error("TensorFlow is not installed. Run: pip install tensorflow")
            sys.exit(1)
        model = keras.models.load_model(model_path)
        log.info("Loaded Keras model from: %s", model_path)
        return model, "keras"

    else:
        log.error("Unsupported model extension '%s'. Expected .pkl or .h5", ext)
        sys.exit(1)


# ---- metrics -----------------------------------------------------------------

def print_report(y_true, y_pred, organ, output_dir):
    """Print and save classification report + confusion matrix."""
    try:
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError:
        log.error("scikit-learn is not installed. Run: pip install scikit-learn")
        sys.exit(1)

    report = classification_report(
        y_true, y_pred, target_names=["normal", "abnormal"], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    log.info("\nClassification Report (%s):\n%s", organ, report)
    log.info("Confusion Matrix:\n%s", str(cm))

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"report_{organ}_{timestamp}.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Evaluation Report - {organ}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")

    log.info("Report saved -> %s", report_path)


# ---- evaluation --------------------------------------------------------------

def evaluate_organ(organ, model, model_type, output_dir):
    """Run evaluation for one organ."""
    X, y = load_features(organ)
    if X is None or len(X) == 0:
        log.warning("No data available for organ='%s'. Skipping.", organ)
        return

    log.info("Evaluating on %d samples for organ='%s'", len(y), organ)

    if model_type == "sklearn":
        y_pred = model.predict(X)
    elif model_type == "keras":
        probs = model.predict(X, verbose=0).squeeze()
        y_pred = (probs >= 0.5).astype(np.int32)
    else:
        log.error("Unknown model type: %s", model_type)
        return

    print_report(y, y_pred, organ, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Model evaluation for Digital Stethoscope ML pipeline"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the saved model file (.pkl for sklearn, .h5 for Keras)",
    )
    parser.add_argument(
        "--organ", choices=["heart", "lung", "all"], default="all",
        help="Which organ to evaluate (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        default=EVAL_DIR,
        help="Directory to save evaluation reports (default: evaluation/)",
    )
    args = parser.parse_args()

    # Resolve model path relative to BASE_DIR if not absolute
    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(BASE_DIR, model_path)

    model, model_type = load_model(model_path)

    organs_to_evaluate = ORGANS if args.organ == "all" else [args.organ]
    log.info("Starting evaluation (organ=%s)", args.organ)

    for organ in organs_to_evaluate:
        evaluate_organ(organ, model, model_type, args.output_dir)

    log.info("All evaluation tasks completed.")


if __name__ == "__main__":
    main()

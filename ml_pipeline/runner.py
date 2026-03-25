"""
runner.py
---------
Orchestration script for the Digital Stethoscope ML pipeline.

Runs the full pipeline in order:
    1. Augmentation  (run_augmentation.py)
    2. MFCC extraction (run_mfcc.py)
    3. Training      (run_train.py)
    4. Evaluation    (run_eval.py)

Each stage is invoked as a subprocess so it inherits the same Python
environment, and failures at any stage stop the pipeline unless
--continue_on_error is given.

Usage:
    python runner.py [--organ heart|lung|all]
                     [--model baseline|cnn]
                     [--n_aug N]
                     [--n_mfcc N]
                     [--epochs N]
                     [--batch_size N]
                     [--test_split F]
                     [--stages aug,mfcc,train,eval]
                     [--continue_on_error]

All paths are relative to this script's location (ml_pipeline/).
"""

import os
import sys
import argparse
import logging
import subprocess
import glob as _glob

# ---- configure logging -------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ---- stage runners -----------------------------------------------------------

def run_stage(label, cmd, continue_on_error):
    """Run a subprocess stage and handle errors."""
    log.info("=" * 60)
    log.info("STAGE: %s", label)
    log.info("Command: %s", " ".join(cmd))
    log.info("=" * 60)

    result = subprocess.run(cmd, cwd=BASE_DIR)

    if result.returncode != 0:
        log.error("Stage '%s' failed with exit code %d.", label, result.returncode)
        if not continue_on_error:
            log.error("Aborting pipeline. Use --continue_on_error to skip failures.")
            sys.exit(result.returncode)
        else:
            log.warning("Continuing despite failure in stage '%s'.", label)
    else:
        log.info("Stage '%s' completed successfully.", label)


def find_latest_model(organ):
    """Return the path of the most recently modified model file for an organ."""
    patterns = [
        os.path.join(MODELS_DIR, f"model_{organ}_*.pkl"),
        os.path.join(MODELS_DIR, f"model_{organ}_*.h5"),
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(_glob.glob(pattern))

    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline runner for Digital Stethoscope ML pipeline"
    )
    parser.add_argument("--organ", choices=["heart", "lung", "all"], default="all")
    parser.add_argument("--model", choices=["baseline", "cnn"], default="baseline")
    parser.add_argument("--n_aug", type=int, default=3,
                        help="Augmented versions per file (default: 3)")
    parser.add_argument("--n_mfcc", type=int, default=40,
                        help="Number of MFCC coefficients (default: 40)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs for CNN (default: 30)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for CNN (default: 32)")
    parser.add_argument("--test_split", type=float, default=0.2,
                        help="Test split fraction (default: 0.2)")
    parser.add_argument(
        "--stages",
        default="aug,mfcc,train,eval",
        help="Comma-separated list of stages to run: aug,mfcc,train,eval (default: all)",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue to the next stage even if a stage fails",
    )
    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(",")]
    python = sys.executable

    log.info("Digital Stethoscope ML Pipeline")
    log.info("Organ  : %s", args.organ)
    log.info("Model  : %s", args.model)
    log.info("Stages : %s", ", ".join(stages))

    # ---- Stage 1: Augmentation -----------------------------------------------
    if "aug" in stages:
        run_stage(
            "Augmentation",
            [python, os.path.join(BASE_DIR, "run_augmentation.py"),
             "--organ", args.organ,
             "--n_aug", str(args.n_aug)],
            args.continue_on_error,
        )

    # ---- Stage 2: MFCC extraction --------------------------------------------
    if "mfcc" in stages:
        run_stage(
            "MFCC Extraction",
            [python, os.path.join(BASE_DIR, "run_mfcc.py"),
             "--organ", args.organ,
             "--n_mfcc", str(args.n_mfcc)],
            args.continue_on_error,
        )

    # ---- Stage 3: Training ---------------------------------------------------
    if "train" in stages:
        run_stage(
            "Training",
            [python, os.path.join(BASE_DIR, "run_train.py"),
             "--organ", args.organ,
             "--model", args.model,
             "--epochs", str(args.epochs),
             "--batch_size", str(args.batch_size),
             "--test_split", str(args.test_split)],
            args.continue_on_error,
        )

    # ---- Stage 4: Evaluation -------------------------------------------------
    if "eval" in stages:
        # Find the latest saved model to evaluate
        organs = ["heart", "lung"] if args.organ == "all" else [args.organ]
        for organ in organs:
            model_path = find_latest_model(organ)
            if model_path is None:
                log.warning("No saved model found for organ='%s'. Skipping eval.", organ)
                continue
            run_stage(
                f"Evaluation ({organ})",
                [python, os.path.join(BASE_DIR, "run_eval.py"),
                 "--model_path", model_path,
                 "--organ", organ],
                args.continue_on_error,
            )

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

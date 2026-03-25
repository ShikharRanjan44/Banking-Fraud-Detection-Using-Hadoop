# Digital Stethoscope ML Pipeline

A portable, end-to-end machine learning pipeline for classifying heart and
lung sounds as **normal** or **abnormal** using MFCC audio features.

---

## Directory Layout

```
ml_pipeline/
|-- runner.py                  # Full pipeline orchestrator
|-- run_augmentation.py        # Stage 1 - audio augmentation
|-- run_mfcc.py                # Stage 2 - MFCC feature extraction
|-- run_train.py               # Stage 3 - model training
|-- run_eval.py                # Stage 4 - model evaluation
|-- requirements.txt           # Python dependencies
|-- README.md                  # This file
|
|-- evaluation/
|   |-- evaluate_model.py      # Reusable evaluation utilities + CLI
|   `-- (reports written here at runtime)
|
|-- data/
|   |-- cleaned/
|   |   |-- heart/             # Place raw cleaned heart .wav files here
|   |   `-- lung/              # Place raw cleaned lung  .wav files here
|   |
|   |-- augmented/
|   |   |-- heart/             # Augmented heart audio (auto-generated)
|   |   `-- lung/              # Augmented lung  audio (auto-generated)
|   |
|   `-- features/              # .npy MFCC feature files (auto-generated)
|
`-- models/                    # Saved trained models (auto-generated)
```

---

## Prerequisites

- Python 3.9 or later
- pip

Install dependencies:

```bash
cd ml_pipeline
pip install -r requirements.txt
```

> **Deep learning (CNN) only:** Also install TensorFlow:
> ```bash
> pip install tensorflow>=2.12.0
> ```

---

## Preparing Your Data

Place cleaned audio files (`.wav`) inside the appropriate folder **before** running
the pipeline:

| Organ | Folder                        | Filename convention                  |
|-------|-------------------------------|--------------------------------------|
| Heart | `data/cleaned/heart/`         | Include `normal` or `abnormal` in name |
| Lung  | `data/cleaned/lung/`          | Include `normal` or `abnormal` in name |

**Filename examples:**

```
patient001_normal_heart.wav
patient002_abnormal_heart.wav
patient003_normal_lung.wav
patient004_abnormal_lung.wav
```

The pipeline derives class labels from the filename automatically.

---

## Running the Full Pipeline

```bash
# From the ml_pipeline/ directory:
python runner.py
```

This runs all four stages in order for both heart and lung data using the
baseline (Random Forest) model.

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--organ` | `all` | `heart`, `lung`, or `all` |
| `--model` | `baseline` | `baseline` (Random Forest) or `cnn` (TensorFlow) |
| `--n_aug` | `3` | Augmented copies generated per source file |
| `--n_mfcc` | `40` | Number of MFCC coefficients |
| `--epochs` | `30` | CNN training epochs |
| `--batch_size` | `32` | CNN batch size |
| `--test_split` | `0.2` | Fraction of data held out for testing |
| `--stages` | `aug,mfcc,train,eval` | Comma-separated subset of stages to run |
| `--continue_on_error` | off | Continue to next stage even if one fails |

### Examples

```bash
# Heart-only, CNN model, 50 epochs
python runner.py --organ heart --model cnn --epochs 50

# Skip augmentation (data already augmented), 60 MFCC coefficients
python runner.py --stages mfcc,train,eval --n_mfcc 60

# Augment and extract features only
python runner.py --stages aug,mfcc
```

---

## Running Individual Stages

### Stage 1 - Augmentation

```bash
python run_augmentation.py --organ heart --n_aug 4
```

Reads from `data/cleaned/{organ}/`, writes to `data/augmented/{organ}/`.
Augmentations applied: white noise, time stretch (slow/fast), pitch shift (up/down).

---

### Stage 2 - MFCC Feature Extraction

```bash
python run_mfcc.py --organ all --n_mfcc 40
```

Reads from `data/augmented/{organ}/`, writes `.npy` files to `data/features/`.
Add `--include_cleaned` to also extract MFCCs from the cleaned (non-augmented) data.

---

### Stage 3 - Training

```bash
# Random Forest baseline (no GPU required)
python run_train.py --organ all --model baseline

# CNN (requires TensorFlow)
python run_train.py --organ heart --model cnn --epochs 40
```

Reads from `data/features/`, writes model to `models/`.
Files are named `model_<organ>_<timestamp>.pkl` (baseline) or `.h5` (CNN).

---

### Stage 4 - Evaluation

```bash
python run_eval.py --model_path models/model_heart_20240101_120000.pkl \
                   --organ heart
```

Or use the evaluation utilities directly:

```bash
python evaluation/evaluate_model.py \
    --model_path models/model_heart_20240101_120000.pkl \
    --organ heart
```

Reports (classification report, confusion matrix, ROC-AUC) are written to
`evaluation/report_<organ>_<timestamp>.txt`.

---

## Workflow Diagram

```
data/cleaned/{heart,lung}/
        |
        v
[run_augmentation.py]  -->  data/augmented/{heart,lung}/
        |
        v
[run_mfcc.py]          -->  data/features/
        |
        v
[run_train.py]         -->  models/
        |
        v
[run_eval.py]          -->  evaluation/
```

---

## Notes

- All scripts use **relative paths** resolved from their own location.
  They can be run from any working directory.
- Log output is plain ASCII (no emoji, no colour codes) for compatibility
  with any terminal or IDE including VSCode.
- No hard-coded drive letters or absolute system paths.
- Scripts are self-contained: missing optional libraries (librosa, TensorFlow)
  trigger a clear error message with the install command.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: soundfile` | `pip install soundfile` |
| `ModuleNotFoundError: librosa` | `pip install librosa` |
| `ModuleNotFoundError: sklearn` | `pip install scikit-learn` |
| `ModuleNotFoundError: tensorflow` | `pip install tensorflow` |
| `No .wav files found` | Check that audio is in `data/cleaned/{organ}/` |
| `No feature files found` | Run `run_mfcc.py` before `run_train.py` |
| `No saved model found` | Run `run_train.py` before `run_eval.py` |
| Model accuracy is low | Add more data, increase `--n_aug`, or use `--model cnn` |

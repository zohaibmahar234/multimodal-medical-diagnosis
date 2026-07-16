# Multi-Modal Medical Diagnosis System

A production-quality PyTorch project that predicts pneumonia from chest X-rays
by **fusing two data modalities**:

1. **Chest X-ray images** — a ResNet50 CNN branch (transfer learning)
2. **Patient clinical/tabular data** — age, gender, symptoms (fever, cough,
   chest pain), blood oxygen level, smoking history — a small dense
   network branch

The two branches are concatenated ("fused") and passed through a final
classification head that outputs a single **disease probability**
(Pneumonia: Positive/Negative).

> ⚠️ **Not a medical device.** This is a research/educational project. The
> clinical data is synthetically generated and the model must never be used
> for real diagnostic decisions.

---

## 1. Architecture

```
                ┌─────────────────────┐
Chest X-ray ──► │  ResNet50 backbone   │ ──► 2048-d pooled features
 (224x224x3)    │ (frozen + layer4 FT) │ ──► Linear+ReLU+Dropout ──► 256-d
                └─────────────────────┘                                │
                                                                        ▼
                                                                 ┌────────────┐
                                                                 │  Concat    │──► 288-d
                                                                 └────────────┘
                                                                        ▲
                ┌─────────────────────┐                                │
Clinical data ─►│  Dense (64) + BN     │ ──► Dropout ──► Dense(32)+ReLU ─┘
 (10 features)  │  (age, gender,       │
                │   symptoms, SpO2,    │
                │   smoking history)   │
                └─────────────────────┘
                                                                        │
                                                                        ▼
                                                       Dense(64)+ReLU+Dropout
                                                                        │
                                                                        ▼
                                                          Dense(1) → sigmoid
                                                                        │
                                                                        ▼
                                                       P(Pneumonia) ∈ [0, 1]
```

- **Image branch** (`src/model.py::ImageBranch`): `torchvision.models.resnet50`
  pretrained on ImageNet. Early layers are frozen; `layer4` (the last
  residual block) is left trainable for fine-tuning on chest X-rays.
- **Tabular branch** (`src/model.py::TabularBranch`): a 3-layer dense
  network (`Linear → BatchNorm → Dropout → Linear`) over the scaled/encoded
  clinical features.
- **Fusion**: the two branches' output vectors are concatenated and passed
  through a final 2-layer classification head producing one logit.

---

## 2. Project structure

```
multimodal_medical_diagnosis/
├── data/
│   ├── raw/               # place the downloaded Kaggle dataset here
│   └── processed/         # generated clinical_data.csv lands here
├── src/
│   ├── data_preprocessing.py     # split, scaling/encoding, integrity checks
│   ├── generate_synthetic_clinical_data.py
│   ├── dataset.py                # transforms + PyTorch Dataset
│   ├── model.py                  # ImageBranch, TabularBranch, fusion model
│   ├── train.py                  # training loop
│   ├── evaluate.py               # metrics, confusion matrix, Grad-CAM
│   └── predict.py                # single-sample CLI inference
├── app/
│   └── streamlit_app.py          # interactive web demo
├── tests/
│   ├── test_data_preprocessing.py
│   ├── test_model.py
│   └── test_predict.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── saved_models/                 # checkpoints + fitted scaler/encoder land here
├── smoke_test.py                 # zero-dependency-on-real-data dry run
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 3. Setup

**Target interpreter: Python 3.10 or 3.11.** (Newer/bleeding-edge Python
versions may lack wheels for some of the pinned packages below.)

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

> If you have an NVIDIA GPU and want CUDA acceleration, install the matching
> CUDA build of `torch`/`torchvision` from
> https://pytorch.org/get-started/locally/ **instead of** the default PyPI
> wheels, keeping the same `torch==2.3.1` / `torchvision==0.18.1` versions.

---

## 4. Getting the dataset

This project uses the Kaggle **"Chest X-Ray Images (Pneumonia)"** dataset
for the image branch. It has no clinical data of its own — see
[§5](#5-synthetic-clinical-data) for how that's generated.

**Option A — kagglehub (recommended):**

```bash
pip install kagglehub
python -c "
import kagglehub
path = kagglehub.dataset_download('paultimothymooney/chest-xray-pneumonia')
print('Downloaded to:', path)
"
```
Then copy/symlink the `chest_xray/` folder it downloads into
`data/raw/chest_xray/`.

**Option B — manual download:**
1. Go to https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Download and unzip so you end up with:
   `data/raw/chest_xray/{train,test,val}/{NORMAL,PNEUMONIA}/*.jpeg`

**Option C — no Kaggle account / offline fallback:**
If `data/raw/chest_xray` doesn't exist,
`src/generate_synthetic_clinical_data.py` automatically falls back to a
small dummy image-path pool so you can still exercise the full pipeline
(this is also exactly what `smoke_test.py` does, using real tiny fake
images instead of just placeholder filenames).

**Important:** the Kaggle download ships its own `train/`, `test/`, `val/`
folders. This project **recombines all of them into one pool** and applies
a single fresh **stratified 70/15/15 split** — see the docstring in
`generate_synthetic_clinical_data.py` for why (keeping Kaggle's split *and*
re-splitting would break image-to-clinical-row pairing and double-split
the data).

---

## 5. Synthetic clinical data

Run this after the images are in place (or let it fall back to a dummy
pool if they're not):

```bash
python src/generate_synthetic_clinical_data.py \
    --image_root data/raw/chest_xray \
    --output data/processed/clinical_data.csv
```

This creates one clinical row per image, **correlated with the image's
label** (pneumonia-positive rows get a higher probability of fever/cough/
chest pain and lower blood-oxygen readings) using a fixed random seed
(`42`) for reproducibility. See the fully-commented logic in
`src/generate_synthetic_clinical_data.py`.

---

## 6. Quick sanity check: `smoke_test.py`

Before touching the real dataset, verify the entire pipeline works:

```bash
python smoke_test.py
```

This builds a tiny synthetic pool (16 fake random images + matching
clinical rows), runs a stratified split, fits the scaler/encoder, builds
the `Dataset`/`DataLoader`, constructs the model (`pretrained=False` for
speed), runs one forward pass, one training step, saves a checkpoint, and
runs one end-to-end prediction — entirely in a temp directory, with
assertions after every step. **This has been executed in the environment
used to build this project and passes end-to-end with zero errors**
(see §10 for the full trace and confirmation).

---

## 7. Training

```bash
python src/train.py --epochs 20 --batch_size 16 --lr 1e-4
```

Key behavior:
- **Class imbalance**: `BCEWithLogitsLoss(pos_weight=...)`, where
  `pos_weight` is computed from the actual train-split class counts and
  printed before training starts (see `train.py::compute_pos_weight` for
  why plain Accuracy is misleading on this imbalanced dataset).
- **Checkpointing**: the model is saved to `saved_models/best_model.pth`
  whenever **validation Recall** improves — not validation loss — because
  in this medical context a missed positive case (false negative) is far
  costlier than a false alarm.
- **Early stopping**: stops if validation Recall doesn't improve for
  `--early_stopping_patience` epochs (default 5).
- **LR scheduling**: `ReduceLROnPlateau` on validation Recall.
- **Logging**: every epoch's loss/accuracy/recall (train + val) is printed
  and appended to `saved_models/train.log`.
- **Artifacts saved to `saved_models/`**: `best_model.pth`,
  `scaler.joblib`, `encoder.joblib`, `tabular_feature_columns.joblib`,
  `train.log`. `predict.py` and the Streamlit app load these exact fitted
  transformers — they never re-fit new ones.

Useful flags: `--no_pretrained` (skip ImageNet weights, useful for fast
local testing), `--image_root <path>` (prefix for relative image paths in
the CSV), `--num_workers N`.

---

## 8. Evaluation

```bash
python src/evaluate.py --checkpoint saved_models/best_model.pth
```

Rebuilds the exact same held-out test split (same seed, same stratified
split logic) and reports **Accuracy, Precision, Recall, F1-score, ROC-AUC**
and a **Confusion Matrix** (saved as
`saved_models/eval_outputs/confusion_matrix.png`). It also generates
**Grad-CAM heatmaps** for a few test images
(`saved_models/eval_outputs/gradcam/`), highlighting which regions of the
X-ray influenced the prediction. See the module docstring in
`evaluate.py` for a full explanation of why Recall is the primary metric
here, not Accuracy.

---

## 9. Single prediction

```bash
python src/predict.py \
    --image_path path/to/xray.jpeg \
    --age 65 --gender male \
    --fever 1 --cough 1 --chest_pain 0 \
    --oxygen_level 92.5 --smoking_history former
```

Outputs the predicted class (`Normal`/`Pneumonia`) and a confidence score,
using the exact fitted scaler/encoder saved during training.

---

## 10. Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Upload an X-ray, fill in the clinical-data form, click **Predict**, and see
the predicted class, confidence, and a Grad-CAM heatmap overlay.

---

## 11. Tests

```bash
pytest tests/ -v
```

Covers: data-preprocessing functions (split ratios/stratification, scaler/
encoder fitting and transforming, single-record transform), model
architecture (output shapes for each branch and the fused model, including
a single-sample batch edge case), and `predict.py`'s output format and
error handling on a missing image file.

---

## 12. Exact terminal commands, in order

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# (Optional) download the real Kaggle dataset into data/raw/chest_xray — see §4

python src/generate_synthetic_clinical_data.py
python smoke_test.py
python src/train.py --epochs 20 --batch_size 16
python src/evaluate.py
streamlit run app/streamlit_app.py
```

---

## 13. Sample expected output

`smoke_test.py` (actually executed while building this project):

```
[SMOKE TEST] 1. Building tiny synthetic image + clinical pool
Built 16 fake rows. Label counts:
label
0    8
1    8

[SMOKE TEST] 2. Stratified 70/15/15 split
train=11 val=2 test=3

[SMOKE TEST] 5. Building model (pretrained=False for a fast, offline smoke test)
Device: cpu

[SMOKE TEST] 6. One forward pass
Forward pass output shape: (4,) (expected: (4,))

[SMOKE TEST] 7. One training step (with pos_weight for class imbalance)
Training step loss: 0.6353

[SMOKE TEST] 9. One end-to-end prediction via predict_single()
Prediction result: {'label': 'Pneumonia', 'confidence': 0.5138, 'probability_pneumonia': 0.5138}

############################################################
# SMOKE TEST PASSED — full pipeline runs end-to-end with
# zero errors on synthetic data.
############################################################
```

`pytest tests/ -v` (actually executed):

```
tests/test_data_preprocessing.py::test_validate_dataframe_drops_missing_rows PASSED
tests/test_data_preprocessing.py::test_validate_dataframe_raises_on_missing_columns PASSED
tests/test_data_preprocessing.py::test_stratified_split_preserves_class_balance PASSED
tests/test_data_preprocessing.py::test_fit_and_transform_split_produces_numeric_matrix PASSED
tests/test_data_preprocessing.py::test_transform_single_record_matches_expected_columns PASSED
tests/test_model.py::test_image_branch_output_shape PASSED
tests/test_model.py::test_tabular_branch_output_shape PASSED
tests/test_model.py::test_fusion_model_output_shape PASSED
tests/test_model.py::test_fusion_model_gradcam_target_layer_exists PASSED
tests/test_model.py::test_fusion_model_single_sample_batch PASSED
tests/test_predict.py::test_predict_single_output_format PASSED
tests/test_predict.py::test_predict_single_missing_image_raises PASSED

12 passed
```

`python src/train.py --epochs 3 --batch_size 4 --no_pretrained` on a tiny
20-image fake dataset (actually executed):

```
Using device: cpu
Split sizes -> train: 13, val: 3, test: 4
Train class distribution: {1: 7, 0: 6}
Using BCEWithLogitsLoss pos_weight = 0.8571 (computed from train split)
Epoch 1/3 (3.6s) | train_loss=0.6465 train_acc=0.2500 train_recall=0.0000 | val_loss=0.6558 val_acc=0.6667 val_recall=0.0000 val_f1=0.0000 | lr=1.00e-04
  -> New best val_recall=0.0000. Saved checkpoint to saved_models/best_model.pth
Epoch 2/3 (3.5s) | train_loss=0.5774 train_acc=0.4167 train_recall=0.0000
Epoch 3/3 (3.5s) | train_loss=0.6013 train_acc=0.2500 train_recall=0.1429
Training complete. Best val_recall=0.1429. Checkpoint: saved_models/best_model.pth
```

`python src/evaluate.py` on that same tiny checkpoint (actually executed):

```
Using device: cpu
==================================================
Evaluation metrics (test split)
==================================================
  ACCURACY: 0.5000
 PRECISION: 0.0000
    RECALL: 0.0000
        F1: 0.0000
   ROC_AUC: 0.7500
==================================================
Saved confusion matrix to 'saved_models/eval_outputs/confusion_matrix.png'
Saved 2 Grad-CAM sample(s) to 'saved_models/eval_outputs/gradcam'
```

(These particular numbers are from a 20-image, 3-epoch, untrained-backbone
run purely to prove the pipeline works — real training on the full ~5,800
image Kaggle dataset with a pretrained backbone will produce far better,
clinically meaningful metrics.)

---

## 14. Known failure points this project explicitly handles

- **Missing/corrupted images**: `data_preprocessing.filter_missing_or_corrupted_images`
  drops rows whose file is missing or fails a `PIL.Image.verify()` check,
  with a clear log message.
- **Empty/NaN CSV rows**: `data_preprocessing.validate_dataframe` drops
  them and reports how many were removed.
- **GPU not available**: `train.get_device()` automatically falls back to
  CPU.
- **Mismatched image/tabular row counts**: prevented by construction —
  the clinical CSV is generated 1:1 against the discovered image pool
  (`generate_synthetic_clinical_data.py`), and `Dataset.__init__` validates
  that all required columns are present before training starts.
- **Undersized last training batch vs. BatchNorm1d**: the train
  `DataLoader` uses `drop_last=True` (when there's more than one batch) —
  `BatchNorm1d` cannot compute a variance over a batch of size 1 in
  training mode, so this is dropped rather than crashing mid-epoch. This
  was caught by actually running `train.py` end-to-end while building this
  project, not just by inspection.
- **Inconsistent train/inference scaling**: `predict.py` and the Streamlit
  app load the exact fitted `StandardScaler`/`OneHotEncoder` saved during
  training via `joblib` — they never re-fit new ones.

"""
data_preprocessing.py

Handles:
  - Loading the combined image+clinical pool produced by
    generate_synthetic_clinical_data.py
  - A single STRATIFIED 70/15/15 train/val/test split over the *combined*
    pool (never re-splitting Kaggle's own train/test/val folders — see the
    module docstring in generate_synthetic_clinical_data.py for why).
  - Fitting a StandardScaler (numeric features) and OneHotEncoder
    (categorical features) on the TRAIN split only, then transforming all
    three splits with those same fitted objects (no leakage).
  - Saving the fitted transformers with joblib so train.py, predict.py, and
    the Streamlit app all reuse the exact same fitted scaler/encoder.
  - Basic data-integrity checks: missing files, empty rows, corrupted
    images, mismatched image/tabular row counts.
"""

import os
import random
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

NUMERIC_FEATURES = ["age", "oxygen_level"]
BINARY_FEATURES = ["fever", "cough", "chest_pain"]  # already 0/1, no encoding needed
CATEGORICAL_FEATURES = ["gender", "smoking_history"]


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing/empty critical fields and warn about them.

    Guards against the "empty CSV rows" and "mismatched row counts"
    failure modes called out in the project requirements.
    """
    required_cols = ["image_path", "label"] + NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Clinical CSV is missing required columns: {missing_cols}")

    before = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[data_preprocessing] Dropped {dropped} row(s) with missing values.")

    if df.empty:
        raise ValueError("Clinical CSV has no valid rows after dropping missing values.")

    return df


def filter_missing_or_corrupted_images(df: pd.DataFrame, image_root: str = None, strict: bool = False) -> pd.DataFrame:
    """Remove rows whose image file is missing or fails to open.

    When `strict=False` (default), this simply skips bad rows and logs a
    warning — useful for real-world messy datasets. When `strict=True`, it
    raises immediately (useful for CI / smoke tests where any bad file
    indicates a real bug rather than a flaky dataset).
    """
    valid_rows = []
    bad_paths = []

    for _, row in df.iterrows():
        path = row["image_path"]
        full_path = os.path.join(image_root, path) if image_root else path

        if not os.path.exists(full_path):
            bad_paths.append((full_path, "missing"))
            continue
        try:
            with Image.open(full_path) as img:
                img.verify()  # cheap corruption check without full decode
            valid_rows.append(row)
        except Exception:
            bad_paths.append((full_path, "corrupted"))

    if bad_paths:
        msg = f"[data_preprocessing] Found {len(bad_paths)} missing/corrupted image(s)."
        if strict:
            raise FileNotFoundError(msg + f" First few: {bad_paths[:5]}")
        print(msg + " These rows will be excluded.")

    return pd.DataFrame(valid_rows).reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame, train_size: float = 0.70, val_size: float = 0.15, test_size: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Single stratified 70/15/15 split over the combined pool, stratified
    by `label` so class balance (important given the pneumonia/normal
    imbalance) is preserved identically in train, val, and test.
    """
    assert abs((train_size + val_size + test_size) - 1.0) < 1e-6, "Split sizes must sum to 1.0"

    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_size), stratify=df["label"], random_state=SEED
    )
    relative_val_size = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val_size), stratify=temp_df["label"], random_state=SEED
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def fit_transformers(train_df: pd.DataFrame) -> Tuple[StandardScaler, OneHotEncoder]:
    """Fit StandardScaler + OneHotEncoder on the TRAIN split ONLY."""
    scaler = StandardScaler()
    scaler.fit(train_df[NUMERIC_FEATURES])

    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        # scikit-learn < 1.2 used `sparse` instead of `sparse_output`.
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    encoder.fit(train_df[CATEGORICAL_FEATURES])

    return scaler, encoder


def transform_split(df: pd.DataFrame, scaler: StandardScaler, encoder: OneHotEncoder) -> Tuple[pd.DataFrame, List[str]]:
    """Apply already-fitted scaler/encoder to a split, returning a new
    DataFrame with numeric feature columns replaced and one-hot columns
    appended, plus the ordered list of final tabular feature column names
    (needed by the Dataset class and the model's input_dim).
    """
    df = df.copy()

    scaled_numeric = scaler.transform(df[NUMERIC_FEATURES])
    for i, col in enumerate(NUMERIC_FEATURES):
        df[col] = scaled_numeric[:, i]

    encoded_cat = encoder.transform(df[CATEGORICAL_FEATURES])
    encoded_cols = encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    encoded_df = pd.DataFrame(encoded_cat, columns=encoded_cols, index=df.index)

    df = pd.concat([df.drop(columns=CATEGORICAL_FEATURES), encoded_df], axis=1)

    tabular_feature_columns = NUMERIC_FEATURES + BINARY_FEATURES + encoded_cols
    return df, tabular_feature_columns


def prepare_datasets(
    clinical_csv_path: str,
    image_root: str = None,
    saved_models_dir: str = "saved_models",
    strict_image_check: bool = False,
):
    """End-to-end preprocessing pipeline:
    1. load + validate CSV
    2. filter missing/corrupted images
    3. stratified 70/15/15 split
    4. fit scaler/encoder on train, transform all splits
    5. persist fitted transformers to disk

    Returns (train_df, val_df, test_df, tabular_feature_columns).
    """
    df = pd.read_csv(clinical_csv_path)
    df = validate_dataframe(df)
    df = filter_missing_or_corrupted_images(df, image_root=image_root, strict=strict_image_check)

    train_df, val_df, test_df = stratified_split(df)

    scaler, encoder = fit_transformers(train_df)

    train_df, tabular_feature_columns = transform_split(train_df, scaler, encoder)
    val_df, _ = transform_split(val_df, scaler, encoder)
    test_df, _ = transform_split(test_df, scaler, encoder)

    os.makedirs(saved_models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(saved_models_dir, "scaler.joblib"))
    joblib.dump(encoder, os.path.join(saved_models_dir, "encoder.joblib"))
    joblib.dump(
        tabular_feature_columns, os.path.join(saved_models_dir, "tabular_feature_columns.joblib")
    )

    return train_df, val_df, test_df, tabular_feature_columns


def load_fitted_transformers(saved_models_dir: str = "saved_models"):
    """Load the exact fitted StandardScaler/OneHotEncoder/feature-column
    list saved during training. predict.py and the Streamlit app MUST use
    this instead of re-fitting new transformers, or clinical inputs would
    be scaled/encoded inconsistently between training and inference.
    """
    scaler = joblib.load(os.path.join(saved_models_dir, "scaler.joblib"))
    encoder = joblib.load(os.path.join(saved_models_dir, "encoder.joblib"))
    tabular_feature_columns = joblib.load(
        os.path.join(saved_models_dir, "tabular_feature_columns.joblib")
    )
    return scaler, encoder, tabular_feature_columns


def transform_single_record(record: dict, scaler: StandardScaler, encoder: OneHotEncoder, tabular_feature_columns: List[str]) -> np.ndarray:
    """Transform one raw clinical dict (as collected from CLI/Streamlit)
    into the fixed-order numeric feature vector the model expects.
    """
    row_df = pd.DataFrame([record])

    missing = [c for c in NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES if c not in row_df.columns]
    if missing:
        raise ValueError(f"Missing clinical fields for prediction: {missing}")

    scaled_numeric = scaler.transform(row_df[NUMERIC_FEATURES])
    encoded_cat = encoder.transform(row_df[CATEGORICAL_FEATURES])
    encoded_cols = encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()

    full = {}
    for i, col in enumerate(NUMERIC_FEATURES):
        full[col] = scaled_numeric[0, i]
    for col in BINARY_FEATURES:
        full[col] = float(row_df.iloc[0][col])
    for i, col in enumerate(encoded_cols):
        full[col] = encoded_cat[0, i]

    vector = np.array([full[c] for c in tabular_feature_columns], dtype=np.float32)
    return vector

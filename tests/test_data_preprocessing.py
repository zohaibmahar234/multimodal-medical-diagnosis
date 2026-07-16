import os
import sys

import numpy as np
import pandas as pd
import pytest

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.append(SRC_DIR)

from data_preprocessing import (  # noqa: E402
    fit_transformers,
    stratified_split,
    transform_single_record,
    transform_split,
    validate_dataframe,
)


def make_fake_df(n=40):
    rng = np.random.default_rng(42)
    labels = [0] * (n // 2) + [1] * (n // 2)
    return pd.DataFrame(
        {
            "image_path": [f"img_{i}.jpeg" for i in range(n)],
            "label": labels,
            "age": rng.integers(18, 80, size=n),
            "gender": rng.choice(["male", "female"], size=n),
            "fever": rng.integers(0, 2, size=n),
            "cough": rng.integers(0, 2, size=n),
            "chest_pain": rng.integers(0, 2, size=n),
            "oxygen_level": rng.uniform(85, 100, size=n),
            "smoking_history": rng.choice(["never", "former", "current"], size=n),
        }
    )


def test_validate_dataframe_drops_missing_rows():
    df = make_fake_df()
    df.loc[0, "age"] = np.nan
    cleaned = validate_dataframe(df)
    assert len(cleaned) == len(df) - 1
    assert cleaned["age"].isna().sum() == 0


def test_validate_dataframe_raises_on_missing_columns():
    df = make_fake_df().drop(columns=["age"])
    with pytest.raises(ValueError):
        validate_dataframe(df)


def test_stratified_split_preserves_class_balance():
    df = make_fake_df(n=100)
    train_df, val_df, test_df = stratified_split(df)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    # Roughly 70/15/15 (allow slack for small-sample rounding).
    assert 0.6 < len(train_df) / len(df) < 0.8
    assert 0.05 < len(val_df) / len(df) < 0.25
    assert 0.05 < len(test_df) / len(df) < 0.25

    overall_ratio = df["label"].mean()
    for split_df in (train_df, val_df, test_df):
        assert abs(split_df["label"].mean() - overall_ratio) < 0.15


def test_fit_and_transform_split_produces_numeric_matrix():
    df = make_fake_df()
    train_df, val_df, _ = stratified_split(df)
    scaler, encoder = fit_transformers(train_df)

    transformed_train, feature_cols = transform_split(train_df, scaler, encoder)
    transformed_val, _ = transform_split(val_df, scaler, encoder)

    assert set(feature_cols).issubset(transformed_train.columns)
    assert transformed_train[feature_cols].select_dtypes(include=[np.number]).shape[1] == len(feature_cols)
    assert not transformed_val[feature_cols].isna().any().any()


def test_transform_single_record_matches_expected_columns():
    df = make_fake_df()
    train_df, _, _ = stratified_split(df)
    scaler, encoder = fit_transformers(train_df)
    _, feature_cols = transform_split(train_df, scaler, encoder)

    record = {
        "age": 55,
        "gender": "female",
        "fever": 1,
        "cough": 0,
        "chest_pain": 1,
        "oxygen_level": 93.2,
        "smoking_history": "former",
    }
    vector = transform_single_record(record, scaler, encoder, feature_cols)
    assert vector.shape == (len(feature_cols),)
    assert vector.dtype == np.float32

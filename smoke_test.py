"""
smoke_test.py

End-to-end dry run using ONLY tiny synthetic data (a handful of fake random
images + fake clinical rows). Proves the entire pipeline — data loading,
model construction, one forward pass, one training step, and one
prediction — works with zero errors, WITHOUT needing the real Kaggle
dataset downloaded.

Run:
    python smoke_test.py
"""

import os
import random
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.append(SRC_DIR)

from data_preprocessing import (  # noqa: E402
    fit_transformers,
    stratified_split,
    transform_single_record,
    transform_split,
)
from dataset import ChestXrayMultiModalDataset, get_eval_transforms, get_train_transforms  # noqa: E402
from model import MultiModalDiagnosisModel  # noqa: E402
from train import compute_pos_weight, get_device  # noqa: E402
from predict import predict_single  # noqa: E402

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def make_fake_image(path: str, size=(224, 224)):
    arr = (np.random.rand(*size, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def build_fake_pool(tmp_dir: str, n_per_class: int = 8) -> pd.DataFrame:
    images_dir = os.path.join(tmp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)
    records = []

    for label, class_name in ((0, "normal"), (1, "pneumonia")):
        for i in range(n_per_class):
            fname = f"{class_name}_{i}.jpeg"
            fpath = os.path.join(images_dir, fname)
            make_fake_image(fpath)

            if label == 1:
                fever = int(rng.random() < 0.8)
                cough = int(rng.random() < 0.85)
                chest_pain = int(rng.random() < 0.5)
                oxygen_level = float(np.clip(rng.normal(91, 3), 70, 100))
                age = int(np.clip(rng.normal(55, 15), 0, 95))
                smoking = rng.choice(["never", "former", "current"], p=[0.4, 0.3, 0.3])
            else:
                fever = int(rng.random() < 0.1)
                cough = int(rng.random() < 0.15)
                chest_pain = int(rng.random() < 0.05)
                oxygen_level = float(np.clip(rng.normal(97.5, 1.5), 85, 100))
                age = int(np.clip(rng.normal(35, 12), 0, 95))
                smoking = rng.choice(["never", "former", "current"], p=[0.7, 0.2, 0.1])

            records.append(
                {
                    "image_path": fpath,
                    "label": label,
                    "age": age,
                    "gender": rng.choice(["male", "female"]),
                    "fever": fever,
                    "cough": cough,
                    "chest_pain": chest_pain,
                    "oxygen_level": round(oxygen_level, 1),
                    "smoking_history": smoking,
                }
            )

    return pd.DataFrame(records)


def step(name):
    print(f"\n{'=' * 60}\n[SMOKE TEST] {name}\n{'=' * 60}")


def main():
    tmp_dir = tempfile.mkdtemp(prefix="smoke_test_")
    saved_models_dir = os.path.join(tmp_dir, "saved_models")
    os.makedirs(saved_models_dir, exist_ok=True)

    try:
        step("1. Building tiny synthetic image + clinical pool")
        df = build_fake_pool(tmp_dir, n_per_class=8)
        print(f"Built {len(df)} fake rows. Label counts:\n{df['label'].value_counts()}")
        assert len(df) == 16

        step("2. Stratified 70/15/15 split")
        train_df, val_df, test_df = stratified_split(df)
        print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
        assert len(train_df) + len(val_df) + len(test_df) == len(df)
        assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0

        step("3. Fitting + applying scaler/encoder (train-only fit)")
        scaler, encoder = fit_transformers(train_df)
        train_df, tabular_feature_columns = transform_split(train_df, scaler, encoder)
        val_df, _ = transform_split(val_df, scaler, encoder)
        test_df, _ = transform_split(test_df, scaler, encoder)
        print(f"Tabular feature columns ({len(tabular_feature_columns)}): {tabular_feature_columns}")

        import joblib

        joblib.dump(scaler, os.path.join(saved_models_dir, "scaler.joblib"))
        joblib.dump(encoder, os.path.join(saved_models_dir, "encoder.joblib"))
        joblib.dump(tabular_feature_columns, os.path.join(saved_models_dir, "tabular_feature_columns.joblib"))

        step("4. Building Dataset/DataLoader")
        train_dataset = ChestXrayMultiModalDataset(train_df, tabular_feature_columns, transform=get_train_transforms())
        val_dataset = ChestXrayMultiModalDataset(val_df, tabular_feature_columns, transform=get_eval_transforms())
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

        images, tabular, labels = next(iter(train_loader))
        print(f"Batch shapes -> images: {tuple(images.shape)}, tabular: {tuple(tabular.shape)}, labels: {tuple(labels.shape)}")
        assert images.shape[1:] == (3, 224, 224)
        assert tabular.shape[1] == len(tabular_feature_columns)

        step("5. Building model (pretrained=False for a fast, offline smoke test)")
        device = get_device()
        print(f"Device: {device}")
        model = MultiModalDiagnosisModel(tabular_input_dim=len(tabular_feature_columns), pretrained=False).to(device)

        step("6. One forward pass")
        model.eval()
        with torch.no_grad():
            images_dev, tabular_dev = images.to(device), tabular.to(device)
            logits = model(images_dev, tabular_dev)
        print(f"Forward pass output shape: {tuple(logits.shape)} (expected: ({images.shape[0]},))")
        assert logits.shape == (images.shape[0],)

        step("7. One training step (with pos_weight for class imbalance)")
        pos_weight = compute_pos_weight(train_df["label"])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.train()
        optimizer.zero_grad()
        logits = model(images_dev, tabular_dev)
        labels_dev = labels.to(device)
        loss = criterion(logits, labels_dev)
        loss.backward()
        optimizer.step()
        print(f"Training step loss: {loss.item():.4f}")
        assert torch.isfinite(loss)

        step("8. Saving a checkpoint")
        checkpoint_path = os.path.join(saved_models_dir, "best_model.pth")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "tabular_input_dim": len(tabular_feature_columns),
                "tabular_feature_columns": tabular_feature_columns,
                "epoch": 1,
                "val_recall": 0.0,
                "val_f1": 0.0,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint to {checkpoint_path}")

        step("9. One end-to-end prediction via predict_single()")
        sample_image_path = test_df.iloc[0]["image_path"]
        clinical_record = {
            "age": 60,
            "gender": "male",
            "fever": 1,
            "cough": 1,
            "chest_pain": 0,
            "oxygen_level": 92.0,
            "smoking_history": "former",
        }
        result = predict_single(
            image_path=sample_image_path,
            clinical_record=clinical_record,
            checkpoint_path=checkpoint_path,
            saved_models_dir=saved_models_dir,
        )
        print(f"Prediction result: {result}")
        assert set(result.keys()) == {"label", "confidence", "probability_pneumonia"}
        assert result["label"] in {"Normal", "Pneumonia"}
        assert 0.0 <= result["probability_pneumonia"] <= 1.0

        step("10. Sanity-checking transform_single_record directly")
        vec = transform_single_record(clinical_record, scaler, encoder, tabular_feature_columns)
        assert vec.shape == (len(tabular_feature_columns),)
        print(f"transform_single_record output shape: {vec.shape}")

        print("\n" + "#" * 60)
        print("# SMOKE TEST PASSED — full pipeline runs end-to-end with")
        print("# zero errors on synthetic data.")
        print("#" * 60)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

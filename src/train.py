"""
train.py

Trains the MultiModalDiagnosisModel with:
  - class-imbalance handling via BCEWithLogitsLoss(pos_weight=...)
  - early stopping
  - ReduceLROnPlateau learning-rate scheduling
  - checkpointing on BEST VALIDATION RECALL (medical use case: a missed
    positive/pneumonia case is far more costly than a false alarm, so we
    optimize for recall rather than raw accuracy or even loss)
  - console + file logging of loss/accuracy/recall per epoch
  - automatic GPU-> CPU fallback

Run:
    python src/train.py --epochs 20 --batch_size 16
"""

import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import prepare_datasets  # noqa: E402
from dataset import ChestXrayMultiModalDataset, get_eval_transforms, get_train_transforms  # noqa: E402
from model import MultiModalDiagnosisModel  # noqa: E402

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def get_device() -> torch.device:
    """Automatic GPU-not-available fallback to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logging(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def compute_pos_weight(labels: pd.Series) -> torch.Tensor:
    """Compute pos_weight for BCEWithLogitsLoss from actual class counts.

    IMPORTANT: this dataset is naturally imbalanced (many more PNEUMONIA
    images than NORMAL in the Kaggle set). If we ignored that, a model
    could reach ~75%+ "accuracy" by always predicting the majority class
    while missing most/all NORMAL cases (or vice versa depending on which
    class dominates) — accuracy alone would look great while the model is
    clinically useless. pos_weight up-weights the minority class's
    contribution to the loss so the model doesn't just learn to predict
    the majority class.
    """
    counts = labels.value_counts()
    n_neg = counts.get(0, 0)
    n_pos = counts.get(1, 0)
    if n_pos == 0:
        return torch.tensor(1.0)
    pos_weight = n_neg / n_pos
    return torch.tensor(pos_weight, dtype=torch.float32)


@torch.no_grad()
def evaluate_epoch(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, tabular, labels in dataloader:
        images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
        logits = model(images, tabular)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(dataloader.dataset), 1)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, accuracy, recall, f1


def train_one_epoch(model, dataloader, device, criterion, optimizer):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, tabular, labels in dataloader:
        images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images, tabular)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(len(dataloader.dataset), 1)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    return avg_loss, accuracy, recall


def run_training(config: dict):
    logger = setup_logging(config["log_path"])
    device = get_device()
    logger.info(f"Using device: {device}")

    logger.info("Preparing datasets (stratified 70/15/15 split, fitting scaler/encoder)...")
    train_df, val_df, test_df, tabular_feature_columns = prepare_datasets(
        clinical_csv_path=config["clinical_csv_path"],
        image_root=config.get("image_root"),
        saved_models_dir=config["saved_models_dir"],
    )
    logger.info(
        f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)} "
        f"(saved test split not used here; see evaluate.py)"
    )
    logger.info(f"Tabular feature columns ({len(tabular_feature_columns)}): {tabular_feature_columns}")

    class_counts = train_df["label"].value_counts().to_dict()
    logger.info(f"Train class distribution: {class_counts}")
    pos_weight = compute_pos_weight(train_df["label"])
    logger.info(f"Using BCEWithLogitsLoss pos_weight = {pos_weight.item():.4f} (computed from train split)")

    train_dataset = ChestXrayMultiModalDataset(
        train_df, tabular_feature_columns, transform=get_train_transforms(), image_root=config.get("image_root")
    )
    val_dataset = ChestXrayMultiModalDataset(
        val_df, tabular_feature_columns, transform=get_eval_transforms(), image_root=config.get("image_root")
    )

    # drop_last=True on the TRAIN loader only: TabularBranch uses
    # BatchNorm1d, which raises a ValueError on a batch of size 1 in
    # training mode (no variance can be computed over a single sample).
    # Dropping a possible final undersized batch avoids that crash without
    # affecting val/test evaluation (where we keep every sample, since
    # evaluate_epoch() always runs the model in eval() mode).
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=(len(train_dataset) > config["batch_size"]),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"]
    )

    model = MultiModalDiagnosisModel(
        tabular_input_dim=len(tabular_feature_columns), pretrained=config["pretrained"]
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_recall = -1.0
    epochs_without_improvement = 0
    checkpoint_path = os.path.join(config["saved_models_dir"], "best_model.pth")
    os.makedirs(config["saved_models_dir"], exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        start = time.time()
        train_loss, train_acc, train_recall = train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc, val_recall, val_f1 = evaluate_epoch(model, val_loader, device, criterion)
        scheduler.step(val_recall)
        elapsed = time.time() - start

        logger.info(
            f"Epoch {epoch}/{config['epochs']} ({elapsed:.1f}s) | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_recall={train_recall:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_recall={val_recall:.4f} val_f1={val_f1:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_recall > best_val_recall:
            best_val_recall = val_recall
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "tabular_input_dim": len(tabular_feature_columns),
                    "tabular_feature_columns": tabular_feature_columns,
                    "epoch": epoch,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                },
                checkpoint_path,
            )
            logger.info(f"  -> New best val_recall={val_recall:.4f}. Saved checkpoint to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["early_stopping_patience"]:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"({epochs_without_improvement} epochs without val_recall improvement)."
                )
                break

    logger.info(f"Training complete. Best val_recall={best_val_recall:.4f}. Checkpoint: {checkpoint_path}")
    return checkpoint_path


def build_config_from_args(args) -> dict:
    return {
        "clinical_csv_path": args.clinical_csv_path,
        "image_root": args.image_root,
        "saved_models_dir": args.saved_models_dir,
        "log_path": args.log_path,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "num_workers": args.num_workers,
        "early_stopping_patience": args.early_stopping_patience,
        "pretrained": not args.no_pretrained,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the multi-modal medical diagnosis model.")
    parser.add_argument("--clinical_csv_path", type=str, default=os.path.join("data", "processed", "clinical_data.csv"))
    parser.add_argument("--image_root", type=str, default=None, help="Optional root prefix for image_path entries.")
    parser.add_argument("--saved_models_dir", type=str, default="saved_models")
    parser.add_argument("--log_path", type=str, default=os.path.join("saved_models", "train.log"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--no_pretrained", action="store_true", help="Disable ImageNet pretrained weights (used by tests/smoke_test.py for speed).")
    args = parser.parse_args()

    config = build_config_from_args(args)
    run_training(config)


if __name__ == "__main__":
    main()

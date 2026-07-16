"""
evaluate.py

Loads the best checkpoint (saved on best validation recall) and reports,
on the held-out TEST split:
    Accuracy, Precision, Recall, F1-score, ROC-AUC, and a Confusion Matrix
    (saved as a PNG).

Also generates a handful of Grad-CAM heatmaps on test images to visualize
which regions of the X-ray influenced the prediction (explainability).

WHY RECALL MATTERS MORE THAN ACCURACY HERE:
In this medical screening context, a False Negative (predicting "Normal"
for an actual Pneumonia case) means a sick patient goes undiagnosed and
untreated — potentially fatal. A False Positive (predicting "Pneumonia"
for a Normal patient) just means an extra clinical review, which is far
less costly. Because the dataset is imbalanced, a model can post a high
Accuracy while still missing a large fraction of true positive cases.
Recall = TP / (TP + FN) directly measures "of all the actually-sick
patients, how many did we catch?" — which is the number that matters most
for patient safety, even at the cost of some extra false alarms.

Run:
    python src/evaluate.py --checkpoint saved_models/best_model.pth
"""

import argparse
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import (  # noqa: E402
    filter_missing_or_corrupted_images,
    load_fitted_transformers,
    stratified_split,
    transform_split,
    validate_dataframe,
)
from dataset import ChestXrayMultiModalDataset, get_eval_transforms  # noqa: E402
from model import MultiModalDiagnosisModel  # noqa: E402
from train import get_device  # noqa: E402

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device, pretrained: bool = False):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MultiModalDiagnosisModel(tabular_input_dim=checkpoint["tabular_input_dim"], pretrained=pretrained)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["tabular_feature_columns"]


def rebuild_test_split(clinical_csv_path: str, image_root, saved_models_dir: str):
    """Rebuilds the exact same test split used during training (same seed,
    same stratified split logic) and applies the ALREADY-FITTED scaler/encoder
    (never re-fit at evaluation time).
    """
    import pandas as pd

    df = pd.read_csv(clinical_csv_path)
    df = validate_dataframe(df)
    df = filter_missing_or_corrupted_images(df, image_root=image_root, strict=False)

    _, _, test_df = stratified_split(df)

    scaler, encoder, tabular_feature_columns = load_fitted_transformers(saved_models_dir)
    test_df, _ = transform_split(test_df, scaler, encoder)

    return test_df, tabular_feature_columns


@torch.no_grad()
def get_predictions(model, dataloader, device):
    all_probs, all_labels = [], []
    for images, tabular, labels in dataloader:
        images, tabular = images.to(device), tabular.to(device)
        logits = model(images, tabular)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())
    return np.array(all_probs), np.array(all_labels)


def compute_and_print_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        # Recall matters more than accuracy in this medical context — see
        # module docstring above for the clinical reasoning.
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_probs)
    except ValueError:
        metrics["roc_auc"] = float("nan")  # e.g. only one class present in a tiny smoke-test split

    print("=" * 50)
    print("Evaluation metrics (test split)")
    print("=" * 50)
    for name, value in metrics.items():
        print(f"{name.upper():>10}: {value:.4f}")
    print("=" * 50)

    return metrics


def plot_confusion_matrix(y_true, y_probs, output_path: str, threshold: float = 0.5):
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Pneumonia"], yticklabels=["Normal", "Pneumonia"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved confusion matrix to '{output_path}'")


def generate_gradcam_samples(model, test_dataset, device, output_dir: str, num_samples: int = 4):
    """Generate Grad-CAM heatmap overlays for a few test images."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("[evaluate] pytorch-grad-cam not installed; skipping Grad-CAM generation.")
        return

    os.makedirs(output_dir, exist_ok=True)
    target_layer = model.get_gradcam_target_layer()

    # pytorch-grad-cam needs a model whose forward takes a single tensor arg
    # and returns per-class scores. We wrap the fusion model so Grad-CAM can
    # backprop through the image branch while holding the paired tabular
    # input fixed for that sample.
    class ImageOnlyWrapper(torch.nn.Module):
        def __init__(self, base_model, fixed_tabular):
            super().__init__()
            self.base_model = base_model
            self.fixed_tabular = fixed_tabular

        def forward(self, image):
            tabular = self.fixed_tabular.expand(image.size(0), -1)
            logit = self.base_model(image, tabular)
            return logit.unsqueeze(1)  # (B, 1) "class score" for GradCAM

    n = min(num_samples, len(test_dataset))
    for i in range(n):
        image_tensor, tabular_tensor, label = test_dataset[i]
        image_batch = image_tensor.unsqueeze(0).to(device)
        tabular_batch = tabular_tensor.unsqueeze(0).to(device)

        wrapped = ImageOnlyWrapper(model, tabular_batch).to(device)
        cam = GradCAM(model=wrapped, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=image_batch)[0]  # (H, W) in [0, 1]

        # Undo ImageNet normalization for visualization.
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

        overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        out_path = os.path.join(output_dir, f"gradcam_sample_{i}_label{int(label.item())}.png")
        plt.imsave(out_path, overlay)

    print(f"Saved {n} Grad-CAM sample(s) to '{output_dir}'")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained multi-modal diagnosis model.")
    parser.add_argument("--checkpoint", type=str, default=os.path.join("saved_models", "best_model.pth"))
    parser.add_argument("--clinical_csv_path", type=str, default=os.path.join("data", "processed", "clinical_data.csv"))
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--saved_models_dir", type=str, default="saved_models")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default=os.path.join("saved_models", "eval_outputs"))
    parser.add_argument("--gradcam_samples", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    model, tabular_feature_columns = load_model_from_checkpoint(args.checkpoint, device)

    test_df, _ = rebuild_test_split(args.clinical_csv_path, args.image_root, args.saved_models_dir)
    test_dataset = ChestXrayMultiModalDataset(
        test_df, tabular_feature_columns, transform=get_eval_transforms(), image_root=args.image_root
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    y_probs, y_true = get_predictions(model, test_loader, device)
    compute_and_print_metrics(y_true, y_probs)
    plot_confusion_matrix(y_true, y_probs, os.path.join(args.output_dir, "confusion_matrix.png"))
    generate_gradcam_samples(model, test_dataset, device, os.path.join(args.output_dir, "gradcam"), args.gradcam_samples)


if __name__ == "__main__":
    main()

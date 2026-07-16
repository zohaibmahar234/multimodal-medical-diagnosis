"""
predict.py

Takes one new X-ray image path + one clinical data row and outputs a
prediction (Pneumonia/Normal) with a confidence score.

Uses the EXACT SAME fitted StandardScaler/OneHotEncoder saved during
training (via joblib) — never re-fits new transformers — so clinical
inputs are scaled/encoded identically to how the model was trained.

CLI usage:
    python src/predict.py \
        --image_path path/to/xray.jpeg \
        --age 65 --gender male --fever 1 --cough 1 --chest_pain 0 \
        --oxygen_level 92.5 --smoking_history former

Programmatic usage:
    from predict import predict_single
    result = predict_single(image_path=..., clinical_record={...})
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_fitted_transformers, transform_single_record  # noqa: E402
from dataset import get_eval_transforms  # noqa: E402
from model import MultiModalDiagnosisModel  # noqa: E402
from train import get_device  # noqa: E402

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CLASS_NAMES = {0: "Normal", 1: "Pneumonia"}


def load_model(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. Train the model first with src/train.py."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MultiModalDiagnosisModel(tabular_input_dim=checkpoint["tabular_input_dim"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["tabular_feature_columns"]


def load_and_transform_image(image_path: str) -> torch.Tensor:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: '{image_path}'")
    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"File at '{image_path}' is not a valid/readable image (possibly corrupted).") from exc

    transform = get_eval_transforms()
    return transform(image)


def predict_single(
    image_path: str,
    clinical_record: dict,
    checkpoint_path: str = os.path.join("saved_models", "best_model.pth"),
    saved_models_dir: str = "saved_models",
) -> dict:
    """Runs inference for one (image, clinical_record) pair.

    Returns a dict: {"label": "Pneumonia"/"Normal", "confidence": float in [0,1],
                      "probability_pneumonia": float in [0,1]}
    """
    device = get_device()

    model, tabular_feature_columns = load_model(checkpoint_path, device)
    scaler, encoder, saved_tabular_feature_columns = load_fitted_transformers(saved_models_dir)

    if saved_tabular_feature_columns != tabular_feature_columns:
        raise RuntimeError(
            "Mismatch between the checkpoint's expected tabular feature columns and the "
            "fitted transformers' feature columns. Re-run training to regenerate consistent artifacts."
        )

    image_tensor = load_and_transform_image(image_path).unsqueeze(0).to(device)

    tabular_vector = transform_single_record(clinical_record, scaler, encoder, tabular_feature_columns)
    tabular_tensor = torch.from_numpy(tabular_vector).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(image_tensor, tabular_tensor)
        probability_pneumonia = torch.sigmoid(logit).item()

    predicted_label = 1 if probability_pneumonia >= 0.5 else 0
    confidence = probability_pneumonia if predicted_label == 1 else (1 - probability_pneumonia)

    return {
        "label": CLASS_NAMES[predicted_label],
        "confidence": round(float(confidence), 4),
        "probability_pneumonia": round(float(probability_pneumonia), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Run a single multi-modal prediction.")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=os.path.join("saved_models", "best_model.pth"))
    parser.add_argument("--saved_models_dir", type=str, default="saved_models")

    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--gender", type=str, choices=["male", "female"], required=True)
    parser.add_argument("--fever", type=int, choices=[0, 1], required=True)
    parser.add_argument("--cough", type=int, choices=[0, 1], required=True)
    parser.add_argument("--chest_pain", type=int, choices=[0, 1], required=True)
    parser.add_argument("--oxygen_level", type=float, required=True)
    parser.add_argument("--smoking_history", type=str, choices=["never", "former", "current"], required=True)

    args = parser.parse_args()

    clinical_record = {
        "age": args.age,
        "gender": args.gender,
        "fever": args.fever,
        "cough": args.cough,
        "chest_pain": args.chest_pain,
        "oxygen_level": args.oxygen_level,
        "smoking_history": args.smoking_history,
    }

    result = predict_single(
        image_path=args.image_path,
        clinical_record=clinical_record,
        checkpoint_path=args.checkpoint,
        saved_models_dir=args.saved_models_dir,
    )

    print("=" * 40)
    print(f"Prediction : {result['label']}")
    print(f"Confidence : {result['confidence'] * 100:.2f}%")
    print(f"P(Pneumonia): {result['probability_pneumonia']:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()

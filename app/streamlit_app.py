"""
streamlit_app.py

Streamlit UI for the Multi-Modal Medical Diagnosis System.

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys

import numpy as np
import streamlit as st
import torch
from PIL import Image

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.append(SRC_DIR)

from data_preprocessing import load_fitted_transformers, transform_single_record  # noqa: E402
from dataset import get_eval_transforms  # noqa: E402
from model import MultiModalDiagnosisModel  # noqa: E402
from train import get_device  # noqa: E402

st.set_page_config(page_title="Multi-Modal Medical Diagnosis", page_icon="🩺", layout="centered")

CHECKPOINT_PATH = os.path.join("saved_models", "best_model.pth")
SAVED_MODELS_DIR = "saved_models"
CLASS_NAMES = {0: "Normal", 1: "Pneumonia"}


@st.cache_resource
def load_artifacts():
    """Loads the model + the SAME fitted scaler/encoder used at training
    time (never re-fit here), cached across Streamlit reruns.
    """
    device = get_device()
    if not os.path.exists(CHECKPOINT_PATH):
        return None

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model = MultiModalDiagnosisModel(tabular_input_dim=checkpoint["tabular_input_dim"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    scaler, encoder, tabular_feature_columns = load_fitted_transformers(SAVED_MODELS_DIR)

    return {
        "model": model,
        "scaler": scaler,
        "encoder": encoder,
        "tabular_feature_columns": tabular_feature_columns,
        "device": device,
    }


def run_gradcam(model, image_tensor, tabular_tensor, device):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        return None

    target_layer = model.get_gradcam_target_layer()

    class ImageOnlyWrapper(torch.nn.Module):
        def __init__(self, base_model, fixed_tabular):
            super().__init__()
            self.base_model = base_model
            self.fixed_tabular = fixed_tabular

        def forward(self, image):
            tabular = self.fixed_tabular.expand(image.size(0), -1)
            logit = self.base_model(image, tabular)
            return logit.unsqueeze(1)

    wrapped = ImageOnlyWrapper(model, tabular_tensor.unsqueeze(0)).to(device)
    cam = GradCAM(model=wrapped, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0))[0]

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return overlay


def main():
    st.title("🩺 Multi-Modal Medical Diagnosis System")
    st.caption("Fuses a chest X-ray image with patient clinical data to estimate pneumonia risk.")
    st.warning(
        "This is a research/educational demo trained on a small public dataset with SYNTHETIC "
        "clinical data. It is NOT a certified diagnostic tool and must never be used for real "
        "clinical decisions.",
        icon="⚠️",
    )

    artifacts = load_artifacts()
    if artifacts is None:
        st.error(
            f"No trained model found at '{CHECKPOINT_PATH}'. Run `python src/train.py` first "
            f"(or `python smoke_test.py` for a quick end-to-end dry run)."
        )
        return

    st.subheader("1. Upload a chest X-ray")
    uploaded_file = st.file_uploader("Image file (JPEG/PNG)", type=["jpg", "jpeg", "png"])

    st.subheader("2. Enter clinical data")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        gender = st.selectbox("Gender", ["male", "female"])
        oxygen_level = st.slider("Blood oxygen level (SpO2 %)", min_value=70, max_value=100, value=97)
        smoking_history = st.selectbox("Smoking history", ["never", "former", "current"])
    with col2:
        fever = st.checkbox("Fever")
        cough = st.checkbox("Cough")
        chest_pain = st.checkbox("Chest pain")

    predict_clicked = st.button("Predict", type="primary")

    if predict_clicked:
        if uploaded_file is None:
            st.error("Please upload a chest X-ray image first.")
            return

        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("Could not read the uploaded file as an image. Please upload a valid JPEG/PNG.")
            return

        st.image(image, caption="Uploaded X-ray", use_column_width=True)

        model = artifacts["model"]
        scaler = artifacts["scaler"]
        encoder = artifacts["encoder"]
        tabular_feature_columns = artifacts["tabular_feature_columns"]
        device = artifacts["device"]

        transform = get_eval_transforms()
        image_tensor = transform(image).to(device)

        clinical_record = {
            "age": float(age),
            "gender": gender,
            "fever": int(fever),
            "cough": int(cough),
            "chest_pain": int(chest_pain),
            "oxygen_level": float(oxygen_level),
            "smoking_history": smoking_history,
        }

        try:
            tabular_vector = transform_single_record(clinical_record, scaler, encoder, tabular_feature_columns)
        except ValueError as exc:
            st.error(f"Invalid clinical data: {exc}")
            return

        tabular_tensor = torch.from_numpy(tabular_vector).to(device)

        with torch.no_grad():
            logit = model(image_tensor.unsqueeze(0), tabular_tensor.unsqueeze(0))
            probability_pneumonia = torch.sigmoid(logit).item()

        predicted_label = 1 if probability_pneumonia >= 0.5 else 0
        confidence = probability_pneumonia if predicted_label == 1 else (1 - probability_pneumonia)

        st.subheader("3. Result")
        if predicted_label == 1:
            st.error(f"Prediction: **{CLASS_NAMES[predicted_label]}** (confidence {confidence * 100:.1f}%)")
        else:
            st.success(f"Prediction: **{CLASS_NAMES[predicted_label]}** (confidence {confidence * 100:.1f}%)")
        st.progress(min(max(probability_pneumonia, 0.0), 1.0))
        st.caption(f"P(Pneumonia) = {probability_pneumonia:.4f}")

        st.subheader("4. Explainability (Grad-CAM)")
        overlay = run_gradcam(model, image_tensor, tabular_tensor, device)
        if overlay is not None:
            st.image(overlay, caption="Grad-CAM heatmap overlay", use_column_width=True)
        else:
            st.info("pytorch-grad-cam is not installed; skipping heatmap.")


if __name__ == "__main__":
    main()

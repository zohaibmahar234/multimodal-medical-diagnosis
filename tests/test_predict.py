import os
import sys

import joblib
import numpy as np
import torch
from PIL import Image

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.append(SRC_DIR)

from data_preprocessing import fit_transformers  # noqa: E402
from model import MultiModalDiagnosisModel  # noqa: E402
from predict import predict_single  # noqa: E402

import pandas as pd  # noqa: E402


def _make_dummy_image(path):
    arr = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _build_dummy_artifacts(tmp_path):
    """Builds a tiny, untrained-but-valid checkpoint + fitted transformers
    so predict_single() can be exercised end-to-end without a real training
    run (mirrors what smoke_test.py does).
    """
    saved_models_dir = tmp_path / "saved_models"
    saved_models_dir.mkdir()

    fake_df = pd.DataFrame(
        {
            "age": [30, 65, 45, 70],
            "gender": ["male", "female", "male", "female"],
            "fever": [0, 1, 0, 1],
            "cough": [0, 1, 0, 1],
            "chest_pain": [0, 1, 0, 0],
            "oxygen_level": [98.0, 91.0, 96.5, 90.0],
            "smoking_history": ["never", "former", "current", "never"],
        }
    )
    scaler, encoder = fit_transformers(fake_df)
    encoded_cols = encoder.get_feature_names_out(["gender", "smoking_history"]).tolist()
    tabular_feature_columns = ["age", "oxygen_level", "fever", "cough", "chest_pain"] + encoded_cols

    joblib.dump(scaler, saved_models_dir / "scaler.joblib")
    joblib.dump(encoder, saved_models_dir / "encoder.joblib")
    joblib.dump(tabular_feature_columns, saved_models_dir / "tabular_feature_columns.joblib")

    model = MultiModalDiagnosisModel(tabular_input_dim=len(tabular_feature_columns), pretrained=False)
    checkpoint_path = saved_models_dir / "best_model.pth"
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

    image_path = tmp_path / "dummy_xray.png"
    _make_dummy_image(image_path)

    return str(checkpoint_path), str(saved_models_dir), str(image_path)


def test_predict_single_output_format(tmp_path):
    checkpoint_path, saved_models_dir, image_path = _build_dummy_artifacts(tmp_path)

    clinical_record = {
        "age": 55,
        "gender": "male",
        "fever": 1,
        "cough": 1,
        "chest_pain": 0,
        "oxygen_level": 93.0,
        "smoking_history": "former",
    }

    result = predict_single(
        image_path=image_path,
        clinical_record=clinical_record,
        checkpoint_path=checkpoint_path,
        saved_models_dir=saved_models_dir,
    )

    assert isinstance(result, dict)
    assert set(result.keys()) == {"label", "confidence", "probability_pneumonia"}
    assert result["label"] in {"Normal", "Pneumonia"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["probability_pneumonia"] <= 1.0


def test_predict_single_missing_image_raises(tmp_path):
    checkpoint_path, saved_models_dir, _ = _build_dummy_artifacts(tmp_path)
    clinical_record = {
        "age": 40,
        "gender": "female",
        "fever": 0,
        "cough": 0,
        "chest_pain": 0,
        "oxygen_level": 98.0,
        "smoking_history": "never",
    }

    try:
        predict_single(
            image_path=str(tmp_path / "does_not_exist.jpeg"),
            clinical_record=clinical_record,
            checkpoint_path=checkpoint_path,
            saved_models_dir=saved_models_dir,
        )
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass

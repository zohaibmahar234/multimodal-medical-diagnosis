import os
import sys

import torch

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.append(SRC_DIR)

from model import ImageBranch, MultiModalDiagnosisModel, TabularBranch  # noqa: E402


def test_image_branch_output_shape():
    branch = ImageBranch(output_dim=128, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = branch(dummy)
    assert out.shape == (2, 128)


def test_tabular_branch_output_shape():
    branch = TabularBranch(input_dim=9, output_dim=16)
    dummy = torch.randn(5, 9)
    out = branch(dummy)
    assert out.shape == (5, 16)


def test_fusion_model_output_shape():
    model = MultiModalDiagnosisModel(tabular_input_dim=9, image_feat_dim=64, tabular_feat_dim=16, pretrained=False)
    dummy_img = torch.randn(3, 3, 224, 224)
    dummy_tab = torch.randn(3, 9)
    out = model(dummy_img, dummy_tab)
    assert out.shape == (3,)


def test_fusion_model_gradcam_target_layer_exists():
    model = MultiModalDiagnosisModel(tabular_input_dim=9, pretrained=False)
    layer = model.get_gradcam_target_layer()
    assert layer is not None


def test_fusion_model_single_sample_batch():
    """BatchNorm1d in TabularBranch requires batch_size > 1 in train mode;
    this test confirms eval mode handles a single-sample batch correctly.
    """
    model = MultiModalDiagnosisModel(tabular_input_dim=9, pretrained=False)
    model.eval()
    dummy_img = torch.randn(1, 3, 224, 224)
    dummy_tab = torch.randn(1, 9)
    with torch.no_grad():
        out = model(dummy_img, dummy_tab)
    assert out.shape == (1,)

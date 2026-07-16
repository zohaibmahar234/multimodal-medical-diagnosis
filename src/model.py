"""
model.py

Multi-modal fusion architecture:
  - ImageBranch:  ResNet50 (ImageNet-pretrained) with early layers frozen
                  and `layer4` left trainable for fine-tuning.
  - TabularBranch: small dense network over the clinical features.
  - MultiModalDiagnosisModel: concatenates both branches' feature vectors,
                  passes them through a fusion head, and outputs a single
                  raw logit (use torch.sigmoid(logit) for probability, or
                  BCEWithLogitsLoss directly on the logit during training).
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class ImageBranch(nn.Module):
    """ResNet50 backbone used as the CNN branch (transfer learning).

    Base convolutional layers are frozen initially; only the last residual
    block (`layer4`) is left trainable, so fine-tuning adapts high-level
    features to chest X-rays without wrecking low-level ImageNet features
    learned from millions of natural images.
    """

    def __init__(self, output_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        for param in backbone.parameters():
            param.requires_grad = False
        for param in backbone.layer4.parameters():
            param.requires_grad = True

        in_features = backbone.fc.in_features  # 2048 for resnet50
        backbone.fc = nn.Identity()  # drop the 1000-class ImageNet head; keep pooled 2048-d features
        self.backbone = backbone

        self.projection = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224) -> (B, 2048) -> (B, output_dim)
        features = self.backbone(x)
        features = self.projection(features)
        return features

    def unfreeze_layer3(self):
        """Optional: unfreeze one more residual block for deeper fine-tuning
        once the model has stabilized on the smaller trainable head.
        """
        for param in self.backbone.layer3.parameters():
            param.requires_grad = True


class TabularBranch(nn.Module):
    """2-3 layer dense network processing scaled/encoded clinical features."""

    def __init__(self, input_dim: int, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim) -> (B, output_dim)
        return self.net(x)


class MultiModalDiagnosisModel(nn.Module):
    """Fuses the image and tabular branches for binary disease classification."""

    def __init__(
        self,
        tabular_input_dim: int,
        image_feat_dim: int = 256,
        tabular_feat_dim: int = 32,
        pretrained: bool = True,
    ):
        super().__init__()
        self.image_branch = ImageBranch(output_dim=image_feat_dim, pretrained=pretrained)
        self.tabular_branch = TabularBranch(input_dim=tabular_input_dim, output_dim=tabular_feat_dim)

        fused_dim = image_feat_dim + tabular_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),  # single raw logit; apply sigmoid externally for probability
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_branch(image)          # (B, image_feat_dim)
        tab_feat = self.tabular_branch(tabular)       # (B, tabular_feat_dim)
        fused = torch.cat([img_feat, tab_feat], dim=1)  # (B, image_feat_dim + tabular_feat_dim)
        logit = self.classifier(fused)                # (B, 1)
        return logit.squeeze(1)                        # (B,)

    def get_gradcam_target_layer(self):
        """Returns the convolutional layer pytorch-grad-cam should hook into.
        The last conv block of the ResNet50 backbone gives the highest-resolution,
        most semantically meaningful activations for Grad-CAM.
        """
        return self.image_branch.backbone.layer4[-1]


if __name__ == "__main__":
    # Quick manual shape check when running this file directly.
    model = MultiModalDiagnosisModel(tabular_input_dim=10, pretrained=False)
    dummy_img = torch.randn(4, 3, 224, 224)
    dummy_tab = torch.randn(4, 10)
    out = model(dummy_img, dummy_tab)
    print("Output shape:", out.shape)  # expected: torch.Size([4])

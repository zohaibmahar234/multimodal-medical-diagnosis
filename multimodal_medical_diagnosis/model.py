"""PyTorch modules for multimodal chest X-ray + clinical diagnosis."""

from __future__ import annotations

import torch
from torch import nn


class ImageEncoder(nn.Module):
    """Lightweight CNN feature extractor for chest X-ray images."""

    def __init__(self, in_channels: int = 1, feature_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(64, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.projection(x)


class ClinicalEncoder(nn.Module):
    """MLP feature extractor for tabular clinical values."""

    def __init__(self, input_dim: int, feature_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MultiModalPneumoniaModel(nn.Module):
    """Fuses image and clinical representations for pneumonia prediction."""

    def __init__(
        self,
        clinical_input_dim: int,
        num_classes: int = 2,
        image_in_channels: int = 1,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(in_channels=image_in_channels, feature_dim=128)
        self.clinical_encoder = ClinicalEncoder(input_dim=clinical_input_dim, feature_dim=32)
        self.classifier = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, image: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        clinical_features = self.clinical_encoder(clinical)
        fused = torch.cat([image_features, clinical_features], dim=1)
        return self.classifier(fused)

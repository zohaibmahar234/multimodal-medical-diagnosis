"""
dataset.py

Image transforms + custom PyTorch Dataset that correctly pairs each chest
X-ray image with its corresponding (already-scaled/encoded) clinical row.
"""

import os
import random

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IMAGE_SIZE = 224  # fixed across train/val/test/inference
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms() -> T.Compose:
    """Augmentation (rotation, flip, zoom) — TRAINING SPLIT ONLY."""
    return T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.RandomAffine(degrees=0, scale=(0.9, 1.1)),  # zoom in/out
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms() -> T.Compose:
    """Deterministic resize + normalize only — used for val/test/inference
    so that evaluation numbers are not inflated/deflated by random augmentation.
    """
    return T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class ChestXrayMultiModalDataset(Dataset):
    """Pairs a chest X-ray image with its corresponding clinical-data row.

    Expects a pandas DataFrame with at least:
        - 'image_path' : path to the image file (relative to `image_root`
                          if provided, otherwise absolute/as-is)
        - 'label'       : 0 (Normal) or 1 (Pneumonia)
        - the columns listed in `tabular_feature_columns` (already numeric,
          i.e. scaled numeric features + binary flags + one-hot columns)
    """

    def __init__(self, dataframe, tabular_feature_columns, transform=None, image_root=None):
        self.df = dataframe.reset_index(drop=True)
        self.tabular_feature_columns = list(tabular_feature_columns)
        self.transform = transform or get_eval_transforms()
        self.image_root = image_root

        required = {"image_path", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        missing_tabular = set(self.tabular_feature_columns) - set(self.df.columns)
        if missing_tabular:
            raise ValueError(f"DataFrame is missing tabular feature columns: {missing_tabular}")

        if len(self.df) == 0:
            raise ValueError("Cannot build a Dataset from an empty DataFrame.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        full_path = os.path.join(self.image_root, image_path) if self.image_root else image_path

        try:
            image = Image.open(full_path).convert("RGB")
        except (FileNotFoundError, OSError) as exc:
            raise RuntimeError(
                f"Could not load image at '{full_path}'. It may be missing or corrupted. "
                f"Run data_preprocessing.filter_missing_or_corrupted_images() to clean the CSV first."
            ) from exc

        image_tensor = self.transform(image)

        tabular_values = row[self.tabular_feature_columns].to_numpy(dtype=np.float32)
        tabular_tensor = torch.from_numpy(tabular_values)

        label_tensor = torch.tensor(float(row["label"]), dtype=torch.float32)

        return image_tensor, tabular_tensor, label_tensor

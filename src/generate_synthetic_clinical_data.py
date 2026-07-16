"""
generate_synthetic_clinical_data.py

Generates a synthetic clinical/tabular dataset that is correlated with the
disease label (Pneumonia vs Normal) of the chest X-ray images shipped in the
Kaggle "Chest X-Ray Images (Pneumonia)" dataset.

The Kaggle dataset only contains images + folder-name labels (NORMAL /
PNEUMONIA) — there is no real clinical data attached to each image. Since
this project fuses an image branch with a tabular branch, we need a
plausible clinical record for every image. This script builds that record
synthetically, but not randomly: pneumonia-positive cases are given a
higher probability of fever, cough, chest pain, lower blood-oxygen levels,
and a higher chance of a smoking history, mirroring real clinical patterns.
This gives the fusion model a genuine (if synthetic) signal to learn from
in the tabular branch, rather than pure noise.

Usage:
    python src/generate_synthetic_clinical_data.py \
        --image_root data/raw/chest_xray \
        --output data/processed/clinical_data.csv

If --image_root does not exist yet (real dataset not downloaded), the
script falls back to generating a small synthetic-only pool with dummy
image filenames, purely so the rest of the pipeline (smoke_test.py) can be
exercised end-to-end without the real dataset.
"""

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global reproducibility seed. Every script in this project seeds Python's
# `random`, `numpy`, and (where relevant) the framework's RNG with the same
# value so that dataset splits, synthetic data, weight init, and augmentation
# are all reproducible run-to-run.
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

VALID_IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png")


def discover_images_from_kaggle_layout(image_root: str) -> pd.DataFrame:
    """Walk the Kaggle folder layout and recombine every image into one flat pool.

    The Kaggle "Chest X-Ray Images (Pneumonia)" dataset ships as:
        chest_xray/{train,test,val}/{NORMAL,PNEUMONIA}/*.jpeg

    We deliberately IGNORE the pre-made train/test/val split here and pool
    every image together with its label. A single fresh stratified 70/15/15
    split is applied later (see data_preprocessing.py) on the *combined*
    pool so that:
      1. The clinical CSV rows line up 1:1 with images (no mismatched
         pairing from splitting images and clinical rows independently).
      2. Class balance is preserved identically across train/val/test,
         which matters a lot given how imbalanced this dataset is.
    """
    records = []
    if not os.path.isdir(image_root):
        return pd.DataFrame(columns=["image_path", "label"])

    for split_folder in ("train", "test", "val"):
        split_path = os.path.join(image_root, split_folder)
        if not os.path.isdir(split_path):
            continue
        for class_name, label in (("NORMAL", 0), ("PNEUMONIA", 1)):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue
            for fname in sorted(os.listdir(class_path)):
                if fname.lower().endswith(VALID_IMAGE_EXTENSIONS):
                    records.append(
                        {
                            "image_path": os.path.join(split_path, class_name, fname),
                            "label": label,
                        }
                    )

    return pd.DataFrame(records)


def make_dummy_pool(n_normal: int = 20, n_pneumonia: int = 20) -> pd.DataFrame:
    """Fallback pool of fake image paths, used only when the real dataset
    is not present (e.g. for smoke_test.py, which uses its own tiny fake
    JPEGs generated on the fly). This keeps the CSV schema identical
    regardless of whether real or fake images are used.
    """
    records = []
    for i in range(n_normal):
        records.append({"image_path": f"dummy_normal_{i}.jpeg", "label": 0})
    for i in range(n_pneumonia):
        records.append({"image_path": f"dummy_pneumonia_{i}.jpeg", "label": 1})
    return pd.DataFrame(records)


def generate_clinical_row(label: int, rng: np.random.Generator) -> dict:
    """Sample one clinical record, correlated with the disease label.

    Pneumonia-positive (label == 1) patients get:
      - higher probability of fever, cough, chest pain
      - lower blood-oxygen saturation (SpO2)
      - slightly higher chance of a smoking history
      - slightly older age distribution (comorbidity proxy)
    """
    if label == 1:
        age = int(np.clip(rng.normal(loc=52, scale=18), 0, 95))
        fever = rng.random() < 0.78
        cough = rng.random() < 0.85
        chest_pain = rng.random() < 0.55
        oxygen_level = float(np.clip(rng.normal(loc=91.5, scale=3.5), 70, 100))
        smoking_history = rng.choice(
            ["never", "former", "current"], p=[0.45, 0.30, 0.25]
        )
    else:
        age = int(np.clip(rng.normal(loc=38, scale=16), 0, 95))
        fever = rng.random() < 0.12
        cough = rng.random() < 0.20
        chest_pain = rng.random() < 0.08
        oxygen_level = float(np.clip(rng.normal(loc=97.5, scale=1.2), 85, 100))
        smoking_history = rng.choice(
            ["never", "former", "current"], p=[0.65, 0.22, 0.13]
        )

    gender = rng.choice(["male", "female"], p=[0.5, 0.5])

    return {
        "age": age,
        "gender": gender,
        "fever": int(fever),
        "cough": int(cough),
        "chest_pain": int(chest_pain),
        "oxygen_level": round(oxygen_level, 1),
        "smoking_history": smoking_history,
    }


def generate_synthetic_clinical_dataset(image_root: str, output_path: str) -> pd.DataFrame:
    pool = discover_images_from_kaggle_layout(image_root)

    if pool.empty:
        print(
            f"[generate_synthetic_clinical_data] No images found under "
            f"'{image_root}'. Falling back to a small dummy pool so the "
            f"pipeline can still be exercised end-to-end. Download the real "
            f"dataset and re-run this script for actual training (see README)."
        )
        pool = make_dummy_pool()

    rng = np.random.default_rng(SEED)

    clinical_rows = [generate_clinical_row(label, rng) for label in pool["label"]]
    clinical_df = pd.DataFrame(clinical_rows)

    full_df = pd.concat([pool.reset_index(drop=True), clinical_df], axis=1)

    # Sanity check: every image row must have a matching clinical row and
    # vice versa — this is guaranteed by construction (concat on aligned
    # indices), but we assert it explicitly to fail loudly if that ever
    # changes.
    assert len(full_df) == len(pool) == len(clinical_df), (
        "Mismatched row counts between image pool and generated clinical "
        "data — this should never happen."
    )
    assert full_df["image_path"].notna().all(), "Found empty image_path entries."
    assert full_df.drop(columns=["image_path"]).notna().all().all(), (
        "Found empty/NaN clinical values in generated data."
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    full_df.to_csv(output_path, index=False)

    print(f"[generate_synthetic_clinical_data] Wrote {len(full_df)} rows to '{output_path}'")
    print(full_df["label"].value_counts().rename({0: "NORMAL", 1: "PNEUMONIA"}))

    return full_df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic clinical data matched to chest X-ray images.")
    parser.add_argument(
        "--image_root",
        type=str,
        default=os.path.join("data", "raw", "chest_xray"),
        help="Path to the Kaggle chest_xray folder (containing train/test/val).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("data", "processed", "clinical_data.csv"),
        help="Where to write the generated clinical CSV.",
    )
    args = parser.parse_args()

    generate_synthetic_clinical_dataset(args.image_root, args.output)


if __name__ == "__main__":
    sys.exit(main() or 0)

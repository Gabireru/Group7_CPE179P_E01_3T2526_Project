#preprocess.py

from datasets import load_dataset
from PIL import Image
import numpy as np
import cv2
import os
from pathlib import Path
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_NAME = "yuntian-deng/im2latex-100k"
OUTPUT_DIR = Path(BASE_DIR) / "preprocessed_samples"
NUM_SAMPLES_TO_INSPECT = 8
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

CLEAN_IMAGE_BINARY_RATIO_THRESHOLD = 0.93
NEAR_BLACK_WHITE_MARGIN = 30

def is_already_clean(gray: np.ndarray) -> bool:
    """
    Returns True if the image looks already-clean and should be left alone
    (aside from format conversion), False if it looks like it needs actual
    cleanup (denoise + binarize).
    """
    near_black = gray <= NEAR_BLACK_WHITE_MARGIN
    near_white = gray >= (255 - NEAR_BLACK_WHITE_MARGIN)
    binary_fraction = np.count_nonzero(near_black | near_white) / gray.size
    return binary_fraction >= CLEAN_IMAGE_BINARY_RATIO_THRESHOLD


def preprocess_image(pil_image: Image.Image) -> Image.Image:
    gray = np.array(pil_image.convert("L"))

    if is_already_clean(gray):
        result = gray
    else:
        denoised = cv2.medianBlur(gray, 3)
        _, result = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # Back to PIL, RGB (TrOCRProcessor expects 3 channels)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(result_rgb)


def deskew_image(pil_image: Image.Image) -> Image.Image:
    gray = np.array(pil_image.convert("L"))
    inverted = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(inverted > 0))

    if coords.shape[0] < 10:
        return pil_image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5:
        return pil_image

    (h, w) = gray.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        np.array(pil_image),
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(rotated)


def full_preprocess(pil_image: Image.Image) -> Image.Image:
    image = pil_image
    if image.mode != "RGB":
        image = image.convert("RGB")

    gray = np.array(image.convert("L"))
    if is_already_clean(gray):
        return image

    image = deskew_image(image)
    image = preprocess_image(image)
    return image


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_im2latex():
    """
    Loads the im2latex-100k dataset splits.
    Returns a DatasetDict with 'train', 'validation', 'test' splits, each
    row containing an 'image' (PIL.Image) and 'formula' (str, LaTeX) field.
    """
    print(f"Loading dataset: {DATASET_NAME} ...")
    dataset = load_dataset(DATASET_NAME)
    print("Splits found:", list(dataset.keys()))
    for split in dataset.keys():
        print(f"  {split}: {len(dataset[split])} examples")
    return dataset


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def save_inspection_samples(dataset, split="train", n=NUM_SAMPLES_TO_INSPECT):
    OUTPUT_DIR.mkdir(exist_ok=True)
    split_data = dataset[split]

    indices = random.sample(range(len(split_data)), n)

    print(f"\nSaving {n} before/after samples from '{split}' to {OUTPUT_DIR}/ ...")
    for i, idx in enumerate(indices):
        example = split_data[idx]
        raw_image = example["image"]
        formula = example["formula"]

        processed_image = full_preprocess(raw_image)

        raw_path = OUTPUT_DIR / f"sample_{i}_raw.png"
        processed_path = OUTPUT_DIR / f"sample_{i}_processed.png"

        raw_image.convert("RGB").save(raw_path)
        processed_image.save(processed_path)

        formula_preview = formula if len(formula) <= 100 else formula[:100] + "..."
        print(f"  [{i}] formula: {formula_preview}")
        print(f"       raw -> {raw_path}")
        print(f"       processed -> {processed_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dataset = load_im2latex()
    save_inspection_samples(dataset, split="train", n=NUM_SAMPLES_TO_INSPECT)
    print("\nDone. Inspect the images in preprocessed_samples/ before moving on.")
#inspect_predictionsint8.py

import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import TrOCRProcessor

from preprocess import full_preprocess
from quantize import load_quantized_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = Path(BASE_DIR) / "trocr-im2latex-int8"
DATASET_NAME = "yuntian-deng/im2latex-100k"
NUM_SAMPLES = 10


def main():
    # Quantized int8 model runs on CPU only — torchao's weight-only int8
    # targets CPU inference, not CUDA. This is correct for Pi deployment too.
    device = "cpu"
    print(f"Using device: {device} (int8 quantized model runs on CPU)")

    print(f"\nLoading quantized int8 model + processor from: {MODEL_DIR}")
    processor = TrOCRProcessor.from_pretrained(str(MODEL_DIR))
    model = load_quantized_model(MODEL_DIR)
    model.eval()

    print(f"\nLoading dataset: {DATASET_NAME} (val split) ...")
    dataset = load_dataset(DATASET_NAME)
    val_data = dataset["val"]

    print(f"\nRunning inference on {NUM_SAMPLES} real val examples...\n")
    print("=" * 100)

    for i in range(NUM_SAMPLES):
        example = val_data[i]
        raw_image = example["image"]
        ground_truth = example["formula"]

        cleaned_image = full_preprocess(raw_image)

        pixel_values = processor(
            images=cleaned_image, return_tensors="pt"
        ).pixel_values  # no .to(device) — stays on CPU

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=256,  # matches MAX_LABEL_LENGTH from training
            )

        prediction = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        print(f"[{i}]")
        print(f"  GROUND TRUTH: {ground_truth}")
        print(f"  PREDICTION:   {prediction}")
        print("-" * 100)

    print("\nDone. Look for: are predictions close-but-imperfect (good sign) or")
    print("structurally garbled / repetitive / empty (sign of a deeper issue)?")


if __name__ == "__main__":
    main()
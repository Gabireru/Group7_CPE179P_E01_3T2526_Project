import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from preprocess import full_preprocess

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = Path(BASE_DIR) / "trocr-im2latex-final"
DATASET_NAME = "yuntian-deng/im2latex-100k"
NUM_SAMPLES = 10


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"\nLoading model + processor from: {MODEL_DIR}")
    processor = TrOCRProcessor.from_pretrained(str(MODEL_DIR))
    model = VisionEncoderDecoderModel.from_pretrained(str(MODEL_DIR)).to(device)
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

        # Same conditional preprocessing used everywhere else in the pipeline
        cleaned_image = full_preprocess(raw_image)

        pixel_values = processor(images=cleaned_image, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values)

        prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"[{i}]")
        print(f"  GROUND TRUTH: {ground_truth}")
        print(f"  PREDICTION:   {prediction}")
        print("-" * 100)

    print("\nDone. Look for: are predictions close-but-imperfect (good sign) or")
    print("structurally garbled / repetitive / empty (sign of a deeper issue)?")


if __name__ == "__main__":
    main()

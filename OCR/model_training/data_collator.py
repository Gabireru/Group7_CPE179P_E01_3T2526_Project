#data_collator.py

from transformers import TrOCRProcessor
import numpy as np
import torch

from preprocess import full_preprocess

MAX_LABEL_LENGTH = 256


def check_formula_lengths(dataset, processor: TrOCRProcessor, split="train", sample_size=2000):
    import random

    split_data = dataset[split]
    sample_indices = random.sample(
        range(len(split_data)), min(sample_size, len(split_data))
    )
    formulas = [split_data[i]["formula"] for i in sample_indices]

    lengths = np.array([
        len(processor.tokenizer(f, truncation=False).input_ids) for f in formulas
    ])

    over_limit = np.count_nonzero(lengths > MAX_LABEL_LENGTH)
    pct_over = 100 * over_limit / len(lengths)

    print(f"\nFormula token length check (sample of {len(lengths)} from '{split}'):")
    print(f"  min={lengths.min()}  max={lengths.max()}  mean={lengths.mean():.1f}  median={np.median(lengths):.0f}")
    print(f"  MAX_LABEL_LENGTH={MAX_LABEL_LENGTH}: {over_limit} examples ({pct_over:.2f}%) would be truncated")

    if pct_over > 1.0:
        print(
            f"  WARNING: more than 1% of sampled formulas exceed MAX_LABEL_LENGTH. "
            f"Consider raising it (current max in sample: {lengths.max()})."
        )
    else:
        print("  Looks fine — proceeding with current MAX_LABEL_LENGTH.")


class TrOCRCollator:
    def __init__(self, processor: TrOCRProcessor, max_label_length: int = MAX_LABEL_LENGTH):
        self.processor = processor
        self.max_label_length = max_label_length

    def __call__(self, batch_rows: list[dict]) -> dict:
        images = [row["image"] for row in batch_rows]
        formulas = [row["formula"] for row in batch_rows]

        cleaned_images = [full_preprocess(img) for img in images]

        pixel_values = self.processor(
            images=cleaned_images, return_tensors="pt"
        ).pixel_values

        label_ids = self.processor.tokenizer(
            formulas,
            padding="max_length",
            max_length=self.max_label_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids

        # Mask pad tokens with -100 so cross-entropy loss ignores them.
        pad_id = self.processor.tokenizer.pad_token_id
        label_ids[label_ids == pad_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": label_ids,
        }

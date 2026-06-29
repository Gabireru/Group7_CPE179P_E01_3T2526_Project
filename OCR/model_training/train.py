#train.py

import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

from preprocess import full_preprocess  # noqa: F401 (used indirectly via data_collator)
from data_collator import TrOCRCollator, check_formula_lengths, MAX_LABEL_LENGTH

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_NAME = "yuntian-deng/im2latex-100k"
MODEL_CHECKPOINT = "microsoft/trocr-base-printed"

OUTPUT_DIR = Path(BASE_DIR) / "trocr-im2latex-checkpoints"
LOGGING_DIR = Path(BASE_DIR) / "logs"

TRAIN_SUBSET_SIZE = 3000
EVAL_SUBSET_SIZE = 100

BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
EVAL_STEPS = 200
SAVE_STEPS = 200
LOGGING_STEPS = 50

USE_FP16 = True


# ---------------------------------------------------------------------------
# CER metric
# ---------------------------------------------------------------------------

def build_compute_metrics_fn(processor: TrOCRProcessor):

    cer_metric = evaluate.load("cer")

    def compute_metrics(eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids

        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    return compute_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Device check ---
    if torch.cuda.is_available():
        print(f"CUDA available — using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available, training will run on CPU and be extremely slow.")

    # --- Load dataset (just references the existing ~340MB cache, no new download if already pulled) ---
    print(f"\nLoading dataset: {DATASET_NAME} ...")
    dataset = load_dataset(DATASET_NAME)
    print("Splits found:", list(dataset.keys()))
    for split in dataset.keys():
        print(f"  {split}: {len(dataset[split])} examples")

    train_data = dataset["train"]
    eval_data = dataset["val"]

    if TRAIN_SUBSET_SIZE is not None:
        train_data = train_data.select(range(min(TRAIN_SUBSET_SIZE, len(train_data))))
        print(f"\nUsing TRAIN SUBSET: {len(train_data)} examples (set TRAIN_SUBSET_SIZE=None for full dataset)")
    if EVAL_SUBSET_SIZE is not None:
        eval_data = eval_data.select(range(min(EVAL_SUBSET_SIZE, len(eval_data))))
        print(f"Using EVAL SUBSET: {len(eval_data)} examples")

    # --- Load processor + model ---
    print(f"\nLoading processor + model: {MODEL_CHECKPOINT} ...")
    processor = TrOCRProcessor.from_pretrained(MODEL_CHECKPOINT)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_CHECKPOINT)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = processor.tokenizer.eos_token_id
    model.generation_config.max_length = MAX_LABEL_LENGTH
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 1.0
    model.generation_config.num_beams = 4 

    check_formula_lengths(dataset, processor, split="train")

    # --- Data collator: converts (image, formula) -> tensors on-the-fly, per batch ---
    collator = TrOCRCollator(processor, max_label_length=MAX_LABEL_LENGTH)

    # --- Training arguments ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        logging_dir=str(LOGGING_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        fp16=USE_FP16 and torch.cuda.is_available(),
        predict_with_generate=True,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        # --- Disk safety: never let checkpoints accumulate unbounded ---
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="none",

        dataloader_num_workers=0,

        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collator,
        compute_metrics=build_compute_metrics_fn(processor),
    )

    print("\nStarting training...")
    print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Checkpoints kept: {training_args.save_total_limit} (older ones auto-deleted)")
    trainer.train()

    print("\nTraining complete. Saving final model...")
    final_model_dir = Path(BASE_DIR) / "trocr-im2latex-final"
    trainer.save_model(str(final_model_dir))
    processor.save_pretrained(str(final_model_dir))
    print(f"Final model saved to: {final_model_dir}")

    print("\nRunning final evaluation on val subset...")
    metrics = trainer.evaluate()
    print("Final metrics:", metrics)


if __name__ == "__main__":
    main()

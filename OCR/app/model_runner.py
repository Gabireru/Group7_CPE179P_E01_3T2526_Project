#model_runner.py

import os
import sys
from pathlib import Path

# Add model_training/ to path so preprocess.py can be found
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_TRAINING_DIR = BASE_DIR.parent / "model_training"
sys.path.append(str(MODEL_TRAINING_DIR))

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from preprocess import full_preprocess  # now findable

INT8_DIR = MODEL_TRAINING_DIR / "trocr-im2latex-int8"
FP32_DIR = MODEL_TRAINING_DIR / "trocr-im2latex-final"


class ModelRunner:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.ready = False
        self.error: str | None = None

    def load(self):
        """Blocking load — call via page.run_thread() so it runs off the UI thread."""
        try:
            if INT8_DIR.exists() and (INT8_DIR / "model_int8.pt").exists():
                self._load_int8()
            elif FP32_DIR.exists():
                self._load_fp32()
            else:
                raise FileNotFoundError(
                    "No model directory found.\n"
                    f"Expected one of:\n  {INT8_DIR}\n  {FP32_DIR}"
                )
            self.ready = True
        except Exception as exc:
            self.error = str(exc)

    def _load_fp32(self):
        print(f"[ModelRunner] Loading fp32 model from {FP32_DIR}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = TrOCRProcessor.from_pretrained(str(FP32_DIR))
        self.model = (
            VisionEncoderDecoderModel.from_pretrained(str(FP32_DIR))
            .to(self.device)
        )
        self.model.eval()
        print(f"[ModelRunner] fp32 model ready on {self.device}")

    def _load_int8(self):
        print(f"[ModelRunner] Loading int8 model from {INT8_DIR}")
        from quantize import load_quantized_model
        self.device = "cpu"
        self.processor = TrOCRProcessor.from_pretrained(str(INT8_DIR))
        self.model = load_quantized_model(INT8_DIR)
        self.model.eval()
        print("[ModelRunner] int8 model ready on cpu")

    def predict(self, pil_image: Image.Image) -> str:
        """Run inference. Called from a worker thread, never from the UI thread."""
        if not self.ready:
            raise RuntimeError(self.error or "Model is not loaded yet.")

        cleaned = full_preprocess(pil_image)
        pixel_values = self.processor(
            images=cleaned, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_new_tokens=256,
            )

        return self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

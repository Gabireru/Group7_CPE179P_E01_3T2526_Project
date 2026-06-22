import os
from pathlib import Path

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = Path(BASE_DIR) / "trocr-im2latex-final"
QUANTIZED_DIR = Path(BASE_DIR) / "trocr-im2latex-int8"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Quantization runs on CPU — torchao's weight-only int8 targets CPU
    # inference (which is what the Pi will use), not CUDA.
    print("Device: CPU (torchao weight-only int8 targets CPU inference)")

    # --- Check torchao is installed ---
    try:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
    except ImportError:
        raise ImportError(
            "torchao is not installed.\n"
            "Install it with:  uv add torchao"
        )

    # --- Load the fine-tuned fp32 model ---
    print(f"\nLoading fine-tuned model from: {MODEL_DIR}")
    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}\n"
            "Make sure train.py has completed successfully first."
        )

    processor = TrOCRProcessor.from_pretrained(str(MODEL_DIR))
    model = VisionEncoderDecoderModel.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.float32,
    )
    model.eval()
    model = model.cpu()

    original_size_mb = get_file_size_mb(MODEL_DIR / "model.safetensors")
    print(f"Original model size on disk: {original_size_mb:.1f} MB")

    # --- Quantize ---
    # Int8WeightOnlyConfig: converts all nn.Linear weight tensors to int8.
    # Activations stay in fp32 at runtime — no overhead, just smaller weights.
    print("\nApplying torchao int8 weight-only quantization...")
    quantize_(model, Int8WeightOnlyConfig(version=2))
    print("Quantization complete.")

    # --- Sanity check before saving ---
    print("\nRunning sanity check (one generate() call on a dummy input)...")
    dummy_pixel_values = torch.zeros(1, 3, 384, 384)
    with torch.no_grad():
        try:
            output = model.generate(dummy_pixel_values)
            print(f"Forward pass OK. Output token ids shape: {output.shape}")
        except Exception as e:
            print(f"WARNING: Forward pass failed: {e}")
            print("Weights will still be saved — check if the error is recoverable.")

    # --- Save ---
    QUANTIZED_DIR.mkdir(exist_ok=True)

    weights_path = QUANTIZED_DIR / "model_int8.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"\nQuantized weights saved: {weights_path}")

    # Save processor + configs so the folder is self-contained
    processor.save_pretrained(str(QUANTIZED_DIR))
    model.config.to_json_file(str(QUANTIZED_DIR / "config.json"))
    model.generation_config.save_pretrained(str(QUANTIZED_DIR))

    # --- Size comparison ---
    quantized_size_mb = get_file_size_mb(weights_path)
    print(f"\n--- Size comparison ---")
    print(f"  Original (fp32 safetensors): {original_size_mb:.1f} MB")
    print(f"  Quantized (int8 .pt):        {quantized_size_mb:.1f} MB")
    print(f"  Reduction:                   {(1 - quantized_size_mb / original_size_mb) * 100:.1f}%")
    print(f"\nQuantized model saved to: {QUANTIZED_DIR}")
    print("Use load_quantized_model() from this file to load it for inference.")


# ---------------------------------------------------------------------------
# Load helper
# ---------------------------------------------------------------------------

def load_quantized_model(quantized_dir: str | Path):
    """
    Load the int8 quantized model for inference.

    Usage in other scripts:
        from quantize import load_quantized_model
        from transformers import TrOCRProcessor

        model = load_quantized_model("path/to/trocr-im2latex-int8")
        processor = TrOCRProcessor.from_pretrained("path/to/trocr-im2latex-int8")
    """
    from torchao.quantization import quantize_, Int8WeightOnlyConfig

    quantized_dir = Path(quantized_dir)

    # Step 1: reconstruct the architecture in fp32
    model = VisionEncoderDecoderModel.from_pretrained(
        str(quantized_dir),
        torch_dtype=torch.float32,
    )
    model.eval()
    model = model.cpu()

    # Step 2: re-apply the same quantization structure so layer types match
    quantize_(model, Int8WeightOnlyConfig(version=2))

    # Step 3: load the int8 weights into those layers
    state_dict = torch.load(
        quantized_dir / "model_int8.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == "__main__":
    main()

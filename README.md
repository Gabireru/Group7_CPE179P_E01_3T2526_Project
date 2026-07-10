# CPE179P_E01_3T2526 — Group 7: Mathematical Equation OCR Pipeline

A fine-tuned OCR pipeline that reads images of mathematical equations and outputs LaTeX strings, targeting multiple linear regression equation recognition. Built on Microsoft's TrOCR model, fine-tuned on the im2latex-100k dataset, quantized to int8, and deployed with a Flet-based desktop UI on Raspberry Pi.

**Course:** CPE179P — Section E01, 3T2526  
**Group:** 7

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Setup](#setup)
5. [Running the Pipeline](#running-the-pipeline)
6. [Running the UI](#running-the-ui)
7. [Script Reference](#script-reference)
8. [Notes for the Grader](#notes-for-the-grader)

---

## Project Overview

This pipeline takes an image containing a mathematical equation and outputs the corresponding LaTeX string. For example:

**Input:** An image of `ŷ = β₀ + β₁x₁ + β₂x₂`  
**Output:** `\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2`

### How It Works

1. **Preprocessing** — Images are checked for quality. Already-clean rendered equations (e.g. from textbooks or PDFs) pass through untouched. Photographed or scanned images with noise and skew get deskewed, denoised, and binarized automatically.
2. **Training** — Microsoft's `trocr-base-printed` model is fine-tuned on the `im2latex-100k` dataset (~55,000 LaTeX equation image-formula pairs from arXiv).
3. **Quantization** — The fine-tuned model (~1.3 GB) is compressed to int8 (~522 MB) using `torchao` for deployment on the Raspberry Pi.
4. **UI** — A Flet-based desktop app lets the user pick an equation image, run the model, see the predicted LaTeX, and copy it to clipboard.

---

## Project Structure

```
Group7_CPE179P_E01_3T2526/
├── OCR/
│   ├── model_training/
│   │   ├── preprocess.py               # Conditional image cleanup
│   │   ├── data_collator.py            # On-the-fly tensor conversion (not run directly)
│   │   ├── train.py                    # Fine-tuning script
│   │   ├── quantize.py                 # int8 quantization + load helper
│   │   ├── inspect_predictionsint8.py  # Prediction inspection tool
│   │   ├── verification.py             # CUDA/GPU verification
│   │   ├── trocr-im2latex-int8/        # Quantized model (tracked via Git LFS)
│   │   │   ├── model_int8.pt           # int8 weights (~522 MB, via LFS)
│   │   │   ├── config.json
│   │   │   ├── generation_config.json
│   │   │   ├── processor_config.json
│   │   │   ├── tokenizer.json
│   │   │   └── tokenizer_config.json
│   │   └── trocr-im2latex-final/       # Full fp32 model (NOT in git — too large)
│   │       └── model.safetensors       # ~1.3 GB — generated locally by train.py
│   └── UI/
│       ├── mainui.py                   # App entry point
│       ├── loginui.py                  # Login/cover screen
│       ├── mainpgui.py                 # Main menu screen
│       ├── uploadgui.py                # File upload + inference screen
│       ├── model_runner.py             # Model loading + inference wrapper
│       └── UI images/                  # Background images for the UI
├── .gitattributes                      # Git LFS tracking rules
├── .gitignore
└── README.md
```

> **Note:** `trocr-im2latex-final/model.safetensors` is excluded from the repository (too large even for LFS). The quantized int8 model in `trocr-im2latex-int8/` is self-contained and sufficient for inference — no fp32 model needed at runtime. To regenerate the fp32 model, run `train.py` (expect ~1.5 hours on an RTX 5070 Ti for the 3k subset).

---

## Requirements

### Hardware
- **Training:** A GPU with CUDA support strongly recommended (tested on NVIDIA RTX 5070 Ti, CUDA 13.2). CPU-only works but is extremely slow.
- **Inference / UI:** Any machine including Raspberry Pi 5 (CPU-only, no GPU needed).

### Software
- Python 3.12
- [uv](https://docs.astral.sh/uv/) — Python package and environment manager
- Git with [Git LFS](https://git-lfs.github.com/) — required to pull the quantized model file
- Flet 0.85.3 — UI framework

---

## Setup

### Step 1 — Install Git LFS and clone

Git LFS is required to download `model_int8.pt`. Without it, the file appears as a small text pointer instead of the real weights.

```bash
# Install Git LFS (once per machine)
git lfs install

# Clone — LFS files download automatically
git clone https://github.com/Gabireru/Group7_CPE179P_E01_3T2526_Project.git
cd Group7_CPE179P_E01_3T2526_Project

# If already cloned without LFS, pull manually
git lfs pull
```

### Step 2 — Install uv

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux / Raspberry Pi:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 3 — Install dependencies

```bash
uv python pin 3.12
```

**For training (Windows with CUDA 13.2 / RTX 5070 Ti):**
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
uv pip install transformers datasets evaluate jiwer pillow opencv-python-headless accelerate numpy torchao flet==0.85.3
```

**For inference/UI only (Raspberry Pi or CPU-only machine):**
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install  transformers datasets evaluate jiwer pillow opencv-python-headless accelerate numpy torchao flet==0.85.3
```

> Pinning `flet==0.85.3` is important — the UI was written and tested against this exact version. Newer Flet versions have breaking API changes.

### Step 4 — Verify GPU (training machines only)

```bash
uv run python OCR/model_training/verification.py
```

---

## Running the Pipeline

> Scripts use `os.path.abspath(__file__)` to anchor paths to their own location, so they work regardless of where you call them from.

### 1. Inspect the dataset (optional)

Downloads im2latex-100k (~340 MB, cached after first run) and saves 8 before/after preprocessing samples to `preprocessed_samples/`.

```bash
uv run python OCR/model_training/preprocess.py
```

### 2. Train the model

Open `train.py` and check the config at the top before running:

```python
TRAIN_SUBSET_SIZE = 3000   # 3000 for a test run (~1.5 hrs on RTX 5070 Ti)
                           # None for full 55,033 examples (many hours)
EVAL_SUBSET_SIZE  = 100
NUM_EPOCHS        = 3
BATCH_SIZE        = 8      # Lower if you hit out-of-memory errors
```

```bash
uv run python OCR/model_training/train.py
```

First run downloads `microsoft/trocr-base-printed` (~1.3 GB). Output: `trocr-im2latex-final/` and up to 2 checkpoints in `trocr-im2latex-checkpoints/`.

> **Storage note:** The HF dataset cache (~340 MB) lives in `~/.cache/huggingface/hub/`. Do NOT delete this between runs. Do NOT run `dataset.map()` with large tensor outputs — `data_collator.py` converts tensors on-the-fly to avoid the 100GB+ disk usage this would cause.

### 3. Quantize the model

Compresses the fp32 model (~1.3 GB) to int8 (~522 MB) for Pi deployment.

```bash
uv run python OCR/model_training/quantize.py
```

Output: `trocr-im2latex-int8/` — self-contained, no internet or fp32 model needed at runtime.

### 4. Inspect predictions (optional)

Runs the quantized model on 10 real val examples and prints predictions vs ground truth.

```bash
uv run python OCR/model_training/inspect_predictionsint8.py
```

---

## Running the UI

The UI is a Flet desktop app. Run it from the `OCR/UI/` directory:

```bash
cd OCR/UI
uv run python mainui.py
```

### UI Flow

```
Cover screen → [START] → Main menu → [SCAN] → Upload screen
                                   → [COPY]    Pick image → [SCAN] → LaTeX output → [COPY LaTeX]
                                                                                   → [BACK]
```

**Upload screen walkthrough:**
1. Press **PICK IMAGE** — opens a file picker (PNG, JPG, BMP, TIFF, WebP supported)
2. The preview shows the preprocessed image (what the model actually sees)
3. Press **SCAN** — model runs in background, status updates while waiting
4. Predicted LaTeX appears in the result area (selectable text)
5. Press **COPY LaTeX** to copy to clipboard, or **BACK** to return to the main menu
6. The **COPY** button on the main menu also copies the last scanned result

### Notes for Pi deployment
- The model loads automatically in the background when the app starts
- Inference runs on CPU — expect 10–30 seconds per image on Pi 5
- The app requires `trocr-im2latex-int8/` to be present alongside `model_training/`
- No internet connection needed at runtime

---

## Script Reference

| Script | Purpose | Requires |
|--------|---------|----------|
| `verification.py` | Check CUDA/GPU availability | — |
| `preprocess.py` | Download dataset, inspect preprocessing | Internet (first run) |
| `data_collator.py` | Tensor conversion module (imported by train.py, not run directly) | — |
| `train.py` | Fine-tune TrOCR on im2latex-100k | GPU recommended |
| `quantize.py` | Compress model to int8 | `train.py` completed |
| `inspect_predictionsint8.py` | Inspect model predictions vs ground truth | `quantize.py` completed |
| `mainui.py` | Launch the UI app | `quantize.py` completed |

---

## Additional Notes

- **Dataset:** `yuntian-deng/im2latex-100k` — 55,033 training / 6,072 val / 6,810 test LaTeX equation image-formula pairs from arXiv physics papers.
- **Base model:** `microsoft/trocr-base-printed` (334M parameters, BEiT encoder + RoBERTa decoder).
- **Training:** Fine-tuned on a 3,000-example subset for 3 epochs due to hardware and time constraints (1h 27min on RTX 5070 Ti Laptop). Final validation CER: **0.304** (Character Error Rate — lower is better).
- **Quantization:** int8 weight-only quantization via `torchao`, reducing model size by ~60% with minor accuracy tradeoff, targeting Raspberry Pi 5 CPU inference. The quantized folder is fully self-contained — no fp32 weights or internet needed at inference time.
- **Preprocessing:** Conditional image cleanup — clean rendered equations pass through untouched (preserving anti-aliased edges), while noisy/photographed images receive deskewing, median blur, and Otsu binarization. The UI shows the preprocessed image in the preview so the user can see what the model actually receives.
- **Storage safety:** Tensor conversion happens on-the-fly per batch via `TrOCRCollator` — no pre-computed tensor dataset is ever saved to disk (this would cost 100+ GB). Checkpoints are capped at 2 copies via `save_total_limit=2`.
- **UI:** Built with Flet 0.85.3, cross-compatible with Windows and Raspberry Pi. No camera used — file upload only. LaTeX rendering via matplotlib was omitted for Pi compatibility (matplotlib's math renderer is CPU-heavy and slow on Pi).
- The large model files are managed via Git LFS (`model_int8.pt`, ~522 MB). The fp32 `model.safetensors` (~1.3 GB) is excluded from the repo entirely and must be generated locally via `train.py`.

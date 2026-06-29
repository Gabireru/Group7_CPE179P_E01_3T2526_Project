# CPE179P_E01_3T2526_Project

# CPE179P_E01_3T2526 — Group 7: Mathematical Equation OCR Pipeline

A fine-tuned OCR pipeline that reads images of mathematical equations and outputs LaTeX strings. Built on Microsoft's TrOCR model, fine-tuned on the im2latex-100k dataset, and quantized to int8 for deployment on resource-constrained devices such as the Raspberry Pi.

**Course:** CPE179P — Section E01, 3T2526  
**Group:** 7  

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Setup](#setup)
5. [Running the Pipeline](#running-the-pipeline)
6. [Script Reference](#script-reference)
7. [Notes for the Grader](#notes-for-the-grader)

---

## Project Overview

This pipeline takes an image containing a mathematical equation and outputs the corresponding LaTeX string, to be used later as a model for multiple linear reegression equations. For example:

**Input:** An image of `ŷ = β₀ + β₁x₁ + β₂x₂`  
**Output:** `\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2`

### How It Works

1. **Preprocessing** — Images are checked for quality. Already-clean rendered equations (e.g. from textbooks or PDFs) pass through untouched. Photographed or scanned images with noise and skew get cleaned up automatically.
2. **Training** — Microsoft's `trocr-base-printed` model is fine-tuned on the `im2latex-100k` dataset (~55,000 LaTeX equation image-formula pairs from arXiv).
3. **Quantization** — The fine-tuned model (~1.3 GB) is compressed to int8 (~522 MB) using `torchao` for deployment on the Raspberry Pi.
4. **Inference** — The quantized model reads an equation image and outputs a LaTeX string to the terminal.

---

## Project Structure

```
Group7_CPE179P_E01_3T2526/
├── OCR/
│   └── model_training/
│       ├── preprocess.py               # Image loading and cleanup
│       ├── data_collator.py            # On-the-fly tensor conversion
│       ├── train.py                    # Fine-tuning script
│       ├── quantize.py                 # int8 quantization + load helper
│       ├── inspect_predictionsint8.py  # Prediction inspection tool
│       ├── verification.py             # CUDA/GPU verification
│       ├── trocr-im2latex-int8/        # Quantized model (tracked via Git LFS)
│       │   ├── model_int8.pt           # int8 weights (~522 MB, via LFS)
│       │   ├── config.json
│       │   ├── generation_config.json
│       │   ├── processor_config.json
│       │   ├── tokenizer.json
│       │   └── tokenizer_config.json
│       └── trocr-im2latex-final/       # Full fp32 model (NOT in git, too large)
│           ├── model.safetensors       # ~1.3 GB — must be generated locally
│           └── ...
├── .gitattributes                      # Git LFS tracking rules
├── .gitignore
└── README.md
```

> **Note:** `trocr-im2latex-final/model.safetensors` is excluded from the repository (too large even for LFS). To use the full fp32 model, run `train.py` locally to generate it although it would tkae quite a while for it to do so depending on your hardware (A computer with an RTX 5070Ti training the model even though its just a 3k subset of the 55k took 1 hour and 27 minutes.). The quantized int8 model in `trocr-im2latex-int8/` is sufficient for inference and is included via Git LFS.

---

## Requirements

### Hardware
- A GPU with CUDA support is strongly recommended for training (tested on NVIDIA RTX 5070 Ti with CUDA 13.2)
- CPU-only is supported but training will be extremely slow
- For inference only (no training): any machine including Raspberry Pi 5

### Software
- Python 3.12
- [uv](https://docs.astral.sh/uv/) — Python package and environment manager
- Git with [Git LFS](https://git-lfs.github.com/) — required to pull the quantized model file

---

## Setup

### Step 1 — Install Git LFS and pull the model

Git LFS is required to download `model_int8.pt` when cloning. If you skip this, the model file will appear as a small text pointer instead of the actual weights.

```bash
# Install Git LFS (do this once per machine)
git lfs install

# Clone the repo normally — LFS files download automatically
git clone https://github.com/Gabireru/Group7_CPE179P_E01_3T2526_Project.git
cd Group7_CPE179P_E01_3T2526_Project

# If you already cloned without LFS, pull the LFS files manually
git lfs pull
```

### Step 2 — Install uv

`uv` is a fast Python package manager. Install it with:

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 3 — Create the virtual environment and install dependencies

Navigate to the project root and run:

```bash
# Pin Python version (do this once)
uv python pin 3.12

# Install all dependencies into a virtual environment
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu132
uv add transformers datasets evaluate jiwer pillow opencv-python-headless accelerate numpy torchao
```

> **Note on the CUDA index URL:** `cu132` targets CUDA 13.2 with RTX 5070 Ti (Blackwell, sm_120) support. If you are on a different GPU or CUDA version, adjust accordingly. For CPU-only, omit the `--index-url` flag entirely.

### Step 4 — Verify your GPU (optional but recommended)

```bash
uv run python verification.py
```

Expected output includes `True` for `torch.cuda.is_available()` and your GPU name. If CUDA is not available, training will still run on CPU but will be very slow.

---

## Running the Pipeline

> All scripts must be run from the `OCR/model_training/` directory, or with the full path as shown below. Scripts use `os.path.abspath(__file__)` to anchor all file paths to their own location, so they work correctly regardless of where you call them from.

### 1. Inspect the dataset (optional)

Downloads the im2latex-100k dataset (~340 MB, cached after first run) and saves 8 before/after preprocessing sample images to `preprocessed_samples/` for visual inspection.

```bash
uv run python OCR/model_training/preprocess.py
```

**Expected output:**
```
Loading dataset: yuntian-deng/im2latex-100k ...
Splits found: ['train', 'test', 'val']
  train: 55033 examples
  test: 6810 examples
  val: 6072 examples
Saving 8 before/after samples from 'train' to preprocessed_samples/ ...
Done.
```

---

### 2. Train the model

Fine-tunes `microsoft/trocr-base-printed` on the im2latex-100k dataset.

**Before running**, open `train.py` and check the config section at the top:

```python
TRAIN_SUBSET_SIZE = 3000   # Use 3000 for a quick test run (~1.5 hrs on GPU)
                           # Set to None to train on all 55,033 examples (many hours)
EVAL_SUBSET_SIZE  = 100    # Validation subset size during training
NUM_EPOCHS        = 3      # Number of full passes through the training data
BATCH_SIZE        = 8      # Lower this if you get out-of-memory errors
```

Then run:

```bash
uv run python OCR/model_training/train.py
```

**Expected output (first run downloads the base model ~1.3 GB):**
```
CUDA available — using GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
Loading dataset: yuntian-deng/im2latex-100k ...
Loading processor + model: microsoft/trocr-base-printed ...
Formula token length check (sample of 2000 from 'train'):
  MAX_LABEL_LENGTH=256: 0 examples (0.00%) would be truncated
Starting training...
  Effective batch size: 16
  Epochs: 3
  Checkpoints kept: 2 (older ones auto-deleted)
{'loss': '9.45', ...}
...
Final metrics: {'eval_cer': 0.304, ...}
```

**Output files:**
- `trocr-im2latex-final/` — the trained model in full fp32 precision
- `trocr-im2latex-checkpoints/` — up to 2 intermediate checkpoints (auto-managed)

> **Storage note:** Training itself writes nothing large to disk during the run. The final model (`model.safetensors`) is ~1.3 GB. Checkpoints are capped at 2 copies. The Hugging Face dataset cache (~340 MB) lives in `~/.cache/huggingface/hub/` — do NOT delete this between runs.

---

### 3. Quantize the model

Compresses the trained model from fp32 (~1.3 GB) to int8 (~522 MB) using `torchao`. Required before running inference on the Raspberry Pi.

```bash
uv run python OCR/model_training/quantize.py
```

**Expected output:**
```
Device: CPU (torchao weight-only int8 targets CPU inference)
Loading fine-tuned model from: .../trocr-im2latex-final
Original model size on disk: 1311.4 MB
Applying torchao int8 weight-only quantization...
Quantization complete.
Forward pass OK. Output token ids shape: torch.Size([1, 20])
Quantized weights saved: .../trocr-im2latex-int8/model_int8.pt
--- Size comparison ---
  Original (fp32 safetensors): 1311.4 MB
  Quantized (int8 .pt):         521.9 MB
  Reduction:                    60.2%
```

**Output files:**
- `trocr-im2latex-int8/` — self-contained quantized model folder, ready for Pi deployment

> **Prerequisite:** `trocr-im2latex-final/` must exist (i.e. `train.py` must have completed successfully).

---

### 4. Inspect predictions

Runs the quantized model on 10 real validation examples and prints predicted LaTeX next to ground truth.

```bash
uv run python OCR/model_training/inspect_predictionsint8.py
```

**Expected output:**
```
Using device: cpu (int8 quantized model runs on CPU)
Loading quantized int8 model + processor from: .../trocr-im2latex-int8
Running inference on 10 real val examples...
[0]
  GROUND TRUTH: E ( v ) = \frac { d } { d t } E ( q ) ...
  PREDICTION:   E ( v ) = \frac { d } { d t } E ( q ) ...
```

---

## Script Reference

| Script | Purpose | Requires |
|--------|---------|----------|
| `verification.py` | Check CUDA/GPU availability | — |
| `preprocess.py` | Download dataset, inspect preprocessing | Internet (first run) |
| `data_collator.py` | Tensor conversion module (not run directly!) | — |
| `train.py` | Fine-tune TrOCR on im2latex-100k | `preprocess.py` run first |
| `quantize.py` | Compress model to int8 | `train.py` completed |
| `inspect_predictionsint8.py` | Inspect model predictions | `quantize.py` completed |

---

## Notes for the Grader

- **Dataset:** `yuntian-deng/im2latex-100k` — 55,033 training / 6,072 val / 6,810 test LaTeX equation image-formula pairs from arXiv physics papers.
- **Base model:** `microsoft/trocr-base-printed` (334M parameters, BEiT encoder + RoBERTa decoder).
- **Training:** Fine-tuned on a 3,000-example subset for 3 epochs due to hardware and time constraints. Final validation CER: **0.304** (Character Error Rate — lower is better; 0.0 = perfect).
- **Quantization:** int8 weight-only quantization via `torchao`, reducing model size by ~60% with minor accuracy tradeoff, targeting Raspberry Pi CPU deployment.
- **Preprocessing:** Conditional image cleanup — clean rendered equations pass through untouched (preserving anti-aliased edges), while noisy/photographed images receive deskewing, median blur, and Otsu binarization.
- **No data is pre-saved to disk** during training — tensor conversion happens on-the-fly per batch via `TrOCRCollator`, keeping storage impact minimal beyond the original dataset cache.
- The UI layer is planned for a future phase and is not included in this submission.
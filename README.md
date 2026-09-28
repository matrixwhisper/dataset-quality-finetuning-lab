# Dataset Quality & Fine-Tuning Lab

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Transformers">
  <img src="https://img.shields.io/badge/PEFT-QLoRA-8A2BE2?style=for-the-badge" alt="PEFT">
  <img src="https://img.shields.io/badge/4--bit-NF4-0F766E?style=for-the-badge" alt="4-bit NF4">
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Pytest-Testing-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="Pytest">
</p>

<p align="center">
  <strong>Human-reviewed dataset quality analysis, quality-risk detection, active review, and QLoRA fine-tuning for LLM response evaluation.</strong>
</p>

---

## Overview

**Dataset Quality & Fine-Tuning Lab** is a Python-based machine learning research project for analyzing and improving the quality of instruction-response datasets used for LLM fine-tuning.

The project combines deterministic dataset quality checks, human-review workflows, a learned quality-risk detector, active-review queue generation, and **4-bit NF4 QLoRA fine-tuning**.

The fine-tuning objective is to teach a language model to predict five human quality ratings:

- Helpfulness
- Correctness
- Coherence
- Complexity
- Verbosity

The evaluation pipeline compares model predictions against human-provided ratings and reports metrics such as per-dimension MAE, macro MAE, exact-vector accuracy, within-one accuracy, JSON parsing success, and inference latency.

---

## Key Features

- Dataset quality auditing and structural issue detection
- Duplicate and placeholder detection
- License and provenance checks
- Human-rating coverage analysis
- Quality-risk detector training and evaluation
- Active-review queue generation using uncertainty
- Reviewer decision and review-history support
- 4-bit NF4 QLoRA fine-tuning with PEFT
- Base-model vs fine-tuned adapter benchmarking
- Five-dimensional response-quality evaluation
- Deterministic dataset fingerprints with SHA-256
- GPU-aware inference and training
- JSON-based experiment artifacts and manifests
- Pytest test suite

---

## Dataset

The training pipeline processes human-rated instruction/response examples and preserves the original quality ratings rather than replacing them with project-generated labels.

The current dataset split contains:

| Split | Examples |
|---|---:|
| Training | 18,290 |
| Tuning | 2,030 |
| Held-out Test | 1,038 |
| **Total** | **20,324** |

The QLoRA pipeline trains against the five HelpSteer2 human-rating dimensions used throughout the project.

---

## QLoRA Fine-Tuning

The project implements supervised **QLoRA** fine-tuning using:

- 4-bit NF4 quantization
- Double quantization
- PEFT LoRA adapters
- Gradient checkpointing
- BF16/FP16 computation depending on GPU support
- `all-linear` LoRA target modules
- Gradient accumulation
- AdamW optimization
- Deterministic random seeds
- Adapter and tokenizer artifact saving

The training pipeline also records an adapter manifest containing training metadata, dataset fingerprint, optimizer steps, mean training loss, training duration, GPU information, and LoRA configuration.

---

## Quality Evaluation

The benchmark evaluates both the base model and the fine-tuned adapter using the same quality-judging interface.

For every evaluated response, the system produces structured predictions for:

```text
helpfulness
correctness
coherence
complexity
verbosity
```

The evaluation pipeline calculates:

- Mean Absolute Error (MAE) per dimension
- Macro MAE
- Exact vector accuracy
- Within-one accuracy across all dimensions
- Valid JSON output rate
- Parse errors
- Mean inference latency

Generation settings can also be tuned on the tuning split before final evaluation.

---

## Dataset Quality Pipeline

```text
Raw Dataset
     │
     ▼
Dataset Loading
     │
     ▼
Quality Audit
     │
     ├── Duplicate Detection
     ├── License Checks
     ├── Placeholder Detection
     ├── Short Text Detection
     └── Ambiguity Checks
     │
     ▼
Human Review / Active Review
     │
     ▼
Training Split
     │
     ▼
4-bit NF4 QLoRA
     │
     ▼
LoRA Adapter
     │
     ▼
Benchmark Evaluation
     │
     ▼
Quality Metrics
```

---

## Project Structure

dataset-quality-finetuning-lab/
│
├── .git/
├── .github/
│   └── ci.yml
│
├── quality_lab/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── cli.py
│   ├── config.py
│   ├── dataset.py
│   ├── detector.py
│   ├── finetuning.py
│   ├── quality.py
│   ├── review.py
│   └── splits.py
│
├── scripts/
│   └── run_experiment.py
│
├── tests/
│   ├── test_pipeline.py
│   ├── test_quality.py
│   └── test_splits.py
│
├── README.md
└── requirements.txt

## Installation

```bash
git clone https://github.com/matrixwhisper/dataset-quality-finetuning-lab.git
cd dataset-quality-finetuning-lab

python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The project uses PyTorch, Transformers, Accelerate, PEFT, bitsandbytes, scikit-learn, Joblib, psutil, and Pytest.

---

## CLI Workflow

The project exposes commands for the main dataset-quality and ML workflow, including detector training, validation, review queues, review databases, QLoRA training, and model evaluation.

Example QLoRA command:

```bash
python -m quality_lab.cli train-qlora \
  --input data/train.jsonl \
  --output outputs/qlora_adapter
```

The training command loads the reviewed training split and produces a saved QLoRA adapter.

---

## Training Configuration

QLoRA training uses a configurable model and LoRA setup. The implementation loads the base causal language model in 4-bit NF4 mode, prepares it for k-bit training, attaches LoRA adapters, and performs gradient-based optimization on the selected training examples.

The resulting adapter is saved together with metadata describing the model revision, quantization method, training data fingerprint, LoRA configuration, training statistics, and GPU environment.

---

## Benchmarking

The benchmark is designed to make the comparison between the base model and the fine-tuned adapter reproducible.

Both models use the same scoring prompt and generation interface, while the evaluation code calculates quality errors against the human ratings supplied with each example.

This makes it possible to investigate whether fine-tuning improves the model's ability to reproduce human quality judgments across multiple dimensions rather than relying on a single aggregate score.

---

## Testing

Run the test suite with:

```bash
pytest -q
```

The repository includes automated tests covering the project's core dataset, quality, training, and evaluation components.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| PyTorch | Model training and inference |
| Hugging Face Transformers | Model and tokenizer handling |
| PEFT | LoRA / QLoRA adapters |
| bitsandbytes | 4-bit NF4 quantization |
| scikit-learn | Quality-risk detection |
| Joblib | Model persistence |
| Pytest | Automated testing |
| CUDA | GPU acceleration |

---

## What This Project Demonstrates

This project demonstrates an end-to-end ML workflow rather than only model fine-tuning:

**Dataset Engineering → Quality Auditing → Human Review → Risk Detection → QLoRA Fine-Tuning → Benchmarking → Evaluation**

It focuses on making dataset quality measurable and connecting dataset curation decisions directly to downstream LLM fine-tuning and evaluation.

---

## Author

**Idris**

Machine Learning Engineer focused on Python, model fine-tuning, and reliable ML systems.

[GitHub](https://github.com/matrixwhisper)

---


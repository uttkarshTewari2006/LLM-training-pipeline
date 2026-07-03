# LLM Training Pipeline

Reliable ML training platform scaffold focused on distributed PyTorch training, experiment tracking, feature serving, model monitoring, and SRE-style operational practices.

## Project Direction

This project is framed as an ML platform, not just a model training script. The initial model is intentionally small so the engineering work can focus on repeatable training, observability, recoverability, and deployment hygiene.

## Phase 1: Distributed Training

Phase 1 starts with PyTorch Distributed Data Parallel (DDP) using `torchrun`.

Core goals:

- Train a CIFAR-10 image classifier with one process per GPU.
- Support CPU-only and single-GPU development locally.
- Support two-GPU execution on Kaggle with `torchrun --nproc_per_node=2`.
- Save checkpoints only from rank 0.
- Keep the training script cloud-agnostic through environment variables.

## Tech Stack

- Python 3.10+
- PyTorch and TorchVision
- PyTorch Distributed Data Parallel
- MLflow for experiment tracking
- Docker Compose for local platform services
- FastAPI for future inference and feature APIs
- Evidently AI for future drift monitoring

## Quick Start

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run a local CPU smoke test:

```bash
python src/training/train_ddp.py --epochs 1 --batch-size 32 --limit-train-batches 5 --limit-val-batches 2 --device cpu
```

Run an offline CPU smoke test without downloading CIFAR-10:

```bash
python src/training/train_ddp.py --dataset fake --epochs 1 --batch-size 4 --num-workers 0 --limit-train-batches 1 --limit-val-batches 1 --device cpu
```

Run DDP on two GPUs:

```bash
torchrun --nproc_per_node=2 src/training/train_ddp.py --epochs 5 --batch-size 128 --device cuda
```

## Repository Layout

```text
.
├── configs/
│   └── train_cifar10.yaml
├── docs/
│   └── architecture.md
├── scripts/
│   └── kaggle_train.sh
├── src/
│   └── training/
│       └── train_ddp.py
├── .env.example
├── docker-compose.yml
├── requirements.txt
└── README.md
```

The local Phase 1 learning guide is intentionally ignored by git at `docs/phase1_ddp_7_day_guide.md`.

# LLM Training Pipeline

Reliable ML training platform scaffold focused on distributed PyTorch training,
experiment tracking, recoverability, and SRE-style operational practices.

The first phase intentionally uses a small CIFAR-10 classifier. The model is not
the point; the point is a repeatable training path that can run locally, scale to
single-node multi-GPU execution, recover from interruption, and emit useful
training signals.

## Phase 1: Distributed Training

Phase 1 implements a PyTorch Distributed Data Parallel (DDP) training entrypoint
launched with `torchrun`.

What this phase proves:

- A single training script supports CPU, single-GPU, and single-node multi-GPU
  runs.
- `DistributedSampler` shards training and validation data across ranks.
- DDP synchronizes gradients during backpropagation.
- Rank 0 owns shared side effects: user-visible logs, checkpoints, and MLflow
  metric writes.
- Checkpoints are reloadable from single-process or distributed runs.
- Local artifacts, datasets, secrets, checkpoints, and planning notes stay out
  of git.

## Architecture Summary

```text
torchrun
  -> rank 0 worker -> device 0 -> training shard -> logs, MLflow, checkpoint
  -> rank 1 worker -> device 1 -> training shard -> training only

DistributedSampler assigns each worker a distinct data shard.
DDP all-reduces gradients during loss.backward().
Rank 0 writes checkpoints to checkpoints/latest.pt after each epoch.
```

The current implementation is single-node DDP. It is designed to be cloud-agnostic
and easy to run on a local machine or a Kaggle 2x T4 notebook, but it does not
claim multi-node production orchestration yet.

## Phase 2: Durable Experiment Tracking

Phase 2 adds a local MLflow tracking stack backed by Postgres for run metadata
and MinIO for model artifacts.

Day 2 implementation includes:

- Docker Compose services for Postgres, MinIO, MLflow, and artifact bucket
  bootstrap.
- A custom MLflow image with S3 and Postgres client dependencies.
- Environment-driven MLflow tracking URI and experiment selection.
- Checkpoint artifact uploads from rank 0 to MLflow.
- A local validation guide at `docs/phase2_mlflow_day2.md`.

## Tech Stack

- Python 3.10+
- PyTorch and TorchVision
- PyTorch Distributed Data Parallel
- MLflow for experiment tracking
- Docker Compose for local MLflow infrastructure
- FastAPI and Evidently AI dependencies reserved for future platform phases

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

Run a local CPU smoke test without downloading CIFAR-10:

```bash
python src/training/train_ddp.py --dataset fake --epochs 1 --batch-size 32 --limit-train-batches 5 --limit-val-batches 2 --device cpu --disable-mlflow
```

Run a local CPU DDP smoke test with two worker processes:

```bash
torchrun --nproc_per_node=2 src/training/train_ddp.py --dataset fake --epochs 1 --batch-size 32 --limit-train-batches 5 --limit-val-batches 2 --device cpu --disable-mlflow
```

On Windows, some PyTorch builds do not include libuv support for the local
`torchrun` rendezvous store. If that environment error appears, use WSL, Linux,
or the Kaggle command below for the DDP smoke test.

Run CIFAR-10 locally:

```bash
python src/training/train_ddp.py --epochs 1 --batch-size 64 --limit-train-batches 20 --limit-val-batches 5 --device cpu
```

Start the local MLflow tracking stack:

```bash
docker compose up -d postgres minio minio-create-bucket mlflow
```

Run training against the stack:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME=ddp-cifar10-platform
python src/training/train_ddp.py --dataset fake --epochs 1 --batch-size 32 --limit-train-batches 5 --limit-val-batches 2 --device cpu
```

The MLflow UI runs at `http://localhost:5000`, and the MinIO console runs at
`http://localhost:9001`.

Run on Kaggle 2x T4:

```bash
pip install -r requirements.txt
torchrun --nproc_per_node=2 src/training/train_ddp.py --epochs 5 --batch-size 128 --device cuda --checkpoint-dir checkpoints/kaggle-ddp
```

## Phase 1 End-to-End Results

Full CIFAR-10 DDP training was completed on July 17, 2026 with two workers,
batch size 128, MLflow tracking enabled, and a rank-0 checkpoint written at the
end of each epoch.

```text
epoch=1 train_loss=1.3795 train_acc=0.4957 val_loss=1.3988 val_acc=0.5472
epoch=2 train_loss=0.9010 train_acc=0.6797 val_loss=1.1010 val_acc=0.6359
epoch=3 train_loss=0.7061 train_acc=0.7507 val_loss=0.8482 val_acc=0.7114
epoch=4 train_loss=0.6023 train_acc=0.7891 val_loss=0.9164 val_acc=0.7068
epoch=5 train_loss=0.5202 train_acc=0.8189 val_loss=0.7675 val_acc=0.7515
```

Final MLflow run summary:

```text
run_id: a19602c69090406eb74bc2d047b24797
status: FINISHED
world_size: 2
batch_size: 128
train_loss: 0.520242
val_loss: 0.767518
train_accuracy: 0.81892
val_accuracy: 0.7515
```

Checkpoint artifact:

```text
latest.pt 128M
```

## Failure Recovery

Rank 0 saves the latest checkpoint after each epoch. By default the checkpoint is
written to:

```text
checkpoints/latest.pt
```

To resume a failed run:

```bash
python src/training/train_ddp.py --resume checkpoints/latest.pt
```

The checkpoint contains:

- Completed epoch number
- Model state dict
- Optimizer state dict

For DDP jobs, restart with the same `torchrun` shape and pass the same
`--resume` path. On Kaggle, `/kaggle/working` persists during the active session,
so a notebook cell can be rerun with the checkpoint path as long as the session
has not been reset.

## SRE Features

- Reproducible CLI entrypoint for local and Kaggle runs.
- Rank-aware side effects prevent duplicate checkpoints and duplicate metric
  writes.
- Checkpointing provides a simple recovery path after interrupted training.
- MLflow logging records loss, accuracy, throughput, device, and world size.
- `.gitignore` excludes `.env`, local data, checkpoints, MLflow runs, runtime
  files, and the private learning guide.

## Repository Layout

```text
.
|-- configs/
|   `-- train_cifar10.yaml
|-- docs/
|   |-- architecture.md
|   |-- phase2_mlflow_day2.md
|   `-- phase1_demo.md
|-- infra/
|   `-- mlflow/
|       `-- Dockerfile
|-- scripts/
|   `-- kaggle_train.sh
|-- src/
|   `-- training/
|       `-- train_ddp.py
|-- .env.example
|-- docker-compose.yml
|-- requirements.txt
`-- README.md
```

The local Phase 1 learning guide is intentionally ignored by git at
`docs/phase1_ddp_7_day_guide.md`.

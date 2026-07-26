# ML Training Pipeline

Reliable ML training platform scaffold focused on distributed PyTorch training,
experiment tracking, recoverability, and SRE-style operational practices.

The project uses a small CIFAR-10 classifier so the platform work is easy to
run and inspect. The model is not the point; the point is a repeatable path from
training to tracking, serving, and monitoring with clear local validation.

## Project Snapshot

- Distributed PyTorch training with CPU, single-GPU, and single-node multi-GPU
  support.
- Checkpoint recovery for interrupted runs.
- MLflow tracking with Postgres metadata and MinIO artifact storage.
- FastAPI serving for a trained checkpoint.
- Evidently drift reporting over reference and live-like image feature batches.
- Focused tests covering serving and monitoring paths.

## Results Summary

| Stage | Outcome | Validation |
| --- | --- | --- |
| Phase 1 | DDP CIFAR-10 training completed with a reloadable rank-0 checkpoint. | 5 epochs, final validation accuracy `0.7515`, checkpoint `latest.pt` at `128M`. |
| Phase 2 | MLflow captured training params, metrics, throughput, and checkpoint artifacts. | Run `fba131b9424f4855ac97b0c217eaf799`, exported DB at `mlflow-logged-runs-db/mlflow.db`. |
| Phase 3 | FastAPI served predictions from `checkpoints/latest.pt`. | `/health` returned healthy and `/predict` returned top-3 CIFAR-10 probabilities. |
| Phase 4 | Evidently generated a local drift report for reference vs live-like image features. | `64` reference rows, `64` current rows, `11` features, full test suite `7 passed`. |

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

Day 3 final validation includes a completed MLflow-logged CIFAR-10 run with the
tracking database exported under `mlflow-logged-runs-db/`.

## Phase 3: Model Serving API

Phase 3 turns the trained checkpoint into a callable FastAPI service.

What this phase proves:

- The service loads a training checkpoint at startup.
- `/health` reports model loading status, checkpoint path, epoch, and device.
- `/predict` accepts CIFAR-10-shaped image arrays and returns class id, label,
  confidence, and top-k probabilities.
- Serving preprocessing matches the training validation normalization.
- API tests cover health, prediction response shape, and input validation.

## Phase 4: Monitoring And Drift Reports

Phase 4 adds a local monitoring path for comparing a reference batch against a
live-like batch after the model has a serving boundary.

The monitoring scripts summarize CIFAR-10-shaped images into tabular features
such as channel means, channel standard deviations, brightness, dark pixel share,
bright pixel share, and a saturation proxy. Evidently compares the reference and
current feature tables and writes report artifacts under `outputs/monitoring/`.

## Tech Stack

- Python 3.10+
- PyTorch and TorchVision
- PyTorch Distributed Data Parallel
- MLflow for experiment tracking
- Docker Compose for local MLflow infrastructure
- FastAPI for serving
- Evidently AI for local drift reports

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

Run the model serving API:

```bash
export MODEL_CHECKPOINT_PATH=checkpoints/latest.pt
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

On Windows PowerShell:

```powershell
$env:MODEL_CHECKPOINT_PATH = "checkpoints/latest.pt"
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

Check health:

```bash
curl http://localhost:8000/health
```

Run a prediction smoke request:

```bash
python scripts/smoke_predict.py
```

Prediction request shape:

```json
{
  "image": "32x32x3 numeric array",
  "top_k": 3
}
```

Build monitoring reference/current inputs:

```bash
python src/monitoring/build_reference.py
```

Generate a drift report:

```bash
python src/monitoring/generate_drift_report.py
```

Monitoring outputs are local artifacts and are ignored by git:

```text
outputs/monitoring/reference_features.csv
outputs/monitoring/current_features.csv
outputs/monitoring/drift_report.html
outputs/monitoring/drift_report.json
outputs/monitoring/drift_summary.json
```

The report answers whether the current batch still resembles the reference batch
on simple image summary statistics. It is an early warning signal for
investigation; it does not prove that model accuracy changed.

The full `image` value must be either `[32, 32, 3]` channel-last pixels or
`[3, 32, 32]` channel-first pixels. Pixel values may be `0..1` floats or
`0..255` integer-like values.

Prediction response shape:

```json
{
  "class_id": 6,
  "class_label": "frog",
  "confidence": 0.21,
  "probabilities": [
    {"class_id": 6, "class_label": "frog", "confidence": 0.21}
  ],
  "model_epoch": 5
}
```

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

## Phase 2 End-to-End Results

Full CIFAR-10 training was logged to MLflow on July 23, 2026. The exported
tracking database is stored at `mlflow-logged-runs-db/mlflow.db`.

```text
epoch=1 train_loss=2.3538 train_acc=0.1734 val_loss=2.4878 val_acc=0.1547 throughput=19.92 samples/sec
epoch=2 train_loss=2.0306 train_acc=0.2703 val_loss=3.9170 val_acc=0.1313 throughput=20.18 samples/sec
```

Final MLflow run summary:

```text
run_id: fba131b9424f4855ac97b0c217eaf799
run_name: phase2-ddp-cifar10
status: FINISHED
source_commit: d2995096a6a5a55a5b10e7aae24955776c6371b0
world_size: 1
device: cpu
dataset: cifar10
epochs: 2
batch_size: 128
train_loss: 2.030552
val_loss: 3.917007
train_accuracy: 0.2703125
val_accuracy: 0.13125
throughput_samples_per_sec: 20.175143
```

This validates Phase 2's final path: the trainer can create a named MLflow run,
record parameters and metrics, and persist the run metadata in a reloadable
tracking database.

## Phase 3 End-to-End Results

The serving API was validated locally against `checkpoints/latest.pt`.

```text
GET /health -> status=healthy model_loaded=true checkpoint_path=checkpoints/latest.pt model_epoch=1
```

Representative prediction response:

```json
{
  "class_id": 1,
  "class_label": "automobile",
  "confidence": 0.10819769650697708,
  "probabilities": [
    {"class_id": 1, "class_label": "automobile", "confidence": 0.10819769650697708},
    {"class_id": 8, "class_label": "ship", "confidence": 0.1043490469455719},
    {"class_id": 4, "class_label": "deer", "confidence": 0.1031981110572815}
  ],
  "model_epoch": 1
}
```

Phase 3 connects training to production-style usage: clients call a stable API
contract and do not need to know how the model was trained, checkpointed, or
loaded.

## Phase 4 End-to-End Results

The local monitoring path generates deterministic reference and live-like
batches, then writes an Evidently HTML report and JSON summaries under
`outputs/monitoring/`.

```text
python src/monitoring/build_reference.py
python src/monitoring/generate_drift_report.py
```

Validated local report output:

```text
reference_rows: 64
current_rows: 64
feature_count: 11
html_report_path: outputs/monitoring/drift_report.html
json_report_path: outputs/monitoring/drift_report.json
```

Largest mean shifts from the generated summary:

```text
red_mean: 0.054777
saturation_proxy_mean: 0.049361
blue_mean: -0.033604
bright_pixel_share: 0.029022
green_std: 0.021630
```

Validation:

```text
pytest tests/test_monitoring.py -q -> 4 passed
pytest -q -> 7 passed
```

The live-like batch intentionally changes color and contrast statistics so the
report has a visible signal to inspect. In a real serving workflow, the current
batch would come from recent prediction traffic or batch scoring inputs.

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
- FastAPI serving exposes health and prediction endpoints for a trained
  checkpoint.
- Evidently drift reports compare reference and live-like image feature batches.
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
|-- mlflow-logged-runs-db/
|   `-- mlflow.db
|-- scripts/
|   |-- kaggle_train.sh
|   `-- smoke_predict.py
|-- src/
|   |-- monitoring/
|   |   |-- __init__.py
|   |   |-- build_reference.py
|   |   |-- data.py
|   |   |-- generate_drift_report.py
|   |   `-- report.py
|   |-- serving/
|   |   |-- __init__.py
|   |   |-- app.py
|   |   `-- model.py
|   `-- training/
|       `-- train_ddp.py
|-- tests/
|   |-- conftest.py
|   |-- test_monitoring.py
|   `-- test_serving_api.py
|-- .env.example
|-- docker-compose.yml
|-- requirements.txt
`-- README.md
```

The local Phase 1 learning guide is intentionally ignored by git at
`docs/phase1_ddp_7_day_guide.md`.

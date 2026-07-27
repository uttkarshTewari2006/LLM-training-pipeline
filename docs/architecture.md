# Architecture

## Platform Narrative

The project is designed as a small but realistic ML platform:

1. Distributed training runs with PyTorch DDP and `torchrun`.
2. Experiments are tracked in MLflow.
3. Model artifacts are stored outside the training process.
4. Monitoring compares reference and live-like data with Evidently AI.
5. Future feature serving separates offline feature generation from online lookup.

## Phase 1 Runtime Flow

```text
torchrun
  ├── rank 0 process -> GPU 0 -> training shard -> metric logging/checkpoints
  └── rank 1 process -> GPU 1 -> training shard

DistributedSampler splits CIFAR-10 batches across ranks.
DDP synchronizes gradients during backward pass.
Rank 0 owns user-visible side effects such as checkpoints.
```

## Phase 2 Tracking Flow

```text
rank 0 training process
  -> MLflow tracking API
     -> Postgres stores run metadata, params, and metrics
     -> MLflow artifact proxy writes checkpoints to MinIO
```

The training process reads `MLFLOW_TRACKING_URI` and
`MLFLOW_EXPERIMENT_NAME` from the environment. Rank 0 starts the MLflow run,
logs training parameters and metrics, saves `checkpoints/latest.pt`, and uploads
that checkpoint as an MLflow artifact.

## Phase 3 Serving Flow

```text
client
  -> FastAPI /predict
     -> CIFAR-10 preprocessing
     -> ResNet-18 checkpoint loaded at startup
     -> class id, label, confidence, and top-k probabilities
```

The serving API loads a checkpoint from `MODEL_CHECKPOINT_PATH`, defaulting to
`checkpoints/latest.pt`. The model architecture is reused from training so
training checkpoints and serving artifacts stay compatible. Input preprocessing
uses the same CIFAR-10 normalization as validation.

## Phase 4 Monitoring Flow

```text
reference image batch
  -> image summary feature table
     -> Evidently drift report
live-like image batch
  -> image summary feature table
```

The monitoring scripts build deterministic reference and current feature tables
from CIFAR-10-shaped image batches. The current batch intentionally shifts color
and contrast statistics so the generated report has a visible monitoring signal.
Reports are written under `outputs/monitoring/` as local artifacts.

## Phase 5 Operations Flow

```text
operator
  -> runbook commands
     -> training smoke
     -> serving live/health checks
     -> prediction smoke
     -> monitoring report smoke
     -> repo hygiene checks
```

Phase 5 hardens the local scaffold with explicit runbooks, smoke automation, and
health boundaries. `/live` reports that the API process is running, while
`/health` reports whether the model checkpoint loaded and includes the checkpoint
path, epoch, device, and load error when applicable.

## SRE Signals

- Reproducible entrypoints for local and cloud runs.
- Checkpointing for recovery after interrupted training.
- Separation between local-only data/artifacts and committed source.
- Durable experiment metadata and artifact storage through MLflow, Postgres,
  and MinIO.
- A FastAPI serving boundary for loading a checkpoint and returning predictions.
- Basic drift monitoring through generated Evidently reports.
- Operational runbooks and smoke automation for local validation.
- Planned observability through Prometheus, Grafana, and production metrics.

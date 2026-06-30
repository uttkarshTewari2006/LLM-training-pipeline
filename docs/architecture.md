# Architecture

## Platform Narrative

The project is designed as a small but realistic ML platform:

1. Distributed training runs with PyTorch DDP and `torchrun`.
2. Experiments are tracked in MLflow.
3. Model artifacts are stored outside the training process.
4. Future feature serving separates offline feature generation from online lookup.
5. Future monitoring compares reference and live-like data with Evidently AI.

## Phase 1 Runtime Flow

```text
torchrun
  ├── rank 0 process -> GPU 0 -> training shard -> metric logging/checkpoints
  └── rank 1 process -> GPU 1 -> training shard

DistributedSampler splits CIFAR-10 batches across ranks.
DDP synchronizes gradients during backward pass.
Rank 0 owns user-visible side effects such as checkpoints.
```

## SRE Signals

- Reproducible entrypoints for local and cloud runs.
- Checkpointing for recovery after interrupted training.
- Separation between local-only data/artifacts and committed source.
- Planned observability through MLflow, Prometheus, Grafana, and drift reports.

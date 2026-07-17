# Phase 1 Demo And Retrospective

## Demo Command

Use fake data for a deterministic, dependency-light smoke test that exercises the
same training, checkpointing, and rank-aware logging path without downloading
CIFAR-10:

```bash
python src/training/train_ddp.py --dataset fake --epochs 1 --batch-size 32 --limit-train-batches 5 --limit-val-batches 2 --device cpu --disable-mlflow --checkpoint-dir checkpoints/demo
```

Representative rank-0 output from the Day 8 smoke run:

```text
epoch=1 train_loss=2.4184 train_acc=0.0938 val_loss=3.1943 val_acc=0.0469
```

Expected checkpoint:

```text
checkpoints/demo/latest.pt
```

## What Phase 1 Proves

Phase 1 proves the training path, not model quality. The project can launch a
PyTorch training job through one reusable entrypoint, run locally without GPUs,
scale to single-node DDP with `torchrun`, shard data through
`DistributedSampler`, synchronize gradients through DDP, and keep shared side
effects on rank 0.

The operational value is that an interrupted run can be resumed from a checkpoint
and the metrics path is already MLflow-compatible. CIFAR-10 keeps iteration cheap
while the platform practices remain realistic.

## Interview Pitch

I built the first stage of a reliable ML platform around PyTorch DDP. The model
is intentionally small, but the training path uses production-style distributed
primitives: `torchrun`, one process per GPU, `DistributedSampler`, rank-aware
side effects, checkpointing, and MLflow-compatible metrics. The next phases add
MLflow infrastructure, feature serving, and drift monitoring.

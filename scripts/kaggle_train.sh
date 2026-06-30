#!/usr/bin/env bash
set -euo pipefail

torchrun --nproc_per_node=2 src/training/train_ddp.py \
  --epochs 5 \
  --batch-size 128 \
  --device cuda \
  --checkpoint-dir checkpoints/kaggle-ddp

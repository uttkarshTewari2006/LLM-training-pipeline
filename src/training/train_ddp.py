import argparse
import os
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import resnet18


def parse_args() -> argparse.Namespace:
    # Personal note: keep CLI defaults modest so local smoke runs and DDP jobs use the same entrypoint.
    parser = argparse.ArgumentParser(description="CIFAR-10 training with optional PyTorch DDP.")
    parser.add_argument("--data-dir", default="data", help="Directory for dataset downloads.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for saved checkpoints.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["cifar10", "fake"], default="cifar10")
    parser.add_argument("--fake-train-size", type=int, default=256)
    parser.add_argument("--fake-val-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    parser.add_argument("--disable-mlflow", action="store_true")
    return parser.parse_args()


def is_distributed() -> bool:
    # Personal note: torchrun sets WORLD_SIZE; a value above one means every rank must coordinate.
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed() -> tuple[int, int, int]:
    # Personal note: initialize process groups only for multi-rank launches and return single-rank defaults otherwise.
    if not is_distributed():
        return 0, 0, 1

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    # Personal note: guard teardown so normal single-process training can call this safely.
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    # Personal note: seed every RNG used here so repeated runs are easier to compare.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(requested: str, local_rank: int) -> torch.device:
    # Personal note: map each DDP worker to its local GPU, while still allowing CPU-only smoke tests.
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if torch.cuda.is_available() and requested in {"auto", "cuda"}:
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def build_model(num_classes: int = 10) -> nn.Module:
    # Personal note: adapt ResNet-18 for CIFAR-10's 32x32 images by removing the ImageNet-style stem.
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_loaders(args: argparse.Namespace, rank: int, world_size: int) -> tuple[DataLoader, DataLoader]:
    # Personal note: use distributed samplers when needed so ranks see separate dataset shards.
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    if args.dataset == "fake":
        train_dataset = torchvision.datasets.FakeData(
            size=args.fake_train_size,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=train_transform,
        )
        val_dataset = torchvision.datasets.FakeData(
            size=args.fake_val_size,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=eval_transform,
        )
    else:
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            download=True,
            transform=eval_transform,
        )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    # Personal note: average metric tensors across ranks so rank 0 logs global results, not local ones.
    if world_size == 1:
        return value
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value / world_size


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    world_size: int,
    limit_batches: int,
) -> tuple[float, float, float]:
    # Personal note: run one training pass and aggregate loss/accuracy in a DDP-compatible format.
    model.train()
    start_time = time.time()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        if limit_batches and batch_idx >= limit_batches:
            break

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_seen += inputs.size(0)

    metrics = torch.tensor([total_loss, total_correct, total_seen], dtype=torch.float64, device=device)
    metrics = reduce_mean(metrics, world_size)
    loss = metrics[0].item() / max(metrics[2].item(), 1)
    accuracy = metrics[1].item() / max(metrics[2].item(), 1)
    throughput = metrics[2].item() * world_size / max(time.time() - start_time, 1e-9)
    return loss, accuracy, throughput


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    world_size: int,
    limit_batches: int,
) -> tuple[float, float]:
    # Personal note: mirror training metrics without gradient work so validation stays cheap and comparable.
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        if limit_batches and batch_idx >= limit_batches:
            break

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_seen += inputs.size(0)

    metrics = torch.tensor([total_loss, total_correct, total_seen], dtype=torch.float64, device=device)
    metrics = reduce_mean(metrics, world_size)
    loss = metrics[0].item() / max(metrics[2].item(), 1)
    accuracy = metrics[1].item() / max(metrics[2].item(), 1)
    return loss, accuracy


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, checkpoint_dir: str) -> None:
    # Personal note: unwrap DDP before saving so checkpoints can be reloaded in single-process jobs.
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        Path(checkpoint_dir) / "latest.pt",
    )


def main() -> None:
    # Personal note: rank 0 owns logging/checkpoint side effects; every rank still trains and validates.
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    set_seed(args.seed + rank)
    device = select_device(args.device, local_rank)

    train_loader, val_loader = build_loaders(args, rank, world_size)
    model = build_model().to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model_to_load = model.module if isinstance(model, DDP) else model
        model_to_load.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}")

    run_context = mlflow.start_run(run_name="phase1-ddp-cifar10") if rank == 0 and not args.disable_mlflow else None
    if rank == 0 and not args.disable_mlflow:
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "world_size": world_size,
                "device": str(device),
                "dataset": args.dataset,
            }
        )

    try:
        for epoch in range(start_epoch, args.epochs):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            train_loss, train_acc, throughput = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                world_size,
                args.limit_train_batches,
            )
            val_loss, val_acc = evaluate(
                model,
                val_loader,
                criterion,
                device,
                world_size,
                args.limit_val_batches,
            )

            if rank == 0:
                print(
                    f"epoch={epoch + 1} "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                )
                save_checkpoint(model, optimizer, epoch + 1, args.checkpoint_dir)
                if not args.disable_mlflow:
                    mlflow.log_metrics(
                        {
                            "train_loss": train_loss,
                            "train_accuracy": train_acc,
                            "val_loss": val_loss,
                            "val_accuracy": val_acc,
                            "throughput_samples_per_sec": throughput,
                        },
                        step=epoch + 1,
                    )
    finally:
        if run_context is not None:
            mlflow.end_run()
        cleanup_distributed()


if __name__ == "__main__":
    main()

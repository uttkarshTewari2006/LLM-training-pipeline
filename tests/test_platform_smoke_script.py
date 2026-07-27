from pathlib import Path

import torch

from scripts import platform_smoke
from src.training.train_ddp import build_model


def test_platform_smoke_serving_step_uses_checkpoint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(platform_smoke, "SMOKE_OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(platform_smoke, "CHECKPOINT_DIR", tmp_path / "checkpoints")

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "latest.pt"
    model = build_model()
    torch.save({"epoch": 1, "model_state_dict": model.state_dict()}, checkpoint)

    assert checkpoint.exists()
    platform_smoke.run_serving_smoke(checkpoint)

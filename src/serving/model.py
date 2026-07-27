from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.training.train_ddp import build_model


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

DEFAULT_CHECKPOINT_PATH = "checkpoints/latest.pt"
MEAN = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32).view(1, 3, 1, 1)


@dataclass(frozen=True)
class Prediction:
    class_id: int
    class_label: str
    confidence: float
    probabilities: list[dict[str, float | int | str]]


class ModelService:
    def __init__(self, checkpoint_path: str | None = None, device: str | None = None) -> None:
        self.checkpoint_path = checkpoint_path or os.environ.get("MODEL_CHECKPOINT_PATH", DEFAULT_CHECKPOINT_PATH)
        requested_device = device or os.environ.get("MODEL_DEVICE", "cpu")
        self.device = torch.device(requested_device)
        self.model = build_model()
        self.model_epoch: int | None = None
        self.loaded = False
        self.load_error: str | None = None

    def load(self) -> None:
        path = Path(self.checkpoint_path)
        if not path.exists():
            self.loaded = False
            self.load_error = f"Checkpoint not found: {path}"
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.model_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
        self.loaded = True
        self.load_error = None

    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy" if self.loaded else "not_loaded",
            "model_loaded": self.loaded,
            "checkpoint_path": self.checkpoint_path,
            "model_epoch": self.model_epoch,
            "device": str(self.device),
            "load_error": self.load_error,
        }

    def predict(self, image: Any, top_k: int = 3) -> Prediction:
        if not self.loaded:
            raise RuntimeError("Model is not loaded.")

        tensor = preprocess_image(image).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        k = min(max(top_k, 1), len(CIFAR10_CLASSES))
        values, indices = torch.topk(probabilities, k=k)
        top_predictions = [
            {
                "class_id": int(class_id),
                "class_label": CIFAR10_CLASSES[int(class_id)],
                "confidence": float(confidence),
            }
            for confidence, class_id in zip(values.cpu(), indices.cpu())
        ]
        winner = top_predictions[0]
        return Prediction(
            class_id=int(winner["class_id"]),
            class_label=str(winner["class_label"]),
            confidence=float(winner["confidence"]),
            probabilities=top_predictions,
        )


def preprocess_image(image: Any) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)

    if array.shape == (32, 32, 3):
        array = np.transpose(array, (2, 0, 1))
    elif array.shape != (3, 32, 32):
        raise ValueError("image must have shape [32, 32, 3] or [3, 32, 32]")

    if not np.isfinite(array).all():
        raise ValueError("image must contain only finite numeric values")

    if array.max(initial=0.0) > 1.0:
        array = array / 255.0

    if array.min(initial=0.0) < 0.0 or array.max(initial=0.0) > 1.0:
        raise ValueError("image values must be in the range 0..1 or 0..255")

    tensor = torch.from_numpy(array).unsqueeze(0)
    return (tensor - MEAN) / STD

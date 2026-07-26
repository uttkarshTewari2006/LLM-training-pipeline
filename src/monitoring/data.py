from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs/monitoring")
REFERENCE_FILE = "reference_features.csv"
CURRENT_FILE = "current_features.csv"
INPUT_SUMMARY_FILE = "monitoring_input_summary.json"

CIFAR10_REFERENCE_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_REFERENCE_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


def generate_image_batch(
    sample_count: int,
    seed: int,
    mean_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    contrast_scale: float = 1.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(
        loc=CIFAR10_REFERENCE_MEAN.reshape(1, 3, 1, 1),
        scale=CIFAR10_REFERENCE_STD.reshape(1, 3, 1, 1),
        size=(sample_count, 3, 32, 32),
    ).astype(np.float32)
    centered = (base - CIFAR10_REFERENCE_MEAN.reshape(1, 3, 1, 1)) * contrast_scale
    shifted = centered + CIFAR10_REFERENCE_MEAN.reshape(1, 3, 1, 1)
    shifted += np.array(mean_shift, dtype=np.float32).reshape(1, 3, 1, 1)
    return np.clip(shifted, 0.0, 1.0)


def summarize_images(images: np.ndarray) -> pd.DataFrame:
    if images.ndim != 4 or images.shape[1:] != (3, 32, 32):
        raise ValueError("images must have shape [n, 3, 32, 32]")

    channel_mean = images.mean(axis=(2, 3))
    channel_std = images.std(axis=(2, 3))
    brightness = images.mean(axis=1)
    saturation_proxy = images.max(axis=1) - images.min(axis=1)

    return pd.DataFrame(
        {
            "red_mean": channel_mean[:, 0],
            "green_mean": channel_mean[:, 1],
            "blue_mean": channel_mean[:, 2],
            "red_std": channel_std[:, 0],
            "green_std": channel_std[:, 1],
            "blue_std": channel_std[:, 2],
            "brightness_mean": brightness.mean(axis=(1, 2)),
            "brightness_std": brightness.std(axis=(1, 2)),
            "dark_pixel_share": (images < 0.1).mean(axis=(1, 2, 3)),
            "bright_pixel_share": (images > 0.9).mean(axis=(1, 2, 3)),
            "saturation_proxy_mean": saturation_proxy.mean(axis=(1, 2)),
        }
    )


def build_monitoring_frames(sample_count: int = 256) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_images = generate_image_batch(sample_count=sample_count, seed=42)
    current_images = generate_image_batch(
        sample_count=sample_count,
        seed=314,
        mean_shift=(0.06, 0.02, -0.04),
        contrast_scale=1.12,
    )
    return summarize_images(reference_images), summarize_images(current_images)


def write_monitoring_inputs(output_dir: Path = DEFAULT_OUTPUT_DIR, sample_count: int = 256) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference, current = build_monitoring_frames(sample_count=sample_count)

    reference_path = output_dir / REFERENCE_FILE
    current_path = output_dir / CURRENT_FILE
    reference.to_csv(reference_path, index=False)
    current.to_csv(current_path, index=False)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_path": str(reference_path),
        "current_path": str(current_path),
        "row_count": sample_count,
        "feature_columns": list(reference.columns),
        "current_batch_profile": {
            "mean_shift": {"red": 0.06, "green": 0.02, "blue": -0.04},
            "contrast_scale": 1.12,
        },
    }
    summary_path = output_dir / INPUT_SUMMARY_FILE
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_monitoring_inputs(output_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_path = output_dir / REFERENCE_FILE
    current_path = output_dir / CURRENT_FILE
    if not reference_path.exists() or not current_path.exists():
        write_monitoring_inputs(output_dir=output_dir)
    return pd.read_csv(reference_path), pd.read_csv(current_path)

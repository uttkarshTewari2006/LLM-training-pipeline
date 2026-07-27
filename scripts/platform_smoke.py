from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from src.monitoring.report import generate_drift_report
from src.serving.app import create_app
from src.serving.model import ModelService


SMOKE_OUTPUT_DIR = ROOT / "outputs" / "phase5-smoke"
CHECKPOINT_DIR = SMOKE_OUTPUT_DIR / "checkpoints"


def run_command(command: list[str]) -> None:
    print(f"$ {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, check=True)


def run_training_smoke() -> Path:
    run_command(
        [
            sys.executable,
            "src/training/train_ddp.py",
            "--dataset",
            "fake",
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--fake-train-size",
            "64",
            "--fake-val-size",
            "32",
            "--num-workers",
            "0",
            "--limit-train-batches",
            "2",
            "--limit-val-batches",
            "1",
            "--device",
            "cpu",
            "--disable-mlflow",
            "--checkpoint-dir",
            str(CHECKPOINT_DIR),
        ]
    )
    checkpoint = CHECKPOINT_DIR / "latest.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"Training smoke did not create {checkpoint}")
    return checkpoint


def run_serving_smoke(checkpoint: Path) -> None:
    service = ModelService(checkpoint_path=str(checkpoint), device="cpu")
    app = create_app(service)
    with TestClient(app) as client:
        live_response = client.get("/live")
        health_response = client.get("/health")
        payload = {"image": [[[128, 128, 128] for _ in range(32)] for _ in range(32)], "top_k": 3}
        prediction_response = client.post("/predict", json=payload)

    live_response.raise_for_status()
    health_response.raise_for_status()
    prediction_response.raise_for_status()
    health = health_response.json()
    prediction = prediction_response.json()
    if health["status"] != "healthy" or not health["model_loaded"]:
        raise RuntimeError(f"Serving smoke health failed: {health}")
    if len(prediction["probabilities"]) != 3:
        raise RuntimeError(f"Serving smoke prediction shape failed: {prediction}")
    print(f"serving_health={health['status']} model_epoch={health['model_epoch']}")


def run_monitoring_smoke() -> None:
    summary = generate_drift_report(output_dir=SMOKE_OUTPUT_DIR / "monitoring")
    print(
        "monitoring_report="
        f"{summary['html_report_path']} rows={summary['reference_rows']}/{summary['current_rows']}"
    )


def main() -> None:
    SMOKE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = run_training_smoke()
    run_serving_smoke(checkpoint)
    run_monitoring_smoke()
    print("phase5_smoke=passed")


if __name__ == "__main__":
    main()

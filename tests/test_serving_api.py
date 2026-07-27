from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.model import CIFAR10_CLASSES, ModelService
from src.training.train_ddp import build_model


@pytest.fixture()
def checkpoint_path(tmp_path: Path) -> Path:
    path = tmp_path / "latest.pt"
    model = build_model()
    torch.save({"epoch": 7, "model_state_dict": model.state_dict()}, path)
    return path


@pytest.fixture()
def client(checkpoint_path: Path) -> TestClient:
    service = ModelService(checkpoint_path=str(checkpoint_path), device="cpu")
    app = create_app(service)
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_loaded_checkpoint(client: TestClient, checkpoint_path: Path) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["checkpoint_path"] == str(checkpoint_path)
    assert body["model_epoch"] == 7
    assert body["load_error"] is None


def test_live_reports_app_process_without_model_load(tmp_path: Path) -> None:
    service = ModelService(checkpoint_path=str(tmp_path / "missing.pt"), device="cpu")
    app = create_app(service)

    with TestClient(app) as test_client:
        live_response = test_client.get("/live")
        health_response = test_client.get("/health")

    assert live_response.status_code == 200
    assert live_response.json() == {"status": "alive"}
    assert health_response.status_code == 200
    body = health_response.json()
    assert body["status"] == "not_loaded"
    assert body["model_loaded"] is False
    assert "Checkpoint not found" in body["load_error"]


def test_predict_returns_cifar10_prediction_shape(client: TestClient) -> None:
    image = [[[0.5, 0.5, 0.5] for _ in range(32)] for _ in range(32)]

    response = client.post("/predict", json={"image": image, "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["class_id"] < len(CIFAR10_CLASSES)
    assert body["class_label"] in CIFAR10_CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["probabilities"]) == 3
    assert body["model_epoch"] == 7


def test_predict_rejects_wrong_image_shape(client: TestClient) -> None:
    response = client.post("/predict", json={"image": [[0.0]], "top_k": 1})

    assert response.status_code == 422
    assert "shape" in response.json()["detail"]

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.serving.model import ModelService


class PredictRequest(BaseModel):
    image: Any = Field(..., description="CIFAR-10 image as [32, 32, 3] or [3, 32, 32] numeric array.")
    top_k: int = Field(default=3, ge=1, le=10)


class ClassProbability(BaseModel):
    class_id: int
    class_label: str
    confidence: float


class PredictResponse(BaseModel):
    class_id: int
    class_label: str
    confidence: float
    probabilities: list[ClassProbability]
    model_epoch: int | None


def create_app(model_service: ModelService | None = None) -> FastAPI:
    service = model_service or ModelService()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            service.load()
        except FileNotFoundError:
            pass
        app.state.model_service = service
        yield

    app = FastAPI(title="LLM Training Pipeline Model Serving API", version="0.1.0", lifespan=lifespan)

    @app.get("/live")
    def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return service.health()

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        try:
            prediction = service.predict(request.image, top_k=request.top_k)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return PredictResponse(
            class_id=prediction.class_id,
            class_label=prediction.class_label,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            model_epoch=service.model_epoch,
        )

    return app


app = create_app()

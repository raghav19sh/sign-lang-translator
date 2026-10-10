"""Local FastAPI inference service for ASL Bridge.

Raw camera frames never reach this service. The frontend may send a normalized
63-number landmark vector only after the user enables the local API connection.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.environ.get("ASL_MODEL_PATH", str(ROOT / "models" / "sign_classifier.joblib")))
ALLOWED_ORIGINS = os.environ.get("ASL_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

app = FastAPI(
    title="ASL Bridge Inference API",
    version="1.0.0",
    description="Optional local classifier API. Accepts normalized landmarks, never video frames.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_model = None
_model_error: str | None = None


def load_model() -> None:
    global _model, _model_error
    if not MODEL_PATH.exists():
        _model = None
        _model_error = "No trained classifier found. Run python ml/train.py after exporting samples."
        return
    try:
        candidate = joblib.load(MODEL_PATH)
        if not hasattr(candidate, "predict"):
            raise ValueError("Model artifact does not implement predict().")
        _model = candidate
        _model_error = None
    except Exception as exc:  # log model load failures without exposing file details to clients
        _model = None
        _model_error = f"Model could not be loaded ({type(exc).__name__})."


load_model()


class FeatureRequest(BaseModel):
    features: list[float] = Field(min_length=63, max_length=63)

    @field_validator("features")
    @classmethod
    def features_must_be_finite(cls, values: list[float]) -> list[float]:
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Features must be finite numbers.")
        return values


@app.get("/api/v1/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "asl-bridge-inference",
        "model_loaded": _model is not None,
        "feature_schema": "wrist-relative-63-v1",
    }


@app.get("/api/v1/model-info")
def model_info() -> dict[str, object]:
    metadata_path = ROOT / "models" / "evaluation.json"
    metadata = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = None
    return {
        "model_loaded": _model is not None,
        "feature_schema": "wrist-relative-63-v1",
        "metadata": metadata,
        "message": None if _model is not None else _model_error,
    }


@app.post("/api/v1/recognize")
def recognize(payload: FeatureRequest) -> dict[str, object]:
    if _model is None:
        return {"status": "model_unavailable", "prediction": None, "detail": _model_error}
    try:
        prediction = str(_model.predict([payload.features])[0])
        confidence = None
        status: Literal["recognized", "uncertain"] = "recognized"
        if hasattr(_model, "predict_proba"):
            probabilities = _model.predict_proba([payload.features])[0]
            confidence = float(max(probabilities))
            if confidence < 0.55:
                status = "uncertain"
        return {
            "status": status,
            "prediction": {"label": prediction, "confidence": confidence if confidence is not None else 0.0},
        }
    except Exception as exc:
        return {"status": "inference_error", "prediction": None, "detail": type(exc).__name__}

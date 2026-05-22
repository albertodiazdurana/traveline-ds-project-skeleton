"""FastAPI serving layer for the rebooking model.

Loads the joblib-serialized (transformer, model, metadata) bundle at
startup via FastAPI's lifespan, stores it on app.state, and exposes:

    GET  /health    -> service liveness + model metadata
    POST /predict   -> probability and thresholded label for one booking

Run from repo root with the project venv active:

    uvicorn rebooking.api.main:app --reload

Requires that training has produced artifacts/model.joblib:

    python -m rebooking.models.train
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import AsyncIterator, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

ARTIFACT_PATH = Path("artifacts/model.joblib")
PREDICT_THRESHOLD = 0.5


class BookingRequest(BaseModel):
    """One booking record. Fields mirror the training schema; the loader
    drops customer_id and the transformer drops nothing else."""

    customer_id: int
    age: int = Field(ge=0, le=120)
    destination: str
    booking_channel: Literal["web", "app", "agent", "phone"]
    budget: float = Field(ge=0.0)
    days_since_last_booking: float | None
    booking_month: int = Field(ge=1, le=12)
    device_type: Literal["mobile", "desktop", "tablet"]
    loyalty_tier: Literal["none", "silver", "gold", "platinum"]
    prior_complaints: int = Field(ge=0)
    party_size: int = Field(ge=1)


class PredictionResponse(BaseModel):
    customer_id: int
    rebook_probability: float = Field(ge=0.0, le=1.0)
    rebook_predicted: bool
    model_version: str
    trained_at: str
    served_at: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
    trained_at: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model bundle once at startup; release on shutdown."""
    bundle = joblib.load(ARTIFACT_PATH)
    app.state.transformer = bundle["transformer"]
    app.state.model = bundle["model"]
    app.state.metadata = bundle["metadata"]
    yield
    # No teardown needed; objects are GC'd on shutdown.


app = FastAPI(title="rebooking", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    meta = request.app.state.metadata
    return HealthResponse(
        status="ok",
        model_version=meta["rebooking_version"],
        trained_at=meta["trained_at"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(req: BookingRequest, request: Request) -> PredictionResponse:
    transformer = request.app.state.transformer
    model = request.app.state.model
    meta = request.app.state.metadata

    # Transformer expects a DataFrame with the training schema (minus
    # rebooked, which the loader strips). customer_id passes through;
    # the transformer ignores it because it's not in CATEGORICAL/NUMERIC.
    row = pd.DataFrame([req.model_dump()])
    X = transformer.transform(row)
    prob = float(model.predict_proba(X)[0, 1])

    return PredictionResponse(
        customer_id=req.customer_id,
        rebook_probability=prob,
        rebook_predicted=prob >= PREDICT_THRESHOLD,
        model_version=meta["rebooking_version"],
        trained_at=meta["trained_at"],
        served_at=datetime.now(UTC).isoformat(),
    )

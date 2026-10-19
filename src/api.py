"""FastAPI service that serves the trained trip-duration model.

Scope: this is a historical replay. `trip_distance` is the taximeter distance
measured during the trip, so the API predicts the duration of a completed trip,
not an estimate available at pickup.

Run locally:
    uv run uvicorn src.api:app --port 8000
"""

import csv
import json
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from time import perf_counter

import joblib
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from src import config
from src.preprocess import MAX_DISTANCE

HTTP_REQUESTS = Counter(
    "taxi_api_http_requests_total",
    "HTTP requests handled by the API.",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "taxi_api_http_request_duration_seconds",
    "End-to-end HTTP request latency.",
    ["path"],
)
PREDICTED_DURATION = Histogram(
    "taxi_api_predicted_duration_minutes",
    "Distribution of predicted trip durations.",
    buckets=(2, 5, 10, 15, 20, 30, 45, 60, 90),
)
TRIP_DISTANCE = Histogram(
    "taxi_api_trip_distance_miles",
    "Distribution of trip_distance received in prediction requests.",
    buckets=(0.5, 1, 2, 3, 5, 8, 12, 20, 50),
)
MODEL_INFO = Gauge(
    "taxi_api_model_info", "Currently loaded model (value is always 1).", ["run_id"]
)

_log_lock = threading.Lock()
_state: dict = {}


def load_model() -> None:
    if not config.MODEL_PATH.exists():
        raise RuntimeError("No trained model found. Run `uv run dvc repro` first.")
    _state["model"] = joblib.load(config.MODEL_PATH)
    _state["metadata"] = json.loads(config.MODEL_METADATA_PATH.read_text())
    MODEL_INFO.labels(run_id=_state["metadata"]["run_id"]).set(1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="Green Taxi Trip Duration API", version="1.0.0", lifespan=lifespan)


class Trip(BaseModel):
    PULocationID: int = Field(..., ge=1, le=265, description="TLC pickup zone", examples=[74])
    DOLocationID: int = Field(..., ge=1, le=265, description="TLC dropoff zone", examples=[75])
    trip_distance: float = Field(
        ...,
        gt=0,
        le=MAX_DISTANCE,
        allow_inf_nan=False,
        description="Taximeter distance in miles (measured after the trip)",
        examples=[1.8],
    )


class Prediction(BaseModel):
    duration_minutes: float
    model_run_id: str


@app.middleware("http")
async def track_requests(request: Request, call_next):
    start = perf_counter()
    response = await call_next(request)
    path = request.url.path
    if path != "/metrics":
        HTTP_LATENCY.labels(path=path).observe(perf_counter() - start)
        HTTP_REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    return response


def log_prediction(trip: Trip, duration: float) -> None:
    """Append inputs + prediction to the CSV that feeds the Evidently report."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **trip.model_dump(),
        "prediction": round(duration, 3),
    }
    path = config.REQUEST_LOG_PATH
    with _log_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists()
        with path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if is_new:
                writer.writeheader()
            writer.writerow(row)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_run_id": _state["metadata"]["run_id"]}


@app.get("/model")
def model_info() -> dict:
    return _state["metadata"]


@app.post("/predict", response_model=Prediction)
def predict(trip: Trip) -> Prediction:
    record = {
        "PULocationID": str(trip.PULocationID),
        "DOLocationID": str(trip.DOLocationID),
        "trip_distance": trip.trip_distance,
    }
    duration = float(_state["model"].predict([record])[0])
    PREDICTED_DURATION.observe(duration)
    TRIP_DISTANCE.observe(trip.trip_distance)
    log_prediction(trip, duration)
    return Prediction(duration_minutes=round(duration, 2), model_run_id=_state["metadata"]["run_id"])


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

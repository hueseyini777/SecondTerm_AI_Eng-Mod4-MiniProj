"""Shared paths and parameters for every pipeline step."""

from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT_DIR / "params.yaml"

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
REFERENCE_DATA_PATH = ROOT_DIR / "data" / "reference" / "reference.parquet"
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
MODEL_METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"
METRICS_PATH = ROOT_DIR / "reports" / "train_metrics.json"
REQUEST_LOG_PATH = ROOT_DIR / "monitoring" / "requests.csv"
DRIFT_REPORT_PATH = ROOT_DIR / "monitoring" / "reports" / "drift_report.html"

CATEGORICAL_FEATURES = ["PULocationID", "DOLocationID"]
NUMERICAL_FEATURES = ["trip_distance"]
FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET = "duration"


def load_params() -> dict[str, Any]:
    """Read params.yaml, the single source of truth tracked by DVC."""
    with PARAMS_PATH.open() as stream:
        return yaml.safe_load(stream)


def raw_data_path(params: dict[str, Any] | None = None) -> Path:
    """Location of the monthly TLC parquet file selected in params.yaml."""
    data = (params or load_params())["data"]
    filename = f"{data['color']}_tripdata_{data['year']}-{data['month']:02d}.parquet"
    return RAW_DATA_DIR / filename

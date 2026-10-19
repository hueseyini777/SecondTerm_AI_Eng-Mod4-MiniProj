"""Train the trip-duration model, save it locally and log the run to MLflow.

Usage (from the repository root):
    uv run python -m src.train
"""

import json
import logging
import os
import platform

import joblib
import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

from src.config import (
    FEATURES,
    METRICS_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    REFERENCE_DATA_PATH,
    TARGET,
    load_params,
    raw_data_path,
)
from src.preprocess import preprocess, to_records

logging.getLogger("mlflow.utils").setLevel(logging.WARNING)


def build_model() -> Pipeline:
    return make_pipeline(DictVectorizer(), LinearRegression())


def evaluate(y_true: pd.Series, y_pred) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def register(model_uri: str, run_id: str, name: str, alias: str) -> str:
    """Register the logged model and move the alias to the new version."""
    client = MlflowClient()
    try:
        client.create_registered_model(
            name=name,
            tags={"framework": "scikit-learn"},
            description="Predicts NYC green-taxi ride duration (historical replay).",
        )
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
    version = client.create_model_version(name=name, source=model_uri, run_id=run_id)
    client.set_registered_model_alias(name=name, alias=alias, version=version.version)
    return version.version


def main() -> dict:
    load_dotenv()
    params = load_params()
    data_cfg, train_cfg, mlflow_cfg = params["data"], params["train"], params["mlflow"]

    df = preprocess(pd.read_parquet(raw_data_path(params)))
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_cfg["test_size"], random_state=train_cfg["random_state"]
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name=f"linreg-{data_cfg['year']}-{data_cfg['month']:02d}") as run:
        mlflow.set_tags(
            {
                "model": "linear regression",
                "dataset": f"{data_cfg['color']}-taxi",
                "year": data_cfg["year"],
                "month": data_cfg["month"],
                "features": FEATURES,
                "target": TARGET,
                "trigger": os.getenv("GITHUB_EVENT_NAME", "local"),
                "git_sha": os.getenv("GITHUB_SHA", "unknown"),
            }
        )
        mlflow.log_params(
            {
                **train_cfg,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "python": platform.python_version(),
            }
        )

        model = build_model()
        model.fit(to_records(X_train), y_train)
        test_pred = model.predict(to_records(X_test))
        metrics = evaluate(y_test, test_pred)
        mlflow.log_metrics(metrics)

        # Persist local artifacts for the API and DVC.
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        metadata = {
            "run_id": run.info.run_id,
            "features": FEATURES,
            "target": TARGET,
            "trained_on": f"{data_cfg['color']} {data_cfg['year']}-{data_cfg['month']:02d}",
            "metrics": metrics,
        }
        MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))

        # Held-out rows + model predictions = baseline for Evidently drift checks.
        reference = X_test.assign(**{TARGET: y_test, "prediction": test_pred})
        reference = reference.sample(
            n=min(train_cfg["reference_rows"], len(reference)),
            random_state=train_cfg["random_state"],
        )
        REFERENCE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        reference.to_parquet(REFERENCE_DATA_PATH, index=False)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="mlflow-model",
            serialization_format="skops",
        )
        mlflow.log_artifact(str(MODEL_METADATA_PATH), artifact_path="metadata")
        version = register(
            model_info.model_uri,
            run.info.run_id,
            mlflow_cfg["registered_model_name"],
            mlflow_cfg["alias"],
        )

    print(json.dumps(metrics, indent=2))
    print(
        f"Registered {mlflow_cfg['registered_model_name']} v{version} "
        f"as '{mlflow_cfg['alias']}' (run {run.info.run_id})"
    )
    return metrics


if __name__ == "__main__":
    main()

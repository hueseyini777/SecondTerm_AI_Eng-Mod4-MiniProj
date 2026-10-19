"""Compare logged API traffic with the training reference using Evidently.

    uv run python -m src.drift_report             # all logged requests
    uv run python -m src.drift_report --last 300  # only the most recent batch
"""

import argparse

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from src.config import (
    CATEGORICAL_FEATURES,
    DRIFT_REPORT_PATH,
    NUMERICAL_FEATURES,
    REFERENCE_DATA_PATH,
    REQUEST_LOG_PATH,
)

COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + ["prediction"]
DEFINITION = DataDefinition(
    categorical_columns=CATEGORICAL_FEATURES,
    numerical_columns=NUMERICAL_FEATURES + ["prediction"],
)


def as_dataset(df: pd.DataFrame) -> Dataset:
    df = df[COLUMNS].copy()
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(str)
    return Dataset.from_pandas(df, data_definition=DEFINITION)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=None, help="use only the last N requests")
    args = parser.parse_args()

    if not REQUEST_LOG_PATH.exists():
        raise SystemExit("No request log yet - send traffic with `python -m src.send_requests`.")

    current = pd.read_csv(REQUEST_LOG_PATH)
    if args.last:
        current = current.tail(args.last)
    if len(current) < 30:
        raise SystemExit(f"Only {len(current)} logged requests; send at least 30 first.")
    reference = pd.read_parquet(REFERENCE_DATA_PATH)

    # Evidently's defaults for >1000 reference rows (Wasserstein / Jensen-Shannon,
    # threshold 0.1) flag drift on a plain 300-row resample of the reference,
    # because ~260 pickup zones are sparse. Hypothesis tests (p < 0.05) don't.
    report = Report(
        [DataDriftPreset(num_method="ks", cat_method="chisquare"), DataSummaryPreset()]
    )
    snapshot = report.run(reference_data=as_dataset(reference), current_data=as_dataset(current))

    DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(DRIFT_REPORT_PATH))

    metrics = snapshot.dict()["metrics"]
    drift = next(m for m in metrics if m["metric_name"].startswith("DriftedColumnsCount"))
    for m in metrics:
        if m["metric_name"].startswith("ValueDrift"):
            column = m["metric_name"].split("column=")[1].split(",")[0]
            print(f"  {column:<14} p-value {float(m['value']):.4f}")
    print(
        f"Compared {len(current)} requests to {len(reference)} reference rows: "
        f"{drift['value']['count']:.0f} of {len(COLUMNS)} columns drifted "
        f"(share {drift['value']['share']:.0%})."
    )
    print(f"Report written to {DRIFT_REPORT_PATH}")


if __name__ == "__main__":
    main()

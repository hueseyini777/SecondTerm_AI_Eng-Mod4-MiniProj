"""Feature engineering shared by training and serving.

The rules mirror the original notebook so refactored runs stay comparable
with the notebook baseline.
"""

import pandas as pd

from src.config import CATEGORICAL_FEATURES, FEATURES, TARGET

MIN_DURATION, MAX_DURATION = 1, 60  # minutes
MIN_DISTANCE, MAX_DISTANCE = 0.1, 100  # miles


def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Trip duration in minutes from the green-taxi pickup/dropoff timestamps."""
    df = df.copy()
    elapsed = df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    df[TARGET] = elapsed.dt.total_seconds() / 60
    return df


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    keep = df[TARGET].between(MIN_DURATION, MAX_DURATION) & df[
        "trip_distance"
    ].between(MIN_DISTANCE, MAX_DISTANCE)
    return df.loc[keep]


def cast_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Zone IDs are labels, not quantities: one-hot them via strings."""
    df = df.copy()
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(str)
    return df


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw TLC rows -> clean feature/target frame."""
    df = filter_outliers(add_duration(raw))
    return cast_categories(df)[FEATURES + [TARGET]].reset_index(drop=True)


def to_records(features: pd.DataFrame) -> list[dict]:
    """DictVectorizer input: one dict per trip."""
    return cast_categories(features[FEATURES]).to_dict(orient="records")

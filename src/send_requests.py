"""Replay trips against the running API to populate metrics and the request log.

    uv run python -m src.send_requests --n 300                 # baseline batch
    uv run python -m src.send_requests --n 300 --shift 2.5     # distances x2.5
"""

import argparse

import pandas as pd
import requests

from src.config import FEATURES, REFERENCE_DATA_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=200, help="number of requests")
    parser.add_argument(
        "--shift",
        type=float,
        default=1.0,
        help="multiply trip_distance by this factor to simulate drift",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    trips = pd.read_parquet(REFERENCE_DATA_PATH)[FEATURES]
    trips = trips.sample(n=args.n, replace=args.n > len(trips), random_state=args.seed)
    trips["trip_distance"] = (trips["trip_distance"] * args.shift).clip(upper=100)

    predictions = []
    with requests.Session() as session:
        for trip in trips.to_dict(orient="records"):
            payload = {
                "PULocationID": int(trip["PULocationID"]),
                "DOLocationID": int(trip["DOLocationID"]),
                "trip_distance": round(float(trip["trip_distance"]), 2),
            }
            response = session.post(f"{args.url}/predict", json=payload, timeout=5)
            response.raise_for_status()
            predictions.append(response.json()["duration_minutes"])

    mean = sum(predictions) / len(predictions)
    print(f"Sent {len(predictions)} requests (shift x{args.shift}); mean prediction {mean:.2f} min")


if __name__ == "__main__":
    main()

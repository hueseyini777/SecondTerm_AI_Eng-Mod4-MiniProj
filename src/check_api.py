"""In-process smoke test of the API, used by CI. Does not touch the real request log."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from src import api, config


def main() -> None:
    trip = {"PULocationID": 74, "DOLocationID": 75, "trip_distance": 1.8}
    with TemporaryDirectory() as tmp:
        config.REQUEST_LOG_PATH = Path(tmp) / "requests.csv"
        with TestClient(api.app) as client:
            assert client.get("/health").status_code == 200

            response = client.post("/predict", json=trip)
            assert response.status_code == 200, response.text
            minutes = response.json()["duration_minutes"]
            assert 0 < minutes < 120, minutes

            bad = client.post("/predict", json={**trip, "trip_distance": -3})
            assert bad.status_code == 422

            assert "taxi_api_http_requests_total" in client.get("/metrics").text

        with config.REQUEST_LOG_PATH.open(newline="") as stream:
            assert len(list(csv.DictReader(stream))) == 1

    print(f"API smoke test passed (sample prediction: {minutes} min)")


if __name__ == "__main__":
    main()

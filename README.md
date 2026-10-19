# Green Taxi Trip Duration – MLOps Pipeline

_Original assignment instructions: [COURSE-README.md](COURSE-README.md) · brief: [project-description.md](project-description.md)_

The notebook model ([01-train-ml-model.ipynb](01-train-ml-model.ipynb)) refactored into a reproducible pipeline:
DVC-versioned data → training with MLflow tracking → GitHub Actions retraining → FastAPI serving → Prometheus/Grafana + Evidently monitoring.

> **Scope:** historical replay. `trip_distance` is the taximeter distance measured *during* the trip, so the API predicts the duration of a completed trip, not an estimate at pickup.

## Layout

| Path | Purpose |
|---|---|
| `params.yaml` | Data month, split, MLflow names – the only place to change settings |
| `dvc.yaml` / `dvc.lock` | Pipeline `download → train`, with hashes of data, code and outputs |
| `src/config.py` | Paths and feature lists shared by all steps |
| `src/download_data.py` | Fetches the TLC parquet for the month in `params.yaml` |
| `src/preprocess.py` | Duration target, outlier filter, zone IDs as categories (same rules as notebook) |
| `src/train.py` | Trains `DictVectorizer + LinearRegression`, logs to MLflow, registers `green-taxi-ride-duration@production` |
| `src/api.py` | FastAPI: `/predict`, `/health`, `/model`, `/metrics` |
| `src/check_api.py` | In-process API smoke test (used in CI) |
| `src/send_requests.py` | Replays held-out trips against the API, optionally with shifted distances |
| `src/drift_report.py` | Evidently report: logged requests vs. training reference |
| `monitoring/` | Prometheus config, Grafana datasource + dashboard |
| `.github/workflows/train.yaml` | CI retraining workflow |

## Run it

```bash
uv sync
cp .env.example .env            # MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# 1. Download data + train (DVC skips stages whose inputs did not change)
uv run dvc repro
uv run dvc metrics show

# 2. Inspect runs and the registered model
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# 3. Serve the model
uv run uvicorn src.api:app --port 8000
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"PULocationID": 74, "DOLocationID": 75, "trip_distance": 1.8}'
# {"duration_minutes":10.55,"model_run_id":"..."}
```

Invalid input (e.g. negative distance, zone outside 1–265) returns HTTP 422 and is not logged.

### Monitoring stack (Docker)

Stop the local API first (port 8000), then:

```bash
docker compose up --build -d
```

| Service | URL |
|---|---|
| API docs | <http://127.0.0.1:8000/docs> |
| Prometheus | <http://127.0.0.1:9090/targets> |
| Grafana (`admin`/`admin`) | <http://127.0.0.1:3000> → dashboard **Green Taxi API** |

The model in `models/` is mounted into the container, so after retraining just `docker compose restart api`.

Exported metrics: request count by path/status, request latency histogram, predicted-duration histogram, request `trip_distance` histogram, and the loaded model's MLflow run id.
The dashboard shows request rate, p50/p95 latency, mean predicted duration, mean request distance, totals, rejected requests and the prediction distribution.

### Drift report (Evidently)

```bash
uv run python -m src.send_requests --n 300                          # baseline batch
uv run python -m src.drift_report --last 300
uv run python -m src.send_requests --n 300 --shift 2.5 --seed 1     # distances x2.5
uv run python -m src.drift_report --last 300
```

The report (`monitoring/reports/drift_report.html`) compares the API request log (`monitoring/requests.csv`) to `data/reference/reference.parquet`: 5,000 held-out test rows with the model's predictions, written by the train stage.

## Results

Training on green taxi January 2024 (test split 20 %):

| Metric | Value |
|---|---|
| RMSE | 5.07 min |
| MAE | 3.45 min |
| R² | 0.665 |

Drift experiment (300 requests each, KS test for numeric columns, chi-square for zones, p < 0.05):

| Column | Baseline batch p-value | Shifted batch (distance x2.5) p-value |
|---|---|---|
| `trip_distance` | 0.45 | 0.0000 → drift |
| `prediction` | 0.69 | 0.0000 → drift |
| `PULocationID` | 0.54 | 0.19 |
| `DOLocationID` | 0.36 | 0.40 |
| **Drifted columns** | **0 / 4** | **2 / 4 → dataset drift** |

Interpretation: only distance was manipulated, and the report finds exactly that. Distance drifts, and the prediction drifts with it because the linear model is dominated by distance (mean prediction rose from 14.1 to 22.8 min). The zone mix is unchanged. Grafana shows the same shift live in the *Mean trip_distance* and *Mean predicted duration* panels.
There are no ground-truth labels in the request log, so drift is a signal to investigate, not proof the model got worse. The appropriate response would be to set `data.month` in `params.yaml` to a newer month and let the workflow retrain.

Note on method choice: with a reference of more than 1,000 rows, Evidently defaults to Wasserstein / Jensen-Shannon distance with a 0.1 threshold. With those defaults the *baseline* batch (a plain resample of the reference) was flagged as 4/4 drifted, because ~260 pickup zones are too sparse for 300 requests. Switching to hypothesis tests removed those false alarms.

## CI

`.github/workflows/train.yaml` runs on pushes that touch code, params or dependencies, on pull requests, monthly, and manually. It uses Python 3.13 with `uv sync --locked`, runs `dvc repro` (downloads the public TLC data, trains, logs to MLflow), prints metrics to the job summary, runs the API smoke test, and uploads the model and the MLflow store as a build artifact.

## Retraining with new data

```bash
# params.yaml → data.month: 2
uv run dvc repro        # re-downloads, retrains, registers a new version as @production
git add params.yaml dvc.lock && git commit -m "Retrain on Feb 2024"
```

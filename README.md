# NEXYGEN Net-Zero Emissions Forecasting Platform

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![CI](https://github.com/stephenogodo/Nexygen-Net-Zero-Project/actions/workflows/ci.yml/badge.svg)
![Status](https://img.shields.io/badge/Status-Active-success)

A full-stack, containerized time-series forecasting platform that predicts
**Scope 1 and Scope 2 carbon emissions** for Nexygen Energy, a UK energy
provider targeting net-zero operational emissions by 2040. It automatically
selects the better-performing model between SARIMA and Prophet per emission
scope, serves forecasts and scenario analysis through a FastAPI backend, and
exposes a Streamlit dashboard for forecasting, historical trends, and
net-zero target-gap tracking.

---

## Table of Contents

1. [Business Context](#business-context)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Data](#data)
5. [Pipeline Scripts Overview](#pipeline-scripts-overview)
6. [Data Prep Pipeline](#data-prep-pipeline)
7. [EDA Pipeline](#eda-pipeline)
8. [Training Pipeline](#training-pipeline)
9. [Model Selection: SARIMA vs. Prophet](#model-selection-sarima-vs-prophet)
10. [API Reference](#api-reference)
11. [Authentication](#authentication)
12. [Frontend (Streamlit)](#frontend-streamlit)
13. [Running the Project](#running-the-project)
14. [Testing](#testing)
15. [CI/CD](#cicd)
16. [Deployment](#deployment)
17. [Known Limitations & Future Work](#known-limitations--future-work)
18. [Presentation](#presentation)
19. [License](#license)

---

## Business Context

Nexygen Energy Ltd. is a mid-to-large UK energy provider (electricity + gas)
serving 1.5M+ customers across England, Scotland, and Wales, founded in 2009
and publicly committed to net-zero operational emissions by 2040. As its
renewable portfolio and regulatory/investor scrutiny grew, it needed to move
from *historical* emissions tracking to *forward-looking* forecasting --
predicting Scope 1 (direct: on-site fuel combustion, vehicles) and Scope 2
(indirect: purchased electricity) emissions, and quantifying progress
against its published reduction targets.

This project delivers that capability end-to-end: data validation →
candidate model training & selection → a forecasting/scenario/target-gap API
→ an interactive dashboard, all containerized and reproducible.

## Architecture

```
                    ┌─────────────────────┐
                    │   Streamlit UI      │
                    │  (Frontend :8501)   │
                    └──────────┬──────────┘
                               │ HTTP (X-API-Key if auth enabled)
                               ▼
                    ┌─────────────────────┐
                    │   FastAPI Backend   │
                    │      (:8000)        │
                    │  /forecast          │
                    │  /scenario          │
                    │  /historical        │
                    │  /target-gap        │
                    │  /model-info        │
                    │  /health            │
                    └──────────┬──────────┘
                               │ loads via model_manifest.json
                               ▼
                    ┌─────────────────────┐
                    │  models/            │
                    │  scope1_model.*     │ ← winner: SARIMA or Prophet
                    │  scope2_model.*     │ ← winner: SARIMA or Prophet
                    │  model_manifest.json│
                    └──────────▲──────────┘
                               │ produced by
                    ┌─────────────────────┐
                    │  pipeline/          │
                    │  data_prep_pipeline │ → validated train/test splits
                    │  train_pipeline     │ → fits both candidates, picks winner
                    └─────────────────────┘
                               ▲
                    ┌─────────────────────┐
                    │  data/ESG_Data.csv  │
                    └─────────────────────┘
```

Both services also run directly as local processes (no Docker) via
`run.py`; see [Running the Project](#running-the-project).

## Project Structure

```
Nexygen-Net-Zero-Project/
├── README.md
├── RUNBOOK.md                      # deployment/ops runbook
├── run.py                          # single entry point (local/docker × fastapi/streamlit)
├── ESG_Data.csv                    # source dataset
├── LICENSE
├── .github/workflows/ci.yml        # tests, pipeline smoke test, docker build
├── presentation/
│   └── NEXYGEN_Presentation.pptx   # combined exec/technical deck
├── docker-compose.yml
├── .env.example
├── backend/
│   ├── app.py                  # FastAPI app
│   ├── model_registry.py       # uniform SARIMA/Prophet loading + forecasting
│   ├── pipeline/
│   │   ├── data_prep_pipeline.py
│   │   ├── eda_report.py       # trend/seasonality/anomaly/driver analysis
│   │   └── train_pipeline.py
│   ├── data/
│   │   ├── ESG_Data.csv
│   │   └── processed/          # train/test splits + validation summary (generated)
│   ├── reports/                # eda_report.py output: 4 PNGs + findings.md (generated)
│   ├── models/
│   │   ├── model_manifest.json # which model won per scope, both candidates' metrics
│   │   ├── scope1_model.*      # .pkl (SARIMA) or .json (Prophet), whichever won
│   │   └── scope2_model.*
│   ├── tests/test_api.py       # 27 pytest cases
│   ├── dockerfile
│   ├── requirements.txt        # runtime deps (incl. prophet, statsmodels)
│   ├── requirements-train.txt  # + scikit-learn, pmdarima, matplotlib (training-only)
│   ├── requirements-dev.txt    # + pytest, httpx
│   └── .env.example
└── frontend/
    ├── streamlit_app.py        # Forecast / Scenario / Historical / Target-Gap tabs
    ├── dockerfile.streamlit
    ├── requirements.txt
    └── .env.example
```

## Data

`ESG_Data.csv`: 236,736 rows, daily granularity, 2020-01-01 to 2025-12-31,
18 assets (`A001`-`A018`) across 5 asset types, 3 UK regions, 3 energy types
(electricity/gas/diesel), split into Scope 1 and Scope 2 emission rows. Also
carries `Target_Emissions_tCO2e` and `Reduction_Percentage_vs_BaseYear` --
NEXYGEN's published annual reduction path (5.5%/year from a 2020 baseline).

**Known data caveat:** `Target_Emissions_tCO2e` is on a different absolute
scale than the summed per-asset actuals (roughly 3x off -- a data-generation
artifact). `/target-gap` deliberately compares **% reduction vs. baseline**
on both sides instead of the raw absolute figures, which sidesteps the scale
mismatch; see that endpoint's docstring in `app.py` for the full rationale.

## Pipeline Scripts Overview

The project originally used a single exploratory notebook
(`Preprocess.ipynb`); it has been replaced by three focused, runnable
pipeline scripts, all importable and CLI-runnable.

| Script | Purpose | Run |
|---|---|---|
| `data_prep_pipeline.py` | Validate the raw CSV, split into train/test | `python pipeline/data_prep_pipeline.py --input data/ESG_Data.csv --output data/processed` |
| `eda_report.py` | Trend, seasonality, anomaly, and driver analysis | `python pipeline/eda_report.py --input data/ESG_Data.csv --output reports` |
| `train_pipeline.py` | Fit SARIMA + Prophet, ship the better one per scope | `python pipeline/train_pipeline.py --input data/ESG_Data.csv --output models` |

## Data Prep Pipeline

**File:** `pipeline/data_prep_pipeline.py`

Loads and validates the raw CSV (required columns, nulls, duplicates),
aggregates it into monthly totals per scope, and splits each into
train/test at the 80th-percentile date (train ends 2024-10-01, matching the
API's forecast anchor date).

```bash
python pipeline/data_prep_pipeline.py --input data/ESG_Data.csv --output data/processed
```

Writes `{scope}_train.csv`, `{scope}_test.csv`, and `data_summary.json`
(row counts, date range, null/duplicate counts) to `--output`.

## EDA Pipeline

**File:** `pipeline/eda_report.py`

Trend, seasonality, anomaly, and driver analysis -- saved as PNGs + a
`findings.md` summary rather than an interactive notebook, for the same
reason the other two pipeline scripts are scripts: reproducible, diffable,
runnable in CI (see `pipeline-smoke-test` in CI/CD below).

```bash
python pipeline/eda_report.py --input data/ESG_Data.csv --output reports
```

Two methodology choices worth knowing about, because the naive version of
each gives a misleading answer on this data:
- **Anomaly detection** uses a 30-day rolling-median local baseline, not
  raw z-score against the whole series' mean. Scope 2 has a strong
  multi-year downward trend; raw z-score flags "high values from early in
  the series, before the trend declined" as anomalies -- that's trend
  drift, not a real local anomaly.
- **Seasonality strength** is reported as peak-to-trough swing as a % of
  the mean, not just the peak/trough month. Scope 1 and Scope 2 both
  numerically "peak" in June, but Scope 1's swing is 18.1% of its mean
  (genuinely seasonal) while Scope 2's is 1.4% (essentially flat) --
  reporting only the peak month would make these look equally seasonal
  when they aren't.

Key findings (full detail in `reports/findings.md`): Scope 1 emissions fell
6.3% and Scope 2 fell 37.6% from 2020-2025; Asset Type explains 59% of
emissions variance via a Random Forest driver analysis, more than Location
(25%) and Emission Type (16%) combined; `Operational_Status` is excluded
from driver analysis entirely -- it's always `"Active"` in this dataset, so
it has zero variance and can't explain anything.

## Training Pipeline

**File:** `pipeline/train_pipeline.py`

For each scope, fits **two candidate models** on the same train split and
evaluates both on the same held-out test split:

- **SARIMA/SARIMAX** (`statsmodels`) -- orders found via an `auto_arima`
  search, hardcoded as defaults for fast/reproducible runs
- **Prophet** -- hyperparameters found via a small grid search over
  `changepoint_prior_scale` × `seasonality_mode`, likewise hardcoded as
  defaults

The candidate with the higher **R²** on the test split is refit and shipped;
both candidates' full metrics are recorded either way.

```bash
python pipeline/train_pipeline.py --input data/ESG_Data.csv --output models
# add --search to re-run the hyperparameter search instead of using the defaults
```

Output: `models/{scope}_model.{pkl|json}` (statsmodels pickle for SARIMA,
Prophet's own JSON serialization for Prophet -- pickling isn't stable across
Prophet's Stan backend versions) and `models/model_manifest.json`:

```json
{
  "split_date": "2024-10-19",
  "scopes": {
    "scope1": {
      "model_type": "prophet",
      "model_file": "scope1_model.json",
      "params": {"changepoint_prior_scale": 0.1, "seasonality_mode": "additive"},
      "metrics": {"r2": 0.982, "mse": 286.2, "mae": 14.93},
      "candidates_evaluated": { "sarima": {...}, "prophet": {...} }
    },
    "scope2": { "...": "..." }
  }
}
```

## Model Selection: SARIMA vs. Prophet

Both candidates are fit on the identical 58-month train split and scored on
the identical 14-month held-out test split -- an apples-to-apples
comparison, not a default-settings fit for either side.

| Scope | SARIMA (best found) | Prophet (best found) | Winner |
|---|---|---|---|
| Scope 1 | SARIMAX(3,1,2)(2,1,0,12) — **R²=0.964**, MSE=566, MAE=20.6 | `cps=0.1`, additive — **R²=0.982**, MSE=286, MAE=14.9 | **Prophet** |
| Scope 2 | SARIMAX(0,1,1)(1,0,0,12) — **R²=0.792**, MSE=523, MAE=16.5 | `cps=0.05`, multiplicative — **R²=0.908**, MSE=232, MAE=9.2 | **Prophet** |

Prophet currently wins both scopes -- clearly on Scope 1, decisively on
Scope 2. Full detail (both candidates, all metrics) is always available live
at `GET /model-info`, not just in this table, so it stays accurate as the
pipeline is rerun on new data.

**Why not always assume Prophet wins:** the selection logic is genuinely
data-driven, not hardcoded to prefer either model -- if a future retrain on
different/more data flips the winner for either scope, `train_pipeline.py`
will pick it up automatically and `app.py` will serve whichever model type
won without any code changes, via `model_registry.py`'s uniform
`.forecast(steps)` interface over either model type.

**Verified, not assumed:** Prophet's `cmdstanpy` backend needs a compiled
Stan model, which typically requires a C++ toolchain -- but the PyPI wheel
ships a prebuilt binary, so it installs and runs cleanly on the bare
`python:3.11-slim` base image with no extra build tools. This was confirmed
by installing Prophet into a clean venv with the system C++ compiler hidden
from `PATH` and successfully fitting/predicting a model.

## API Reference

Interactive docs (Swagger UI) are always available at `/docs` once the
backend is running, e.g. `http://localhost:8000/docs`.

| Endpoint | Method | Auth | Purpose |
|---|---|---|---|
| `/` | GET | No | Liveness check |
| `/health` | GET | No | Readiness: model/data load status, active model type + R² per scope, auth state |
| `/forecast` | POST | Yes* | Forecast `emission_type` (`scope1`/`scope2`) for `steps` months ahead |
| `/scenario` | POST | Yes* | Forecast with an adjustable `annual_reduction_rate_pct` assumption layered on top |
| `/historical` | GET | Yes* | Historical emissions, filterable by scope/granularity/date range/asset/location |
| `/target-gap` | GET | Yes* | Actual vs. target % reduction vs. baseline year, per year, with an on-track verdict |
| `/model-info` | GET | Yes* | Full training manifest: winning model + both candidates' metrics, per scope |

\* Only enforced if `API_KEY` is set; see [Authentication](#authentication).

### `POST /forecast`

```json
// request
{"emission_type": "scope1", "steps": 6}
// response
{
  "emission_type": "scope1",
  "model_type": "prophet",
  "forecast": [1636.07, 1646.62, 1653.25, 1519.42, 1762.03, 1778.33],
  "dates": ["2024-11-01", "2024-12-01", "2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"],
  "last_training_date": "2024-10-01"
}
```

### `POST /scenario`

```json
// request
{"emission_type": "scope2", "steps": 12, "annual_reduction_rate_pct": 10}
// response includes both baseline_forecast and adjusted_forecast for the same dates
```

### `GET /target-gap?year=2024`

```json
[{
  "year": 2024, "emission_type": "all",
  "actual_emissions_tco2e": 37055.6, "target_emissions_tco2e": 11965.87,
  "actual_reduction_pct_vs_baseline": 17.68, "target_reduction_pct_vs_baseline": 20.25,
  "gap_pct": 2.57, "on_track": false
}]
```

As of the current dataset, **NEXYGEN is behind its published reduction path
every year since 2020**, and the gap has widened over time (1.1 points
behind in 2021 → 2.6 points behind in 2025) -- the kind of finding this
project exists to surface.

## Authentication

Opt-in, API-key based (`X-API-Key` header): unset `API_KEY` = open (default,
frictionless for local dev/CI); set it = enforced on every endpoint except
`/` and `/health`. See [RUNBOOK.md](RUNBOOK.md#enabling-auth) for the exact
steps. The Streamlit frontend forwards its own `API_KEY` env var
automatically -- no separate toggle needed.

This is intentionally a single shared key, not per-user identity/roles;
adequate for a small internal deployment. For multi-user access control,
replace the `require_api_key` dependency in `app.py` with a proper
OAuth2/JWT scheme.

## Frontend (Streamlit)

Four tabs, each backed by a live endpoint above:

- **📈 Forecast** -- pick a scope + horizon, see the projection and which
  model produced it
- **🧭 Scenario** -- slider for an assumed extra annual reduction rate;
  plots baseline vs. adjusted side by side
- **🗂️ Historical Data** -- filterable by scope/granularity/date range,
  multi-line trend chart
- **🎯 Target Gap** -- actual vs. target reduction-% line chart, an
  on-track/behind-target metric, and an explicit caveat about the
  absolute-vs-percentage scale issue in the source data

The sidebar shows the active model + R² per scope (from `/model-info`) and
an API health check button.

## Running the Project

Single entry point, all four combinations:

```bash
python run.py --env local  --app both        # uvicorn + streamlit as local subprocesses
python run.py --env local  --app fastapi      # backend only, local
python run.py --env local  --app streamlit    # frontend only, local (needs a backend already running)
python run.py --env docker --app both         # docker compose up --build (both services)
python run.py --env docker --app fastapi      # docker compose up --build backend
python run.py --env docker --app streamlit    # docker compose up --build frontend (also brings up backend)
python run.py --env docker --down             # docker compose down
```

Local mode expects dependencies already installed
(`pip install -r backend/requirements.txt` and
`frontend/requirements.txt`); Docker mode only needs Docker
itself. Both were tested directly: local `both` starts backend and
frontend concurrently and shuts both down cleanly on SIGINT/SIGTERM (a
bounded grace period, then a force-kill fallback); Docker mode
fails with a clear "install Docker" message rather than a stack trace when
Docker isn't present.

## Testing

```bash
cd backend
pip install -r requirements-dev.txt
pytest tests/ -v
```

27 test cases covering: health, forecast (incl. input validation bounds),
scenario (incl. the zero-rate-matches-baseline invariant), historical data
(filters, date ranges), target-gap (single year, all years, unknown year),
auth (both open and enforced modes, correct/wrong/missing key), and model
selection (`/health` and `/model-info` report a valid model type, and the
shipped model is verifiably the higher-R² candidate).

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`), three jobs on every push/PR:

1. **backend-tests** -- installs `requirements-dev.txt`, runs `pytest`
2. **pipeline-smoke-test** -- actually re-runs `data_prep_pipeline.py` and
   `train_pipeline.py` end-to-end and asserts a valid model was selected
   per scope, independent of whatever models happen to be committed
3. **docker-build** -- builds both Docker images and boots the backend
   container, polling `/health` until it reports healthy

Job 3 specifically would have caught an earlier real bug in this project
(a `dockerfile: Dockerfile` vs. actual lowercase `dockerfile` filename
mismatch that broke `docker compose up --build`) automatically, on every
push.

## Deployment

Full build/run/rollback/troubleshooting steps live in
**[RUNBOOK.md](RUNBOOK.md)**. Quick version:

```bash
python run.py --env docker --app both
curl http://localhost:8000/health
```

## Known Limitations & Future Work

- **No database** -- `ESG_Data.csv` is read directly at API startup. Fine at
  this data volume; a database becomes worth it if write-access or larger
  data volumes are needed.
- **No CI-driven auto-deploy** -- CI validates but does not deploy; adding
  cloud deployment (AWS/Azure/GCP) is a natural next step.
- **SHAP interpretability is intentionally not implemented** -- it doesn't
  natively apply to SARIMA or Prophet, which don't expose per-feature
  attribution in the way SHAP is built for. Would require a genuinely
  different, feature-based model (e.g. gradient-boosted regressor on lag
  and seasonal features) to add meaningfully rather than for show.
- **Single shared API key**, no per-user roles -- see
  [Authentication](#authentication).
- **Hyperparameter search is a small grid/auto-order search**, not a full
  AutoML sweep -- appropriate for two ~72-month monthly series, would need
  rethinking for a much larger model set.

## Presentation

`presentation/NEXYGEN_Presentation.pptx` -- a combined executive/technical
deck: business context, the target-gap finding, EDA (trend, seasonality,
anomalies, driver analysis), the SARIMA-vs-Prophet model bake-off,
architecture, API reference, deployment/CI, and known limitations. Every
number in it is pulled from this repo's actual pipeline output and live
API responses, not written from memory.

## License

MIT -- see [LICENSE](LICENSE).
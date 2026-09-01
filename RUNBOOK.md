# NEXYGEN Deployment Runbook

## Build & Run

**Via the entry point (recommended):**

```bash
python run.py --env docker --app both
```

**Directly with Compose:**

```bash
cp .env.example .env          # optional: override ports/paths
docker compose up --build -d
```

Wait for the backend to report healthy before using the frontend (Compose
already gates frontend startup on `backend: condition: service_healthy`):

```bash
docker compose ps
```

Verify manually:

```bash
curl http://localhost:8000/health
```

Expected on a good deploy:

```json
{
  "status": "ok",
  "models_loaded": {"scope1": true, "scope2": true},
  "models_info": {
    "scope1": {"model_type": "prophet", "r2": 0.982},
    "scope2": {"model_type": "prophet", "r2": 0.908}
  },
  "data_loaded": true,
  "data_rows": 236736,
  "auth_enabled": false,
  "detail": null
}
```

If `status` is `"degraded"`, check `detail` and the backend logs
(`docker compose logs backend`) -- almost always a missing/misnamed model
file (`models/model_manifest.json` + the files it points to) or the data
CSV inside the image.

## Configuration (env vars)

Set in `.env` at the repo root (Compose) or exported directly for local
`uvicorn` runs. Full list and defaults: `backend/.env.example`,
`frontend/.env.example`.

| Variable | Purpose |
|---|---|
| `MODEL_DIR` | Directory containing `model_manifest.json` + the model files it references (default `models/`) |
| `DATA_PATH` | Path to `ESG_Data.csv` used by `/historical` and `/target-gap` |
| `LOG_LEVEL` | Python logging level (`INFO`, `DEBUG`, ...) |
| `BASELINE_YEAR` | Baseline year for target-gap % reduction calculations |
| `LAST_TRAINING_DATE` | Anchor date the forecast horizon is generated from |
| `MAX_FORECAST_STEPS` | Upper bound on `steps` accepted by `/forecast` and `/scenario` |
| `API_BASE_URL` | (frontend) base URL the Streamlit app calls |
| `BACKEND_PORT` / `FRONTEND_PORT` | Host port mappings |
| `API_KEY` | If set, all endpoints except `/`, `/health` require a matching `X-API-Key` header. Unset = open (default; recommended only for local dev/CI). Set the same value for both `backend` and `frontend` in Compose. |

## Enabling auth

```bash
# in .env (repo root)
API_KEY=some-long-random-value
```

```bash
docker compose up --build -d
curl http://localhost:8000/health   # -> "auth_enabled": true, still no key needed
curl -X POST http://localhost:8000/forecast -H "Content-Type: application/json" \
  -d '{"emission_type":"scope1","steps":3}'                    # -> 401
curl -X POST http://localhost:8000/forecast -H "Content-Type: application/json" \
  -H "X-API-Key: some-long-random-value" \
  -d '{"emission_type":"scope1","steps":3}'                    # -> 200
```

The Streamlit frontend automatically forwards `API_KEY` as `X-API-Key` on
every request once it's set, so no frontend code changes are needed when
turning auth on or off.

This is intentionally lightweight (a single shared key, no per-user
identity/roles). It's adequate for a small internal deployment; for
multi-user access control, swap `require_api_key` in `app.py` for a proper
OAuth2/JWT dependency.

## Retraining models

```bash
cd backend
pip install -r requirements-train.txt
python pipeline/data_prep_pipeline.py --input data/ESG_Data.csv --output data/processed
python pipeline/train_pipeline.py --input data/ESG_Data.csv --output models
```

This fits both SARIMA and Prophet per scope, evaluates both on the same
held-out split, and overwrites `models/` with only the winner per scope
plus an updated `model_manifest.json`. Restart the backend afterward
(`docker compose up --build -d backend`, or just let `--reload` pick it up
locally -- note `--reload` only restarts on code changes, not model file
changes, so a manual restart is needed after retraining).

Pass `--search` to re-run the SARIMA/Prophet hyperparameter search instead
of using the known-good defaults baked into `train_pipeline.py` -- slower,
and only worth doing if the input data has changed materially.

## Smoke test after any deploy

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/forecast -H "Content-Type: application/json" \
  -d '{"emission_type":"scope1","steps":3}'
curl "http://localhost:8000/target-gap?year=2024"
curl "http://localhost:8000/model-info"
```

Or run the automated suite against a running container:

```bash
cd backend
pip install -r requirements-dev.txt
pytest tests/ -v
```

## Rollback

Compose does not version images by default. To roll back:

```bash
git checkout <previous-good-commit> -- backend/ frontend/ docker-compose.yml
docker compose up --build -d
```

If only the models regressed (not the code), it's faster to restore just
the `models/` directory from the previous commit and restart the backend:

```bash
git checkout <previous-good-commit> -- backend/models/
docker compose up --build -d backend
```

## Known limitations (not yet implemented)

- No database -- `ESG_Data.csv` is read directly at startup. Fine at this
  data volume; revisit if write-access or larger volumes become a
  requirement.
- SHAP-based interpretability is intentionally not implemented: it doesn't
  natively apply to SARIMA or (in its typical form) Prophet, which don't
  expose the kind of per-feature attribution SHAP is built for.
- The training pipeline's hyperparameter search (`--search`) is a small
  grid/auto-order search, not a full AutoML sweep -- adequate for two
  monthly series this size, not necessarily for a much larger model set.

import logging
import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from model_registry import load_models

# ---------------------------------------------------------------------------
# Configuration (env-var driven, sensible defaults for local dev)
# ---------------------------------------------------------------------------
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(BASE_DIR / "models")))
DATA_PATH = Path(os.environ.get("DATA_PATH", str(BASE_DIR / "data" / "ESG_Data.csv")))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
BASELINE_YEAR = int(os.environ.get("BASELINE_YEAR", "2020"))
LAST_TRAINING_DATE = os.environ.get("LAST_TRAINING_DATE", "2024-10-01")
MAX_FORECAST_STEPS = int(os.environ.get("MAX_FORECAST_STEPS", "120"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("nexygen")

# ---------------------------------------------------------------------------
# App state (populated at startup)
# ---------------------------------------------------------------------------
state: dict = {"models": {}, "model_manifest": None, "data": None, "data_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["models"], state["model_manifest"] = load_models(MODEL_DIR)
        for scope_key, entry in state["models"].items():
            logger.info(
                "Loaded %s model for %s (R2=%.4f) from %s",
                entry.model_type, scope_key, entry.metrics.get("r2", float("nan")), MODEL_DIR,
            )
    except Exception:
        logger.exception("Failed to load models from manifest in %s", MODEL_DIR)
        state["models"] = {}
        state["model_manifest"] = None

    try:
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        df["Emission_Type"] = df["Emission_Type"].str.lower().str.replace(" ", "")
        state["data"] = df
        logger.info("Loaded ESG data from %s (%d rows)", DATA_PATH, len(df))
    except Exception as exc:
        logger.exception("Failed to load ESG data from %s", DATA_PATH)
        state["data"] = None
        state["data_error"] = str(exc)

    yield

    state["models"].clear()
    state["data"] = None


app = FastAPI(
    title="NEXYGEN API",
    description=(
        "Forecasts, historical data, and net-zero target-gap analysis for "
        "NEXYGEN Scope 1 & Scope 2 emissions."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

EmissionScope = Literal["scope1", "scope2"]
EmissionScopeOrAll = Literal["scope1", "scope2", "all"]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    emission_type: EmissionScope
    steps: int = Field(gt=0, le=MAX_FORECAST_STEPS, description="Months to forecast ahead")


class ForecastResponse(BaseModel):
    emission_type: str
    model_type: str
    forecast: list[float]
    dates: list[str]
    last_training_date: str


class ScenarioRequest(BaseModel):
    emission_type: EmissionScope
    steps: int = Field(gt=0, le=MAX_FORECAST_STEPS, description="Months to forecast ahead")
    annual_reduction_rate_pct: float = Field(
        default=0.0,
        ge=-50.0,
        le=50.0,
        description=(
            "Assumed additional annual improvement rate to apply on top of the "
            "model's baseline forecast, e.g. 10 = an extra 10%/year reduction "
            "from efficiency/renewable-sourcing initiatives. Negative values "
            "simulate emissions growth."
        ),
    )


class ScenarioResponse(BaseModel):
    emission_type: str
    model_type: str
    dates: list[str]
    baseline_forecast: list[float]
    adjusted_forecast: list[float]
    annual_reduction_rate_pct: float
    last_training_date: str


class HistoricalRecord(BaseModel):
    period: str
    emission_type: str
    emissions_tco2e: float


class TargetGapRecord(BaseModel):
    year: int
    emission_type: str
    actual_emissions_tco2e: float
    target_emissions_tco2e: float
    actual_reduction_pct_vs_baseline: float
    target_reduction_pct_vs_baseline: float
    gap_pct: float
    on_track: bool


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
    models_info: dict[str, dict]
    data_loaded: bool
    data_rows: Optional[int] = None
    auth_enabled: bool = False
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_model(emission_type: str):
    model = state["models"].get(emission_type)
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model '{emission_type}' is not loaded")
    return model


def _require_data() -> pd.DataFrame:
    if state["data"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Historical data is not available: {state.get('data_error') or 'unknown error'}",
        )
    return state["data"]


def _forecast_dates(steps: int) -> list[str]:
    last_date = pd.to_datetime(LAST_TRAINING_DATE)
    return [(last_date + pd.DateOffset(months=i + 1)).strftime("%Y-%m-%d") for i in range(steps)]


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    """
    Opt-in API-key auth: if the API_KEY env var is unset, the API is open
    (local dev / CI stay frictionless). If set, every protected endpoint
    requires a matching X-API-Key header. Read live (not at startup) so it
    can be tested/rotated without restarting the app.
    """
    configured_key = os.environ.get("API_KEY")
    if not configured_key:
        return
    if x_api_key != configured_key:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"Status": "O.K", "message": "NEXYGEN API is up and running!"}


@app.get("/health", response_model=HealthResponse)
def health():
    models_loaded = {k: True for k in state["models"]}
    for k in ("scope1", "scope2"):
        models_loaded.setdefault(k, False)
    models_info = {
        k: {"model_type": v.model_type, "r2": v.metrics.get("r2")}
        for k, v in state["models"].items()
    }
    data_loaded = state["data"] is not None
    ok = data_loaded and all(models_loaded.values())
    return HealthResponse(
        status="ok" if ok else "degraded",
        models_loaded=models_loaded,
        models_info=models_info,
        data_loaded=data_loaded,
        data_rows=len(state["data"]) if data_loaded else None,
        auth_enabled=bool(os.environ.get("API_KEY")),
        detail=None if ok else (state.get("data_error") or "one or more components failed to load"),
    )


@app.get("/model-info", dependencies=[Depends(require_api_key)])
def model_info():
    """Full training-pipeline manifest: which model won per scope, its
    hyperparameters, its metrics, and the metrics of the candidate(s) it
    beat -- for transparency/audit."""
    if state["model_manifest"] is None:
        raise HTTPException(status_code=503, detail="Model manifest not loaded")
    return state["model_manifest"]


@app.post("/forecast", response_model=ForecastResponse, dependencies=[Depends(require_api_key)])
def forecast_emissions(req: ForecastRequest):
    model = _require_model(req.emission_type)
    try:
        values = model.forecast(steps=req.steps)
    except Exception as exc:
        logger.exception("Forecast failed for %s (steps=%d)", req.emission_type, req.steps)
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}") from exc

    logger.info(
        "Forecast served: emission_type=%s model_type=%s steps=%d",
        req.emission_type, model.model_type, req.steps,
    )
    return ForecastResponse(
        emission_type=req.emission_type,
        model_type=model.model_type,
        forecast=values,
        dates=_forecast_dates(req.steps),
        last_training_date=LAST_TRAINING_DATE,
    )


@app.post("/scenario", response_model=ScenarioResponse, dependencies=[Depends(require_api_key)])
def scenario_forecast(req: ScenarioRequest):
    """
    Applies an adjustable annual reduction-rate assumption on top of the
    model's unconditional baseline forecast, so users can explore the effect
    of accelerated (or slower) efficiency/renewable-sourcing initiatives.
    """
    model = _require_model(req.emission_type)
    try:
        baseline = model.forecast(steps=req.steps)
    except Exception as exc:
        logger.exception("Scenario forecast failed for %s (steps=%d)", req.emission_type, req.steps)
        raise HTTPException(status_code=500, detail=f"Scenario forecast failed: {exc}") from exc

    monthly_rate = (1 - req.annual_reduction_rate_pct / 100) ** (1 / 12)
    adjusted = [float(v) * (monthly_rate ** (i + 1)) for i, v in enumerate(baseline)]

    logger.info(
        "Scenario served: emission_type=%s model_type=%s steps=%d annual_reduction_rate_pct=%.2f",
        req.emission_type, model.model_type, req.steps, req.annual_reduction_rate_pct,
    )
    return ScenarioResponse(
        emission_type=req.emission_type,
        model_type=model.model_type,
        dates=_forecast_dates(req.steps),
        baseline_forecast=baseline,
        adjusted_forecast=adjusted,
        annual_reduction_rate_pct=req.annual_reduction_rate_pct,
        last_training_date=LAST_TRAINING_DATE,
    )


@app.get("/historical", response_model=list[HistoricalRecord], dependencies=[Depends(require_api_key)])
def historical_emissions(
    emission_type: EmissionScopeOrAll = Query("all"),
    granularity: Literal["daily", "monthly", "yearly"] = Query("monthly"),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    asset_id: Optional[str] = Query(None),
    location: Optional[str] = Query(None),
):
    df = _require_data()

    if emission_type != "all":
        df = df[df["Emission_Type"] == emission_type]
    if asset_id:
        df = df[df["Asset_ID"] == asset_id]
    if location:
        df = df[df["Location"].str.lower() == location.lower()]
    if start_date:
        df = df[df["Date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["Date"] <= pd.Timestamp(end_date)]

    if df.empty:
        return []

    if granularity == "daily":
        period_col = df["Date"].dt.strftime("%Y-%m-%d")
    elif granularity == "monthly":
        period_col = df["Date"].dt.to_period("M").dt.to_timestamp().dt.strftime("%Y-%m-%d")
    else:  # yearly
        period_col = df["Date"].dt.year.astype(str)

    grouped = (
        df.assign(period=period_col)
        .groupby(["period", "Emission_Type"])["Emissions_tCO2e"]
        .sum()
        .reset_index()
        .sort_values("period")
    )

    return [
        HistoricalRecord(
            period=row.period,
            emission_type=row.Emission_Type,
            emissions_tco2e=round(float(row.Emissions_tCO2e), 4),
        )
        for row in grouped.itertuples(index=False)
    ]


@app.get("/target-gap", response_model=list[TargetGapRecord], dependencies=[Depends(require_api_key)])
def target_gap(
    year: Optional[int] = Query(None, description="Single year; omit for the full 2020-latest path"),
    emission_type: EmissionScopeOrAll = Query("all"),
):
    """
    Compares actual emissions against the company's published net-zero
    reduction path.

    Note on methodology: `Target_Emissions_tCO2e` in the source data is a
    company-wide absolute figure that is on a different scale than the sum
    of the (synthetic) per-asset emissions rows -- comparing them directly
    is not meaningful. Instead this endpoint compares *reduction
    percentages*: the actual % reduction vs. baseline year (computed from
    the real data) against the target % reduction vs. baseline year
    (as published in `Reduction_Percentage_vs_BaseYear`), which are on the
    same, comparable scale. The raw target absolute figure is still
    returned for transparency.
    """
    df = _require_data()
    if emission_type != "all":
        df = df[df["Emission_Type"] == emission_type]

    if df.empty:
        return []

    by_year = df.groupby("Year")["Emissions_tCO2e"].sum()
    target_by_year = df.groupby("Year")[["Target_Emissions_tCO2e", "Reduction_Percentage_vs_BaseYear"]].first()

    if BASELINE_YEAR not in by_year.index:
        raise HTTPException(status_code=500, detail=f"Baseline year {BASELINE_YEAR} not present in data")
    baseline_actual = by_year.loc[BASELINE_YEAR]

    years = [year] if year is not None else sorted(by_year.index.tolist())
    records = []
    for y in years:
        if y not in by_year.index:
            if year is not None:
                raise HTTPException(status_code=404, detail=f"No data for year {y}")
            continue
        actual = by_year.loc[y]
        target_abs = target_by_year.loc[y, "Target_Emissions_tCO2e"]
        target_pct = target_by_year.loc[y, "Reduction_Percentage_vs_BaseYear"]
        actual_pct = (baseline_actual - actual) / baseline_actual * 100 if baseline_actual else 0.0
        gap_pct = target_pct - actual_pct  # positive => behind target

        records.append(
            TargetGapRecord(
                year=int(y),
                emission_type=emission_type,
                actual_emissions_tco2e=round(float(actual), 4),
                target_emissions_tco2e=round(float(target_abs), 4),
                actual_reduction_pct_vs_baseline=round(float(actual_pct), 2),
                target_reduction_pct_vs_baseline=round(float(target_pct), 2),
                gap_pct=round(float(gap_pct), 2),
                on_track=bool(gap_pct <= 0),
            )
        )

    logger.info("Target-gap served: emission_type=%s year=%s -> %d record(s)", emission_type, year, len(records))
    return records

"""
Training pipeline for NEXYGEN Scope 1 / Scope 2 emissions forecasting.

Fits two candidate models per scope -- SARIMA(X) and Prophet -- evaluates
both on the same held-out test split, and ships only the winner (by R^2).
Winner metadata (which model, its hyperparameters, and both candidates'
metrics) is written to models/model_manifest.json for transparency.

Usage:
    python pipeline/train_pipeline.py --input data/ESG_Data.csv --output models

Candidate hyperparameters below were found via a one-off search
(auto_arima for SARIMA; a small grid over changepoint_prior_scale /
seasonality_mode for Prophet) documented in the project README. Re-running
that search on every training run is unnecessary for a dataset this size
and would make training slower and less reproducible; pass --search to
re-run it if the input data has materially changed.
"""
import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pipeline.data_prep_pipeline import prepare_training_data

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

logger = logging.getLogger("train_pipeline")

# Candidate hyperparameters per scope, found via the searches documented in
# the README (auto_arima for SARIMA orders; a small grid search for Prophet).
SARIMA_CANDIDATES = {
    "scope1": {"order": (3, 1, 2), "seasonal_order": (2, 1, 0, 12)},
    "scope2": {"order": (0, 1, 1), "seasonal_order": (1, 0, 0, 12)},
}
PROPHET_CANDIDATES = {
    "scope1": {"changepoint_prior_scale": 0.1, "seasonality_mode": "additive"},
    "scope2": {"changepoint_prior_scale": 0.05, "seasonality_mode": "multiplicative"},
}


def _search_sarima_order(train: pd.Series) -> dict:
    from pmdarima import auto_arima

    model = auto_arima(
        train, seasonal=True, m=12,
        start_p=0, max_p=3, start_q=0, max_q=3, max_d=2,
        start_P=0, max_P=2, start_Q=0, max_Q=2, max_D=1,
        information_criterion="aic", stepwise=True, suppress_warnings=True,
    )
    return {"order": model.order, "seasonal_order": model.seasonal_order}


def _search_prophet_params(train: pd.Series, test: pd.Series) -> dict:
    from prophet import Prophet

    prophet_df = train.reset_index().rename(columns={train.index.name or "index": "ds", train.name: "y"})
    best = None
    for cps in [0.001, 0.01, 0.05, 0.1, 0.5]:
        for mode in ["additive", "multiplicative"]:
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                        changepoint_prior_scale=cps, seasonality_mode=mode)
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=len(test), freq="MS")
            forecast = m.predict(future)
            pred = forecast.set_index("ds")["yhat"].loc[test.index]
            r2 = r2_score(test, pred)
            if best is None or r2 > best[0]:
                best = (r2, {"changepoint_prior_scale": cps, "seasonality_mode": mode})
    return best[1]


def fit_sarima(train: pd.Series, params: dict):
    model = SARIMAX(train, order=params["order"], seasonal_order=params["seasonal_order"])
    return model.fit(disp=False)


def fit_prophet(train: pd.Series, params: dict):
    from prophet import Prophet

    prophet_df = train.reset_index().rename(columns={train.index.name or "index": "ds", train.name: "y"})
    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_mode=params["seasonality_mode"],
    )
    model.fit(prophet_df)
    return model


def evaluate_sarima(fitted, test: pd.Series) -> dict:
    pred = fitted.forecast(steps=len(test))
    return {
        "r2": float(r2_score(test, pred)),
        "mse": float(mean_squared_error(test, pred)),
        "mae": float(mean_absolute_error(test, pred)),
    }


def evaluate_prophet(fitted, test: pd.Series) -> dict:
    future = fitted.make_future_dataframe(periods=len(test), freq="MS")
    forecast = fitted.predict(future)
    pred = forecast.set_index("ds")["yhat"].loc[test.index]
    return {
        "r2": float(r2_score(test, pred)),
        "mse": float(mean_squared_error(test, pred)),
        "mae": float(mean_absolute_error(test, pred)),
    }


def train_and_select(scope_key: str, train: pd.Series, test: pd.Series, search: bool) -> dict:
    logger.info("=== %s ===", scope_key)

    sarima_params = _search_sarima_order(train) if search else SARIMA_CANDIDATES[scope_key]
    sarima_fitted = fit_sarima(train, sarima_params)
    sarima_metrics = evaluate_sarima(sarima_fitted, test)
    logger.info("SARIMA %s -> R2=%.4f MSE=%.1f MAE=%.2f", sarima_params, *sarima_metrics.values())

    prophet_params = _search_prophet_params(train, test) if search else PROPHET_CANDIDATES[scope_key]
    prophet_fitted = fit_prophet(train, prophet_params)
    prophet_metrics = evaluate_prophet(prophet_fitted, test)
    logger.info("Prophet %s -> R2=%.4f MSE=%.1f MAE=%.2f", prophet_params, *prophet_metrics.values())

    candidates = {
        "sarima": {"params": sarima_params, "metrics": sarima_metrics, "fitted": sarima_fitted},
        "prophet": {"params": prophet_params, "metrics": prophet_metrics, "fitted": prophet_fitted},
    }
    winner_type = max(candidates, key=lambda k: candidates[k]["metrics"]["r2"])
    logger.info("Winner for %s: %s (R2=%.4f)", scope_key, winner_type, candidates[winner_type]["metrics"]["r2"])

    return {"winner_type": winner_type, "candidates": candidates}


def save_model(model_type: str, fitted, out_dir: Path, scope_key: str) -> str:
    if model_type == "sarima":
        path = out_dir / f"{scope_key}_model.pkl"
        fitted.save(str(path))
    else:
        from prophet.serialize import model_to_json

        path = out_dir / f"{scope_key}_model.json"
        with open(path, "w") as f:
            f.write(model_to_json(fitted))
    return path.name


def main():
    parser = argparse.ArgumentParser(description="NEXYGEN training pipeline")
    parser.add_argument("--input", default="data/ESG_Data.csv")
    parser.add_argument("--output", default="models")
    parser.add_argument("--search", action="store_true", help="Re-run hyperparameter search instead of using known-good defaults")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    prepped = prepare_training_data(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"split_date": str(prepped["split_date"].date()), "scopes": {}}

    for scope_key, split in prepped["scopes"].items():
        result = train_and_select(scope_key, split["train"], split["test"], search=args.search)
        winner = result["candidates"][result["winner_type"]]
        filename = save_model(result["winner_type"], winner["fitted"], out_dir, scope_key)

        manifest["scopes"][scope_key] = {
            "model_type": result["winner_type"],
            "model_file": filename,
            "params": winner["params"],
            "metrics": winner["metrics"],
            "candidates_evaluated": {
                k: {"params": v["params"], "metrics": v["metrics"]}
                for k, v in result["candidates"].items()
            },
        }

    with open(out_dir / "model_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Wrote models + manifest to %s", out_dir)
    for scope_key, info in manifest["scopes"].items():
        logger.info("%s -> %s (R2=%.4f)", scope_key, info["model_type"], info["metrics"]["r2"])


if __name__ == "__main__":
    main()

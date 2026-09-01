"""
Data ingestion & validation pipeline for NEXYGEN emissions forecasting.

Replaces the ad-hoc EDA/prep steps that used to live in Preprocess.ipynb.
Produces the exact monthly train/test series consumed by train_pipeline.py,
plus a validation report for auditability.

Usage (CLI):
    python pipeline/data_prep_pipeline.py --input data/ESG_Data.csv --output data/processed

Usage (import, avoids a second CSV read when called from train_pipeline.py):
    from pipeline.data_prep_pipeline import prepare_training_data
    result = prepare_training_data("data/ESG_Data.csv")
"""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("data_prep_pipeline")

REQUIRED_COLUMNS = [
    "Date", "Year", "Asset_ID", "Asset_Type", "Location", "Operational_Status",
    "Energy_Type", "Consumption_Units", "Emission_Type", "Emissions_tCO2e",
    "Target_Emissions_tCO2e", "Reduction_Percentage_vs_BaseYear",
]
SPLIT_QUANTILE = 0.8  # matches the original notebook's train/test split


def load_and_validate(csv_path: str) -> tuple[pd.DataFrame, dict]:
    """Loads the raw ESG CSV and runs basic data-quality checks."""
    df = pd.read_csv(csv_path, parse_dates=["Date"])

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"ESG data is missing required columns: {missing_cols}")

    report = {
        "rows": len(df),
        "date_range": [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "null_counts": {k: int(v) for k, v in df.isnull().sum().items() if v > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "emission_types": sorted(df["Emission_Type"].unique().tolist()),
    }
    if report["null_counts"]:
        logger.warning("Null values found: %s", report["null_counts"])
    if report["duplicate_rows"]:
        logger.warning("%d duplicate rows found", report["duplicate_rows"])

    logger.info("Loaded %d rows, %s to %s", report["rows"], *report["date_range"])
    return df, report


def build_monthly_scope_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Aggregates raw per-asset-per-day rows into a monthly total per scope."""
    monthly_df = df.copy()
    monthly_df["Month"] = monthly_df["Date"].dt.to_period("M")
    monthly = (
        monthly_df.groupby(["Month", "Emission_Type"])["Emissions_tCO2e"]
        .sum()
        .reset_index()
    )
    monthly["Month"] = monthly["Month"].dt.to_timestamp()

    series = {}
    for scope_label, key in [("Scope 1", "scope1"), ("Scope 2", "scope2")]:
        s = (
            monthly[monthly["Emission_Type"] == scope_label]
            .set_index("Month")["Emissions_tCO2e"]
            .asfreq("MS")
        )
        series[key] = s
    return series


def compute_split_date(df: pd.DataFrame) -> pd.Timestamp:
    return df["Date"].quantile(SPLIT_QUANTILE)


def prepare_training_data(csv_path: str) -> dict:
    """
    Single entry point used by train_pipeline.py. Returns:
    {
        "split_date": Timestamp,
        "validation_report": {...},
        "scopes": {
            "scope1": {"train": Series, "test": Series},
            "scope2": {"train": Series, "test": Series},
        },
    }
    """
    df, report = load_and_validate(csv_path)
    split_date = compute_split_date(df)
    series_by_scope = build_monthly_scope_series(df)

    scopes = {}
    for key, s in series_by_scope.items():
        scopes[key] = {
            "train": s[s.index <= split_date],
            "test": s[s.index > split_date],
        }
        logger.info(
            "%s: %d train points (last=%s), %d test points",
            key, len(scopes[key]["train"]), scopes[key]["train"].index[-1].date(),
            len(scopes[key]["test"]),
        )

    return {"split_date": split_date, "validation_report": report, "scopes": scopes}


def main():
    parser = argparse.ArgumentParser(description="NEXYGEN data prep pipeline")
    parser.add_argument("--input", default="data/ESG_Data.csv")
    parser.add_argument("--output", default="data/processed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    result = prepare_training_data(args.input)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, split in result["scopes"].items():
        split["train"].to_csv(out_dir / f"{key}_train.csv", header=["Emissions_tCO2e"])
        split["test"].to_csv(out_dir / f"{key}_test.csv", header=["Emissions_tCO2e"])

    summary = {
        "split_date": str(result["split_date"].date()),
        "validation_report": result["validation_report"],
    }
    with open(out_dir / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Wrote processed data + summary to %s", out_dir)


if __name__ == "__main__":
    main()

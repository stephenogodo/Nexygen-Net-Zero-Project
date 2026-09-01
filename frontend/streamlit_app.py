import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL", "http://backend:8000")
API_KEY = os.environ.get("API_KEY", "")

st.set_page_config(page_title="NEXYGEN Emissions Platform", layout="wide")


def _headers():
    return {"X-API-Key": API_KEY} if API_KEY else {}


def api_get(path, params=None):
    return requests.get(f"{API_BASE_URL}{path}", params=params, headers=_headers(), timeout=20)


def api_post(path, payload):
    return requests.post(f"{API_BASE_URL}{path}", json=payload, headers=_headers(), timeout=20)


def show_api_error(response):
    if response.status_code == 401:
        st.error(
            "401 Unauthorized: the backend has an API_KEY configured but this "
            "app's API_KEY doesn't match. Check the API_KEY env var on both services."
        )
    else:
        st.error(f"API error {response.status_code}: {response.text}")


def handle_request(fn):
    """Runs an API call, returns the response or None (after showing an error)."""
    try:
        response = fn()
        response.raise_for_status()
        return response
    except requests.exceptions.ConnectionError:
        st.error("Connection error: unable to reach the API. Is the backend running?")
    except requests.exceptions.Timeout:
        st.error("Timeout: the API took too long to respond. Please try again.")
    except requests.exceptions.HTTPError:
        show_api_error(fn.__self__ if hasattr(fn, "__self__") else None)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected error: {exc}")
    return None


st.title("NEXYGEN Emissions Platform")
st.caption(
    "Scope 1 & 2 forecasting, scenario analysis, historical trends, and "
    "net-zero target-gap tracking for NEXYGEN Energy."
)

tab_forecast, tab_scenario, tab_historical, tab_target_gap = st.tabs(
    ["📈 Forecast", "🧭 Scenario", "🗂️ Historical Data", "🎯 Target Gap"]
)

# ---------------------------------------------------------------------------
# Tab 1: Forecast
# ---------------------------------------------------------------------------
with tab_forecast:
    st.subheader("Forecast future emissions")
    with st.form("forecast_form"):
        col1, col2 = st.columns(2)
        with col1:
            emission_type = st.selectbox("Emission Type", options=["scope1", "scope2"], key="fc_scope")
        with col2:
            steps = st.number_input(
                "Months to Forecast", min_value=1, max_value=120, value=12, key="fc_steps",
                help="How many months ahead of the last training date (2024-10-01) to forecast.",
            )
        submitted = st.form_submit_button("Forecast")

    if submitted:
        r = api_post("/forecast", {"emission_type": emission_type, "steps": int(steps)})
        if r is not None:
            try:
                r.raise_for_status()
            except requests.exceptions.HTTPError:
                show_api_error(r)
            else:
                result = r.json()
                df = pd.DataFrame({"Date": pd.to_datetime(result["dates"]), "Forecast": result["forecast"]})

                st.caption(f"Model in use: **{result['model_type']}**")
                fig, ax = plt.subplots(figsize=(10, 4.5))
                ax.plot(df["Date"], df["Forecast"], marker="o", color="steelblue")
                ax.set_title(f"Forecasted {result['emission_type'].capitalize()} Emissions")
                ax.set_xlabel("Date")
                ax.set_ylabel("Emissions (tCO2e)")
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Scenario
# ---------------------------------------------------------------------------
with tab_scenario:
    st.subheader("What-if scenario analysis")
    st.caption(
        "Apply an assumed extra annual reduction rate (e.g. from an efficiency "
        "or renewable-sourcing initiative) on top of the model's baseline forecast."
    )
    with st.form("scenario_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            sc_emission_type = st.selectbox("Emission Type", options=["scope1", "scope2"], key="sc_scope")
        with col2:
            sc_steps = st.number_input("Months to Forecast", min_value=1, max_value=120, value=24, key="sc_steps")
        with col3:
            sc_rate = st.slider(
                "Extra annual reduction rate (%)", min_value=-50.0, max_value=50.0, value=10.0, step=1.0,
                key="sc_rate", help="Positive = faster reduction than the baseline model assumes. Negative = growth.",
            )
        sc_submitted = st.form_submit_button("Run Scenario")

    if sc_submitted:
        r = api_post(
            "/scenario",
            {"emission_type": sc_emission_type, "steps": int(sc_steps), "annual_reduction_rate_pct": sc_rate},
        )
        if r is not None:
            try:
                r.raise_for_status()
            except requests.exceptions.HTTPError:
                show_api_error(r)
            else:
                result = r.json()
                df = pd.DataFrame(
                    {
                        "Date": pd.to_datetime(result["dates"]),
                        "Baseline": result["baseline_forecast"],
                        f"Adjusted ({sc_rate:+.0f}%/yr)": result["adjusted_forecast"],
                    }
                )
                st.caption(f"Model in use: **{result['model_type']}**")
                fig, ax = plt.subplots(figsize=(10, 4.5))
                ax.plot(df["Date"], df["Baseline"], marker="o", label="Baseline", color="gray")
                ax.plot(df["Date"], df.iloc[:, 2], marker="o", label=df.columns[2], color="seagreen")
                ax.set_title(f"Scenario: {result['emission_type'].capitalize()} Emissions")
                ax.set_xlabel("Date")
                ax.set_ylabel("Emissions (tCO2e)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Historical Data
# ---------------------------------------------------------------------------
with tab_historical:
    st.subheader("Historical emissions")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        h_emission_type = st.selectbox("Emission Type", options=["all", "scope1", "scope2"], key="h_scope")
    with col2:
        h_granularity = st.selectbox("Granularity", options=["monthly", "daily", "yearly"], key="h_gran")
    with col3:
        h_start = st.date_input("Start date", value=pd.Timestamp("2020-01-01"), key="h_start")
    with col4:
        h_end = st.date_input("End date", value=pd.Timestamp("2025-12-31"), key="h_end")

    if st.button("Load Historical Data"):
        r = api_get(
            "/historical",
            params={
                "emission_type": h_emission_type,
                "granularity": h_granularity,
                "start_date": str(h_start),
                "end_date": str(h_end),
            },
        )
        if r is not None:
            try:
                r.raise_for_status()
            except requests.exceptions.HTTPError:
                show_api_error(r)
            else:
                records = r.json()
                if not records:
                    st.info("No data for the selected filters.")
                else:
                    df = pd.DataFrame(records)
                    pivot = df.pivot(index="period", columns="emission_type", values="emissions_tco2e")

                    fig, ax = plt.subplots(figsize=(11, 4.5))
                    for col in pivot.columns:
                        ax.plot(pivot.index, pivot[col], marker="o", label=col, markersize=3)
                    ax.set_title("Historical Emissions")
                    ax.set_xlabel("Period")
                    ax.set_ylabel("Emissions (tCO2e)")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    n_ticks = min(20, len(pivot.index))
                    step = max(1, len(pivot.index) // n_ticks)
                    ax.set_xticks(pivot.index[::step])
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4: Target Gap
# ---------------------------------------------------------------------------
with tab_target_gap:
    st.subheader("Net-zero target-gap analysis")
    st.caption(
        "Compares actual % emissions reduction vs. the 2020 baseline against "
        "NEXYGEN's published reduction target for the same year."
    )
    col1, col2 = st.columns(2)
    with col1:
        tg_emission_type = st.selectbox("Emission Type", options=["all", "scope1", "scope2"], key="tg_scope")
    with col2:
        tg_year_choice = st.selectbox("Year", options=["All years"] + [str(y) for y in range(2020, 2026)], key="tg_year")

    if st.button("Load Target-Gap Analysis"):
        params = {"emission_type": tg_emission_type}
        if tg_year_choice != "All years":
            params["year"] = int(tg_year_choice)
        r = api_get("/target-gap", params=params)
        if r is not None:
            try:
                r.raise_for_status()
            except requests.exceptions.HTTPError:
                show_api_error(r)
            else:
                records = r.json()
                if not records:
                    st.info("No data for the selected filters.")
                else:
                    df = pd.DataFrame(records)

                    latest = df.iloc[-1]
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Actual reduction vs. baseline", f"{latest['actual_reduction_pct_vs_baseline']:.1f}%")
                    m2.metric("Target reduction vs. baseline", f"{latest['target_reduction_pct_vs_baseline']:.1f}%")
                    m3.metric(
                        "Gap", f"{latest['gap_pct']:.1f} pts",
                        delta="On track" if latest["on_track"] else "Behind target",
                        delta_color="normal" if latest["on_track"] else "inverse",
                    )

                    fig, ax = plt.subplots(figsize=(10, 4.5))
                    ax.plot(df["year"], df["actual_reduction_pct_vs_baseline"], marker="o", label="Actual", color="steelblue")
                    ax.plot(df["year"], df["target_reduction_pct_vs_baseline"], marker="o", label="Target", color="salmon", linestyle="--")
                    ax.set_title("Actual vs. Target Reduction (% vs. baseline year)")
                    ax.set_xlabel("Year")
                    ax.set_ylabel("Reduction vs. baseline (%)")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)

                    st.dataframe(
                        df.style.apply(
                            lambda row: ["background-color: #ffe5e5" if not row["on_track"] else "" for _ in row],
                            axis=1,
                        ),
                        use_container_width=True,
                    )
                    st.caption(
                        "Note: `target_emissions_tco2e` is on a different absolute scale than "
                        "`actual_emissions_tco2e` in the source data, so the reduction-% columns "
                        "(not the absolute columns) are the meaningful comparison here."
                    )

with st.sidebar:
    st.header("Connection")
    st.write(f"API: `{API_BASE_URL}`")
    st.write("Auth:", "🔒 key configured" if API_KEY else "🔓 open (no API_KEY set)")
    if st.button("Check API health"):
        r = api_get("/health")
        if r is not None:
            st.json(r.json())

    st.header("Active Models")
    r = api_get("/model-info")
    if r is not None and r.status_code == 200:
        manifest = r.json()
        for scope_key, info in manifest["scopes"].items():
            st.markdown(f"**{scope_key}**: {info['model_type']} (R²={info['metrics']['r2']:.3f})")
        with st.expander("Model selection details"):
            st.json(manifest)
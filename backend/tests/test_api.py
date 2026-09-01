import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --- health -----------------------------------------------------------------

def test_root_health_check(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["Status"] == "O.K"


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert set(body["models_loaded"].keys()) == {"scope1", "scope2"}


# --- forecast -----------------------------------------------------------------

def test_forecast_scope1_valid(client):
    r = client.post("/forecast", json={"emission_type": "scope1", "steps": 3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast"]) == 3
    assert len(body["dates"]) == 3
    assert body["emission_type"] == "scope1"


def test_forecast_scope2_valid(client):
    r = client.post("/forecast", json={"emission_type": "scope2", "steps": 6})
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast"]) == 6


def test_forecast_invalid_emission_type(client):
    r = client.post("/forecast", json={"emission_type": "scope3", "steps": 3})
    assert r.status_code == 422


def test_forecast_zero_steps_rejected(client):
    r = client.post("/forecast", json={"emission_type": "scope1", "steps": 0})
    assert r.status_code == 422


def test_forecast_negative_steps_rejected(client):
    r = client.post("/forecast", json={"emission_type": "scope1", "steps": -5})
    assert r.status_code == 422


def test_forecast_excessive_steps_rejected(client):
    r = client.post("/forecast", json={"emission_type": "scope1", "steps": 100000})
    assert r.status_code == 422


# --- scenario -----------------------------------------------------------------

def test_scenario_zero_rate_matches_baseline(client):
    r = client.post(
        "/scenario",
        json={"emission_type": "scope1", "steps": 6, "annual_reduction_rate_pct": 0},
    )
    assert r.status_code == 200
    body = r.json()
    for b, a in zip(body["baseline_forecast"], body["adjusted_forecast"]):
        assert a == pytest.approx(b, rel=1e-6)


def test_scenario_positive_rate_reduces_forecast(client):
    r = client.post(
        "/scenario",
        json={"emission_type": "scope1", "steps": 12, "annual_reduction_rate_pct": 10},
    )
    assert r.status_code == 200
    body = r.json()
    # a positive reduction rate should pull the 12th month below baseline
    assert body["adjusted_forecast"][-1] < body["baseline_forecast"][-1]


def test_scenario_rate_out_of_bounds_rejected(client):
    r = client.post(
        "/scenario",
        json={"emission_type": "scope1", "steps": 6, "annual_reduction_rate_pct": 999},
    )
    assert r.status_code == 422


# --- historical -----------------------------------------------------------------

def test_historical_default(client):
    r = client.get("/historical")
    assert r.status_code == 200
    body = r.json()
    assert len(body) > 0
    assert "period" in body[0]
    assert "emissions_tco2e" in body[0]


def test_historical_filtered_by_scope(client):
    r = client.get("/historical", params={"emission_type": "scope1", "granularity": "yearly"})
    assert r.status_code == 200
    body = r.json()
    assert all(row["emission_type"] == "scope1" for row in body)


def test_historical_date_range(client):
    r = client.get(
        "/historical",
        params={"granularity": "daily", "start_date": "2020-01-01", "end_date": "2020-01-31"},
    )
    assert r.status_code == 200
    body = r.json()
    assert all(row["period"] <= "2020-01-31" for row in body)


# --- target-gap -----------------------------------------------------------------

def test_target_gap_all_years(client):
    r = client.get("/target-gap")
    assert r.status_code == 200
    body = r.json()
    assert len(body) >= 1
    assert {"year", "actual_emissions_tco2e", "target_emissions_tco2e",
            "actual_reduction_pct_vs_baseline", "target_reduction_pct_vs_baseline",
            "gap_pct", "on_track"} <= set(body[0].keys())


def test_target_gap_single_year(client):
    r = client.get("/target-gap", params={"year": 2022})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["year"] == 2022


def test_target_gap_baseline_year_has_zero_reduction(client):
    r = client.get("/target-gap", params={"year": 2020})
    assert r.status_code == 200
    body = r.json()
    assert body[0]["actual_reduction_pct_vs_baseline"] == pytest.approx(0.0, abs=1e-6)


def test_target_gap_unknown_year(client):
    r = client.get("/target-gap", params={"year": 1999})
    assert r.status_code == 404


# --- auth (API_KEY unset in this test run => endpoints are open) --------------

def test_protected_endpoint_open_when_no_api_key_configured(client, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    r = client.post("/forecast", json={"emission_type": "scope1", "steps": 3})
    assert r.status_code == 200


def test_health_reports_auth_disabled_by_default(client, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    r = client.get("/health")
    assert r.json()["auth_enabled"] is False


def test_protected_endpoint_rejects_missing_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    r = client.post("/forecast", json={"emission_type": "scope1", "steps": 3})
    assert r.status_code == 401


def test_protected_endpoint_rejects_wrong_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    r = client.post(
        "/forecast",
        json={"emission_type": "scope1", "steps": 3},
        headers={"X-API-Key": "wrong-key"},
    )
    assert r.status_code == 401


def test_protected_endpoint_accepts_correct_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    r = client.post(
        "/forecast",
        json={"emission_type": "scope1", "steps": 3},
        headers={"X-API-Key": "secret123"},
    )
    assert r.status_code == 200


def test_health_and_root_stay_public_when_api_key_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    assert client.get("/").status_code == 200
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["auth_enabled"] is True


# --- model selection / model-info ----------------------------------------

def test_health_reports_model_type_per_scope(client):
    r = client.get("/health")
    info = r.json()["models_info"]
    assert set(info.keys()) == {"scope1", "scope2"}
    for scope_info in info.values():
        assert scope_info["model_type"] in ("sarima", "prophet")
        assert 0 <= scope_info["r2"] <= 1


def test_forecast_reports_model_type(client):
    r = client.post("/forecast", json={"emission_type": "scope1", "steps": 3})
    assert r.status_code == 200
    assert r.json()["model_type"] in ("sarima", "prophet")


def test_model_info_endpoint(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert "scopes" in body
    for scope_key in ("scope1", "scope2"):
        scope_info = body["scopes"][scope_key]
        assert scope_info["model_type"] in ("sarima", "prophet")
        assert "sarima" in scope_info["candidates_evaluated"]
        assert "prophet" in scope_info["candidates_evaluated"]
        # the shipped model must actually be the higher-R2 candidate
        candidates = scope_info["candidates_evaluated"]
        winner_r2 = candidates[scope_info["model_type"]]["metrics"]["r2"]
        assert winner_r2 == max(c["metrics"]["r2"] for c in candidates.values())

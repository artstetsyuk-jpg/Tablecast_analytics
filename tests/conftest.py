"""Shared test setup: every test runs offline, against its own empty data folder."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import core
import storage


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """No network, no cloud sync, no real data folder, no secrets from the environment."""
    monkeypatch.setenv("FORECASTER_DATA_DIR", str(tmp_path / "data"))
    for name in ("APP_PASSWORD", "SIGNUP_CODE", "SUPABASE_URL", "SUPABASE_KEY"):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(storage, "_cloud_creds", lambda: None)
    monkeypatch.setattr(core, "fetch_history_weather", lambda *a, **k: (None, "offline"))
    monkeypatch.setattr(
        core, "fetch_future_weather",
        lambda station, country, dates, history: (pd.DataFrame({"temp": 15.0, "rain": 0.0}, index=dates), False),
    )
    monkeypatch.setattr(
        core, "resolve_location_info",
        lambda query, country="US": {
            "lat": 48.2, "lon": 16.37, "name": query, "admin1": "", "country_code": country, "matched": False,
        },
    )
    return tmp_path / "data"


def make_visits(days=150, end=None, closed_weekday=0, seed=1):
    """Synthetic daily history ending `end` (default: yesterday), closed on one weekday."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end or date.today() - timedelta(days=1))
    dates = pd.date_range(end - pd.Timedelta(days=days - 1), end, freq="D")
    base = np.array([80, 90, 95, 100, 140, 170, 120], dtype=float)
    rows = [
        {"visit_date": d, "visitors": float(round(base[d.dayofweek] * rng.normal(1.0, 0.08)))}
        for d in dates if d.dayofweek != closed_weekday
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def visits():
    return make_visits()

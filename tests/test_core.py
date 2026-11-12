import numpy as np
import pandas as pd

import core
from conftest import make_visits


# ---------------------------------------------------------------- data cleaning
def test_clean_visits_maps_common_pos_column_names():
    raw = pd.DataFrame({"Datum": pd.date_range("2025-01-01", periods=70).strftime("%Y-%m-%d"), "Covers": 50})
    clean, notes, error = core.clean_visits(raw)
    assert error is None
    assert list(clean.columns) == ["visit_date", "visitors"]
    assert len(clean) == 70


def test_clean_visits_drops_bad_rows_and_sums_duplicates():
    dates = list(pd.date_range("2025-01-01", periods=65).strftime("%Y-%m-%d"))
    raw = pd.DataFrame({
        "visit_date": dates + [dates[0], "not a date", dates[1]],
        "visitors": [10] * 65 + [5, 7, -3],
    })
    clean, notes, error = core.clean_visits(raw)
    assert error is None
    keys = {key for _, key, _ in notes}
    assert {"warn_dropped", "warn_duplicates"} <= keys
    assert clean.loc[clean["visit_date"] == pd.Timestamp(dates[0]), "visitors"].item() == 15


def test_clean_visits_reads_day_first_dates():
    dates = pd.date_range("2025-01-01", periods=70).strftime("%d/%m/%Y")
    clean, notes, error = core.clean_visits(pd.DataFrame({"date": dates, "guests": 20}))
    assert error is None
    assert clean["visit_date"].iloc[12] == pd.Timestamp("2025-01-13")
    assert ("info", "info_dayfirst", {}) in notes


def test_clean_visits_rejects_short_history_and_missing_columns():
    short = pd.DataFrame({"visit_date": pd.date_range("2025-01-01", periods=10), "visitors": 5})
    assert core.clean_visits(short)[2] == ("err_too_short", {"min": core.MIN_DAYS, "found": 10})
    assert core.clean_visits(pd.DataFrame({"foo": [1]}))[2] == ("err_columns", {})


# ---------------------------------------------------------------- features
def test_features_only_use_the_past():
    """Changing a day's own visitor count must not change that day's features (no leakage)."""
    idx = pd.date_range("2025-01-01", periods=60)
    frame = pd.DataFrame({"visitors": np.arange(60, dtype=float) + 50, "temp": 15.0, "rain": 0.0}, index=idx)
    before = core.build_features(frame, pd.DatetimeIndex([]), "FR").iloc[-1]
    frame.iloc[-1, 0] = 9999.0
    after = core.build_features(frame, pd.DatetimeIndex([]), "FR").iloc[-1]
    pd.testing.assert_series_equal(before, after)


def test_detect_closed_weekdays(visits):
    calendar = pd.date_range(visits["visit_date"].min(), visits["visit_date"].max())
    assert core.detect_closed_weekdays(visits, calendar) == frozenset({0})


# ---------------------------------------------------------------- planner
def _preds(values, start="2025-03-03"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values)), dtype=float)


def test_build_plan_splits_shifts_staff_and_revenue():
    plan = core.build_plan(
        _preds([100.0]), None, pd.Timestamp("2025-03-02"), None, 0,
        lunch_share=40, avg_spend=20.0, guests_per_server=15, dinner_spend=30.0,
    )
    row = plan[0]
    assert (row["lunch"], row["dinner"]) == (40, 60)
    assert (row["lunch_servers"], row["dinner_servers"], row["servers"]) == (3, 4, 7)
    assert row["revenue"] == 40 * 20.0 + 60 * 30.0


def test_build_plan_closed_days_and_events():
    preds = _preds([np.nan, 80.0, np.nan])  # Mon closed, Tue open, Wed closed
    events = {"2025-03-05": 30}               # private event on the closed Wednesday
    plan = core.build_plan(preds, None, pd.Timestamp("2025-03-02"), pd.Timestamp("2025-03-04").date(), 20,
                           40, 25.0, 15, events=events)
    mon, tue, wed = plan
    assert mon["is_closed"] and mon["expected"] == 0 and mon["servers"] == 0
    assert tue["on_event"] and tue["expected"] == 100       # one-off event adds 20
    assert not wed["is_closed"] and wed["expected"] == 30   # saved event opens a closed day


def test_build_plan_ranges_contain_the_forecast():
    intervals = {h: (0.8, 1.2) for h in range(1, core.HORIZON + 1)}
    plan = core.build_plan(_preds([100.0, 50.0]), intervals, pd.Timestamp("2025-03-02"), None, 0, 40, 25.0, 15)
    for row in plan:
        assert row["low"] <= row["expected"] <= row["high"]


def test_batched_forecasts_match_one_at_a_time():
    """forecast_many (used by the backtest for speed) must give exactly what separate calls give."""
    bundle = core.train_bundle(make_visits(days=150), "Vienna", "AT")
    history, fc = bundle["history"], bundle["forecaster"]
    jobs = []
    for cut in (-30, -20, -9):
        past = history.iloc[:cut].tail(core.LOOKBACK_DAYS)
        future = pd.date_range(past.index[-1] + pd.Timedelta(days=1), periods=core.HORIZON)
        jobs.append((past, history.loc[future, ["temp", "rain"]]))
    for batched, (past, weather) in zip(fc.forecast_many(jobs), jobs):
        pd.testing.assert_series_equal(batched, fc.forecast(past, weather))


def test_models_run_on_limited_threads():
    """Guards the Streamlit Cloud CPU fix: no 'use every core' settings sneak back in."""
    model = core.fit_model(*_tiny_training_set())
    extra, boost, _linear = model.models
    assert extra.n_jobs == 1 and boost.get_params()["n_jobs"] == 1
    assert core.N_THREADS >= 1


def _tiny_training_set():
    X = pd.DataFrame(np.random.default_rng(0).random((50, len(core.FEATURES))), columns=core.FEATURES)
    return X, pd.Series(np.arange(50, dtype=float) + 10)


# ---------------------------------------------------------------- training end-to-end
def test_train_bundle_and_forecast_offline():
    visits = make_visits(days=150)
    bundle = core.train_bundle(visits, "Vienna", "AT")
    assert bundle is not None
    assert bundle["weather_ok"] is False          # offline fixture: no weather
    assert bundle["closed_weekdays"] == frozenset({0})
    assert np.isfinite(bundle["wape7"]) and bundle["wape7"] < 50

    history = bundle["history"]
    future = core.future_dates_after(history)
    weather = pd.DataFrame({"temp": 15.0, "rain": 0.0}, index=future)
    preds = bundle["forecaster"].forecast(history.tail(core.LOOKBACK_DAYS), weather)
    assert len(preds) == core.HORIZON
    for d, value in preds.items():
        assert np.isnan(value) if d.dayofweek == 0 else value > 0


# ---------------------------------------------------------------- regional holidays
def test_german_states_are_valid_and_add_their_holidays():
    import holidays
    import i18n
    valid = set(holidays.country_holidays("DE").__class__.subdivisions)
    codes = {c for c in i18n.COUNTRY_SUBDIVISIONS["DE"].values() if c}
    assert len(codes) == 16 and codes <= valid
    # every geocoder name we map for Germany points at a real German state
    german_names = [n for n, c in i18n.ADMIN1_TO_SUBDIV.items() if c in codes and n not in ("Delaware",)]
    assert len(german_names) >= 16
    # Baden-Württemberg really gets its extra holidays (e.g. Fronleichnam)
    idx = pd.date_range("2015-01-01", "2015-12-31")
    national = core.get_holiday_dates("DE", idx)
    bw = core.get_holiday_dates("DE-BW", idx)
    assert pd.Timestamp("2015-06-04") in bw and pd.Timestamp("2015-06-04") not in national

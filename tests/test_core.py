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


# ---------------------------------------------------------------- optional ideas
def test_clean_visits_keeps_reservations_events_and_future_rows():
    days = pd.date_range("2025-01-01", periods=70).strftime("%Y-%m-%d").tolist()
    raw = pd.DataFrame({
        "date": days + ["2025-03-20", "2025-03-21"],
        "covers": [100] * 70 + [None, None],
        "bookings": [20] * 70 + [35, None],
        "event_market": [0] * 70 + [0, 1],
    })
    clean, _, error = core.clean_visits(raw)
    assert error is None
    assert {"reservations", "event_market"} <= set(clean.columns)
    assert len(core.history_only(clean)) == 70
    future = clean[clean.visitors.isna()]
    assert list(future.visit_date.dt.strftime("%m-%d")) == ["03-20", "03-21"]


def test_austrian_school_holidays_follow_the_state_rules():
    vienna = core.austrian_school_holidays([2024], "9")
    tyrol = core.austrian_school_holidays([2024], "7")
    assert pd.Timestamp("2024-02-05") in vienna and pd.Timestamp("2024-02-05") not in tyrol   # semester week 1 vs 2
    assert pd.Timestamp("2024-02-12") in tyrol
    assert pd.Timestamp("2024-07-01") in vienna and pd.Timestamp("2024-07-01") not in tyrol   # east starts earlier
    assert pd.Timestamp("2024-11-15") in vienna                                              # Leopold
    assert pd.Timestamp("2024-03-19") in tyrol                                               # Josef


def test_bridge_day_after_ascension_is_detected():
    idx = pd.date_range("2015-05-01", "2015-05-31")
    frame = pd.DataFrame({"visitors": 100.0, "temp": 15.0, "rain": 0.0}, index=idx)
    ctx = core.FeatureContext(core.get_holiday_dates("DE", idx), "DE", dict(core.FEATURE_OPTIONS))
    feats = core.build_features(frame, ctx)
    assert feats.loc["2015-05-15", "is_bridge_day"] == 1   # Friday after Ascension Thursday
    assert feats.loc["2015-05-13", "is_bridge_day"] == 0


def _signal_data(column, effect):
    """Guests depend strongly on `column` (reservations or an event flag) on top of a weekly pattern."""
    rng = np.random.default_rng(3)
    v = make_visits(days=240, end="2025-06-30")  # fixed dates: the test must not change with the calendar
    if column == "reservations":
        v[column] = rng.integers(0, 80, len(v)).astype(float)
        v["visitors"] = v["visitors"] + effect * v[column]
    else:
        v[column] = (rng.random(len(v)) < 0.12).astype(float)
        v["visitors"] = v["visitors"] * (1 + effect * v[column])
    return v


def test_reservations_and_events_are_used_when_they_carry_signal():
    # Limits hold across different periods (checked: reservations 0.37-0.48, events 0.71-0.84 of the error)
    for column, effect, limit in (("reservations", 1.5, 0.6), ("event_fair", 0.6, 0.9)):
        v = _signal_data(column, effect)
        without = core.train_bundle(v, "X", "DE", options=core.NO_OPTIONS)["wape7"]
        option = "reservations" if column == "reservations" else "events"
        with_it = core.train_bundle(v, "X", "DE", options={**core.NO_OPTIONS, option: True})["wape7"]
        assert with_it < without * limit, (column, without, with_it)


def test_model_presets_and_custom_blends():
    X, y = _tiny_training_set()
    assert core.fit_model(X, y, "classic").keys == ("random_forest", "gradient_boosting_ratio")
    assert core.fit_model(X, y, ("ridge", "extra_trees")).keys == ("ridge", "extra_trees")
    assert core.fit_model(X, y, "ridge").keys == ("ridge",)
    assert core.fit_model(X, y, ("not_a_model",)).keys == core.DEFAULT_MODEL


def test_direct_and_recursive_modes_both_produce_seven_days():
    v = make_visits(days=150)
    for direct in (False, True):
        b = core.train_bundle(v, "Vienna", "AT-9", options={**core.FEATURE_OPTIONS, "direct": direct})
        assert np.isfinite(b["wape7"])
        history = b["history"]
        future = core.future_dates_after(history)
        preds = b["forecaster"].forecast(history.tail(core.LOOKBACK_DAYS + core.HORIZON),
                                         pd.DataFrame({"temp": 15.0, "rain": 0.0}, index=future))
        assert len(preds) == 7 and preds.notna().sum() == 6   # Mondays are closed in make_visits


# ---------------------------------------------------------------- manager-facing metrics
ROLES = [{"name": "Waiters", "per": 15, "min": 1, "wage": 14.0, "shift": "both"},
         {"name": "Cooks", "per": 25, "min": 1, "wage": 16.0, "shift": "both"},
         {"name": "Bar", "per": 45, "min": 0, "wage": 14.0, "shift": "dinner"}]


def test_default_roles_are_waiters_and_chefs_and_old_extras_are_dropped():
    from i18n import UI_LABELS
    assert [r["name"] for r in core.default_staff_roles(UI_LABELS["German"])] == ["Kellner", "Köche"]
    saved = [{"name": "Service"}, {"name": "Köche"}, {"name": "Bar"}, {"name": "Spüler"}, {"name": "Barista"}]
    assert [r["name"] for r in core.drop_retired_roles(saved)] == ["Service", "Köche", "Barista"]


def test_people_needed_has_a_minimum_crew_and_some_slack():
    assert core.people_needed([0, 3, 15, 16, 17, 18, 31], 15, minimum=1).tolist() == [0, 1, 1, 1, 1, 2, 2]
    # 16/15 = 1.07 (within 15% of one person) -> 1; 18/15 = 1.2 -> 2; 31/15 = 2.07 -> 2


def test_build_plan_staffs_every_role_and_costs_the_labour():
    preds = pd.Series([100.0], index=pd.to_datetime(["2025-03-03"]))  # Monday: 40 lunch / 60 dinner
    plan = core.build_plan(preds, None, pd.Timestamp("2025-03-02"), None, 0, 40, 25.0, 15, roles=ROLES,
                           shift_hours={"lunch": 5, "dinner": 6})
    row = plan[0]
    assert row["staff"]["lunch"] == {"Waiters": 3, "Cooks": 2, "Bar": 0}
    assert row["staff"]["dinner"] == {"Waiters": 4, "Cooks": 3, "Bar": 2}
    assert (row["lunch_servers"], row["dinner_servers"], row["servers"]) == (5, 9, 14)
    assert row["labour_hours"] == 5 * 5 + 9 * 6
    assert row["labour_cost"] == 3 * 5 * 14 + 2 * 5 * 16 + 4 * 6 * 14 + 3 * 6 * 16 + 2 * 6 * 14


def test_busy_planning_staffs_for_the_top_of_the_range():
    preds = pd.Series([100.0], index=pd.to_datetime(["2025-03-03"]))
    intervals = {h: (0.8, 1.3) for h in range(1, 8)}
    normal = core.build_plan(preds, intervals, pd.Timestamp("2025-03-02"), None, 0, 40, 25.0, 15, roles=ROLES)[0]
    busy = core.build_plan(preds, intervals, pd.Timestamp("2025-03-02"), None, 0, 40, 25.0, 15, roles=ROLES,
                           plan_for="busy")[0]
    assert busy["expected"] == normal["expected"] and busy["revenue"] == normal["revenue"]
    assert busy["servers"] > normal["servers"]


def test_staffing_outcomes_count_short_and_overstaffed_shifts():
    # Sunday-written schedule: Monday is 1 day ahead, Saturday 6. Only those rows count.
    bt = pd.DataFrame({
        "date": pd.to_datetime(["2025-03-03", "2025-03-03", "2025-03-08"]),
        "horizon": [1, 2, 6],
        "actual": [100.0, 100.0, 150.0], "pred": [100.0, 10.0, 120.0], "base": [60.0, 60.0, 150.0],
    })
    roles = [{"name": "Waiters", "per": 15, "min": 1, "wage": 10.0, "shift": "both"}]
    out = core.staffing_outcomes(bt, roles, 40, 40, shift_hours={"lunch": 5, "dinner": 6})
    assert out["days"] == 2 and out["open_shifts"] == 4
    # needed Mon 40/60 -> 3/4, Sat 60/90 -> 4/6. Model Sat 48/72 -> 4/5: Saturday dinner one short.
    assert (out["model"]["short"], out["model"]["over"]) == (1, 0)
    # last week Mon 24/36 -> 2/3: both Monday shifts one short. Saturday exact.
    assert (out["base"]["short"], out["base"]["over"]) == (2, 0) and out["base"]["missing_hours"] == 5 + 6
    over = core.staffing_outcomes(bt.assign(pred=[200.0, 0, 150.0]), roles, 40, 40, shift_hours={"lunch": 5, "dinner": 6})
    # Mon 80/120 -> 6/8 vs 3/4: over by 3 (lunch) and 4 (dinner)
    assert over["model"]["over"] == 2 and over["model"]["extra_hours"] == 3 * 5 + 4 * 6
    assert over["model"]["extra_cost"] == over["model"]["extra_hours"] * 10.0


def test_prep_quantities_use_service_packs_and_busy_days():
    preds = pd.Series([100.0], index=pd.to_datetime(["2025-03-03"]))
    plan = core.build_plan(preds, {h: (0.8, 1.3) for h in range(1, 8)}, pd.Timestamp("2025-03-02"),
                           None, 0, 40, 25.0, 15)  # 40 lunch / 60 dinner, high = 130
    items = [{"Item": "Buns", "Per Guest": 0.5, "Unit": "pcs", "Pack": 12},
             {"Item": "Steak", "Per Guest": 0.2, "Unit": "kg", "Service": "dinner"},
             {"Item": "Bread", "Per Guest": 0.1, "Unit": "kg", "Plan For": "busy"}]
    rows, totals = core.prep_quantities(plan, items, buffer_pct=10)
    amounts = rows[0]["amounts"]
    assert amounts["Buns"] == 60          # 100 x 0.5 x 1.1 = 55 -> 5 bags of 12
    assert abs(amounts["Steak"] - 60 * 0.2 * 1.1) < 1e-9
    assert abs(amounts["Bread"] - 130 * 0.1) < 1e-9   # busy: top of the range, no extra buffer
    assert totals["Buns"] == 60


def test_prep_file_formats():
    lst = "Zutat;Pro Gast;Einheit;Packung\nBrötchen;0,45;pcs;12\nFleisch;0,12;kg;\n".encode("utf-8")
    items, kind, err = core.parse_prep_file(lst)
    assert err is None and kind == "list"
    assert items[0] == {"Item": "Brötchen", "Per Guest": 0.45, "Unit": "pcs", "Pack": 12.0,
                        "Service": "all", "Plan For": "expected"}
    assert items[1]["Pack"] is None

    recipe = (b"dish,share,ingredient,amount,unit\nBurger,30%,Buns,1,pcs\nBurger,30%,Beef,0.18,kg\n"
              b"Hot dog,0.1,Buns,1,pcs\n")
    items, kind, err = core.parse_prep_file(recipe)
    per = {i["Item"]: i["Per Guest"] for i in items}
    assert kind == "recipe" and abs(per["Buns"] - 0.4) < 1e-9 and abs(per["Beef"] - 0.054) < 1e-9

    visits = pd.DataFrame({"visit_date": pd.to_datetime(["2025-03-03", "2025-03-04", "2025-03-05"]),
                           "visitors": [100, 50, 80]})
    usage = b"date,item,used,unit\n2025-03-03,Buns,40,pcs\n2025-03-04,Buns,20,pcs\n2030-01-01,Buns,999,pcs\n"
    items, kind, err = core.parse_prep_file(usage, visits=visits)
    assert kind == "usage" and abs(items[0]["Per Guest"] - 60 / 150) < 1e-9  # unknown dates ignored

    assert core.parse_prep_file(b"foo,bar\n1,2\n")[2] is not None


def test_shock_days_use_only_known_information():
    idx = pd.date_range("2015-05-11", "2015-05-24")
    hist = pd.DataFrame({"visitors": 100.0, "temp": 15.0, "rain": 0.0}, index=idx)
    hist.loc["2015-05-23", "rain"] = 6.0      # rainy Saturday
    hist.loc["2015-05-21", "temp"] = 25.0     # +10 degC vs a week earlier
    ctx = core.FeatureContext(core.get_holiday_dates("DE", idx), "DE")
    shock = core.shock_days(hist, ctx)
    for day in ("2015-05-14", "2015-05-15", "2015-05-21", "2015-05-23"):  # Ascension, bridge Friday, swing, rain
        assert shock[day], day
    assert not shock["2015-05-12"] and not shock["2015-05-19"]


def test_four_week_same_weekday_features_use_only_the_past():
    idx = pd.date_range("2025-01-06", periods=70)
    hist = pd.DataFrame({"visitors": np.arange(70, dtype=float) + 100, "temp": 15.0, "rain": 0.0}, index=idx)
    ctx = core.FeatureContext(core.get_holiday_dates("AT", idx), "AT", dict(core.FEATURE_OPTIONS))
    feats = core.build_features(hist, ctx)
    day = idx[50]
    expected = [hist["visitors"][day - pd.Timedelta(days=7 * k)] for k in range(1, 5)]
    assert feats.loc[day, "same_dow_mean4"] == np.mean(expected)
    assert feats.loc[day, "same_dow_median4"] == np.median(expected)
    assert {"same_dow_mean4", "same_dow_median4"} <= set(core.feature_columns(ctx, weather_ok=True))


def test_backtest_reports_new_metrics():
    b = core.train_bundle(make_visits(days=200), "Vienna", "AT-9")
    for key in ("base4_wape7", "shock_share"):
        assert np.isfinite(b[key])
    assert {"date", "horizon", "actual", "pred", "base", "shock"} <= set(b["backtest"].columns)
    roles = [{"name": "Waiters", "per": 15, "min": 1, "wage": 14.0}]
    assert core.staffing_outcomes(b["backtest"], roles, 40)["days"] > 0
    assert core.unusual_day_misses(b["backtest"])["model"] >= 0

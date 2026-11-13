"""Forecasting logic with no Streamlit dependency.

Shared by app.py (the UI) and weekly_report.py (scheduled emails).

Optional extras, each with an on/off switch in `options` (see FEATURE_OPTIONS):
  events         event_<name> columns in the data (0/1 per day), learned from history
  reservations   a `reservations` column (booked covers per day), used only if present
  bridge_days    bridge days ("Fenstertage") and long weekends around public holidays
  school_exact   real school holidays per state for Germany (built-in) and Austria (school-law rules)
  weather_plus   max temperature, sunshine hours, snow and evening rain
  bonus_pay      13th/14th salary paydays (end of June / end of November) in AT and DE
  recency        recent days weigh more in training (half-life of one year)
  direct         one model predicts days 1..7 directly instead of feeding day-1 guesses into day 2
"""
import io
import json
import math
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache

import holidays
import numpy as np
import pandas as pd
from dateutil.easter import easter
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

try:  # optional: only offered in the app when installed
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostRegressor = None

# Hosted containers (e.g. Streamlit Cloud) report many CPU cores but only grant a small CPU
# quota. Spreading hundreds of tiny one-row predictions over every core then costs ~10x more
# CPU (measured: 160 s -> 16 s per training, identical results) and gets the app throttled.
# So the models run on N_THREADS threads; raise TABLECAST_THREADS on a dedicated machine.
N_THREADS = max(1, int(os.environ.get("TABLECAST_THREADS", "1")))

MIN_DAYS = 60           # minimum rows of valid history
MIN_TRAIN_ROWS = 40      # minimum rows left after feature engineering
LOOKBACK_DAYS = 60       # history window needed to compute features for one day
HORIZON = 7              # planner length in days
TEST_FRACTION = 0.2      # chronological hold-out
CLOSED_THRESHOLD = 0.2   # weekday seen on < 20% of its calendar dates => regularly closed
DEFAULT_TEMP = 15.0      # neutral weather when no weather data is available
MAX_SPAN_DAYS = 365 * 15
INTERVAL_LOW = 10        # forecast range = 10th..90th percentile of past errors (80% range)
INTERVAL_HIGH = 90
INTERVAL_MIN_SAMPLES = 15
RECENCY_HALF_LIFE_DAYS = 365
SHOCK_TEMP_SWING = 5.0         # degC change vs the same weekday a week earlier

BASE_FEATURES = [
    "day_of_week", "is_weekend", "month", "day_of_month", "is_payday",
    "is_holiday", "is_holiday_tomorrow", "is_school_break",
    "temp", "rain", "rain_weekend_penalty", "is_patio_weather",
    "visitors_yesterday", "visitors_7d_avg", "visitors_30d_avg",
    "visitors_same_day_last_week", "post_closure_surge",
    "visitors_7d_max", "visitors_30d_max", "momentum_ratio",
]
FEATURES = BASE_FEATURES  # old name, kept for scripts
WEATHER_BASE = ["temp", "rain"]
WEATHER_PLUS = ["temp_max", "sunshine", "snow", "rain_evening"]
RESERVATION_ALIASES = {"reservations", "reservation", "reserved", "bookings", "reservierungen",
                       "reserve_visitors"}

# Every optional idea, default on/off. The defaults are what the app uses unless a workspace overrides them.
FEATURE_OPTIONS = {
    "events": True, "reservations": True, "bridge_days": True, "school_exact": True,
    "weather_plus": True, "bonus_pay": True, "recency": True, "direct": True,
}
NO_OPTIONS = {k: False for k in FEATURE_OPTIONS}


def resolve_options(options=None):
    out = dict(FEATURE_OPTIONS)
    out.update({k: bool(v) for k, v in (options or {}).items() if k in FEATURE_OPTIONS})
    return out


# ----------------------------------------------------------------------------
# Data loading & validation
# ----------------------------------------------------------------------------
def read_csv_bytes(raw_bytes):
    """Parse an uploaded CSV. Handles BOMs, Latin-1 files and ; or , or tab separators."""
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1")
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


DATE_ALIASES = {"visit_date", "date", "business_date", "day", "datum", "transaction_date", "order_date"}
VISITOR_ALIASES = {"visitors", "covers", "guests", "pax", "headcount", "customers", "client_count", "anzahl"}


def _standardize_pos_columns(raw_df):
    """Map common POS export column names to 'visit_date', 'visitors' and 'reservations'."""
    df = raw_df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    col_map = {}
    if "visit_date" not in df.columns:
        for col in df.columns:
            if col in DATE_ALIASES:
                col_map[col] = "visit_date"
                break
    if "visitors" not in df.columns:
        for col in df.columns:
            if col in VISITOR_ALIASES:
                col_map[col] = "visitors"
                break
    if "reservations" not in df.columns:
        for col in df.columns:
            if col in RESERVATION_ALIASES:
                col_map[col] = "reservations"
                break
    return df.rename(columns=col_map) if col_map else df


def extra_columns(df):
    """Optional known-in-advance inputs present in a visits table."""
    return [c for c in df.columns if c == "reservations" or str(c).startswith("event_")]


def history_only(visits):
    """Rows with a real guest count (future rows that only carry events/reservations are dropped)."""
    return visits[visits["visitors"].notna()].reset_index(drop=True)


def clean_visits(raw_df, min_days=MIN_DAYS):
    """Validate and clean an upload.

    Returns (clean_df, notes, error). `notes` is a list of (level, message_key, kwargs);
    `error` is None or (message_key, kwargs). Messages are rendered by the UI (localized).
    Optional columns `reservations` and `event_*` are kept. Rows AFTER the last guest count that
    only carry those columns (empty guest count) are kept too: known future events/bookings.
    """
    raw_df = _standardize_pos_columns(raw_df)
    if "visit_date" not in raw_df.columns or "visitors" not in raw_df.columns:
        return None, [], ("err_columns", {})

    notes = []
    dates_raw = raw_df["visit_date"].astype(str).str.strip()
    # Decide by majority: one junk row must not flip an ISO file to day-first parsing
    # (which would silently drop every date with a day above 12).
    if dates_raw.str.match(r"^\d{4}-\d{2}-\d{2}").mean() >= 0.5:
        parsed = pd.to_datetime(dates_raw, errors="coerce", format="ISO8601")
    else:
        parsed = pd.to_datetime(dates_raw, errors="coerce", dayfirst=True)
        notes.append(("info", "info_dayfirst", {}))
    parsed = parsed.dt.normalize()
    visitors = pd.to_numeric(raw_df["visitors"], errors="coerce")
    extras = extra_columns(raw_df)
    extra_vals = pd.DataFrame({c: pd.to_numeric(raw_df[c], errors="coerce") for c in extras}, index=raw_df.index)

    valid = parsed.notna() & visitors.notna() & (visitors >= 0)
    if not valid.any():
        return None, notes, ("err_no_valid_rows", {})
    future_known = (parsed.notna() & visitors.isna() & extra_vals.notna().any(axis=1)
                    & (parsed > parsed[valid].max())) if extras else pd.Series(False, index=raw_df.index)
    dropped = int((~valid & ~future_known).sum())
    if dropped:
        notes.append(("warning", "warn_dropped", {"n": dropped}))

    rows = valid | future_known
    clean = pd.DataFrame({"visit_date": parsed[rows], "visitors": visitors[rows].astype(float)})
    for c in extras:
        clean[c] = extra_vals.loc[rows, c].astype(float)
    duplicates = int(clean.loc[clean["visitors"].notna(), "visit_date"].duplicated().sum())
    if duplicates:
        notes.append(("info", "warn_duplicates", {"n": duplicates}))
    agg = {"visitors": lambda s: s.sum(min_count=1)}
    agg.update({c: "max" for c in extras})
    clean = clean.groupby("visit_date", as_index=False).agg(agg).sort_values("visit_date")

    # Treat explicit 0-visitor days from POS exports as closed days (absent in `visits` -> NaN on calendar)
    clean = clean[clean["visitors"].isna() | (clean["visitors"] > 0)].reset_index(drop=True)
    hist = history_only(clean)
    if hist.empty:
        return None, notes, ("err_no_valid_rows", {})

    if (hist["visit_date"].max() - hist["visit_date"].min()).days > MAX_SPAN_DAYS:
        return None, notes, ("err_dates", {})
    if len(hist) < min_days:
        return None, notes, ("err_too_short", {"min": min_days, "found": len(hist)})
    return clean, notes, None


CAPITAL_COORDS = {
    "US": (40.7128, -74.0060),   # New York
    "AU": (-33.8688, 151.2093),  # Sydney
    "GB": (51.5074, -0.1278),    # London
    "CA": (43.6532, -79.3832),   # Toronto
    "NZ": (-36.8485, 174.7633),  # Auckland
    "FR": (48.8566, 2.3522),     # Paris
    "DE": (52.5200, 13.4050),    # Berlin
    "AT": (48.2082, 16.3738),    # Vienna
    "CH": (47.3769, 8.5417),     # Zurich
    "BE": (50.8503, 4.3517),     # Brussels
    "IT": (41.9028, 12.4964),    # Rome
    "ES": (40.4168, -3.7038),    # Madrid
    "NL": (52.3676, 4.9041),     # Amsterdam
    "PH": (14.5995, 120.9842),
}


@lru_cache(maxsize=512)
def resolve_location_info(location_query, country="US"):
    """Resolve any city name or postal/ZIP code via urllib + Open-Meteo Geocoding API.
    Returns dict with: lat, lon, name, admin1 (state/region), country_code, matched (bool)."""
    query = str(location_query or "").strip()
    base_country = str(country or "US").split("-", 1)[0].strip().upper()
    fallback_lat, fallback_lon = CAPITAL_COORDS.get(base_country, CAPITAL_COORDS["US"])

    if not query:
        return {
            "lat": fallback_lat, "lon": fallback_lon,
            "name": base_country, "admin1": "", "country_code": base_country, "matched": False,
        }

    for use_country in (True, False):
        params = {"name": query, "count": 5, "language": "en", "format": "json"}
        if use_country and base_country:
            params["countryCode"] = base_country
        url = f"https://geocoding-api.open-meteo.com/v1/search?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TableCastAI/2.5"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            results = payload.get("results")
            if results:
                hit = results[0]
                return {
                    "lat": float(hit["latitude"]),
                    "lon": float(hit["longitude"]),
                    "name": hit.get("name", query),
                    "admin1": hit.get("admin1", ""),
                    "country_code": hit.get("country_code", base_country),
                    "matched": True,
                }
        except Exception:
            break

    return {
        "lat": fallback_lat, "lon": fallback_lon,
        "name": query, "admin1": "", "country_code": base_country, "matched": False,
    }


@lru_cache(maxsize=512)
def resolve_location_coords(location_query, country="FR"):
    """Return (lat, lon) tuple for backward compatibility with training and weather fetchers."""
    info = resolve_location_info(location_query, country)
    return info["lat"], info["lon"]


@lru_cache(maxsize=256)
def _fetch_open_meteo_cached(lat, lon, start_iso, end_iso, today_iso):
    """Cached worker for Open-Meteo Archive & Forecast APIs.
    Columns: temp (daily mean), rain (mm), temp_max, sunshine (hours), snow (cm), rain_evening (mm 17-22h)."""
    start_d = date.fromisoformat(start_iso)
    end_d = date.fromisoformat(end_iso)
    today_d = date.fromisoformat(today_iso)
    max_forecast_d = today_d + timedelta(days=15)
    if start_d > max_forecast_d:
        return None

    cutoff_d = today_d - timedelta(days=5)
    frames = []

    def _call_api(base_url, s_date, e_date):
        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "start_date": s_date.isoformat(),
            "end_date": e_date.isoformat(),
            "daily": "temperature_2m_mean,precipitation_sum,temperature_2m_max,sunshine_duration,snowfall_sum",
            "hourly": "precipitation",
            "timezone": "auto",
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        daily_data = payload.get("daily", {})
        times = daily_data.get("time", [])
        if not times:
            return None
        n = len(times)

        def col(name, scale=1.0):
            return pd.to_numeric(pd.Series(daily_data.get(name, [np.nan] * n)), errors="coerce").to_numpy() * scale

        out = pd.DataFrame({
            "date": pd.to_datetime(times).normalize(),
            "temp": col("temperature_2m_mean"),
            "rain": col("precipitation_sum"),
            "temp_max": col("temperature_2m_max"),
            "sunshine": col("sunshine_duration", 1 / 3600),
            "snow": col("snowfall_sum"),
        })
        hourly = payload.get("hourly", {})
        if hourly.get("time"):
            h = pd.DataFrame({"t": pd.to_datetime(hourly["time"]),
                              "p": pd.to_numeric(pd.Series(hourly.get("precipitation")), errors="coerce")})
            evening = h[(h.t.dt.hour >= 17) & (h.t.dt.hour < 22)]
            per_day = evening.groupby(evening.t.dt.normalize())["p"].sum(min_count=1)
            out["rain_evening"] = out["date"].map(per_day).to_numpy(dtype=float)
        else:
            out["rain_evening"] = np.nan
        return out

    if start_d <= cutoff_d:
        arch_end = min(end_d, cutoff_d)
        df_arch = _call_api("https://archive-api.open-meteo.com/v1/archive", start_d, arch_end)
        if df_arch is not None:
            frames.append(df_arch)

    if end_d > cutoff_d:
        fc_start = max(start_d, cutoff_d + timedelta(days=1))
        fc_end = min(end_d, max_forecast_d)
        if fc_start <= fc_end:
            df_fc = _call_api("https://api.open-meteo.com/v1/forecast", fc_start, fc_end)
            if df_fc is not None:
                frames.append(df_fc)

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True).drop_duplicates("date", keep="last")
    return combined.set_index("date").sort_index()


def _fetch_open_meteo_range(lat, lon, start_iso, end_iso):
    res = _fetch_open_meteo_cached(float(lat), float(lon), str(start_iso), str(end_iso), date.today().isoformat())
    return res.copy() if res is not None else None


def fetch_history_weather(station, start_iso, end_iso, country="FR"):
    """Daily historical weather from Open-Meteo (used as a model feature).
    Returns (DataFrame indexed by date, None) or (None, reason)."""
    try:
        lat, lon = resolve_location_coords(station, country)
        out = _fetch_open_meteo_range(lat, lon, start_iso, end_iso)
        if out is not None and len(out) > 0:
            return out, None
        return None, "no data"
    except Exception as exc:
        return None, type(exc).__name__


def fetch_future_weather(station, country, future_dates, history):
    """Fetch day-by-day weather forecast from Open-Meteo for `future_dates`.
    Falls back to `default_weather(history)` for any missing dates.
    Returns (DataFrame indexed by future_dates, live_weather_ok)."""
    def_temp, def_rain = default_weather(history)
    fallback = pd.DataFrame({"temp": def_temp, "rain": def_rain}, index=future_dates)
    if len(future_dates) == 0:
        return fallback, False
    try:
        lat, lon = resolve_location_coords(station, country)
        raw = _fetch_open_meteo_range(
            lat, lon, future_dates[0].date().isoformat(), future_dates[-1].date().isoformat()
        )
        if raw is None or raw.empty:
            return fallback, False
        reindexed = raw.reindex(future_dates)
        live_ok = bool(reindexed["temp"].notna().any())
        reindexed["temp"] = reindexed["temp"].fillna(def_temp).round(1)
        reindexed["rain"] = reindexed["rain"].fillna(def_rain).round(1)
        return reindexed, live_ok
    except Exception:
        return fallback, False


# ----------------------------------------------------------------------------
# Calendars: public holidays, school holidays, bridge days
# ----------------------------------------------------------------------------
def school_break_flags(idx, country="FR"):
    """Rough school-holiday / tourism-season flag (supports both Northern and Southern hemispheres)."""
    m, d = idx.month.to_numpy(), idx.day.to_numpy()
    base_country = str(country or "FR").split("-", 1)[0].strip().upper()

    if base_country in {"AU", "NZ"}:
        # Southern Hemisphere school holidays: Dec 15 - Jan 28 (Summer), mid-Apr, early Jul, late Sep/early Oct
        flag = (
            ((m == 12) & (d >= 15))
            | ((m == 1) & (d <= 28))
            | ((m == 4) & (d >= 10) & (d <= 25))
            | ((m == 7) & (d >= 1) & (d <= 16))
            | ((m == 9) & (d >= 20))
            | ((m == 10) & (d <= 8))
        )
    else:
        # Northern Hemisphere (US / Europe / Canada / UK)
        flag = (
            ((m == 4) & (d >= 10) & (d <= 25))
            | np.isin(m, [7, 8])
            | ((m == 12) & (d >= 20))
            | ((m == 1) & (d <= 6))
            | ((m == 2) & (d >= 10) & (d <= 25))
            | ((m == 10) & (d >= 24))
        )
    return flag.astype(int)


def get_holiday_dates(country, calendar_index):
    years = list(range(int(calendar_index.year.min()), int(calendar_index.year.max()) + 2))
    parts = str(country or "FR").split("-", 1)
    base_country = parts[0].strip()
    subdiv = parts[1].strip() if len(parts) > 1 else None
    try:
        hol = holidays.country_holidays(base_country, subdiv=subdiv, years=years)
    except Exception:
        hol = holidays.country_holidays(base_country, years=years)
    return pd.DatetimeIndex(pd.to_datetime(list(hol.keys())))


# Austrian semester break week by state (Schulzeitgesetz): 1st, 2nd or 3rd Monday of February.
AT_SEMESTER_WEEK = {"9": 1, "3": 1, "1": 2, "2": 2, "5": 2, "7": 2, "8": 2, "4": 3, "6": 3}
AT_EAST = {"1", "3", "9"}   # Burgenland, Niederösterreich, Wien start summer one week earlier
# School-free state patron days (month, day): Josef, Florian, Rupert, Volksabstimmung, Martin, Leopold
AT_PATRON_DAYS = {"2": [(3, 19), (10, 10)], "6": [(3, 19)], "7": [(3, 19)], "8": [(3, 19)], "4": [(5, 4)],
                  "5": [(9, 24)], "1": [(11, 11)], "3": [(11, 15)], "9": [(11, 15)]}


def _days(start, end):
    return pd.date_range(start, end, freq="D")


def austrian_school_holidays(years, state="9"):
    """Austrian school holidays from the school-law rules (state = holidays subdivision '1'..'9')."""
    state = state if state in AT_SEMESTER_WEEK else "9"
    out = []
    for y in years:
        out.append(_days(date(y, 1, 1), date(y, 1, 6)))                     # Christmas (end)
        xmas_start = date(y, 12, 23) if date(y, 12, 23).weekday() == 0 else date(y, 12, 24)
        out.append(_days(xmas_start, date(y, 12, 31)))                      # Christmas (start)
        out.append(_days(date(y, 11, 2), date(y, 11, 2)))                   # Allerseelen, school-free
        for m, d in AT_PATRON_DAYS.get(state, []):
            out.append(_days(date(y, m, d), date(y, m, d)))
        first_mon = date(y, 2, 1) + timedelta(days=(7 - date(y, 2, 1).weekday()) % 7)
        sem = first_mon + timedelta(weeks=AT_SEMESTER_WEEK[state] - 1)
        out.append(_days(sem - timedelta(days=2), sem + timedelta(days=6)))  # semester break Sat..Sun
        e = easter(y)
        out.append(_days(e - timedelta(days=8), e + timedelta(days=1)))      # Sat before Palm Sunday..Easter Mon
        p = e + timedelta(days=49)
        out.append(_days(p - timedelta(days=1), p + timedelta(days=1)))      # Pentecost Sat..Mon
        if y >= 2020:
            out.append(_days(date(y, 10, 27), date(y, 10, 31)))              # autumn break
        window = (date(y, 6, 28), date(y, 7, 4)) if state in AT_EAST else (date(y, 7, 5), date(y, 7, 11))
        start = window[0] + timedelta(days=(5 - window[0].weekday()) % 7)   # first Saturday in the window
        out.append(_days(start, start + timedelta(days=64)))                # 9 weeks summer
    return pd.DatetimeIndex(sorted(set(np.concatenate([d.values for d in out])))) if out else pd.DatetimeIndex([])


def get_school_dates(country, calendar_index):
    """Exact school holiday dates when known (DE per state, AT per state), else None (-> rough flags)."""
    years = list(range(int(calendar_index.year.min()), int(calendar_index.year.max()) + 2))
    parts = str(country or "").split("-", 1)
    base, subdiv = parts[0].strip().upper(), (parts[1].strip() if len(parts) > 1 else None)
    if base == "AT":
        return austrian_school_holidays(years, subdiv or "9")
    if base == "DE" and subdiv:
        try:
            school = holidays.country_holidays("DE", subdiv=subdiv, years=years, categories=("school",))
        except Exception:
            return None
        if len(school) == 0:
            return None
        return pd.DatetimeIndex(pd.to_datetime(list(school.keys())))
    return None


# ----------------------------------------------------------------------------
# Features (single source of truth for training AND forecasting)
# ----------------------------------------------------------------------------
@dataclass
class FeatureContext:
    holiday_dates: pd.DatetimeIndex
    country: str = "FR"
    options: dict = field(default_factory=lambda: dict(NO_OPTIONS))
    school_dates: pd.DatetimeIndex = None
    known: pd.DataFrame = None          # index = date, columns = reservations / event_* (incl. future)


def _off_day(idx, holiday_dates):
    return (idx.dayofweek >= 5) | idx.isin(holiday_dates)


def build_features(frame, ctx_or_holidays, country="FR", horizon=1):
    """frame: daily-calendar DataFrame with columns visitors (NaN = closed/unknown) and weather.
    Every lag uses only data from at least `horizon` days BEFORE the row's own day."""
    ctx = ctx_or_holidays if isinstance(ctx_or_holidays, FeatureContext) else FeatureContext(
        holiday_dates=ctx_or_holidays, country=country)
    opts = ctx.options
    idx = frame.index
    dow = idx.dayofweek.to_numpy()
    day = idx.day.to_numpy()
    month = idx.month.to_numpy()
    days_in_month = idx.days_in_month.to_numpy()
    is_weekend = (dow >= 5).astype(int)
    temp = frame["temp"].to_numpy(dtype=float)
    rain = frame["rain"].to_numpy(dtype=float)
    holiday_dates = ctx.holiday_dates

    visitors = frame["visitors"]
    past = visitors.shift(horizon)
    avg7 = past.rolling(7, min_periods=3).mean()
    avg30 = past.rolling(30, min_periods=10).mean()

    if opts.get("school_exact") and ctx.school_dates is not None:
        school = idx.isin(ctx.school_dates).astype(int)
    else:
        school = school_break_flags(idx, ctx.country)

    cols = {
        "day_of_week": dow,
        "is_weekend": is_weekend,
        "month": month,
        "day_of_month": day,
        "is_payday": ((day == 1) | (day == 15) | (day >= 25) | (day >= days_in_month - 2)).astype(int),
        "is_holiday": idx.isin(holiday_dates).astype(int),
        "is_holiday_tomorrow": (idx + pd.Timedelta(days=1)).isin(holiday_dates).astype(int),
        "is_school_break": school,
        "temp": temp,
        "rain": rain,
        "rain_weekend_penalty": rain * is_weekend,
        "is_patio_weather": ((temp >= 18.0) & (rain <= 0.5)).astype(int),
        "visitors_7d_avg": avg7.to_numpy(),
        "visitors_30d_avg": avg30.to_numpy(),
        "visitors_7d_max": past.rolling(7, min_periods=3).max().to_numpy(),
        "visitors_30d_max": past.rolling(30, min_periods=10).max().to_numpy(),
        "post_closure_surge": past.isna().astype(int).to_numpy(),
        "visitors_yesterday": past.fillna(avg7).to_numpy(),
        "visitors_same_day_last_week": visitors.shift(7).fillna(avg7).to_numpy(),
        "momentum_ratio": (avg7 / avg30.replace(0, np.nan)).fillna(1.0).clip(0.5, 2.0).to_numpy(),
        "horizon": np.full(len(idx), horizon),
    }

    if opts.get("bridge_days"):
        one = pd.Timedelta(days=1)
        off_today = _off_day(idx, holiday_dates)
        off_prev, off_next = _off_day(idx - one, holiday_dates), _off_day(idx + one, holiday_dates)
        near_holiday = (idx.isin(holiday_dates) | (idx - one).isin(holiday_dates) | (idx + one).isin(holiday_dates)
                        | (idx - 2 * one).isin(holiday_dates) | (idx + 2 * one).isin(holiday_dates))
        cols["is_bridge_day"] = (~off_today & off_prev & off_next).astype(int)
        cols["is_long_weekend"] = (off_today & near_holiday & ((dow >= 4) | (dow <= 0))).astype(int)

    if opts.get("bonus_pay") and str(ctx.country).split("-")[0].upper() in {"AT", "DE"}:
        cols["is_bonus_payday"] = ((((month == 6) | (month == 11)) & (day >= 25))
                                   | (((month == 7) | (month == 12)) & (day <= 5))).astype(int)

    # Same weekday over the last 4 weeks (all known at forecast time for horizons up to 7 days).
    # Benchmarked: lower 7-day error on 3 of 3 datasets, and -0.4 points on an earlier hold-out period.
    same = pd.concat([visitors.shift(7 * k) for k in range(1, 5)], axis=1)
    cols["same_dow_mean4"] = same.mean(axis=1).fillna(avg7).to_numpy()
    cols["same_dow_median4"] = same.median(axis=1).fillna(avg7).to_numpy()

    if opts.get("weather_plus"):
        for c in WEATHER_PLUS:
            cols[c] = frame[c].to_numpy(dtype=float) if c in frame else np.full(len(idx), np.nan)
        trailing = frame["temp"].shift(1).rolling(30, min_periods=7).mean()
        cols["temp_anomaly"] = (frame["temp"] - trailing).to_numpy()

    if ctx.known is not None and (opts.get("events") or opts.get("reservations")):
        known = ctx.known.reindex(idx)
        for c in ctx.known.columns:
            if c.startswith("event_") and opts.get("events"):
                cols[c] = known[c].fillna(0.0).to_numpy()
            elif c == "reservations" and opts.get("reservations"):
                res = known[c].fillna(0.0).to_numpy()
                cols["reservations"] = res
                cols["reservations_share"] = res / np.maximum(1.0, np.nan_to_num(avg30.to_numpy(), nan=1.0))
        if opts.get("events"):
            ev = [c for c in ctx.known.columns if c.startswith("event_")]
            if ev:
                cols["event_any"] = (known[ev].fillna(0.0).to_numpy() > 0).any(axis=1).astype(int)

    # Built in one go (not column by column): this runs hundreds of times per training.
    return pd.DataFrame(cols, index=idx)


def shock_days(history, ctx):
    """Days where 'same as last week' is likely wrong, using only information known in advance:
    public holidays, bridge days, known events, rainy weekends, big temperature swings."""
    idx = history.index
    one = pd.Timedelta(days=1)
    hol = idx.isin(ctx.holiday_dates)
    off_today, off_prev, off_next = (_off_day(idx, ctx.holiday_dates), _off_day(idx - one, ctx.holiday_dates),
                                     _off_day(idx + one, ctx.holiday_dates))
    bridge = ~off_today & off_prev & off_next
    weekend = idx.dayofweek >= 5
    rainy_weekend = weekend & (history["rain"].to_numpy(dtype=float) >= 3.0)
    swing = (history["temp"] - history["temp"].shift(7)).abs().fillna(0).to_numpy() >= SHOCK_TEMP_SWING
    event = np.zeros(len(idx), dtype=bool)
    if ctx.known is not None:
        ev = [c for c in ctx.known.columns if c.startswith("event_")]
        if ev:
            event = (ctx.known[ev].reindex(idx).fillna(0).to_numpy() > 0).any(axis=1)
    return pd.Series(hol | bridge | rainy_weekend | swing | event, index=idx)


def manager_baselines(visitors):
    """'Same day last week' (the usual manager guess) and the 4-week same-weekday average."""
    avg7 = visitors.shift(1).rolling(7, min_periods=3).mean()
    last_week = visitors.shift(7).fillna(avg7)
    four_weeks = pd.concat([visitors.shift(7 * k) for k in range(1, 5)], axis=1).mean(axis=1).fillna(last_week)
    return last_week, four_weeks


# ----------------------------------------------------------------------------
# Staff per role
# ----------------------------------------------------------------------------
SHIFTS = ("lunch", "dinner")
DEFAULT_SHIFT_HOURS = {"lunch": 5.0, "dinner": 6.0}
# A person can absorb a little extra: the next person is only added once a shift is more than 15%
# of one person's load over capacity (31 guests at 15 per waiter = 2 waiters, not 3).
STAFF_SLACK = 0.15
# Sensible casual-restaurant defaults; every restaurant edits these (key = translation key).
# Tuned so a typical casual restaurant lands at ~30% labour cost / revenue.
DEFAULT_ROLES = [
    {"key": "role_waiters", "per": 20.0, "min": 1, "wage": 14.0, "shift": "both"},
    {"key": "role_cooks", "per": 40.0, "min": 1, "wage": 16.0, "shift": "both"},
    {"key": "role_bartenders", "per": 80.0, "min": 0, "wage": 14.0, "shift": "dinner"},
    {"key": "role_dishwashers", "per": 120.0, "min": 1, "wage": 13.0, "shift": "both"},
]


def default_staff_roles(labels, waiter_per=None):
    """DEFAULT_ROLES with names in the user's language; an older saved 'guests per server' carries over."""
    return [{"name": labels.get(r["key"], r["key"]),
             "per": float(waiter_per) if waiter_per and r["key"] == "role_waiters" else r["per"],
             "min": r["min"], "wage": r["wage"], "shift": r["shift"]} for r in DEFAULT_ROLES]


def clean_roles(roles):
    """Validate a role list (from the editor, settings or an upload). Unusable rows are dropped."""
    out = []
    for r in roles or []:
        name = str(r.get("name") or "").strip()
        try:
            per = float(r.get("per") or 0)
            minimum = int(float(r.get("min") or 0))
            wage = float(r.get("wage") or 0)
        except (TypeError, ValueError):
            continue
        shift = str(r.get("shift") or "both").strip().lower()
        if not name or per <= 0:
            continue
        out.append({"name": name, "per": per, "min": max(0, minimum), "wage": max(0.0, wage),
                    "shift": shift if shift in ("both",) + SHIFTS else "both"})
    return out


def people_needed(guests, per_person, minimum=0, slack=STAFF_SLACK):
    """People for one role and one shift: guests / guests-per-person, rounded up only when the shift is
    clearly over capacity (see STAFF_SLACK), never below the minimum crew of an open shift."""
    g = np.asarray(guests, dtype=float)
    n = np.ceil(np.maximum(g / per_person - slack, 0.0))
    return np.where(g > 0, np.maximum(n, minimum), 0).astype(int)


def split_shifts(guests, dates, lunch_share, weekend_lunch_share=None):
    """Guests per day -> (lunch guests, dinner guests), with a separate weekend lunch share."""
    g = np.maximum(0, np.round(np.asarray(guests, dtype=float))).astype(int)
    weekend = pd.DatetimeIndex(dates).dayofweek >= 5
    wknd = lunch_share if weekend_lunch_share is None else weekend_lunch_share
    lunch = np.round(g * np.where(weekend, wknd, lunch_share) / 100.0).astype(int)
    return lunch, g - lunch


def shift_staff(lunch_guests, dinner_guests, roles, slack=STAFF_SLACK):
    """{"lunch": {role: people}, "dinner": {role: people}} for arrays of guests (one entry per day)."""
    out = {}
    for shift, guests in (("lunch", lunch_guests), ("dinner", dinner_guests)):
        out[shift] = {}
        for r in roles:
            works = r["shift"] in ("both", shift)
            people = people_needed(guests, r["per"], r["min"], slack) if works else np.zeros(len(np.atleast_1d(guests)), int)
            out[shift][r["name"]] = np.atleast_1d(people)
    return out


def staffing_outcomes(backtest, roles, lunch_share, weekend_lunch_share=None, shift_hours=None,
                      intervals=None, plan_for="expected"):
    """What a schedule written from the forecast would have meant on past days the model never saw.

    The schedule is written on Sunday for Monday..Sunday (Monday is 1 day ahead, Sunday 7), so every
    test day is counted once. A shift is SHORT if any role had fewer people than the real guests
    needed, OVER if nobody was short but some role had too many. Compared with copying last week."""
    roles = clean_roles(roles)
    if backtest is None or len(backtest) == 0 or not roles:
        return None
    bt = backtest[backtest["horizon"] == pd.to_datetime(backtest["date"]).dt.dayofweek + 1]
    if bt.empty:
        return None
    hours = {**DEFAULT_SHIFT_HOURS, **(shift_hours or {})}
    dates = pd.to_datetime(bt["date"])
    model = bt["pred"].to_numpy(dtype=float)
    if plan_for == "busy" and intervals is not None:
        model = model * np.array([intervals[int(h)][1] for h in bt["horizon"]])

    def staff(guests):
        return shift_staff(*split_shifts(guests, dates, lunch_share, weekend_lunch_share), roles)

    need = staff(bt["actual"])
    out = {"days": int(len(bt)), "weeks": len(bt) / 7.0}
    for name, guesses in (("model", model), ("base", bt["base"].to_numpy(dtype=float))):
        got = staff(guesses)
        short = over = 0
        extra_hours = missing_hours = extra_cost = 0.0
        for shift in SHIFTS:
            diff = np.stack([got[shift][r["name"]] - need[shift][r["name"]] for r in roles])
            is_short = (diff < 0).any(axis=0)
            short += int(is_short.sum())
            over += int((~is_short & (diff > 0).any(axis=0)).sum())
            extra_hours += float(np.clip(diff, 0, None).sum() * hours[shift])
            missing_hours += float(np.clip(-diff, 0, None).sum() * hours[shift])
            wages = np.array([r["wage"] for r in roles])[:, None]
            extra_cost += float((np.clip(diff, 0, None) * wages).sum() * hours[shift])
        out[name] = {"short": short, "over": over, "extra_hours": extra_hours,
                     "missing_hours": missing_hours, "extra_cost": extra_cost}
    out["open_shifts"] = int(sum((np.asarray(v) > 0).sum() for v in split_shifts(bt["actual"], dates, lunch_share,
                                                                                weekend_lunch_share)))
    return out


def unusual_day_misses(backtest):
    """Average miss in guests on unusual days (holidays, events, ...): forecast vs 'same day last week'."""
    if backtest is None or len(backtest) == 0 or "shock" not in backtest:
        return None
    sh = backtest[backtest["shock"]]
    if sh.empty:
        return None
    return {"model": float((sh["actual"] - sh["pred"]).abs().mean()),
            "base": float((sh["actual"] - sh["base"]).abs().mean()),
            "days": int(sh["date"].nunique()),
            "share": float(backtest.drop_duplicates("date")["shock"].mean() * 100)}


def feature_columns(ctx, weather_ok):
    """Which feature columns a model trains on, given the options and the data available."""
    opts = ctx.options
    cols = list(BASE_FEATURES)
    if opts.get("direct"):
        cols.append("horizon")
    cols += ["same_dow_mean4", "same_dow_median4"]
    if opts.get("bridge_days"):
        cols += ["is_bridge_day", "is_long_weekend"]
    if opts.get("bonus_pay") and str(ctx.country).split("-")[0].upper() in {"AT", "DE"}:
        cols.append("is_bonus_payday")
    if opts.get("weather_plus") and weather_ok:
        cols += WEATHER_PLUS + ["temp_anomaly"]
    if ctx.known is not None:
        ev = [c for c in ctx.known.columns if c.startswith("event_")]
        if opts.get("events") and ev:
            cols += ev + ["event_any"]
        if opts.get("reservations") and "reservations" in ctx.known.columns:
            cols += ["reservations", "reservations_share"]
    return cols


def detect_closed_weekdays(visits, calendar_index):
    observed = visits["visit_date"].dt.dayofweek.value_counts()
    expected = pd.Series(calendar_index.dayofweek).value_counts()
    closed = {
        d for d in range(7)
        if expected.get(d, 0) > 0 and observed.get(d, 0) / expected[d] < CLOSED_THRESHOLD
    }
    return frozenset() if len(closed) >= 7 else frozenset(closed)


def wape(actual, predicted):
    """Weighted absolute percentage error: sum|error| / sum(actual), in %."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    total = actual.sum()
    return float(np.abs(actual - predicted).sum() / total * 100) if total > 0 else float("nan")


def fmt_pct(value):
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.1f}%"


# ----------------------------------------------------------------------------
# Models: single models, presets and custom blends
# ----------------------------------------------------------------------------
class RatioModel:
    """Predicts guests relative to the 30-day average (e.g. 1.2x normal), so trees can follow growth."""

    def __init__(self, model):
        self.model = model

    def fit(self, X, y, sample_weight=None):
        level = np.maximum(1.0, X["visitors_30d_avg"].to_numpy(dtype=float))
        self.model.fit(X, np.asarray(y, dtype=float) / level, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(X) * np.maximum(1.0, X["visitors_30d_avg"].to_numpy(dtype=float))


MODEL_MEMBERS = ["extra_trees", "xgboost", "ridge", "random_forest", "gradient_boosting",
                 "gradient_boosting_ratio"] + (["catboost"] if CatBoostRegressor is not None else [])
MODEL_PRESETS = {
    "recommended": ("extra_trees", "xgboost", "ridge"),   # benchmarked best for restaurants
    "classic": ("random_forest", "gradient_boosting_ratio"),  # the original TableCast model
}
DEFAULT_MODEL = MODEL_PRESETS["recommended"]


def _make_member(key):
    if key == "extra_trees":
        return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=N_THREADS)
    if key == "xgboost":
        return XGBRegressor(n_estimators=500, learning_rate=0.04, max_depth=5, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=3, n_jobs=N_THREADS, random_state=42)
    if key == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
    if key == "random_forest":
        return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=N_THREADS)
    if key == "gradient_boosting":
        return HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=42)
    if key == "gradient_boosting_ratio":
        return RatioModel(HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=42))
    if key == "catboost" and CatBoostRegressor is not None:
        return CatBoostRegressor(iterations=600, depth=6, learning_rate=0.05, random_seed=42,
                                 verbose=False, thread_count=N_THREADS)
    raise ValueError(f"unknown model: {key}")


def _fit_member(key, X, y, sample_weight=None):
    model = _make_member(key)
    if key == "ridge":
        model.fit(X, y, ridgecv__sample_weight=sample_weight)
    else:
        model.fit(X, y, sample_weight=sample_weight)
    if key == "random_forest":
        model.set_params(n_jobs=1)
    return model


class BlendedModel:
    """Equal-weight average of its members (one member = a single model).

    Default: two tree models (ExtraTrees, XGBoost) + one linear model (Ridge). The trees capture
    combinations such as "Saturday + sunny + payday"; the linear model follows trends smoothly.
    Benchmarked against the original RandomForest + gradient-boosting model in TableCast's own
    7-day backtest: better on 12 of 12 real restaurants/bakeries."""

    def __init__(self, *models, keys=()):
        self.models = models
        self.keys = tuple(keys)

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


def normalize_model_spec(spec):
    if isinstance(spec, str):
        spec = MODEL_PRESETS.get(spec, (spec,))
    keys = tuple(k for k in (spec or DEFAULT_MODEL) if k in MODEL_MEMBERS)
    return keys or DEFAULT_MODEL


def fit_model(X, y, model_spec=None, sample_weight=None):
    keys = normalize_model_spec(model_spec)
    return BlendedModel(*[_fit_member(k, X, y, sample_weight) for k in keys], keys=keys)


fit_forest = fit_model  # old name, kept so older scripts keep working


def recency_weights(dates, last_date):
    age = (pd.Timestamp(last_date) - pd.DatetimeIndex(dates)).days.to_numpy(dtype=float)
    return np.power(0.5, age / RECENCY_HALF_LIFE_DAYS)


@dataclass
class Forecaster:
    model: object
    ctx: FeatureContext
    closed_weekdays: frozenset
    medians: pd.Series
    columns: list = field(default_factory=lambda: list(BASE_FEATURES))

    @property
    def holiday_dates(self):
        return self.ctx.holiday_dates

    @property
    def country(self):
        return self.ctx.country

    @property
    def direct(self):
        return bool(self.ctx.options.get("direct"))

    def forecast(self, history: pd.DataFrame, future_weather: pd.DataFrame) -> pd.Series:
        return self.forecast_many([(history, future_weather)])[0]

    def forecast_many(self, jobs):
        """Forecasts for several (history, future_weather) pairs at once.

        Recursive mode: each day's prediction feeds the next day's features; step k of every pair is
        predicted in ONE model call. Direct mode: day k is predicted straight from the history (the
        model was trained for k days ahead), so nothing is fed back. Either way, results are identical
        to forecasting each pair alone; the backtest just makes 7 model calls instead of ~800."""
        with threadpool_limits(N_THREADS):
            frames, preds = [], [{} for _ in jobs]
            for history, weather in jobs:
                cols = ["visitors"] + [c for c in history.columns if c != "visitors"]
                frames.append(history[cols].copy())
            for step in range(max((len(w) for _, w in jobs), default=0)):
                rows, owners = [], []
                for i, (_, weather) in enumerate(jobs):
                    if step >= len(weather):
                        continue
                    d, w = weather.index[step], weather.iloc[step]
                    frame = frames[i]
                    new_row = {c: (float(w[c]) if c in w.index and pd.notna(w[c]) else np.nan) for c in frame.columns}
                    new_row["visitors"] = np.nan
                    frame.loc[d] = [new_row[c] for c in frame.columns]
                    if d.dayofweek in self.closed_weekdays:
                        preds[i][d] = np.nan
                        continue
                    horizon = step + 1 if self.direct else 1
                    feats = build_features(frame.tail(LOOKBACK_DAYS + horizon), self.ctx, horizon=horizon)
                    rows.append(feats[self.columns].iloc[[-1]].fillna(self.medians))
                    owners.append((i, d))
                if rows:
                    values = self.model.predict(pd.concat(rows))
                    for (i, d), value in zip(owners, values):
                        pred = max(0.0, float(value))
                        if not self.direct:
                            frames[i].loc[d, "visitors"] = pred
                        preds[i][d] = pred
            return [pd.Series(p, dtype=float) for p in preds]


def validate_seven_day(forecaster, history, features, first_test_date, baseline=None, baseline4=None, shock=None):
    """Rolling-origin 7-day-ahead backtest across the test period (stepping by 1 day).
    Stepping by 1 day provides 7x more out-of-sample error observations than non-overlapping
    windows, enabling reliable empirical prediction intervals even on 60-to-90-day datasets."""
    step = pd.Timedelta(days=1)
    horizon_delta = pd.Timedelta(days=HORIZON)
    last_date = history.index[-1]
    origin = first_test_date - pd.Timedelta(days=1)
    weather_cols = [c for c in history.columns if c != "visitors"]
    windows, jobs = [], []
    while origin + horizon_delta <= last_date:
        future = pd.date_range(origin + pd.Timedelta(days=1), periods=HORIZON, freq="D")
        windows.append(future)
        jobs.append((history.loc[:origin].tail(LOOKBACK_DAYS + HORIZON), history.loc[future, weather_cols]))
        origin += step

    if baseline is None:
        baseline = features["visitors_same_day_last_week"]
    if baseline4 is None:
        baseline4 = baseline
    rows = []
    ratios = {h: [] for h in range(1, HORIZON + 1)}
    for future, preds in zip(windows, forecaster.forecast_many(jobs)):
        preds = preds.reindex(future)
        for h, d in enumerate(future, start=1):
            act, pred = history.at[d, "visitors"], preds[d]
            if np.isfinite(pred) and np.isfinite(act) and pred > 0:
                ratios[h].append(float(act / pred))
            rows.append((d, h, act, pred, baseline.get(d, np.nan), baseline4.get(d, np.nan),
                         bool(shock.get(d, False)) if shock is not None else False))
    bt = pd.DataFrame(rows, columns=["date", "horizon", "actual", "pred", "base", "base4", "shock"])
    bt = bt[bt["actual"].notna() & bt["pred"].notna() & bt["base"].notna()].reset_index(drop=True)
    if bt.empty:
        nan = float("nan")
        return {"wape7": nan, "base_wape7": nan, "base4_wape7": nan, "shock_wape7": nan, "base_shock_wape7": nan,
                "shock_share": nan, "ratios_by_h": ratios, "backtest": bt}
    bt["base4"] = bt["base4"].fillna(bt["base"])
    sh = bt[bt["shock"]]
    return {
        "wape7": wape(bt["actual"], bt["pred"]),
        "base_wape7": wape(bt["actual"], bt["base"]),
        "base4_wape7": wape(bt["actual"], bt["base4"]),
        "shock_wape7": wape(sh["actual"], sh["pred"]) if len(sh) else float("nan"),
        "base_shock_wape7": wape(sh["actual"], sh["base"]) if len(sh) else float("nan"),
        "shock_share": float(bt["shock"].mean() * 100),
        "ratios_by_h": ratios,
        "backtest": bt,
    }


def compute_intervals(ratios_by_h):
    """Turn past actual/predicted ratios into {h: (low_ratio, high_ratio)} for h = 1..HORIZON."""
    arrays = [np.asarray(v, dtype=float) for v in ratios_by_h.values() if len(v)]
    pooled = np.concatenate(arrays) if arrays else np.array([])
    if len(pooled) < INTERVAL_MIN_SAMPLES:
        return None
    pooled_q = np.percentile(pooled, [INTERVAL_LOW, INTERVAL_HIGH])
    out = {}
    for h in range(1, HORIZON + 1):
        values = np.asarray(ratios_by_h.get(h, []), dtype=float)
        q = np.percentile(values, [INTERVAL_LOW, INTERVAL_HIGH]) if len(values) >= INTERVAL_MIN_SAMPLES else pooled_q
        out[h] = (float(q[0]), float(q[1]))
    return out


def train_bundle(visits, station, country, events=None, model_spec=None, options=None):
    """Everything expensive, computed once per (data, station, country, events, model, options).
    Returns None when there is not enough usable history."""
    with threadpool_limits(N_THREADS):
        return _train_bundle(visits, station, country, events, normalize_model_spec(model_spec),
                             resolve_options(options))


def _training_set(history, ctx, columns, horizons):
    """Stacked (X, y, target dates) for the given horizons; rows whose features are incomplete are dropped."""
    Xs, ys = [], []
    for h in horizons:
        feats = build_features(history, ctx, horizon=h)
        usable = history["visitors"].notna() & feats[columns].notna().all(axis=1)
        Xs.append(feats.loc[usable, columns])
        ys.append(history.loc[usable, "visitors"])
    return pd.concat(Xs), pd.concat(ys)


def _train_bundle(visits, station, country, events, model_spec, options):
    known_cols = extra_columns(visits)
    known = visits.set_index("visit_date")[known_cols].sort_index() if known_cols else None
    visits = history_only(visits)
    calendar = pd.date_range(visits["visit_date"].min(), visits["visit_date"].max(), freq="D")
    raw_visitors = visits.set_index("visit_date")["visitors"].reindex(calendar)
    if events:
        extra_series = pd.Series(
            [float(events.get(d.strftime("%Y-%m-%d"), 0)) for d in calendar],
            index=calendar,
            dtype=float,
        )
        adjusted = np.where(
            raw_visitors.notna(),
            np.maximum(1.0, raw_visitors - extra_series),
            np.nan,
        )
        raw_visitors = pd.Series(adjusted, index=calendar)
    history = pd.DataFrame({"visitors": raw_visitors})

    weather, weather_reason = fetch_history_weather(
        station, calendar[0].date().isoformat(), calendar[-1].date().isoformat(), country=country
    )
    weather_ok = weather is not None and bool(weather["temp"].notna().any() or weather["rain"].notna().any())
    if weather_ok:
        temp = weather["temp"].reindex(calendar)
        rain = weather["rain"].reindex(calendar)
        history["temp"] = temp.fillna(temp.mean() if temp.notna().any() else DEFAULT_TEMP)
        history["rain"] = rain.fillna(0.0)
        if options.get("weather_plus"):
            for c in WEATHER_PLUS:
                series = weather[c].reindex(calendar) if c in weather else pd.Series(np.nan, index=calendar)
                history[c] = series.fillna(series.mean() if series.notna().any() else 0.0)
    else:
        history["temp"] = DEFAULT_TEMP
        history["rain"] = 0.0

    holiday_dates = get_holiday_dates(country, calendar)
    school_dates = get_school_dates(country, calendar) if options.get("school_exact") else None
    ctx = FeatureContext(holiday_dates=holiday_dates, country=country, options=options,
                         school_dates=school_dates, known=known)
    columns = feature_columns(ctx, weather_ok)
    closed = detect_closed_weekdays(visits, calendar)
    features = build_features(history, ctx)  # day-1 features: charts, next-day error, baseline

    usable = history["visitors"].notna() & features[columns].notna().all(axis=1)
    usable_dates = history.index[usable]
    if len(usable_dates) < MIN_TRAIN_ROWS:
        return None

    # 1) Honest validation: train on the first 80%, test on the most recent 20%.
    cut = usable_dates[int(len(usable_dates) * (1 - TEST_FRACTION))]
    horizons = range(1, HORIZON + 1) if options.get("direct") else (1,)
    X_all, y_all = _training_set(history, ctx, columns, horizons)
    train_mask = X_all.index < cut
    X_train, y_train = X_all[train_mask], y_all[train_mask]
    w_train = recency_weights(X_train.index, cut) if options.get("recency") else None
    val_model = fit_model(X_train, y_train, model_spec, w_train)
    val_forecaster = Forecaster(val_model, ctx, closed, X_train.median(), columns)

    X_test = features.loc[usable & (history.index >= cut), columns]
    y_test = history.loc[X_test.index, "visitors"]
    preds = val_model.predict(X_test)
    last_week, four_weeks = manager_baselines(history["visitors"])
    baseline = last_week.loc[X_test.index].to_numpy()
    validation = validate_seven_day(val_forecaster, history, features, X_test.index[0],
                                    baseline=last_week, baseline4=four_weeks, shock=shock_days(history, ctx))

    n_chart = min(30, len(X_test))
    chart = pd.DataFrame(
        {"actual": y_test.to_numpy()[-n_chart:], "pred": preds[-n_chart:], "baseline": baseline[-n_chart:]},
        index=X_test.index[-n_chart:],
    )

    # 2) The model used for the planner is refit on ALL the data.
    w_all = recency_weights(X_all.index, history.index[-1]) if options.get("recency") else None
    final_model = fit_model(X_all, y_all, model_spec, w_all)
    dow_means = visits.groupby(visits["visit_date"].dt.dayofweek)["visitors"].mean().reindex(range(7)).fillna(0.0)

    return {
        "forecaster": Forecaster(final_model, ctx, closed, X_all.median(), columns),
        "history": history,
        "closed_weekdays": closed,
        "mean_visitors": float(visits["visitors"].mean()),
        "mae": float(mean_absolute_error(y_test, preds)),
        "wape": wape(y_test, preds),
        "base_wape": wape(y_test, baseline),
        "wape7": validation["wape7"],
        "base_wape7": validation["base_wape7"],
        "base4_wape7": validation["base4_wape7"],
        "shock_wape7": validation["shock_wape7"],
        "base_shock_wape7": validation["base_shock_wape7"],
        "shock_share": validation["shock_share"],
        "backtest": validation["backtest"],
        "intervals": compute_intervals(validation["ratios_by_h"]),
        "chart": chart,
        "dow_means": dow_means,
        "weather_ok": weather_ok,
        "weather_reason": weather_reason,
        "model": model_spec,
        "options": options,
        "features": columns,
    }


# ----------------------------------------------------------------------------
# Planning helpers (shared by the app and the weekly email)
# ----------------------------------------------------------------------------
def future_dates_after(history, days=HORIZON):
    return pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=days, freq="D")


def default_weather(history):
    """Neutral weather assumption: the last week's average temperature, no rain."""
    return float(np.clip(round(history["temp"].tail(7).mean()), -10, 40)), 0.0


def build_plan(
    base_preds, intervals, last_date, event_date, event_extra,
    lunch_share, avg_spend, guests_per_server, events=None,
    weekend_lunch_share=None, dinner_spend=None, roles=None, shift_hours=None, plan_for="expected",
):
    """Turn raw model predictions into the planner rows (numbers only, no formatting).

    Supports distinct weekday vs. weekend lunch shares and separate lunch vs. dinner check averages.
    Staff: one entry per role (see DEFAULT_ROLES); without roles, a single server role at
    guests_per_server (the original behaviour). plan_for="busy" staffs for the top of the range.
    """
    plan = []
    events_map = events or {}
    wknd_share = lunch_share if weekend_lunch_share is None else weekend_lunch_share
    din_spend = avg_spend if dinner_spend is None else dinner_spend
    roles = clean_roles(roles)
    slack = STAFF_SLACK
    if not roles:
        roles, slack = [{"name": "servers", "per": float(guests_per_server), "min": 0, "wage": 0.0, "shift": "both"}], 0.0
    hours = {**DEFAULT_SHIFT_HOURS, **(shift_hours or {})}
    wage = {r["name"]: r["wage"] for r in roles}

    for d, base in base_preds.items():
        d_iso = d.strftime("%Y-%m-%d")
        saved_extra = int(events_map.get(d_iso, 0))
        single_extra = int(event_extra) if (event_date is not None and d.date() == event_date) else 0
        extra = single_extra if single_extra > 0 else saved_extra
        on_event = extra > 0
        regularly_closed = bool(pd.isna(base))
        is_closed = regularly_closed and not on_event
        expected = extra if regularly_closed else int(round(base)) + extra

        low = high = None
        if intervals is not None:
            if regularly_closed:
                low = high = expected
            else:
                h = min(HORIZON, max(1, (d - last_date).days))
                q_lo, q_hi = intervals[h]
                low = min(expected, int(round(base * q_lo)) + extra)
                high = max(expected, int(round(base * q_hi)) + extra)

        staff = {s: {r["name"]: 0 for r in roles} for s in SHIFTS}
        if is_closed or expected == 0:
            lunch = dinner = 0
            revenue = 0.0
        else:
            active_share = wknd_share if d.dayofweek >= 5 else lunch_share
            lunch = int(round(expected * active_share / 100))
            dinner = expected - lunch
            revenue = round(lunch * avg_spend + dinner * din_spend, 2)
            staff_guests = high if plan_for == "busy" and high is not None else expected
            staff_lunch = int(round(staff_guests * active_share / 100))
            people = shift_staff(np.array([staff_lunch]), np.array([staff_guests - staff_lunch]), roles, slack)
            staff = {s: {name: int(v[0]) for name, v in people[s].items()} for s in SHIFTS}

        lunch_staff, dinner_staff = sum(staff["lunch"].values()), sum(staff["dinner"].values())
        plan.append({
            "date": d, "expected": expected, "low": low, "high": high,
            "is_closed": is_closed, "on_event": on_event,
            "lunch": lunch, "dinner": dinner,
            "staff": staff,
            "lunch_servers": lunch_staff, "dinner_servers": dinner_staff,
            "servers": lunch_staff + dinner_staff,
            "labour_hours": sum(n * hours[s] for s in SHIFTS for n in staff[s].values()),
            "labour_cost": round(sum(n * hours[s] * wage[name] for s in SHIFTS for name, n in staff[s].items()), 2),
            "revenue": revenue,
        })
    return plan


# ----------------------------------------------------------------------------
# Kitchen prep: per-guest amounts, recipes, or learned from past usage
# ----------------------------------------------------------------------------
PREP_SERVICES = ("all", "lunch", "dinner")
_ITEM_ALIASES = ("item", "ingredient", "zutat", "artikel", "product", "produkt", "ingrédient", "article")
_PER_GUEST_ALIASES = ("per guest", "per_guest", "per cover", "per_cover", "pro gast", "pro_gast", "par client", "rate")
_UNIT_ALIASES = ("unit", "einheit", "unité")
_PACK_ALIASES = ("pack", "pack size", "pack_size", "packung", "gebinde", "packungsgröße", "conditionnement")
_SERVICE_ALIASES = ("service", "shift", "schicht")
_PLAN_ALIASES = ("plan for", "plan_for", "planen für")
_DISH_ALIASES = ("dish", "gericht", "menu item", "plat", "speise")
_AMOUNT_ALIASES = ("amount", "menge", "quantity per dish", "qty", "quantité", "per dish", "pro gericht")
_SHARE_ALIASES = ("share", "anteil", "orders per guest", "% of guests", "popularity", "part")
_USED_ALIASES = ("used", "usage", "verbrauch", "quantity", "menge verbraucht", "consumed", "consommé")
_DATE_ALIASES_PREP = ("date", "datum", "day", "tag")


def _pick(columns, aliases):
    for a in aliases:
        if a in columns:
            return a
    return None


def _num(series):
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")


def parse_prep_file(raw_bytes, visits=None):
    """Read a manager's own prep file. Three formats are recognised by their columns:

      list    item, per guest, unit [, pack, service, plan for]       amounts per guest, as typed
      recipe  dish, share, ingredient, amount, unit [, pack]          share = orders per guest (0.3 or 30%)
      usage   date, item, used, unit [, pack]                         past usage; per-guest amounts are
                                                                      learned from the guests on those days
    Returns (items, kind, error). Items use the keys Item / Per Guest / Unit / Pack / Service / Plan For."""
    try:
        df = read_csv_bytes(raw_bytes)
    except Exception as exc:  # noqa: BLE001 - any parse problem is shown to the user
        return [], None, f"could not read the file ({type(exc).__name__})"
    cols = list(df.columns)
    item_c, unit_c, pack_c = _pick(cols, _ITEM_ALIASES), _pick(cols, _UNIT_ALIASES), _pick(cols, _PACK_ALIASES)
    dish_c, date_c = _pick(cols, _DISH_ALIASES), _pick(cols, _DATE_ALIASES_PREP)
    if item_c is None:
        return [], None, "no item / ingredient column"

    def unit_of(frame):
        return frame[unit_c].astype(str).str.strip() if unit_c else pd.Series("pcs", index=frame.index)

    def pack_of(frame):
        return _num(frame[pack_c]).fillna(0.0) if pack_c else pd.Series(0.0, index=frame.index)

    if dish_c is not None:  # recipe: per guest = sum over dishes of (orders per guest x amount per dish)
        share_c, amount_c = _pick(cols, _SHARE_ALIASES), _pick(cols, _AMOUNT_ALIASES)
        if share_c is None or amount_c is None:
            return [], "recipe", "a recipe file needs a share (orders per guest) and an amount column"
        share = _num(df[share_c].astype(str).str.rstrip("%"))
        share = share.where(share <= 1.0, share / 100.0)  # 30 or 30% -> 0.30
        frame = pd.DataFrame({"Item": df[item_c].astype(str).str.strip(), "Unit": unit_of(df),
                              "per": share * _num(df[amount_c]), "Pack": pack_of(df)}).dropna(subset=["per"])
        grouped = frame.groupby(["Item", "Unit"], sort=False).agg(per=("per", "sum"), Pack=("Pack", "max")).reset_index()
        kind = "recipe"
    elif date_c is not None:  # usage log: per guest = total used / total guests on the same days
        used_c = _pick(cols, _USED_ALIASES)
        if used_c is None:
            return [], "usage", "a usage file needs a used / quantity column"
        if visits is None or len(visits) == 0:
            return [], "usage", "upload your guest numbers first"
        guests = visits.set_index(pd.to_datetime(visits["visit_date"]).dt.normalize())["visitors"]
        frame = pd.DataFrame({"date": pd.to_datetime(df[date_c], errors="coerce", dayfirst=True).dt.normalize(),
                              "Item": df[item_c].astype(str).str.strip(), "Unit": unit_of(df),
                              "used": _num(df[used_c]), "Pack": pack_of(df)}).dropna(subset=["date", "used"])
        frame = frame[frame["date"].isin(guests[guests > 0].index)]
        if frame.empty:
            return [], "usage", "none of the dates match days in your guest data"
        rows = []
        for (name, unit), g in frame.groupby(["Item", "Unit"], sort=False):
            days = g["date"].unique()
            rows.append({"Item": name, "Unit": unit, "per": g["used"].sum() / guests.loc[days].sum(),
                         "Pack": g["Pack"].max()})
        grouped = pd.DataFrame(rows)
        kind = "usage"
    else:
        per_c = _pick(cols, _PER_GUEST_ALIASES)
        if per_c is None:
            return [], "list", "no per-guest column"
        grouped = pd.DataFrame({"Item": df[item_c].astype(str).str.strip(), "Unit": unit_of(df),
                                "per": _num(df[per_c]), "Pack": pack_of(df)}).dropna(subset=["per"])
        kind = "list"

    service_c, plan_c = _pick(cols, _SERVICE_ALIASES), _pick(cols, _PLAN_ALIASES)
    items = []
    for i, r in grouped.reset_index(drop=True).iterrows():
        if not r["Item"] or r["Item"].lower() == "nan" or r["per"] <= 0:
            continue
        item = {"Item": r["Item"], "Per Guest": round(float(r["per"]), 4), "Unit": r["Unit"] or "pcs",
                "Pack": float(r["Pack"]) if r["Pack"] > 0 else None, "Service": "all", "Plan For": "expected"}
        if kind == "list":
            src = df.loc[grouped.index[i]]
            if service_c and str(src[service_c]).strip().lower() in PREP_SERVICES:
                item["Service"] = str(src[service_c]).strip().lower()
            if plan_c and str(src[plan_c]).strip().lower() in ("expected", "busy"):
                item["Plan For"] = str(src[plan_c]).strip().lower()
        items.append(item)
    if not items:
        return [], kind, "no usable rows"
    return items, kind, None


def prep_quantities(plan, items, buffer_pct=0.0):
    """Amounts per open day and item, plus week totals.

    guests  expected guests, or the top of the range for items marked "busy" (never run out)
    service all day, or only the lunch / dinner guests
    amount  guests x per guest x (1 + buffer) for "expected" items (busy items already have their margin),
            rounded UP to whole packs when a pack size is given (buns in bags of 12, ...)
    Returns (rows, totals): rows = [{"date", "closed", "amounts": {item: qty}}], totals = {item: qty}."""
    rows, totals = [], {}
    for row in plan:
        amounts = {}
        for it in items:
            name = str(it.get("Item") or "").strip()
            try:
                per = float(it.get("Per Guest") or 0)
            except (TypeError, ValueError):
                per = 0.0
            if not name or per <= 0:
                continue
            busy = str(it.get("Plan For") or "expected") == "busy"
            guests = row["high"] if busy and row.get("high") is not None else row["expected"]
            service = str(it.get("Service") or "all")
            if service in ("lunch", "dinner") and row["expected"] > 0:
                guests = guests * row[service] / row["expected"]
            qty = 0.0 if row["is_closed"] else guests * per * (1.0 if busy else 1.0 + buffer_pct / 100.0)
            try:
                pack = float(it.get("Pack") or 0)
            except (TypeError, ValueError):
                pack = 0.0
            if pack > 0 and qty > 0:
                qty = math.ceil(qty / pack - 1e-9) * pack
            amounts[name] = qty
            totals[name] = totals.get(name, 0.0) + qty
        rows.append({"date": row["date"], "closed": row["is_closed"], "amounts": amounts})
    return rows, totals

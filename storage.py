"""Saved locations (history + settings) and the forecast track record.

Everything is plain files under a data folder (default ./data, override with the
FORECASTER_DATA_DIR environment variable):

    data/locations/<slug>/visits.csv      cleaned history (visit_date, visitors)
    data/locations/<slug>/settings.json   name, language, country, station, prices, recipients
    data/locations/<slug>/events.json     saved private events
    data/forecast_log.csv                 every forecast you chose to save
"""
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

LOG_COLUMNS = ["location", "made_on", "target_date", "predicted", "low", "high"]


def data_dir():
    return Path(os.environ.get("FORECASTER_DATA_DIR", "data"))


def slugify(name):
    """'Café Zürich #2' -> 'cafe-zurich-2' (safe folder name). Empty names become 'default'."""
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return slug or "default"


def _location_dir(slug):
    return data_dir() / "locations" / slug


# ----------------------------------------------------------------------------
# Saved locations
# ----------------------------------------------------------------------------
def load_settings(slug):
    path = _location_dir(slug) / "settings.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(slug, settings):
    folder = _location_dir(slug)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "settings.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")


def jsonable(obj):
    """Settings as plain JSON values: NaN (empty cells) -> None, numpy numbers -> Python numbers."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def load_events(slug):
    """Return saved custom events for a location as {'YYYY-MM-DD': extra_guests_int}."""
    path = _location_dir(slug) / "events.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in raw.items() if int(v) > 0}
    except (OSError, ValueError, TypeError):
        return {}


def save_events(slug, events):
    """Persist custom events dictionary {'YYYY-MM-DD': extra_guests_int} for a location."""
    folder = _location_dir(slug)
    folder.mkdir(parents=True, exist_ok=True)
    clean = {str(k): int(v) for k, v in (events or {}).items() if int(v) > 0}
    (folder / "events.json").write_text(json.dumps(clean, indent=2, sort_keys=True), encoding="utf-8")
    return clean


STAFF_COLUMNS = ["date", "shift", "role", "people"]


def load_staff(slug):
    """Who actually worked (see core.parse_staff_file), or None."""
    path = _location_dir(slug) / "staff_actual.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df[STAFF_COLUMNS] if set(STAFF_COLUMNS) <= set(df.columns) else None


def save_staff(slug, rows):
    """Merge newly uploaded staff rows into the saved ones; days in the new file replace the old days."""
    old = load_staff(slug)
    if old is not None and len(old):
        old = old[~old["date"].isin(set(rows["date"]))]
        rows = pd.concat([old, rows], ignore_index=True)
    rows = rows.sort_values(["date", "shift", "role"]).reset_index(drop=True)
    folder = _location_dir(slug)
    folder.mkdir(parents=True, exist_ok=True)
    rows[STAFF_COLUMNS].to_csv(folder / "staff_actual.csv", index=False)
    return rows


def list_locations():
    """[(slug, display_name), ...] for every location that has saved history or settings."""
    root = data_dir() / "locations"
    if not root.exists():
        return []
    found = []
    for folder in sorted(root.iterdir()):
        if folder.is_dir() and ((folder / "visits.csv").exists() or (folder / "settings.json").exists()):
            found.append((folder.name, load_settings(folder.name).get("name", folder.name)))
    return found


def location_exists(slug: str) -> bool:
    """Return True if a restaurant slug already has saved settings or password."""
    folder = _location_dir(slugify(slug))
    return (folder / "settings.json").exists() or (folder / "visits.csv").exists()


def load_visits(slug):
    path = _location_dir(slug) / "visits.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["visit_date"])
    df["visitors"] = pd.to_numeric(df["visitors"], errors="coerce")
    extras = [c for c in df.columns if c == "reservations" or str(c).startswith("event_")]
    # Keep rows with a guest count, plus future rows that only carry events/reservations.
    keep = df["visit_date"].notna() & (df["visitors"].notna() | df[extras].notna().any(axis=1) if extras
                                       else df["visitors"].notna())
    return df[keep].sort_values("visit_date").reset_index(drop=True)


def merge_visits(existing, new):
    """Combine saved history with newly uploaded days; for the same date the new value wins.
    Optional columns (reservations, event_*) from either side are kept."""
    if existing is None or len(existing) == 0:
        return new.sort_values("visit_date").reset_index(drop=True)
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates("visit_date", keep="last").sort_values("visit_date")
    return combined.reset_index(drop=True)


def set_known_value(slug, day, column, value):
    """Mark a known input for one day, e.g. set_known_value(slug, date, 'event_home_match', 1).
    Creates the column / a future row when needed."""
    visits = load_visits(slug)
    if visits is None:
        return None
    day = pd.Timestamp(day).normalize()
    if column not in visits.columns:
        visits[column] = 0.0
    if (visits["visit_date"] == day).any():
        visits.loc[visits["visit_date"] == day, column] = float(value)
    else:
        row = {c: np.nan for c in visits.columns}
        row.update({"visit_date": day, column: float(value)})
        visits = pd.concat([visits, pd.DataFrame([row])], ignore_index=True)
    folder = _location_dir(slug)
    visits.sort_values("visit_date").to_csv(folder / "visits.csv", index=False)
    return visits


def save_visits(slug, visits):
    """Save history, merged with whatever is already stored for this location."""
    merged = merge_visits(load_visits(slug), visits)
    folder = _location_dir(slug)
    folder.mkdir(parents=True, exist_ok=True)
    merged.to_csv(folder / "visits.csv", index=False)
    return merged


# ----------------------------------------------------------------------------
# Forecast track record
# ----------------------------------------------------------------------------
def _log_path():
    return data_dir() / "forecast_log.csv"


def load_log(slug):
    path = _log_path()
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    log = pd.read_csv(path)
    return log[log["location"] == slug].reset_index(drop=True)


def _merge_into_global_log(new_rows_df):
    """Merge a DataFrame of forecast log rows into data/forecast_log.csv without duplicates."""
    if new_rows_df is None or new_rows_df.empty:
        return
    clean_new = new_rows_df[LOG_COLUMNS].copy()
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            old = pd.read_csv(path)
            combined = pd.concat([old, clean_new], ignore_index=True)
            combined = combined.drop_duplicates(subset=["location", "made_on", "target_date"], keep="last")
        except Exception:
            combined = clean_new
    else:
        combined = clean_new
    combined.to_csv(path, index=False)


def log_forecast(slug, made_on, plan):
    """Save the open days of a plan (see core.build_plan). Saving the same forecast twice on the
    same day replaces it instead of duplicating it. Returns the number of days saved."""
    new = pd.DataFrame([
        {
            "location": slug,
            "made_on": made_on.isoformat(),
            "target_date": row["date"].date().isoformat(),
            "predicted": row["expected"],
            "low": row["low"],
            "high": row["high"],
        }
        for row in plan if not row["is_closed"]
    ], columns=LOG_COLUMNS)
    if new.empty:
        return 0
    _merge_into_global_log(new)
    return len(new)


def compare_log_to_actuals(log, visits):
    """Match saved forecasts to what really happened."""
    columns = ["target_date", "predicted", "low", "high", "actual"]
    if log is None or len(log) == 0:
        return pd.DataFrame(columns=columns)
    log = log.copy()
    log["target_date"] = pd.to_datetime(log["target_date"])
    log["made_on"] = pd.to_datetime(log["made_on"])
    log = log[log["made_on"] < log["target_date"]]
    latest = log.sort_values("made_on").drop_duplicates("target_date", keep="last")
    actuals = (visits[visits["visitors"].notna()][["visit_date", "visitors"]]
               .rename(columns={"visit_date": "target_date", "visitors": "actual"}))
    merged = latest.merge(actuals, on="target_date", how="inner").sort_values("target_date")
    return merged[columns].reset_index(drop=True)


def track_record_metrics(compared):
    """{'n': days compared, 'wape': %, 'coverage': % of actuals inside the saved range (or None)}"""
    if len(compared) == 0:
        return {"n": 0, "wape": float("nan"), "coverage": None}
    actual = compared["actual"].to_numpy(dtype=float)
    predicted = compared["predicted"].to_numpy(dtype=float)
    total = actual.sum()
    wape = float(np.abs(actual - predicted).sum() / total * 100) if total > 0 else float("nan")
    has_range = compared["low"].notna() & compared["high"].notna()
    coverage = None
    if has_range.any():
        sub = compared[has_range]
        inside = (sub["actual"] >= sub["low"]) & (sub["actual"] <= sub["high"])
        coverage = float(inside.mean() * 100)
    return {"n": int(len(compared)), "wape": wape, "coverage": coverage}


# ============================================================================
# Per-Location Passwords & Optional Supabase Cloud Sync
# ============================================================================
def hash_password(password: str) -> str:
    salt = os.urandom(16).hex()
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return f"pbkdf2_sha256${salt}${dk.hex()}"


def save_location_password(slug: str, raw_password: str):
    """Write password_hash directly into the location's settings.json file."""
    slug = slugify(slug)
    p = _location_dir(slug) / "settings.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    data["password_hash"] = hash_password(raw_password)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _get_stored_hash(slug: str) -> str:
    p = _location_dir(slugify(slug)) / "settings.json"
    if not p.exists():
        return ""
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("password_hash", "")
    except Exception:
        return ""


def verify_location_password(slug: str, password: str) -> bool:
    if not password:
        return False
    stored_hash = _get_stored_hash(slug)
    if not stored_hash:
        return False
    if stored_hash.startswith("pbkdf2_sha256$"):
        _, salt, expected_hex = stored_hash.split("$", 2)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
        return hmac.compare_digest(dk.hex(), expected_hex)
    return hmac.compare_digest(password.encode("utf-8"), str(stored_hash).encode("utf-8"))


def has_any_location_passwords() -> bool:
    return any(bool(_get_stored_hash(s)) for s, _ in list_locations())


# ----------------------------------------------------------------------------
# Sign-in lockout (kept on disk so opening a new browser tab does not reset it)
# ----------------------------------------------------------------------------
MIN_PASSWORD_LENGTH = 8
MAX_FAILED_LOGINS = 5            # failures allowed before the account is locked
LOGIN_LOCK_SECONDS = 15 * 60     # how long the lock lasts


def _attempts_path():
    return data_dir() / "login_attempts.json"


def _load_attempts():
    try:
        return json.loads(_attempts_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_attempts(attempts):
    path = _attempts_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(attempts), encoding="utf-8")


def login_locked_seconds(account, now=None):
    """Seconds until `account` may try to sign in again (0 = not locked)."""
    now = time.time() if now is None else now
    entry = _load_attempts().get(account, {})
    return max(0, int(entry.get("locked_until", 0) - now))


def record_failed_login(account, now=None):
    """Count a failed sign-in; lock the account after MAX_FAILED_LOGINS failures."""
    now = time.time() if now is None else now
    attempts = _load_attempts()
    entry = attempts.get(account, {})
    if entry.get("locked_until", 0) <= now and entry.get("fails", 0) >= MAX_FAILED_LOGINS:
        entry = {}  # previous lock expired: start counting again
    entry["fails"] = entry.get("fails", 0) + 1
    if entry["fails"] >= MAX_FAILED_LOGINS:
        entry["locked_until"] = now + LOGIN_LOCK_SECONDS
    attempts[account] = entry
    _save_attempts(attempts)


def clear_failed_logins(account):
    attempts = _load_attempts()
    if attempts.pop(account, None) is not None:
        _save_attempts(attempts)


# ----------------------------------------------------------------------------
# Password reset codes: 6 digits, emailed to the account's recovery address,
# stored only as a hash, single use, short-lived, limited guesses.
# ----------------------------------------------------------------------------
RESET_CODE_MINUTES = 15
RESET_MAX_ATTEMPTS = 5
RESET_RESEND_SECONDS = 60


def _resets_path():
    return data_dir() / "password_resets.json"


def _load_resets():
    try:
        return json.loads(_resets_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_resets(resets):
    path = _resets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(resets), encoding="utf-8")


def _hash_code(slug, code):
    return hashlib.sha256(f"{slug}:{code}".encode("utf-8")).hexdigest()


def create_reset_code(slug, now=None):
    """New code for `slug`, replacing any older one. Returns None if one was sent < RESET_RESEND_SECONDS ago."""
    now = time.time() if now is None else now
    resets = _load_resets()
    if now - resets.get(slug, {}).get("sent_at", -1e12) < RESET_RESEND_SECONDS:
        return None
    code = f"{secrets.randbelow(10**6):06d}"
    resets[slug] = {"hash": _hash_code(slug, code), "expires": now + RESET_CODE_MINUTES * 60,
                    "attempts": 0, "sent_at": now}
    _save_resets(resets)
    return code


def use_reset_code(slug, code, now=None):
    """True (and the code is used up) if `code` is the current, unexpired code for `slug`."""
    now = time.time() if now is None else now
    resets = _load_resets()
    entry = resets.get(slug)
    if not entry or now > entry["expires"] or entry["attempts"] >= RESET_MAX_ATTEMPTS:
        return False
    if hmac.compare_digest(entry["hash"], _hash_code(slug, str(code).strip())):
        resets.pop(slug)
        _save_resets(resets)
        return True
    entry["attempts"] += 1
    _save_resets(resets)
    return False


def _cloud_creds():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not (url and key):
        try:
            import streamlit as st
            url = url or st.secrets.get("SUPABASE_URL")
            key = key or st.secrets.get("SUPABASE_KEY")
        except Exception:
            pass
    return (url.rstrip("/"), key.strip()) if (url and key) else None


def _cloud_headers(api_key):
    """Old Supabase keys are JWTs (eyJ...) and also go in Authorization; new sb_secret_ keys must not."""
    headers = {"apikey": api_key}
    if api_key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def is_cloud_enabled() -> bool:
    return _cloud_creds() is not None


def sync_to_cloud(slug: str):
    """Push local location files and this location's forecast log to Supabase."""
    creds = _cloud_creds()
    if not creds:
        return
    base_url, api_key = creds
    slug = slugify(slug)
    loc_folder = _location_dir(slug)
    if not loc_folder.exists():
        return

    payload = {"slug": slug}
    for fname, col in (
        ("settings.json", "settings_json"),
        ("events.json", "events_json"),
        ("visits.csv", "visits_csv"),
        ("staff_actual.csv", "staff_csv"),
    ):
        fpath = loc_folder / fname
        if fpath.exists():
            payload[col] = fpath.read_text(encoding="utf-8")

    loc_log = load_log(slug)
    if not loc_log.empty:
        payload["log_csv"] = loc_log.to_csv(index=False)

    req = urllib.request.Request(
        f"{base_url}/rest/v1/tablecast_locations",
        data=json.dumps(payload).encode("utf-8"),
        headers={**_cloud_headers(api_key), "Content-Type": "application/json",
                 "Prefer": "resolution=merge-duplicates"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8):
            pass
    except Exception:
        pass


def pull_all_from_cloud():
    """Restore and merge cloud-saved locations and forecast logs into ./data/ on startup."""
    creds = _cloud_creds()
    if not creds:
        return
    base_url, api_key = creds
    req = urllib.request.Request(
        f"{base_url}/rest/v1/tablecast_locations?select=*",
        headers=_cloud_headers(api_key),
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
        for row in rows or []:
            slug = slugify(row.get("slug", ""))
            if not slug:
                continue
            loc_folder = _location_dir(slug)
            loc_folder.mkdir(parents=True, exist_ok=True)

            for fname, col in (("settings.json", "settings_json"), ("events.json", "events_json")):
                if row.get(col):
                    if fname in ("settings.json", "events.json") or not (loc_folder / fname).exists():
                        (loc_folder / fname).write_text(row[col], encoding="utf-8")

            if row.get("visits_csv"):
                try:
                    cloud_visits = pd.read_csv(io.StringIO(row["visits_csv"]), parse_dates=["visit_date"])
                    cloud_visits["visitors"] = pd.to_numeric(cloud_visits["visitors"], errors="coerce")
                    # Keep days with empty cells (no sales value, no event flag, future booking rows):
                    # only rows without a date are unusable.
                    save_visits(slug, cloud_visits.dropna(subset=["visit_date"]))
                except Exception:
                    pass

            if row.get("staff_csv"):
                try:
                    cloud_staff = pd.read_csv(io.StringIO(row["staff_csv"]), parse_dates=["date"])
                    if set(STAFF_COLUMNS) <= set(cloud_staff.columns):
                        save_staff(slug, cloud_staff[STAFF_COLUMNS])
                except Exception:
                    pass

            if row.get("log_csv"):
                try:
                    cloud_log = pd.read_csv(io.StringIO(row["log_csv"]))
                    if set(LOG_COLUMNS).issubset(cloud_log.columns):
                        _merge_into_global_log(cloud_log)
                except Exception:
                    pass
    except Exception:
        pass

"""TableCast: restaurant guest, staffing, prep and revenue forecasts (Streamlit UI).

Forecasting logic lives in core.py, files/history in storage.py, PDF + calendar in reports.py,
email in mailer.py, text in i18n.py, and the look in ui.py + assets/style.css + .streamlit/config.toml.
"""
import csv
import hashlib
import hmac
import importlib
import os
import sys
from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


def _reload_changed_local_modules():
    """After a deploy, Streamlit re-runs this file but may keep OLD copies of our own modules in
    memory, which caused "cannot import name ..." / AttributeError crashes until a manual reboot.
    Reload any of our modules whose file changed on disk. Dependencies come first (reports uses core).
    Skipped under pytest, where tests replace functions (mail, weather) that a reload would undo."""
    if "pytest" in sys.modules:
        return
    seen = sys.__dict__.setdefault("_tablecast_module_mtimes", {})
    for name in ("i18n", "ui", "mailer", "storage", "core", "reports"):
        module = sys.modules.get(name)
        if module is None or not getattr(module, "__file__", None):
            continue
        try:
            mtime = os.path.getmtime(module.__file__)
        except OSError:
            continue
        if seen.get(name) != mtime:
            importlib.reload(module)
            seen[name] = mtime


_reload_changed_local_modules()

import core  # noqa: E402  (after the reload above, on purpose)
import mailer
import reports
import storage
import ui
from i18n import (
    ADMIN1_TO_SUBDIV,
    COUNTRY_CITIES,
    COUNTRY_CURRENCIES,
    COUNTRY_NAMES,
    COUNTRY_SUBDIVISIONS,
    CURRENCY_OPTIONS,
    DAY_SHORT,
    DAY_TRANSLATIONS,
    LANGUAGE_ORDER,
    MONTH_TRANSLATIONS,
    TRANSLATIONS,
    UI_LABELS,
)

BRAND = "TableCast"
MAKER = "TableCast Labs"
CUSTOM_CITY = "✍️ Type any city or ZIP code..."
ADMIN_NAMES = {"admin", "administrator"}
ADMIN_ACCOUNT = "__admin__"  # lockout key for the admin sign-in
DEFAULT_COUNTRY = "AT"       # preselected country for new workspaces and the demo
DEFAULT_CITY = "Vienna"

st.set_page_config(
    page_title="TableCast · Restaurant Forecasts",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.load_css()


# ----------------------------------------------------------------------------
# Cached wrappers & demo data
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_read_csv(raw_bytes):
    return core.read_csv_bytes(raw_bytes)


@st.cache_resource(show_spinner=False)
def cached_train(visits, station, country, events=None):
    return core.train_bundle(visits, station, country, events=events)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_future_weather(station, country, _future_dates, _history):
    return core.fetch_future_weather(station, country, _future_dates, _history)


@st.cache_data(show_spinner=False)
def generate_demo_visits():
    """180 days of realistic bistro visitor history so anyone can try the app immediately."""
    rng = np.random.default_rng(42)
    end_d = date.today() - timedelta(days=1)
    dates = pd.date_range(end_d - timedelta(days=179), end_d, freq="D")
    dow_base = np.array([0, 75, 85, 95, 135, 165, 120])  # closed Mondays
    rows = []
    for i, d in enumerate(dates):
        if d.dayofweek == 0:
            continue
        trend = 1.0 + 0.12 * (i / len(dates))
        seasonal = 1.0 + 0.10 * np.sin(2 * np.pi * d.dayofyear / 365.0)
        noise = rng.normal(1.0, 0.11)
        guests = max(15, int(round(dow_base[d.dayofweek] * trend * seasonal * noise)))
        rows.append({"visit_date": d, "visitors": float(guests)})
    return pd.DataFrame(rows)


def get_secret(name):
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------
def weather_icon(temp, rain):
    if rain >= 5.0:
        return "🌧️"
    if rain >= 0.8:
        return "🌦️"
    if temp >= 20.0:
        return "☀️"
    if temp >= 12.0:
        return "⛅"
    return "☁️"


def c_to_f(temp_c):
    return temp_c * 9.0 / 5.0 + 32.0


def f_to_c(temp_f):
    return (temp_f - 32.0) * 5.0 / 9.0


def fmt_temp(temp_c, unit="C"):
    return f"{c_to_f(temp_c):.0f}°F" if unit == "F" else f"{temp_c:.0f}°C"


def fmt_rain(rain_mm, unit="C"):
    return f"{rain_mm / 25.4:.2f} in" if unit == "F" else f"{rain_mm:.1f} mm"


def day_driver(d, temp, rain, holiday_dates, on_event, is_closed):
    """The main reason behind a day's forecast: (label key, style)."""
    if is_closed:
        return "tag_closed", ""
    if on_event:
        return "tag_event", "accent"
    if d in holiday_dates:
        return "tag_holiday", "accent"
    if rain >= 3.0 and d.dayofweek >= 5:
        return "tag_rain", "warn"
    if temp >= 18.0 and rain <= 0.5:
        return "tag_patio", "good"
    if d.day in {1, 15} or d.day >= 25:
        return "tag_payday", "good"
    if d.dayofweek in {4, 5}:
        return "tag_peak", "accent"
    return "tag_normal", ""


# ============================================================================
# Session state, cloud sync, language
# ============================================================================
for key, default in (
    ("authed", False),
    ("auth_role", "admin"),
    ("locked_slug", None),
    ("force_login", False),
    ("cloud_pulled", False),
    ("use_demo", False),
):
    if key not in st.session_state:
        st.session_state[key] = default

if not st.session_state.cloud_pulled:
    storage.pull_all_from_cloud()
    st.session_state.cloud_pulled = True

ui.md(ui.sidebar_brand(BRAND), sidebar=True)
selected_lang = st.sidebar.selectbox(
    "Language", LANGUAGE_ORDER, key="language_selector", label_visibility="collapsed",
    format_func=lambda lang: UI_LABELS[lang]["lang_name"],
)
t = TRANSLATIONS[selected_lang]
L = UI_LABELS[selected_lang]
DAYS = DAY_TRANSLATIONS[selected_lang]
DAYS_SHORT = DAY_SHORT[selected_lang]
MONTHS = MONTH_TRANSLATIONS[selected_lang]

flash = st.session_state.pop("flash", None)
if flash:
    getattr(st, flash[0])(flash[1])


# ============================================================================
# Sign in / create account
# ============================================================================
saved_locations = storage.list_locations()
saved_names = dict(saved_locations)
app_password = get_secret("APP_PASSWORD")
signup_code = get_secret("SIGNUP_CODE")  # optional: when set, new accounts need this invite code
login_required = (
    bool(app_password)
    or storage.has_any_location_passwords()
    or st.session_state.get("force_login", False)
)
need_login = login_required and not st.session_state.get("use_demo", False)
# A visitor who skipped sign-in via the demo button. They must never see or change saved
# locations, so the sidebar and every save/email action is locked down for them below.
guest_demo = login_required and not st.session_state.authed


def start_session(role, slug):
    st.session_state.authed = True
    st.session_state.auth_role = role
    st.session_state.locked_slug = slug
    st.session_state.force_login = False
    st.session_state.use_demo = False
    st.session_state.pop("password_input", None)
    st.rerun()


def log_out():
    st.session_state.authed = False
    st.session_state.auth_role = "admin"
    st.session_state.locked_slug = None
    st.session_state.use_demo = False
    st.rerun()


def handle_sign_in(name, password):
    if not name:
        st.error(L["err_need_name"])
        return
    is_admin = name.lower() in ADMIN_NAMES
    account = ADMIN_ACCOUNT if is_admin else storage.slugify(name)
    wait = storage.login_locked_seconds(account)
    if wait:
        st.error(L["err_locked"].format(m=max(1, round(wait / 60))))
        return
    if is_admin and app_password and hmac.compare_digest(password.encode("utf-8"), str(app_password).encode("utf-8")):
        storage.clear_failed_logins(account)
        start_session("admin", None)
    elif not is_admin and storage.verify_location_password(account, password):
        storage.clear_failed_logins(account)
        start_session("client", account)
    else:
        # Only count attempts against real accounts, so random names don't fill the file.
        if is_admin or storage.location_exists(account):
            storage.record_failed_login(account)
        st.error(L["err_wrong_login"])


def valid_email(text):
    return bool(text) and mailer.parse_recipients(text) == [text]


def handle_sign_up(name, email, pw1, pw2, code):
    slug = storage.slugify(name) if name else ""
    if signup_code and not hmac.compare_digest(code.encode("utf-8"), str(signup_code).encode("utf-8")):
        st.error(L["err_invite"])
    elif not name or slug in {"default"} | ADMIN_NAMES:
        st.error(L["err_bad_name"])
    elif storage.location_exists(slug):
        st.error(L["err_exists"])
    elif not valid_email(email):
        st.error(L["err_email"])
    elif len(pw1) < storage.MIN_PASSWORD_LENGTH:
        st.error(L["err_pw_short"].format(n=storage.MIN_PASSWORD_LENGTH))
    elif pw1 != pw2:
        st.error(L["err_pw_match"])
    else:
        storage.save_settings(slug, {
            "name": name,
            "language": selected_lang,
            "country": DEFAULT_COUNTRY,
            "station": DEFAULT_CITY,
            "currency": COUNTRY_CURRENCIES.get(DEFAULT_COUNTRY, "€"),
            "temp_unit": "C",
            "owner_email": email,
            "recipients": email,
            "password_hash": storage.hash_password(pw1),
        })
        storage.sync_to_cloud(slug)
        st.session_state["flash"] = ("success", L["welcome"].format(name=name))
        start_session("client", slug)


def handle_send_reset_code(name):
    """Email a reset code. The reply is the same whether or not the account exists (no account probing)."""
    if not name:
        st.error(L["err_need_name"])
        return
    smtp = mailer.smtp_config(get_secret)
    if smtp is None:
        st.error(L["err_no_smtp"])
        return
    slug = storage.slugify(name)
    email = storage.load_settings(slug).get("owner_email", "") if storage.location_exists(slug) else ""
    if valid_email(email):
        code = storage.create_reset_code(slug)
        if code:  # None = a code was sent less than a minute ago; don't spam the inbox
            message = mailer.build_email(
                smtp["from"], [email], L["mail_subject"],
                L["mail_body"].format(
                    name=storage.load_settings(slug).get("name", name), code=code, m=storage.RESET_CODE_MINUTES,
                ),
            )
            try:
                mailer.send_email(message, smtp)
            except OSError:
                st.error(L["err_send"])
                return
    st.session_state["reset_slug"] = slug
    st.session_state["reset_sent"] = True
    st.rerun()


def handle_reset_password(code, pw1, pw2):
    slug = st.session_state.get("reset_slug", "")
    if len(pw1) < storage.MIN_PASSWORD_LENGTH:
        st.error(L["err_new_pw_short"].format(n=storage.MIN_PASSWORD_LENGTH))
    elif pw1 != pw2:
        st.error(L["err_pw_match"])
    elif not slug or not storage.use_reset_code(slug, code):
        st.error(L["err_code"])
    else:
        storage.save_location_password(slug, pw1)
        storage.clear_failed_logins(slug)  # a reset also lifts a sign-in lockout
        storage.sync_to_cloud(slug)
        for key in ("reset_slug", "reset_sent", "show_reset"):
            st.session_state.pop(key, None)
        st.session_state["flash"] = ("success", L["pw_changed"])
        st.rerun()


def render_password_reset():
    """Step 1: ask for the restaurant and email a code. Step 2: code + new password."""
    st.markdown(f"**{L['reset_title']}**")
    if not st.session_state.get("reset_sent"):
        with st.form("reset_request_form", border=False):
            reset_name = st.text_input(L["f_rest_name"], key="reset_rest_name", placeholder=L["ph_rest_name"])
            send = st.form_submit_button(L["btn_send_code"], key="reset_send_btn", type="primary", width="stretch")
        if send:
            handle_send_reset_code(reset_name.strip())
    else:
        st.info(L["reset_sent"].format(m=storage.RESET_CODE_MINUTES))
        with st.form("reset_form", border=False):
            reset_code = st.text_input(L["f_code"], key="reset_code", max_chars=6)
            reset_pw1 = st.text_input(
                L["f_new_pw"].format(n=storage.MIN_PASSWORD_LENGTH), type="password", key="reset_pw1"
            )
            reset_pw2 = st.text_input(L["f_new_pw2"], type="password", key="reset_pw2")
            do_reset = st.form_submit_button(L["btn_set_pw"], key="reset_submit_btn", type="primary", width="stretch")
        if do_reset:
            handle_reset_password(reset_code.strip(), reset_pw1, reset_pw2)
        if st.button(L["btn_resend"], key="reset_restart_btn", type="tertiary"):
            st.session_state.pop("reset_sent", None)
            st.rerun()
    if st.button(L["btn_back"], key="reset_back_btn", type="tertiary"):
        for key in ("reset_slug", "reset_sent", "show_reset"):
            st.session_state.pop(key, None)
        st.rerun()


if need_login and not st.session_state.authed:
    _, mid, _ = st.columns([1, 1.25, 1])
    with mid:
        ui.md(ui.login_header(BRAND, L["login_sub"]))
        tab_signin, tab_signup = st.tabs([L["tab_signin"], L["tab_signup"]])
        with tab_signin:
            if st.session_state.get("show_reset"):
                render_password_reset()
            else:
                with st.form("signin_form", border=False):
                    login_name = st.text_input(L["f_rest_name"], key="login_rest_name", placeholder=L["ph_rest_name"])
                    entered_pw = st.text_input(L["f_password"], type="password", key="password_input")
                    signin = st.form_submit_button(L["btn_signin"], key="login_submit_btn", type="primary", width="stretch")
                if signin:
                    handle_sign_in(login_name.strip(), entered_pw)
                if st.button(L["btn_forgot"], key="forgot_btn", type="tertiary"):
                    st.session_state["show_reset"] = True
                    st.rerun()
        with tab_signup:
            with st.form("signup_form", border=False):
                new_name = st.text_input(L["f_rest_name"], key="signup_rest_name", placeholder=L["ph_new_rest"])
                new_email = st.text_input(
                    L["f_email"], key="signup_email", placeholder="you@restaurant.com", help=L["help_email"],
                )
                new_pw1 = st.text_input(
                    L["f_pw_min"].format(n=storage.MIN_PASSWORD_LENGTH), type="password", key="signup_pw1"
                )
                new_pw2 = st.text_input(L["f_pw_confirm"], type="password", key="signup_pw2")
                invite = st.text_input(L["f_invite"], key="signup_code") if signup_code else ""
                create = st.form_submit_button(L["btn_create"], key="signup_submit_btn", type="primary", width="stretch")
            if create:
                handle_sign_up(new_name.strip(), new_email.strip(), new_pw1, new_pw2, invite.strip())
        st.divider()
        if st.button(L["btn_demo"], key="login_demo_btn", width="stretch"):
            st.session_state.use_demo = True
            st.session_state.force_login = False
            st.rerun()
        ui.md(f'<div class="tc-login-foot">{ui.esc(L["login_foot"])}</div>')
    st.stop()


# ============================================================================
# Sidebar
# ============================================================================
pending_location = st.session_state.pop("pending_location", None)
if pending_location in saved_names:
    st.session_state["saved_location"] = pending_location

ui.md(ui.section_label(L["sec_workspace"]), sidebar=True)
if guest_demo:
    # Demo visitors get no access to saved locations: nothing is loaded from or written to disk.
    saved_names = {}
    slug = "_guest_demo"  # underscore: storage.slugify() can never produce this, so no real location collides
    loc_name = ""
    ui.md(ui.note(L["demo_mode"]), sidebar=True)
    if st.sidebar.button(L["btn_exit_demo"], key="exit_demo_btn", width="stretch"):
        log_out()
elif st.session_state.auth_role == "client" and st.session_state.locked_slug:
    slug = st.session_state.locked_slug
    loc_name = saved_names.get(slug, slug)
    st.sidebar.success(f"🔒 {loc_name}")
    if st.sidebar.button(L["btn_logout"], key="logout_btn", width="stretch"):
        log_out()
else:
    chosen_slug = st.sidebar.selectbox(
        t["location_saved"], [""] + list(saved_names),
        format_func=lambda s: t["location_none"] if not s else saved_names[s], key="saved_location",
    )
    loc_name = st.sidebar.text_input(
        t["location_name"], value=saved_names.get(chosen_slug, ""), key=f"loc_name_{chosen_slug or 'new'}"
    ).strip()
    slug = storage.slugify(loc_name) if loc_name else "default"
    if need_login and st.sidebar.button(L["btn_logout"], key="admin_logout_btn", width="stretch"):
        log_out()

settings = storage.load_settings(slug) if slug in saved_names else {}
saved_visits = storage.load_visits(slug) if slug in saved_names else None

# --- Your data ---
with st.sidebar.expander(L["sec_data"], expanded=saved_visits is None):
    uploaded_file = st.file_uploader(t["upload_prompt"], type=["csv"], key="csv_file_uploader")
    st.caption(t["req_format"])

# --- Location & calendar ---
with st.sidebar.expander(L["sec_location"]):
    # Default country first in the list; a workspace's saved country still wins.
    country_codes = [DEFAULT_COUNTRY] + [c for c in COUNTRY_NAMES if c != DEFAULT_COUNTRY]
    default_country_idx = country_codes.index(settings["country"]) if settings.get("country") in country_codes else 0
    country = st.selectbox(
        ui.plain(t["country_label"]), country_codes, index=default_country_idx,
        format_func=lambda c: f"{COUNTRY_NAMES[c]} ({c})", key=f"country_{slug}",
    )

    preset_cities = COUNTRY_CITIES.get(country, [DEFAULT_CITY])
    saved_station = settings.get("station", preset_cities[0])
    city_options = preset_cities + [CUSTOM_CITY]
    default_city_idx = preset_cities.index(saved_station) if saved_station in preset_cities else len(preset_cities)
    picked_city = st.selectbox(
        L["f_city"], city_options, index=default_city_idx, key=f"city_preset_{slug}_{country}",
        format_func=lambda c: L["custom_city"] if c == CUSTOM_CITY else c,
    )
    if picked_city == CUSTOM_CITY:
        custom_default = saved_station if saved_station not in preset_cities else ""
        station = st.text_input(
            t["station_label"], value=custom_default, placeholder=L["ph_city"], key=f"station_custom_{slug}",
        ).strip() or preset_cities[0]
    else:
        station = picked_city

    geo_info = core.resolve_location_info(station, country)
    subdiv_code = ""
    if country in COUNTRY_SUBDIVISIONS:
        subdiv_map = COUNTRY_SUBDIVISIONS[country]
        subdiv_labels = list(subdiv_map.keys())
        saved_subdiv = settings.get("subdiv", "")
        default_sub_idx = next(
            (i for i, code in enumerate(subdiv_map.values()) if code and code == saved_subdiv), 0
        )
        picked_subdiv_label = st.selectbox(
            L["f_subdiv"], subdiv_labels, index=default_sub_idx, key=f"subdiv_{slug}_{country}",
            format_func=lambda label: L["subdiv_auto"] if not subdiv_map[label] else label,
        )
        subdiv_code = subdiv_map[picked_subdiv_label]
    if not subdiv_code and geo_info.get("admin1"):
        # Only accept a state that belongs to the chosen country (state names alone can collide).
        auto_code = ADMIN1_TO_SUBDIV.get(geo_info["admin1"], "")
        if auto_code in COUNTRY_SUBDIVISIONS.get(country, {}).values():
            subdiv_code = auto_code
    holiday_country = f"{country}-{subdiv_code}" if subdiv_code else country

    if geo_info["matched"]:
        region_str = f", {geo_info['admin1']}" if geo_info["admin1"] else ""
        st.caption(L["weather_for"].format(place=f"{geo_info['name']}{region_str}", code=holiday_country))
    else:
        st.caption(L["city_not_found"].format(country=COUNTRY_NAMES.get(country, country)))

    unit_col1, unit_col2 = st.columns(2)
    default_curr = settings.get("currency") or COUNTRY_CURRENCIES.get(country, "€")
    curr_idx = CURRENCY_OPTIONS.index(default_curr) if default_curr in CURRENCY_OPTIONS else 0
    currency = unit_col1.selectbox(
        L["f_currency"], CURRENCY_OPTIONS, index=curr_idx, key=f"currency_select_{slug}_{country}"
    )
    default_temp_unit = settings.get("temp_unit") or ("F" if country == "US" else "C")
    temp_unit_label = unit_col2.radio(
        L["f_units"], ["°C / mm", "°F / in"], index=1 if default_temp_unit == "F" else 0,
        key=f"temp_unit_{slug}_{country}",
    )
    temp_unit = "F" if "°F" in temp_unit_label else "C"

# --- Service & pricing ---
with st.sidebar.expander(L["sec_service"]):
    lunch_share = st.slider(t["lunch_share"], 0, 100, int(settings.get("lunch_share", 40)), key=f"lunch_{slug}")
    weekend_lunch_share = st.slider(
        L["f_wknd_lunch"],
        0, 100, int(settings.get("weekend_lunch_share", lunch_share)), key=f"wknd_lunch_{slug}",
    )
    avg_spend = st.number_input(
        L["f_lunch_spend"].format(c=currency),
        min_value=0.0, value=float(settings.get("avg_spend", 25.0)), step=1.0, key=f"spend_{slug}",
    )
    dinner_spend = st.number_input(
        L["f_dinner_spend"].format(c=currency),
        min_value=0.0, value=float(settings.get("dinner_spend", max(avg_spend, 32.0))), step=1.0,
        key=f"dinner_spend_{slug}",
    )
    guests_per_server = st.number_input(
        t["guests_per_server"], min_value=1, value=int(settings.get("guests_per_server", 15)), step=1, key=f"gps_{slug}"
    )

# --- Events & bookings ---
saved_events = {} if guest_demo else storage.load_events(slug)
with st.sidebar.expander(L["sec_events"], expanded=bool(saved_events)):
    st.caption(t["events_desc"])
    has_custom_event = st.checkbox(t["checkbox_event"], key="private_event_checkbox")
    event_date = None
    event_extra_visitors = 0
    if has_custom_event:
        event_date = st.date_input(t["event_date"], value=date.today(), key="event_date_input")
        event_extra_visitors = int(
            st.number_input(t["extra_guests"], min_value=1, max_value=500, value=25, key="extra_guests_input")
        )
        ev_col1, ev_col2 = st.columns(2)
        if not guest_demo and ev_col1.button(L["btn_save_event"], key="save_event_btn", width="stretch"):
            saved_events[event_date.isoformat()] = event_extra_visitors
            storage.save_events(slug, saved_events)
            storage.sync_to_cloud(slug)
            st.rerun()
        if saved_events and ev_col2.button(L["btn_clear_events"], key="clear_events_btn", width="stretch"):
            saved_events = storage.save_events(slug, {})
            storage.sync_to_cloud(slug)
            st.rerun()
    if saved_events:
        st.caption(L["saved_events"].format(list=", ".join(f"{k} (+{v})" for k, v in sorted(saved_events.items()))))

# --- Account & email ---
new_loc_password = ""
recipients_text = ""
owner_email = settings.get("owner_email", "")
if not guest_demo:
    with st.sidebar.expander(L["sec_account"], expanded=bool(settings) and not owner_email):
        owner_email = st.text_input(
            L["f_recovery"], value=owner_email, key=f"owner_email_{slug}", help=L["help_recovery"],
        ).strip()
        if settings and not owner_email:
            st.caption(L["add_recovery"])
        recipients_text = st.text_input(
            t["email_recipients"], value=settings.get("recipients", ""), key=f"recipients_{slug}"
        )
        pw_label = L["pw_change"] if settings.get("password_hash") else L["pw_set"]
        new_loc_password = st.text_input(
            pw_label, type="password", key=f"loc_pw_{slug}",
            help=L["help_pw"].format(n=storage.MIN_PASSWORD_LENGTH),
        ).strip()


# ============================================================================
# Data: upload, saved history or demo
# ============================================================================
if uploaded_file is None and saved_visits is None and not st.session_state.use_demo:
    ui.md(ui.empty_state(L["empty_title"], L["empty_body"], [L["step1"], L["step2"], L["step3"]]))
    if st.button(L["demo_button"], type="primary", key="empty_demo_btn"):
        st.session_state.use_demo = True
        st.rerun()
    st.stop()

using_demo_data = False
if uploaded_file is not None:
    if not guest_demo:  # guests can try their own CSV, but stay in (unsaved) demo mode
        st.session_state.use_demo = False
    try:
        raw_df = cached_read_csv(uploaded_file.getvalue())
    except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, csv.Error):
        st.error(t["err_read"])
        st.stop()
    upload_min_days = 1 if saved_visits is not None else core.MIN_DAYS
    upload_clean, notes, error = core.clean_visits(raw_df, min_days=upload_min_days)
    for level, key, kwargs in notes:
        getattr(st, level)(t[key].format(**kwargs))
    if error is not None:
        st.error(t[error[0]].format(**error[1]))
        st.stop()
    visits = storage.merge_visits(saved_visits, upload_clean)
    if len(visits) < core.MIN_DAYS:
        st.error(t["err_too_short"].format(min=core.MIN_DAYS, found=len(visits)))
        st.stop()
    # Auto-save into an existing workspace, once per uploaded file (the uploader keeps the
    # file across reruns, so remember which file was already saved).
    upload_id = f"{slug}:{hashlib.sha1(uploaded_file.getvalue()).hexdigest()}"
    if not guest_demo and slug in saved_names:
        if st.session_state.get("autosaved_upload") != upload_id:
            new_days = len(visits) - (0 if saved_visits is None else len(saved_visits))
            storage.save_visits(slug, upload_clean)
            storage.sync_to_cloud(slug)
            st.session_state["autosaved_upload"] = upload_id
            st.toast(L["autosaved"].format(name=loc_name, total=len(visits), new=new_days), icon="✅")
    else:
        st.toast(t["success_loaded"].format(n=len(visits)), icon="✅")
        if not guest_demo:
            st.sidebar.caption(L["new_loc_hint"])
elif saved_visits is not None:
    st.session_state.use_demo = False
    visits = saved_visits
else:
    visits = generate_demo_visits()
    using_demo_data = True

if not guest_demo:
    st.sidebar.divider()
    if st.sidebar.button(t["save_location_btn"], key="save_location_button", type="primary", width="stretch"):
        if not loc_name:
            st.sidebar.error(t["err_location_name"])
        elif new_loc_password and len(new_loc_password) < storage.MIN_PASSWORD_LENGTH:
            st.sidebar.error(L["err_pw_short_side"].format(n=storage.MIN_PASSWORD_LENGTH))
        elif owner_email and not valid_email(owner_email):
            st.sidebar.error(L["err_recovery"])
        else:
            storage.save_visits(slug, visits)
            storage.save_settings(slug, {
                "name": loc_name,
                "language": selected_lang,
                "country": country,
                "subdiv": subdiv_code,
                "currency": currency,
                "temp_unit": temp_unit,
                "station": station,
                "lunch_share": lunch_share,
                "weekend_lunch_share": weekend_lunch_share,
                "avg_spend": avg_spend,
                "dinner_spend": dinner_spend,
                "guests_per_server": int(guests_per_server),
                "recipients": recipients_text,
                "owner_email": owner_email,
                # The prep tab is drawn further down, so read its current values from session state.
                "prep_items": st.session_state.get(f"prep_items_list_{slug}", settings.get("prep_items")),
                "prep_buffer": int(st.session_state.get(f"prep_buf_{slug}", settings.get("prep_buffer", 10))),
                "password_hash": (
                    storage.hash_password(new_loc_password) if new_loc_password else settings.get("password_hash", "")
                ),
            })
            storage.sync_to_cloud(slug)
            st.session_state["pending_location"] = slug
            st.session_state["flash"] = ("success", t["saved_ok"])
            st.rerun()


# ============================================================================
# Training & forecast
# ============================================================================
with st.spinner(t["spinner"]):
    bundle = cached_train(visits, station, holiday_country, saved_events)
if bundle is None:
    st.error(t["err_training"])
    st.stop()
if not bundle["weather_ok"]:
    st.warning(t["warn_weather"])

history = bundle["history"]
last_date = history.index[-1]
future_dates = core.future_dates_after(history)
auto_weather, live_weather_ok = cached_future_weather(station, country, future_dates, history)

if st.session_state.get("manual_weather_checkbox", not live_weather_ok):
    default_temp, _ = core.default_weather(history)
    pred_temp = st.session_state.get("temp_slider_input", default_temp)
    pred_rain = st.session_state.get("rain_slider_input", 0.0)
    future_weather = pd.DataFrame({"temp": pred_temp, "rain": pred_rain}, index=future_dates)
else:
    future_weather = auto_weather

base_preds = bundle["forecaster"].forecast(history.tail(core.LOOKBACK_DAYS), future_weather)

event_active = bool(has_custom_event and event_date is not None)
if event_active and event_date not in {d.date() for d in future_dates}:
    st.warning(t["warn_event_outside"])
    event_active = False

plan = core.build_plan(
    base_preds, bundle["intervals"], last_date,
    event_date if event_active else None, event_extra_visitors,
    lunch_share, avg_spend, int(guests_per_server),
    events=saved_events,
    weekend_lunch_share=weekend_lunch_share,
    dinner_spend=dinner_spend,
)

week_total_guests = sum(r["expected"] for r in plan)
week_total_rev = sum(r["revenue"] for r in plan)
beats_baseline = bool(
    np.isfinite(bundle["wape7"]) and np.isfinite(bundle["base_wape7"]) and bundle["wape7"] < bundle["base_wape7"]
)
edge_pct = bundle["base_wape7"] - bundle["wape7"] if np.isfinite(bundle["wape7"]) else 0.0


# ============================================================================
# Header & KPIs
# ============================================================================
display_name = loc_name or (L["demo_name"] if using_demo_data else BRAND)
ui.md(ui.page_header(
    L["eyebrow"],
    display_name,
    t["forecast_from"].format(date=last_date.strftime("%Y-%m-%d")),
    [
        (L["chip_data"].format(date=f"{last_date.day} {MONTHS[last_date.month - 1]} {last_date.year}"), ""),
        (L["chip_weather_live"], "good") if live_weather_ok else (L["chip_weather_assumed"], "warn"),
        (L["chip_beats"], "good") if beats_baseline else (L["chip_close"], "warn"),
    ],
))
if using_demo_data:
    ui.md(ui.note(L["demo_note"]))

ui.md(ui.kpi_cards([
    {
        "label": L["kpi_guests"], "value": f"{week_total_guests:,}", "unit": t["unit_people"],
        "sub": L["kpi_avg_sub"].format(n=f"{bundle['mean_visitors']:.0f}"),
    },
    {
        "label": L["kpi_revenue"], "value": f"{currency}{week_total_rev:,.0f}",
        "sub": L["kpi_rev_sub"].format(lunch=f"{currency}{avg_spend:.0f}", dinner=f"{currency}{dinner_spend:.0f}"),
    },
    {
        "label": ui.plain(t["metric_wape7"]), "value": core.fmt_pct(bundle["wape7"]),
        "pill": (L["kpi_good"].format(edge=f"{edge_pct:.1f}"), "good") if beats_baseline else (L["kpi_warn"], "warn"),
    },
    {
        "label": ui.plain(t["metric_mae"]), "value": f"±{bundle['mae']:.1f}", "unit": t["unit_people"],
        "sub": L["kpi_mae_sub"].format(wape=core.fmt_pct(bundle["wape"])),
    },
]))


# ============================================================================
# Tabs
# ============================================================================
tab_plan, tab_prep, tab_ai, tab_track, tab_export = st.tabs(
    [L["tab_plan"], L["tab_prep"], L["tab_accuracy"], L["tab_track"], L["tab_export"]]
)

# ----------------------------------------------------------------------------
# Week plan
# ----------------------------------------------------------------------------
with tab_plan:
    holiday_set = set(bundle["forecaster"].holiday_dates)
    day_rows = []
    for row in plan:
        d = row["date"]
        wx = future_weather.loc[d]
        temp_val, rain_val = float(wx["temp"]), float(wx["rain"])
        tag_key, tag_kind = day_driver(d, temp_val, rain_val, holiday_set, row["on_event"], row["is_closed"])
        has_range = not row["is_closed"] and row["low"] is not None
        day_rows.append({
            "name": DAYS_SHORT[d.dayofweek],
            "date": f"{d.day} {MONTHS[d.month - 1]}",
            "weather": f"{weather_icon(temp_val, rain_val)} {fmt_temp(temp_val, temp_unit)} · {fmt_rain(rain_val, temp_unit)}",
            "guests": t["closed"] if row["is_closed"] else f"{row['expected']}",
            "range": f"{row['low']}–{row['high']}" if has_range else "",
            "tag": L[tag_key], "tag_kind": tag_kind,
            "closed": row["is_closed"], "event": row["on_event"],
        })
    ui.md(ui.day_cards(day_rows))

    chart_df = pd.DataFrame([
        {
            "day": f"{DAYS_SHORT[r['date'].dayofweek]} {r['date'].day}",
            "expected": 0 if r["is_closed"] else r["expected"],
            "low": None if r["is_closed"] else r["low"],
            "high": None if r["is_closed"] else r["high"],
            "kind": "event" if r["on_event"] else "normal",
        }
        for r in plan
    ])
    x_axis = alt.X("day:N", sort=None, title=None, axis=alt.Axis(labelAngle=0))
    bars = alt.Chart(chart_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=x_axis,
        y=alt.Y("expected:Q", title=None),
        color=alt.condition(alt.datum.kind == "event", alt.value(ui.ACCENT_DARK), alt.value(ui.ACCENT)),
        tooltip=[alt.Tooltip("day:N", title=" "), alt.Tooltip("expected:Q", title=ui.plain(t["col_traffic"])),
                 alt.Tooltip("low:Q", title=L["tt_low"]), alt.Tooltip("high:Q", title=L["tt_high"])],
    )
    layers = [bars]
    if bundle["intervals"] is not None:
        layers.append(
            alt.Chart(chart_df.dropna(subset=["low", "high"]))
            .mark_rule(color=ui.INK, strokeWidth=1.5, opacity=0.55)
            .encode(x=x_axis, y="low:Q", y2="high:Q")
        )
    st.markdown(f"**{L['chart_title']}**")
    st.altair_chart(
        alt.layer(*layers).properties(height=240)
        .configure_axis(labelColor=ui.MUTED, gridColor=ui.LINE, domainColor=ui.LINE, tickColor=ui.LINE)
        .configure_view(strokeWidth=0),
        width="stretch",
    )

    show_range = bundle["intervals"] is not None
    headers, table_rows, pdf_rows = reports.format_plan(plan, t, DAYS, MONTHS, show_range, currency=currency)
    forecast_df = pd.DataFrame(table_rows)
    max_traffic = max([r["expected"] for r in plan] + [1])
    rev_col = f"{t['col_revenue']} ({currency})" if currency else t["col_revenue"]
    st.dataframe(
        forecast_df,
        hide_index=True,
        width="stretch",
        column_config={
            t["col_traffic"]: st.column_config.ProgressColumn(
                t["col_traffic"], format="%d", min_value=0, max_value=int(max_traffic * 1.15),
            ),
            rev_col: st.column_config.NumberColumn(rev_col, format=f"{currency}%.2f"),
        },
    )
    if show_range:
        st.caption(t["range_caption"])
    if bundle["closed_weekdays"]:
        st.caption(t["closed_caption"].format(days=", ".join(DAYS[d] for d in sorted(bundle["closed_weekdays"]))))

    with st.expander(L["weather_assumptions"]):
        st.checkbox(
            L["override_weather"],
            value=not live_weather_ok,
            key="manual_weather_checkbox",
        )
        if st.session_state.get("manual_weather_checkbox", False):
            default_temp, _ = core.default_weather(history)
            col_b1, col_b2 = st.columns(2)
            if temp_unit == "F":
                temp_f_val = col_b1.slider(L["f_temp_f"], 14.0, 104.0, round(c_to_f(default_temp), 1), key="temp_slider_f")
                st.session_state["temp_slider_input"] = f_to_c(temp_f_val)
                rain_in_val = col_b2.slider(L["f_rain_in"], 0.0, 2.0, 0.0, step=0.05, key="rain_slider_in")
                st.session_state["rain_slider_input"] = rain_in_val * 25.4
            else:
                col_b1.slider(t["avg_temp"], -10.0, 40.0, default_temp, key="temp_slider_input")
                col_b2.slider(t["avg_rain"], 0.0, 50.0, 0.0, key="rain_slider_input")

# ----------------------------------------------------------------------------
# Kitchen prep (custom ingredients + presets)
# ----------------------------------------------------------------------------
with tab_prep:
    st.caption(L["prep_caption"])

    # Preset labels and item names follow the selected language (stored data keeps whatever was saved).
    PRESET_TEMPLATES = {
        L["preset_bistro"]: [
            {"Item": L["it_buns"], "Per Guest": 0.45, "Unit": "pcs"},
            {"Item": L["it_meat"], "Per Guest": 0.12, "Unit": "kg"},
            {"Item": L["it_fries"], "Per Guest": 0.15, "Unit": "kg"},
        ],
        L["preset_bakery"]: [
            {"Item": L["it_pastry"], "Per Guest": 0.85, "Unit": "pcs"},
            {"Item": L["it_beans"], "Per Guest": 0.018, "Unit": "kg"},
            {"Item": L["it_milk"], "Per Guest": 0.12, "Unit": "L"},
        ],
        L["preset_gym"]: [
            {"Item": L["it_creatine"], "Per Guest": 5.0, "Unit": "g"},
            {"Item": L["it_whey"], "Per Guest": 30.0, "Unit": "g"},
            {"Item": L["it_fruit"], "Per Guest": 0.5, "Unit": "pcs"},
        ],
    }
    UNITS = ["pcs", "kg", "g", "L", "ml", "lb", "oz", "portions"]

    state_key = f"prep_items_list_{slug}"
    if state_key not in st.session_state:
        st.session_state[state_key] = settings.get("prep_items") or PRESET_TEMPLATES[L["preset_bistro"]]

    with st.expander(L["prep_add"]):
        add_c1, add_c2, add_c3, add_c4 = st.columns([2.2, 1.2, 1, 1.2], vertical_alignment="bottom")
        new_item_name = add_c1.text_input(L["f_ingredient"], placeholder=L["ph_ingredient"], key=f"new_ing_name_{slug}")
        new_item_rate = add_c2.number_input(L["f_per_guest"], min_value=0.01, value=0.50, step=0.05, key=f"new_ing_rate_{slug}")
        new_item_unit = add_c3.selectbox(L["f_unit"], UNITS, key=f"new_ing_unit_{slug}")
        if add_c4.button(L["btn_add"], key=f"add_ing_btn_{slug}", type="primary", width="stretch"):
            if new_item_name.strip():
                st.session_state[state_key].append(
                    {"Item": new_item_name.strip(), "Per Guest": float(new_item_rate), "Unit": new_item_unit}
                )
                st.rerun()

        preset_cols = st.columns(4)
        for idx_p, (preset_label, preset_rows) in enumerate(PRESET_TEMPLATES.items()):
            if preset_cols[idx_p].button(preset_label, key=f"pre_{idx_p}_{slug}", width="stretch"):
                st.session_state[state_key] = [dict(r) for r in preset_rows]
                st.rerun()
        if preset_cols[3].button(L["btn_clear_list"], key=f"clear_ing_{slug}", width="stretch"):
            st.session_state[state_key] = []
            st.rerun()

    col_p1, col_p2 = st.columns([3, 1])
    with col_p2:
        safety_buffer = st.slider(
            L["f_buffer"], 0, 30, int(settings.get("prep_buffer", 10)), help=L["help_buffer"],
            key=f"prep_buf_{slug}",
        )
    with col_p1:
        prep_config_df = st.data_editor(
            pd.DataFrame(st.session_state[state_key] or [{"Item": "", "Per Guest": 0.0, "Unit": "pcs"}]),
            num_rows="dynamic",
            width="stretch",
            column_config={
                "Item": st.column_config.TextColumn(L["f_ingredient"], required=True),
                "Per Guest": st.column_config.NumberColumn(L["f_per_guest"], min_value=0.0, format="%.2f"),
                "Unit": st.column_config.SelectboxColumn(L["f_unit"], options=UNITS),
            },
            key=f"prep_editor_{slug}",
        )
        st.session_state[state_key] = prep_config_df.to_dict("records")

    valid_items = [
        r for _, r in prep_config_df.iterrows()
        if str(r.get("Item", "")).strip() and pd.notna(r.get("Per Guest"))
    ]
    if valid_items:
        prep_schedule = []
        for row in plan:
            d = row["date"]
            day_entry = {
                L["col_date"]: f"{DAYS_SHORT[d.dayofweek]} {d.day} {MONTHS[d.month - 1]}",
                L["col_guests"]: 0 if row["is_closed"] else row["expected"],
            }
            for item_row in valid_items:
                item_name = str(item_row["Item"]).strip()
                rate = float(item_row["Per Guest"] or 0.0)
                unit = str(item_row.get("Unit") or "pcs").strip()
                if row["is_closed"]:
                    day_entry[f"{item_name} ({unit})"] = t["closed"]
                else:
                    raw_needed = row["expected"] * rate * (1.0 + safety_buffer / 100.0)
                    day_entry[f"{item_name} ({unit})"] = (
                        f"{raw_needed:.1f}" if unit in {"kg", "L", "lb"} else f"{int(np.ceil(raw_needed))}"
                    )
            prep_schedule.append(day_entry)
        st.dataframe(pd.DataFrame(prep_schedule), hide_index=True, width="stretch")
    else:
        st.info(L["prep_empty"])

# ----------------------------------------------------------------------------
# Accuracy: backtest chart & day-of-week pattern
# ----------------------------------------------------------------------------
with tab_ai:
    _, baseline_text = reports.metric_lines(bundle, t)
    if beats_baseline:
        st.success(ui.plain(t["verdict_better"]))
    else:
        st.warning(ui.plain(t["verdict_worse"]))
    st.caption(baseline_text)

    st.subheader(t["main_chart_header"])
    backtest = (
        bundle["chart"]
        .rename(columns={"actual": t["legend_actual"], "pred": t["legend_pred"], "baseline": t["legend_baseline"]})
        .reset_index()
        .rename(columns={"index": "date"})
        .melt("date", var_name="Series", value_name="Visitors")
    )
    series_domain = [t["legend_actual"], t["legend_pred"], t["legend_baseline"]]
    line_chart = (
        alt.Chart(backtest)
        .mark_line(point=alt.OverlayMarkDef(size=28), strokeWidth=2.2)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("Visitors:Q", title=None),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(domain=series_domain, range=[ui.INK, ui.ACCENT, ui.BASELINE]),
                legend=alt.Legend(orient="top", title=None),
            ),
            strokeDash=alt.condition(
                alt.datum.Series == t["legend_baseline"], alt.value([4, 4]), alt.value([1, 0])
            ),
            tooltip=["date:T", "Series:N", alt.Tooltip("Visitors:Q", format=".0f")],
        )
        .properties(height=320)
        .configure_axis(labelColor=ui.MUTED, gridColor=ui.LINE, domainColor=ui.LINE, tickColor=ui.LINE)
        .configure_legend(labelColor=ui.INK)
        .configure_view(strokeWidth=0)
        .interactive()
    )
    st.altair_chart(line_chart, width="stretch")

    st.subheader(ui.plain(t["dow_header"]))
    st.caption(t["dow_sub"])
    dow_df = pd.DataFrame({"day": DAYS, "visitors": bundle["dow_means"].to_numpy()})
    dow_chart = (
        alt.Chart(dow_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color=ui.ACCENT)
        .encode(
            x=alt.X("day:N", sort=DAYS, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("visitors:Q", title=None),
            tooltip=["day", alt.Tooltip("visitors:Q", format=".1f")],
        )
        .properties(height=260)
        .configure_axis(labelColor=ui.MUTED, gridColor=ui.LINE, domainColor=ui.LINE, tickColor=ui.LINE)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(dow_chart, width="stretch")

# ----------------------------------------------------------------------------
# Track record
# ----------------------------------------------------------------------------
with tab_track:
    st.caption(t["tracking_desc"])
    compared = storage.compare_log_to_actuals(storage.load_log(slug), visits)
    if compared.empty:
        st.info(t["tracking_empty"])
    else:
        record = storage.track_record_metrics(compared)
        st.write(t["tracking_metrics"].format(
            n=record["n"], wape=core.fmt_pct(record["wape"]), cov=core.fmt_pct(record["coverage"]),
        ))
        st.line_chart(
            compared.set_index("target_date")[["actual", "predicted"]].rename(
                columns={"actual": t["legend_actual"], "predicted": t["legend_pred"]}
            ),
            color=[ui.INK, ui.ACCENT],
        )
    if guest_demo:
        st.caption(L["guest_track"])
    elif st.button(ui.plain(t["save_forecast_btn"]), key="save_forecast_button", type="primary"):
        saved_days = storage.log_forecast(slug, date.today(), plan)
        storage.sync_to_cloud(slug)
        st.success(t["forecast_saved"].format(n=saved_days))
        st.rerun()

# ----------------------------------------------------------------------------
# Export & email
# ----------------------------------------------------------------------------
with tab_export:
    summary_lines, _ = reports.metric_lines(bundle, t)
    pdf_bytes = reports.build_pdf(t["pdf_title"] + (f" - {loc_name}" if loc_name else ""), summary_lines, headers, pdf_rows)
    ics_bytes = reports.build_ics(plan, location_name=loc_name or BRAND, currency=currency)

    d_col1, d_col2, d_col3 = st.columns(3)
    d_col1.download_button(
        label=ui.plain(t["download_btn"]),
        data=forecast_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="weekly_forecast_schedule.csv", mime="text/csv",
        key="download_csv_btn", width="stretch", icon=":material/table_view:",
    )
    d_col2.download_button(
        label=ui.plain(t["download_pdf_btn"]),
        data=pdf_bytes,
        file_name="executive_forecast_report.pdf", mime="application/pdf",
        key="download_pdf_btn", width="stretch", type="primary", icon=":material/picture_as_pdf:",
    )
    d_col3.download_button(
        label=L["dl_ics"],
        data=ics_bytes,
        file_name=f"{slug.strip('_')}_7day_schedule.ics", mime="text/calendar",
        key="download_ics_btn", width="stretch", icon=":material/calendar_month:",
    )

    st.divider()
    st.subheader(ui.plain(t["email_header"]))
    if guest_demo:
        st.caption(L["guest_email"])
    else:
        st.caption(f"{t['email_recipients']}: {recipients_text or '-'}")
        # Guests never reach this button, so they can't send mail through the app's SMTP account.
        if st.button(t["email_send_btn"], key="send_email_button", type="primary", icon=":material/send:"):
            smtp = mailer.smtp_config(get_secret)
            recipients = mailer.parse_recipients(recipients_text)
            if smtp is None:
                st.error(t["email_not_configured"])
            elif not recipients:
                st.error(t["email_no_recipients"])
            else:
                message = mailer.build_email(
                    smtp["from"], recipients,
                    t["email_subject"].format(name=loc_name or "-", date=future_dates[0].strftime("%Y-%m-%d")),
                    t["email_body"].format(
                        name=loc_name or "-", date=last_date.strftime("%Y-%m-%d"),
                        total=sum(row["expected"] for row in plan),
                    ),
                    pdf_bytes, "executive_forecast_report.pdf",
                )
                try:
                    mailer.send_email(message, smtp)
                    st.success(t["email_sent"].format(n=len(recipients)))
                except OSError as exc:
                    st.error(t["email_failed"].format(reason=f"{type(exc).__name__}: {str(exc)[:200]}"))
        st.caption(t["auto_email_hint"])

ui.md(ui.footer(L["footer"], MAKER))

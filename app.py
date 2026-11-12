"""AI Business Forecaster — Signature Edition (Streamlit UI).

The forecasting logic lives in core.py, files/history in storage.py, PDF + email in reports.py,
text in i18n.py.
"""
import csv
import hmac
import os
from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import core
import mailer
import reports
import storage
from i18n import (
    ADMIN1_TO_SUBDIV,
    COUNTRY_CITIES,
    COUNTRY_CURRENCIES,
    COUNTRY_NAMES,
    COUNTRY_SUBDIVISIONS,
    CURRENCY_OPTIONS,
    DAY_TRANSLATIONS,
    MONTH_TRANSLATIONS,
    TRANSLATIONS,
)
# Customize your personal brand name here — it appears in the top signature pill & footer
CREATOR_SIGNATURE = "TABLECAST LABS"

st.set_page_config(
    page_title="TableCast AI — Hospitality Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Signature Design System (Custom CSS & Typography)
# ============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Syne:wght@700;800&display=swap');

    :root {
        --bg-card: rgba(22, 25, 32, 0.78);
        --bg-card-hover: rgba(30, 34, 45, 0.92);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-accent: rgba(245, 158, 11, 0.45);
        --accent-gold: #F59E0B;
        --accent-ember: #EA580C;
        --accent-emerald: #10B981;
        --text-primary: #F8FAFC;
        --text-muted: #94A3B8;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main container breathing room */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1380px !important;
    }

    /* Signature Hero Banner */
    .sig-hero {
        background: radial-gradient(circle at 85% 15%, rgba(245, 158, 11, 0.16), transparent 45%),
                    radial-gradient(circle at 15% 85%, rgba(234, 88, 12, 0.12), transparent 40%),
                    linear-gradient(135deg, #0F1117 0%, #171A24 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 28px 34px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
    }
    .sig-hero-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        flex-wrap: wrap;
        gap: 10px;
    }
    .sig-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #FBBF24;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 5px 12px;
        border-radius: 999px;
    }
    .sig-dot {
        width: 7px;
        height: 7px;
        background: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
    }
    .sig-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #FFFFFF;
        margin: 0 0 6px 0;
        line-height: 1.15;
    }
    .sig-subtitle {
        color: #94A3B8;
        font-size: 0.98rem;
        margin: 0;
        max-width: 760px;
    }

    /* Bento KPI Cards */
    .bento-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-bottom: 20px;
    }
    .bento-card {
        background: linear-gradient(180deg, rgba(26, 29, 39, 0.9) 0%, rgba(18, 20, 28, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 22px;
        transition: all 0.22s ease;
        position: relative;
        overflow: hidden;
    }
    .bento-card:hover {
        border-color: rgba(245, 158, 11, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 12px 28px -8px rgba(0, 0, 0, 0.45);
    }
    .bento-label {
        color: #94A3B8;
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 8px;
    }
    .bento-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.1;
    }
    .bento-sub {
        margin-top: 8px;
        font-size: 0.78rem;
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
    }
    .bento-pill-good {
        color: #34D399;
        background: rgba(16, 185, 129, 0.12);
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .bento-pill-warn {
        color: #FBBF24;
        background: rgba(245, 158, 11, 0.12);
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
    }

    /* 7-Day Visual Forecast Strip */
    .day-strip {
        display: grid;
        grid-template-columns: repeat(7, minmax(130px, 1fr));
        gap: 10px;
        margin: 16px 0 22px 0;
        overflow-x: auto;
    }
    .day-card {
        background: rgba(22, 25, 34, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 14px 12px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .day-card:hover {
        border-color: rgba(245, 158, 11, 0.45);
        background: rgba(30, 34, 46, 0.95);
    }
    .day-card-closed {
        opacity: 0.48;
        background: rgba(15, 17, 23, 0.6);
    }
    .day-name {
        font-weight: 700;
        font-size: 0.82rem;
        color: #E2E8F0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .day-date {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #64748B;
        margin-bottom: 8px;
    }
    .day-wx {
        font-size: 1.35rem;
        margin: 4px 0;
    }
    .day-temp {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        color: #94A3B8;
        margin-bottom: 8px;
    }
    .day-guests {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.35rem;
        font-weight: 700;
        color: #FBBF24;
    }
    .day-tag {
        display: inline-block;
        margin-top: 6px;
        font-size: 0.66rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 99px;
        background: rgba(255, 255, 255, 0.06);
        color: #CBD5E1;
    }

    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(18, 20, 28, 0.7);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.07);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(245, 158, 11, 0.18) !important;
        color: #FBBF24 !important;
    }

    /* Signature Footer */
    .sig-footer {
        margin-top: 48px;
        padding: 18px 24px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Cached wrappers & Demo Data Generator
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
    """Generate 180 days of realistic bistro visitor history so anyone can test the app immediately."""
    rng = np.random.default_rng(42)
    end_d = date.today() - timedelta(days=1)
    dates = pd.date_range(end_d - timedelta(days=179), end_d, freq="D")
    dow_base = np.array([0, 75, 85, 95, 135, 165, 120])  # Closed Mondays
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
    return f"{rain_mm / 25.4:.2f}in" if unit == "F" else f"{rain_mm:.1f}mm"


def explain_day_driver(d, temp, rain, holiday_dates, on_event, is_closed):
    """Human-readable 'Why this forecast?' micro-tag for each day card."""
    if is_closed:
        return "🔒 Regular Closure"
    if on_event:
        return "🎉 Private Event"
    if d in holiday_dates:
        return "🏛️ Public Holiday"
    if rain >= 3.0 and d.dayofweek >= 5:
        return "🌧️ Wknd Rain Drag"
    if temp >= 18.0 and rain <= 0.5:
        return "☀️ Patio Boost"
    if d.day in {1, 15} or d.day >= 25:
        return "💰 Payday Window"
    if d.dayofweek in {4, 5}:
        return "🔥 Peak Night"
    return "📊 Organic Trend"


# ============================================================================
# Session state & Cloud Sync
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

st.sidebar.subheader("🌐 Language / Langue / Sprache")
selected_lang = st.sidebar.selectbox(
    "Choose Language", ["English", "French", "German"], key="language_selector", label_visibility="collapsed"
)
t = TRANSLATIONS[selected_lang]
DAYS = DAY_TRANSLATIONS[selected_lang]
MONTHS = MONTH_TRANSLATIONS[selected_lang]

# Signature Hero Header
st.markdown(
    f"""
    <div class="sig-hero">
        <div class="sig-hero-top">
            <span class="sig-badge"><span class="sig-dot"></span> {CREATOR_SIGNATURE} • HOSPITALITY AI</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#64748B;">
                ENSEMBLE ENGINE v2.4 • OPEN-METEO LIVE
            </span>
        </div>
        <h1 class="sig-title">{t["title"]}</h1>
        <p class="sig-subtitle">{t["subtitle"]}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

flash = st.session_state.pop("flash", None)
if flash:
    getattr(st, flash[0])(flash[1])

# ============================================================================
# Private Multi-Tenant Portal (Sign In / Create Restaurant Account)
# ============================================================================
saved_locations = storage.list_locations()
saved_names = dict(saved_locations)
app_password = get_secret("APP_PASSWORD")
login_required = (
    bool(app_password)
    or storage.has_any_location_passwords()
    or st.session_state.get("force_login", False)
)
need_login = login_required and not st.session_state.get("use_demo", False)
# A visitor who skipped sign-in via the demo button. They must never see or change saved
# locations, so the sidebar and every save/email action is locked down for them below.
guest_demo = login_required and not st.session_state.authed

if need_login and not st.session_state.authed:
    st.subheader("📊 TableCast Analytics — Portal")
    tab_signin, tab_signup = st.tabs(["🔑 Sign In", "📝 Register New Business"])

    # --- TAB 1: PRIVATE SIGN IN ---
    with tab_signin:
        st.subheader("🔐 Sign In to Your Restaurant")
        login_name = st.text_input(
            "Restaurant Name",
            placeholder="Enter your restaurant name...",
            key="login_rest_name",
        ).strip()
        entered_pw = st.text_input(
            t["password_prompt"],
            type="password",
            key="password_input",
        )

        if st.button("🔓 Sign In", key="login_submit_btn", type="primary", use_container_width=True):
            target_slug = storage.slugify(login_name) if login_name else ""
            if login_name.lower() in {"admin", "administrator"} and app_password:
                if hmac.compare_digest(entered_pw.encode("utf-8"), str(app_password).encode("utf-8")):
                    st.session_state.authed = True
                    st.session_state.auth_role = "admin"
                    st.session_state.locked_slug = None
                    st.session_state.force_login = False
                    st.session_state.use_demo = False
                    st.session_state.pop("password_input", None)
                    st.rerun()
                else:
                    st.error(t["password_wrong"])
            elif target_slug and storage.verify_location_password(target_slug, entered_pw):
                st.session_state.authed = True
                st.session_state.auth_role = "client"
                st.session_state.locked_slug = target_slug
                st.session_state.force_login = False
                st.session_state.use_demo = False
                st.session_state.pop("password_input", None)
                st.rerun()
            else:
                st.error("❌ Invalid restaurant name or password.")

    # --- TAB 2: SELF-SERVE RESTAURANT REGISTRATION ---
    with tab_signup:
        st.subheader("🏪 Register Your Restaurant")
        st.caption("Create a private workspace for your venue. Only you will have access to your data and forecasts.")
        new_rest_name = st.text_input(
            "Restaurant / Venue Name",
            placeholder="e.g. Harbor Bistro Sydney",
            key="signup_rest_name",
        ).strip()
        new_pw1 = st.text_input(
            "Create Password (min. 6 characters)",
            type="password",
            key="signup_pw1",
        )
        new_pw2 = st.text_input(
            "Confirm Password",
            type="password",
            key="signup_pw2",
        )

        if st.button("🚀 Create Account & Continue", key="signup_submit_btn", type="primary", use_container_width=True):
            new_slug = storage.slugify(new_rest_name) if new_rest_name else ""
            if not new_rest_name or new_slug in {"default", "admin", "administrator"}:
                st.error("Please enter a valid restaurant name.")
            elif storage.location_exists(new_slug):
                st.error("⚠️ An account with this restaurant name already exists. Please sign in.")
            elif len(new_pw1) < 6:
                st.error("⚠️ Password must be at least 6 characters long.")
            elif new_pw1 != new_pw2:
                st.error("⚠️ Passwords do not match.")
            else:
                storage.save_settings(new_slug, {
                    "name": new_rest_name,
                    "language": selected_lang,
                    "country": "US",
                    "station": "New York",
                    "currency": "$",
                    "temp_unit": "F",
                    "password_hash": storage.hash_password(new_pw1),
                })
                storage.sync_to_cloud(new_slug)
                st.session_state.authed = True
                st.session_state.auth_role = "client"
                st.session_state.locked_slug = new_slug
                st.session_state.force_login = False
                st.session_state.use_demo = False
                st.session_state["flash"] = ("success", f"🎉 Welcome, {new_rest_name}! Your private account is ready.")
                st.rerun()

    # --- LIVE DEMO BYPASS (No Account Required) ---
    st.markdown("---")
    if st.button("👀 Try Interactive Live Demo (No Account Needed)", key="login_demo_btn", use_container_width=True):
        st.session_state.use_demo = True
        st.session_state.ai_trained = True
        st.session_state.force_login = False
        st.rerun()

    st.stop()

# ============================================================================
# Sidebar: Location, Upload, Settings, Events
# ============================================================================
pending_location = st.session_state.pop("pending_location", None)
if pending_location in saved_names:
    st.session_state["saved_location"] = pending_location

def log_out():
    st.session_state.authed = False
    st.session_state.auth_role = "admin"
    st.session_state.locked_slug = None
    st.session_state.use_demo = False
    st.rerun()


st.sidebar.header(t["location_header"])
if guest_demo:
    # Demo visitors get no access to saved locations: nothing is loaded from or written to disk.
    saved_names = {}
    slug = "_guest_demo"  # underscore: storage.slugify() can never produce this, so no real location collides
    loc_name = ""
    st.sidebar.info("👀 Demo mode — sign in to use your own workspace.")
    if st.sidebar.button("🔑 Sign in / Register", key="exit_demo_btn"):
        log_out()
elif st.session_state.auth_role == "client" and st.session_state.locked_slug:
    chosen_slug = st.session_state.locked_slug
    loc_name = saved_names.get(chosen_slug, chosen_slug)
    slug = chosen_slug
    st.sidebar.success(f"🔒 {loc_name}")
    if st.sidebar.button("🚪 Log out", key="logout_btn"):
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
    if need_login and st.sidebar.button("🚪 Log out", key="admin_logout_btn"):
        log_out()

settings = storage.load_settings(slug) if slug in saved_names else {}

new_loc_password = ""
if not guest_demo:
    pw_status = "🔑 Change Password (optional)" if settings.get("password_hash") else "🔑 Set Account Password"
    new_loc_password = st.sidebar.text_input(pw_status, type="password", key=f"loc_pw_{slug}").strip()
st.sidebar.header(t["upload_header"])
uploaded_file = st.sidebar.file_uploader(t["upload_prompt"], type=["csv"], key="csv_file_uploader")
st.sidebar.markdown(t["req_format"])

st.sidebar.markdown("---")
st.sidebar.subheader(t["settings_header"])
country_codes = list(COUNTRY_NAMES)
default_country_idx = country_codes.index(settings["country"]) if settings.get("country") in country_codes else 0
country = st.sidebar.selectbox(
    t["country_label"], country_codes,
    index=default_country_idx,
    format_func=lambda c: f"{COUNTRY_NAMES[c]} ({c})", key=f"country_{slug}",
)

# Preset cities for selected country + ability to type any custom city or postal/ZIP code
preset_cities = COUNTRY_CITIES.get(country, ["Vienna"])
saved_station = settings.get("station", preset_cities[0])
city_options = preset_cities + ["✍️ Type any city or ZIP code..."]
default_city_idx = preset_cities.index(saved_station) if saved_station in preset_cities else len(preset_cities)

picked_city = st.sidebar.selectbox(
    "🏙️ Select City",
    city_options,
    index=default_city_idx,
    key=f"city_preset_{slug}_{country}",
)
if picked_city == "✍️ Type any city or ZIP code...":
    custom_default = saved_station if saved_station not in preset_cities else ""
    station = st.sidebar.text_input(
        t["station_label"],
        value=custom_default,
        placeholder="e.g. Scottsdale, Gold Coast, 90210...",
        key=f"station_custom_{slug}",
    ).strip() or preset_cities[0]
else:
    station = picked_city

# Live urllib Geocoding verification badge so the user sees the automatic connection
geo_info = core.resolve_location_info(station, country)
# State / Territory Public Holiday Calendar (Auto-detected from City or Manual Override)
subdiv_code = ""
if country in COUNTRY_SUBDIVISIONS:
    subdiv_map = COUNTRY_SUBDIVISIONS[country]
    subdiv_labels = list(subdiv_map.keys())
    saved_subdiv = settings.get("subdiv", "")
    default_sub_idx = 0
    for idx_s, code_s in enumerate(subdiv_map.values()):
        if code_s and code_s == saved_subdiv:
            default_sub_idx = idx_s
            break
    picked_subdiv_label = st.sidebar.selectbox(
        "🏛️ State / Territory Holidays",
        subdiv_labels,
        index=default_sub_idx,
        key=f"subdiv_{slug}_{country}",
    )
    subdiv_code = subdiv_map[picked_subdiv_label]

if not subdiv_code and geo_info.get("admin1"):
    subdiv_code = ADMIN1_TO_SUBDIV.get(geo_info["admin1"], "")

holiday_country = f"{country}-{subdiv_code}" if subdiv_code else country
if subdiv_code:
    st.sidebar.caption(f"🗓️ **Holiday Calendar:** `{holiday_country}` (includes state holidays)")
if geo_info["matched"]:
    region_str = f", {geo_info['admin1']}" if geo_info["admin1"] else ""
    st.sidebar.caption(
        f"📡 **Weather Connected:** {geo_info['name']}{region_str} ({geo_info['country_code']}) "
        f"`[{geo_info['lat']:.2f}, {geo_info['lon']:.2f}]`"
    )
else:
    st.sidebar.caption(f"⚠️ Using capital coordinates for {COUNTRY_NAMES.get(country, country)}")

# Currency & Temperature Unit Toggles
unit_col1, unit_col2 = st.sidebar.columns(2)
default_curr = settings.get("currency") or COUNTRY_CURRENCIES.get(country, "$")
curr_idx = CURRENCY_OPTIONS.index(default_curr) if default_curr in CURRENCY_OPTIONS else 0
currency = unit_col1.selectbox(
    "💱 Currency",
    CURRENCY_OPTIONS,
    index=curr_idx,
    key=f"currency_select_{slug}_{country}",
)

default_temp_unit = settings.get("temp_unit") or ("F" if country == "US" else "C")
temp_unit_label = unit_col2.radio(
    "🌡️ Units",
    ["°C / mm", "°F / in"],
    index=1 if default_temp_unit == "F" else 0,
    horizontal=True,
    key=f"temp_unit_{slug}_{country}",
)
temp_unit = "F" if "°F" in temp_unit_label else "C"

lunch_share = st.sidebar.slider(t["lunch_share"], 0, 100, int(settings.get("lunch_share", 40)), key=f"lunch_{slug}")
weekend_lunch_share = st.sidebar.slider(
    t.get("weekend_lunch_share", "Weekend Lunch Share (%)"),
    0, 100, int(settings.get("weekend_lunch_share", lunch_share)), key=f"wknd_lunch_{slug}",
)
avg_spend = st.sidebar.number_input(
    t["avg_spend"].replace("€", currency),
    min_value=0.0, value=float(settings.get("avg_spend", 25.0)), step=1.0, key=f"spend_{slug}",
)
dinner_spend = st.sidebar.number_input(
    t.get("dinner_spend", f"{t['shift_dinner']} {t['avg_spend']}").replace("€", currency),
    min_value=0.0, value=float(settings.get("dinner_spend", max(avg_spend, 32.0))), step=1.0, key=f"dinner_spend_{slug}",
)
guests_per_server = st.sidebar.number_input(
    t["guests_per_server"], min_value=1, value=int(settings.get("guests_per_server", 15)), step=1, key=f"gps_{slug}"
)
recipients_text = "" if guest_demo else st.sidebar.text_input(
    t["email_recipients"], value=settings.get("recipients", ""), key=f"recipients_{slug}"
)

st.sidebar.markdown("---")
st.sidebar.subheader(t["events_header"])
st.sidebar.write(t["events_desc"])
saved_events = {} if guest_demo else storage.load_events(slug)
has_custom_event = st.sidebar.checkbox(t["checkbox_event"], key="private_event_checkbox")
event_date = None
event_extra_visitors = 0
if has_custom_event:
    event_date = st.sidebar.date_input(t["event_date"], value=date.today(), key="event_date_input")
    event_extra_visitors = int(
        st.sidebar.number_input(t["extra_guests"], min_value=1, max_value=500, value=25, key="extra_guests_input")
    )
    ev_col1, ev_col2 = st.sidebar.columns(2)
    if not guest_demo and ev_col1.button("➕ Save Event", key="save_event_btn"):
        saved_events[event_date.isoformat()] = event_extra_visitors
        storage.save_events(slug, saved_events)
        storage.sync_to_cloud(slug)
        st.rerun()
    if saved_events and ev_col2.button("🗑️ Clear All", key="clear_events_btn"):
        saved_events = storage.save_events(slug, {})
        storage.sync_to_cloud(slug)
        st.rerun()

if saved_events:
    events_summary = ", ".join(f"{k} (+{v})" for k, v in sorted(saved_events.items()))
    st.sidebar.caption(f"📅 **Saved Events:** {events_summary}")

# ============================================================================
# Data Ingestion or 1-Click Live Demo Mode
# ============================================================================
saved_visits = storage.load_visits(slug) if slug in saved_names else None

if uploaded_file is None and saved_visits is None and not st.session_state.use_demo:
    st.info(t["waiting_msg"])
    col_demo1, col_demo2, _ = st.columns([1.2, 2, 2])
    if col_demo1.button("✨ Load Live Demo Bistro (180 Days)", type="primary", use_container_width=True):
        st.session_state.use_demo = True
        st.rerun()
    st.stop()

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
    st.toast(t["success_loaded"].format(n=len(visits)), icon="✅")
elif saved_visits is not None:
    st.session_state.use_demo = False
    visits = saved_visits
else:
    visits = generate_demo_visits()
    if not loc_name:
        loc_name = "Bistro Lumière (Demo)"
    st.caption(
        "✨ **Live Demo Mode Active:** Showing 180 days of synthetic bistro data (closed Mondays). "
        "Upload a CSV in the sidebar at any time to switch to your own data."
    )

st.sidebar.markdown("---")
if not guest_demo and st.sidebar.button(t["save_location_btn"], key="save_location_button", use_container_width=True):
    if not loc_name:
        st.sidebar.error(t["err_location_name"])
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
            # The prep tab is drawn further down, so read its current values from session state.
            "prep_items": st.session_state.get(f"prep_items_list_{slug}", settings.get("prep_items")),
            "prep_buffer": int(st.session_state.get(f"prep_buf_{slug}", settings.get("prep_buffer", 10))),
            "password_hash": storage.hash_password(new_loc_password) if new_loc_password else settings.get("password_hash", ""),
        })
        storage.sync_to_cloud(slug)
        st.session_state["pending_location"] = slug
        st.session_state["flash"] = ("success", t["saved_ok"])
        st.rerun()

# ============================================================================
# Model Training & Future Forecast Calculation
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

# Pre-compute weather and plan so KPIs and tabs can share them
manual_override = st.session_state.get("manual_weather_checkbox", False)
if manual_override:
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
beats_baseline = (
    np.isfinite(bundle["wape7"])
    and np.isfinite(bundle["base_wape7"])
    and bundle["wape7"] < bundle["base_wape7"]
)
edge_pct = bundle["base_wape7"] - bundle["wape7"] if np.isfinite(bundle["wape7"]) else 0.0

# ============================================================================
# Signature Bento KPI Grid
# ============================================================================
verdict_pill = (
    f'<span class="bento-pill-good">▲ {edge_pct:+.1f}% vs Baseline</span>'
    if beats_baseline
    else '<span class="bento-pill-warn">⚠️ Close to Baseline</span>'
)

st.markdown(
    f"""
    <div class="bento-grid">
        <div class="bento-card">
            <div class="bento-label">7-Day Projected Traffic</div>
            <div class="bento-value">{week_total_guests:,} <span style="font-size:0.95rem;color:#94A3B8;">guests</span></div>
            <div class="bento-sub">Hist. Avg: {bundle['mean_visitors']:.0f} / day</div>
        </div>
        <div class="bento-card">
            <div class="bento-label">7-Day Projected Revenue</div>
            <div class="bento-value" style="color:#FBBF24;">{currency}{week_total_rev:,.0f}</div>
            <div class="bento-sub">Lunch {currency}{avg_spend:.0f} • Dinner {currency}{dinner_spend:.0f}</div>
        </div>
        <div class="bento-card">
            <div class="bento-label">{t["metric_wape7"]}</div>
            <div class="bento-value">{core.fmt_pct(bundle["wape7"])}</div>
            <div class="bento-sub">{verdict_pill}</div>
        </div>
        <div class="bento-card">
            <div class="bento-label">{t["metric_mae"]}</div>
            <div class="bento-value">±{bundle["mae"]:.1f} <span style="font-size:0.95rem;color:#94A3B8;">{t["unit_people"]}</span></div>
            <div class="bento-sub">Next-Day WAPE: {core.fmt_pct(bundle["wape"])}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# Workspace Tabs
# ============================================================================
tab_plan, tab_prep, tab_ai, tab_track, tab_export = st.tabs([
    t["planner_header"],
    "🥖 Kitchen & Prep Sheet",
    "📊 AI Performance & Backtest",
    t["tracking_header"],
    "📤 PDF & Email Dispatch",
])

# ----------------------------------------------------------------------------
# TAB 1: 7-Day Planner + Visual Weather Strip + Explainable AI Tags
# ----------------------------------------------------------------------------
with tab_plan:
    top_c1, top_c2 = st.columns([2.2, 1.2])
    with top_c1:
        st.caption(t["forecast_from"].format(date=last_date.strftime("%Y-%m-%d")))
        if bundle["closed_weekdays"]:
            st.caption(t["closed_caption"].format(days=", ".join(DAYS[d] for d in sorted(bundle["closed_weekdays"]))))
    with top_c2:
        st.checkbox(
            t.get("manual_weather_override", "Override weather forecast manually (What-If)"),
            value=not live_weather_ok,
            key="manual_weather_checkbox",
        )

    if st.session_state.get("manual_weather_checkbox", False):
        default_temp, _ = core.default_weather(history)
        col_b1, col_b2 = st.columns(2)
        if temp_unit == "F":
            temp_f_val = col_b1.slider(
                "Avg Week Temp (°F)", 14.0, 104.0, round(c_to_f(default_temp), 1), key="temp_slider_f"
            )
            st.session_state["temp_slider_input"] = f_to_c(temp_f_val)
            rain_in_val = col_b2.slider("Avg Week Rain (in)", 0.0, 2.0, 0.0, step=0.05, key="rain_slider_in")
            st.session_state["rain_slider_input"] = rain_in_val * 25.4
        else:
            col_b1.slider(t["avg_temp"], -10.0, 40.0, default_temp, key="temp_slider_input")
            col_b2.slider(t["avg_rain"], 0.0, 50.0, 0.0, key="rain_slider_input")

    # 7-Day Visual Weather & Driver Strip
    holiday_set = set(bundle["forecaster"].holiday_dates)
    cards_html = ['<div class="day-strip">']
    for row in plan:
        d = row["date"]
        wx_row = future_weather.loc[d]
        temp_val, rain_val = float(wx_row["temp"]), float(wx_row["rain"])
        icon = weather_icon(temp_val, rain_val)
        driver = explain_day_driver(d, temp_val, rain_val, holiday_set, row["on_event"], row["is_closed"])
        closed_cls = " day-card-closed" if row["is_closed"] else ""
        guest_txt = t["closed"] if row["is_closed"] else f"{row['expected']}"
        cards_html.append(
            f'<div class="day-card{closed_cls}">'
            f'<div class="day-name">{DAYS[d.dayofweek][:3]}</div>'
            f'<div class="day-date">{MONTHS[d.month - 1]} {d.day:02d}</div>'
            f'<div class="day-wx">{icon}</div>'
            f'<div class="day-temp">{fmt_temp(temp_val, temp_unit)} • {fmt_rain(rain_val, temp_unit)}</div>'
            f'<div class="day-guests">{guest_txt}</div>'
            f'<div class="day-tag">{driver}</div>'
            f'</div>'
        )
    cards_html.append("</div>")
    st.markdown("".join(cards_html), unsafe_allow_html=True)

    show_range = bundle["intervals"] is not None
    headers, table_rows, pdf_rows = reports.format_plan(plan, t, DAYS, MONTHS, show_range, currency=currency)
    forecast_df = pd.DataFrame(table_rows)

    max_traffic = max([r["expected"] for r in plan] + [1])
    rev_col = f"{t['col_revenue']} ({currency})" if currency else t["col_revenue"]

    st.dataframe(
        forecast_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            t["col_traffic"]: st.column_config.ProgressColumn(
                t["col_traffic"],
                format="%d",
                min_value=0,
                max_value=int(max_traffic * 1.15),
            ),
            rev_col: st.column_config.NumberColumn(
                rev_col,
                format=f"{currency}%.2f",
            ),
        },
    )
    if show_range:
        st.caption(t["range_caption"])
        


# ----------------------------------------------------------------------------
# TAB 2: Kitchen & Bakery Prep Calculator (Custom Ingredients + Presets)
# ----------------------------------------------------------------------------
with tab_prep:
    st.subheader("🥖 7-Day Kitchen & Inventory Prep Sheet")
    st.caption("Add your own custom ingredients (or load a preset) to automatically calculate daily prep targets.")

    PRESET_TEMPLATES = {
        "🍔 Burger / Bistro": [
            {"Item": "Burger Buns / Bread Rolls", "Per Guest": 0.45, "Unit": "pcs"},
            {"Item": "Beef / Meat Prep", "Per Guest": 0.12, "Unit": "kg"},
            {"Item": "Fries / Potatoes", "Per Guest": 0.15, "Unit": "kg"},
        ],
        "🥐 Bakery / Café": [
            {"Item": "Croissants / Pastries", "Per Guest": 0.85, "Unit": "pcs"},
            {"Item": "Espresso Beans", "Per Guest": 0.018, "Unit": "kg"},
            {"Item": "Milk / Oat Milk", "Per Guest": 0.12, "Unit": "L"},
        ],
        "💪 Gym / Shake Bar": [
            {"Item": "Creatine", "Per Guest": 5.0, "Unit": "g"},
            {"Item": "Whey Protein", "Per Guest": 30.0, "Unit": "g"},
            {"Item": "Bananas / Fruit", "Per Guest": 0.5, "Unit": "pcs"},
        ],
    }

    state_key = f"prep_items_list_{slug}"
    if state_key not in st.session_state:
        st.session_state[state_key] = settings.get(
            "prep_items", PRESET_TEMPLATES["🍔 Burger / Bistro"]
        )

    # --- Quick-Add Custom Ingredient Bar ---
    with st.expander("➕ Add Custom Ingredient or Load Preset", expanded=True):
        add_c1, add_c2, add_c3, add_c4 = st.columns([2.2, 1.2, 1, 1.2])
        new_item_name = add_c1.text_input(
            "Ingredient Name", placeholder="e.g. Creatine, Semmeln, Chicken...", key=f"new_ing_name_{slug}"
        )
        new_item_rate = add_c2.number_input(
            "Amount per Guest", min_value=0.01, value=0.50, step=0.05, key=f"new_ing_rate_{slug}"
        )
        new_item_unit = add_c3.selectbox(
            "Unit", ["pcs", "kg", "g", "L", "ml", "lb", "oz", "portions"], key=f"new_ing_unit_{slug}"
        )
        add_c4.write("")
        if add_c4.button("➕ Add Item", key=f"add_ing_btn_{slug}", type="primary", use_container_width=True):
            if new_item_name.strip():
                st.session_state[state_key].append({
                    "Item": new_item_name.strip(),
                    "Per Guest": float(new_item_rate),
                    "Unit": new_item_unit,
                })
                st.rerun()

        pre_c1, pre_c2, pre_c3, pre_c4 = st.columns(4)
        for idx_p, (preset_label, preset_rows) in enumerate(PRESET_TEMPLATES.items()):
            if [pre_c1, pre_c2, pre_c3][idx_p].button(f"Load {preset_label}", key=f"pre_{idx_p}_{slug}", use_container_width=True):
                st.session_state[state_key] = [dict(r) for r in preset_rows]
                st.rerun()
        if pre_c4.button("🗑️ Clear All Items", key=f"clear_ing_{slug}", use_container_width=True):
            st.session_state[state_key] = []
            st.rerun()

    col_p1, col_p2 = st.columns([3, 1])
    with col_p2:
        safety_buffer = st.slider(
            "🛡️ Safety Buffer (%)", 0, 30, int(settings.get("prep_buffer", 10)),
            help="Adds a cushion so you don't run out during an unexpected rush.",
            key=f"prep_buf_{slug}",
        )
    with col_p1:
        prep_config_df = st.data_editor(
            pd.DataFrame(st.session_state[state_key] or [{"Item": "", "Per Guest": 0.0, "Unit": "pcs"}]),
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Item": st.column_config.TextColumn("Ingredient / Item Name", required=True),
                "Per Guest": st.column_config.NumberColumn("Usage per Guest", min_value=0.0, format="%.2f"),
                "Unit": st.column_config.SelectboxColumn(
                    "Unit", options=["pcs", "kg", "g", "L", "ml", "lb", "oz", "portions"]
                ),
            },
            key=f"prep_editor_{slug}",
        )
        st.session_state[state_key] = prep_config_df.to_dict("records")

    # --- 7-Day Calculated Prep Schedule ---
    valid_items = [
        r for _, r in prep_config_df.iterrows()
        if str(r.get("Item", "")).strip() and pd.notna(r.get("Per Guest"))
    ]
    if valid_items:
        prep_schedule = []
        for row in plan:
            d = row["date"]
            day_entry = {
                "Date": f"{DAYS[d.dayofweek][:3]}, {MONTHS[d.month - 1]} {d.day:02d}",
                "Expected Guests": 0 if row["is_closed"] else row["expected"],
            }
            for item_row in valid_items:
                item_name = str(item_row["Item"]).strip()
                rate = float(item_row["Per Guest"] or 0.0)
                unit = str(item_row.get("Unit") or "pcs").strip()
                if row["is_closed"]:
                    day_entry[f"{item_name} ({unit})"] = t["closed"]
                else:
                    raw_needed = row["expected"] * rate * (1.0 + safety_buffer / 100.0)
                    if unit in {"kg", "L", "lb"}:
                        day_entry[f"{item_name} ({unit})"] = f"{raw_needed:.1f}"
                    else:
                        day_entry[f"{item_name} ({unit})"] = f"{int(np.ceil(raw_needed))}"
            prep_schedule.append(day_entry)

        prep_df = pd.DataFrame(prep_schedule)
        st.dataframe(prep_df, hide_index=True, use_container_width=True)
    else:
        st.info("👆 Add at least one ingredient above to generate your 7-day prep schedule.")

# ----------------------------------------------------------------------------
# TAB 2: AI Backtest Chart & Day-of-Week Signature Altair Visuals
# ----------------------------------------------------------------------------
with tab_ai:
    summary_lines, baseline_text = reports.metric_lines(bundle, t)
    st.caption(baseline_text)
    if beats_baseline:
        st.success(t["verdict_better"])
    else:
        st.warning(t["verdict_worse"])

    st.subheader(t["main_chart_header"])
    chart_df = (
        bundle["chart"]
        .rename(columns={"actual": t["legend_actual"], "pred": t["legend_pred"], "baseline": t["legend_baseline"]})
        .reset_index()
        .rename(columns={"index": "date"})
    )
    melted = chart_df.melt("date", var_name="Series", value_name="Visitors")
    line_chart = (
        alt.Chart(melted)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("Visitors:Q", title=t["metric_visitors"]),
            color=alt.Color(
                "Series:N",
                scale=alt.Scale(
                    domain=[t["legend_actual"], t["legend_pred"], t["legend_baseline"]],
                    range=["#F8FAFC", "#F59E0B", "#64748B"],
                ),
                legend=alt.Legend(orient="top"),
            ),
            strokeDash=alt.condition(
                alt.datum.Series == t["legend_baseline"],
                alt.value([4, 4]),
                alt.value([1, 0]),
            ),
            tooltip=["date:T", "Series:N", alt.Tooltip("Visitors:Q", format=".0f")],
        )
        .properties(height=340)
        .interactive()
    )
    st.altair_chart(line_chart, use_container_width=True)

    st.subheader(t["dow_header"])
    st.write(t["dow_sub"])
    dow_df = pd.DataFrame({"day": DAYS, "visitors": bundle["dow_means"].to_numpy()})
    dow_chart = (
        alt.Chart(dow_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#F59E0B")
        .encode(
            x=alt.X("day:N", sort=DAYS, title=None),
            y=alt.Y("visitors:Q", title=t["metric_visitors"]),
            tooltip=["day", alt.Tooltip("visitors:Q", format=".1f")],
        )
        .properties(height=280)
    )
    st.altair_chart(dow_chart, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3: Forecast Track Record & Proof
# ----------------------------------------------------------------------------
with tab_track:
    st.subheader(t["tracking_header"])
    st.write(t["tracking_desc"])

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
            )
        )
    if not guest_demo and st.button(t["save_forecast_btn"], key="save_forecast_button", type="primary"):
        saved_days = storage.log_forecast(slug, date.today(), plan)
        storage.sync_to_cloud(slug)
        st.success(t["forecast_saved"].format(n=saved_days))
        st.rerun()

# ----------------------------------------------------------------------------
# TAB 4: Exports & Email Dispatch
# ----------------------------------------------------------------------------
with tab_export:
    summary_lines, _ = reports.metric_lines(bundle, t)
    pdf_bytes = reports.build_pdf(
        t["pdf_title"] + (f" - {loc_name}" if loc_name else ""), summary_lines, headers, pdf_rows
    )

    ics_bytes = reports.build_ics(plan, location_name=loc_name or "TableCast", currency=currency)

    d_col1, d_col2, d_col3 = st.columns(3)
    with d_col1:
        st.download_button(
            label=t["download_btn"],
            data=forecast_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="weekly_forecast_schedule.csv",
            mime="text/csv",
            key="download_csv_btn",
            use_container_width=True,
        )
    with d_col2:
        st.download_button(
            label=t["download_pdf_btn"],
            data=pdf_bytes,
            file_name="executive_forecast_report.pdf",
            mime="application/pdf",
            key="download_pdf_btn",
            use_container_width=True,
        )
    with d_col3:
        st.download_button(
            label="📅 Download Calendar (.ics)",
            data=ics_bytes,
            file_name=f"{slug}_7day_schedule.ics",
            mime="text/calendar",
            key="download_ics_btn",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader(t["email_header"])
    st.caption(f"{t['email_recipients']}: {recipients_text or '-'}")
    # Guests must not be able to send mail through the app's SMTP account.
    if not guest_demo and st.button(t["email_send_btn"], key="send_email_button", type="primary"):
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

# ============================================================================
# Signature Maker Watermark Footer
# ============================================================================
st.markdown(
    f"""
    <div class="sig-footer">
        <span>⚡ DESIGNED & ENGINEERED BY <strong style="color:#FBBF24;">{CREATOR_SIGNATURE}</strong></span>
        <span>DATA VAULT: {storage.data_dir().resolve().name}/ • MULTI-TENANT READY</span>
    </div>
    """,
    unsafe_allow_html=True,
)

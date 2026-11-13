"""End-to-end checks of the Streamlit app (sign-in, demo lockdown, saving) with AppTest."""
import json
import re
from datetime import date, timedelta
from pathlib import Path

import pytest
import streamlit
from streamlit.testing.v1 import AppTest

import mailer
import storage
from conftest import make_visits

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PASSWORD = "test-password-123"


def run_app(language="English"):
    """Start the app; tests pick English so they can check the English messages."""
    at = AppTest.from_file(APP, default_timeout=180).run()
    if language:
        at = at.selectbox(key="language_selector").set_value(language).run()
    return at


@pytest.fixture
def account():
    """A saved restaurant with a password, so the app requires sign-in."""
    storage.save_visits("harbor-bistro", make_visits())
    storage.save_settings("harbor-bistro", {
        "name": "Harbor Bistro", "country": "AT", "station": "Vienna",
        "recipients": "team@example.com", "owner_email": "owner@example.com",
        "password_hash": storage.hash_password(PASSWORD),
    })
    return "harbor-bistro"


def sign_in(at, name, password):
    at.text_input(key="login_rest_name").input(name)
    at.text_input(key="password_input").input(password)
    return at.button(key="login_submit_btn").click().run()


def test_demo_visitor_cannot_see_or_change_saved_restaurants(account):
    at = run_app()
    assert not at.exception
    at.button(key="login_demo_btn").click().run()
    assert not at.exception
    keys = {w.key for w in at.selectbox} | {w.key for w in at.text_input} | {w.key for w in at.button}
    for forbidden in ("saved_location", "save_location_button", "save_event_btn",
                      "save_forecast_button", "send_email_button", f"loc_pw_{account}", f"recipients_{account}"):
        assert forbidden not in keys
    assert "Harbor Bistro" not in " ".join(str(m.value) for m in at.markdown)


def test_sign_in_locks_after_repeated_wrong_passwords(account):
    at = run_app()
    for _ in range(storage.MAX_FAILED_LOGINS):
        at = sign_in(at, "Harbor Bistro", "nope")
    at = sign_in(at, "Harbor Bistro", PASSWORD)  # right password, but locked now
    assert any("Too many failed attempts" in e.value for e in at.error)
    assert not at.session_state["authed"]


def test_sign_up_enforces_password_length():
    storage.save_settings("someone-else", {"name": "x", "password_hash": storage.hash_password(PASSWORD)})
    at = run_app()
    at.text_input(key="signup_rest_name").input("New Place")
    at.text_input(key="signup_email").input("me@newplace.com")
    at.text_input(key="signup_pw1").input("short")
    at.text_input(key="signup_pw2").input("short")
    at = at.button(key="signup_submit_btn").click().run()
    assert any("at least" in e.value for e in at.error)
    assert not storage.location_exists("new-place")


def test_signed_in_owner_can_save_all_settings(account):
    at = sign_in(run_app(), "Harbor Bistro", PASSWORD)
    assert not at.exception
    assert at.session_state["locked_slug"] == account
    at = at.button(key="save_location_button").click().run()
    assert not at.exception
    saved = json.loads((storage.data_dir() / "locations" / account / "settings.json").read_text(encoding="utf-8"))
    for key in ("subdiv", "weekend_lunch_share", "dinner_spend", "temp_unit", "prep_items", "prep_buffer"):
        assert key in saved
    assert storage.verify_location_password(account, PASSWORD)  # password kept when not changed


def test_sign_up_needs_a_valid_email():
    storage.save_settings("someone-else", {"name": "x", "password_hash": storage.hash_password(PASSWORD)})
    at = run_app()
    at.text_input(key="signup_rest_name").input("New Place")
    at.text_input(key="signup_email").input("not-an-email")
    at.text_input(key="signup_pw1").input(PASSWORD)
    at.text_input(key="signup_pw2").input(PASSWORD)
    at = at.button(key="signup_submit_btn").click().run()
    assert any("valid email" in e.value for e in at.error)
    assert not storage.location_exists("new-place")


@pytest.fixture
def outbox(monkeypatch):
    """A fake mail server: emails are collected instead of sent."""
    sent = []
    monkeypatch.setattr(mailer, "send_email", lambda message, cfg: sent.append(message))
    for name, value in {"SMTP_HOST": "smtp.test", "SMTP_USER": "user", "SMTP_PASSWORD": "pw"}.items():
        monkeypatch.setenv(name, value)
    return sent


def test_password_reset_by_email(account, outbox):
    at = run_app().button(key="forgot_btn").click().run()
    at.text_input(key="reset_rest_name").input("Harbor Bistro")
    at = at.button(key="reset_send_btn").click().run()
    assert not at.exception
    assert len(outbox) == 1 and outbox[0]["To"] == "owner@example.com"
    code = re.search(r"\b(\d{6})\b", outbox[0].get_content()).group(1)

    at.text_input(key="reset_code").input("000000" if code != "000000" else "111111")
    at.text_input(key="reset_pw1").input("brand-new-pass")
    at.text_input(key="reset_pw2").input("brand-new-pass")
    at = at.button(key="reset_submit_btn").click().run()
    assert any("wrong or has expired" in e.value for e in at.error)

    at.text_input(key="reset_code").input(code)
    at = at.button(key="reset_submit_btn").click().run()
    assert not at.exception
    assert storage.verify_location_password(account, "brand-new-pass")
    assert not storage.verify_location_password(account, PASSWORD)


def test_reset_for_unknown_restaurant_looks_the_same_but_sends_nothing(account, outbox):
    at = run_app().button(key="forgot_btn").click().run()
    at.text_input(key="reset_rest_name").input("Nobody Here")
    at = at.button(key="reset_send_btn").click().run()
    assert not at.exception
    assert outbox == []
    assert at.session_state["reset_sent"]  # same "we sent a code if..." screen as a real account


class FakeUpload:
    def __init__(self, data):
        self._data = data

    def getvalue(self):
        return self._data


def test_upload_is_saved_automatically_for_signed_in_owner(account, monkeypatch):
    old = make_visits(end=date.today() - timedelta(days=40))
    storage.save_visits(account, old)
    (storage.data_dir() / "locations" / account / "visits.csv").write_text(old.to_csv(index=False), encoding="utf-8")
    recent = make_visits(days=45, seed=9)
    csv_bytes = recent.rename(columns={"visit_date": "date", "visitors": "covers"}).to_csv(index=False).encode()
    monkeypatch.setattr(streamlit, "file_uploader", lambda *a, **k: FakeUpload(csv_bytes))

    at = sign_in(run_app(), "Harbor Bistro", PASSWORD)
    assert not at.exception
    saved = storage.load_visits(account)
    assert saved["visit_date"].max() == recent["visit_date"].max()       # new days are on disk
    assert saved["visit_date"].min() == old["visit_date"].min()          # old history kept
    assert not saved["visit_date"].duplicated().any()


def test_austria_is_the_default_for_new_accounts_and_the_demo():
    storage.save_settings("someone-else", {"name": "x", "password_hash": storage.hash_password(PASSWORD)})
    at = run_app()
    at.text_input(key="signup_rest_name").input("Wiener Cafe")
    at.text_input(key="signup_email").input("owner@wienercafe.at")
    at.text_input(key="signup_pw1").input(PASSWORD)
    at.text_input(key="signup_pw2").input(PASSWORD)
    at = at.button(key="signup_submit_btn").click().run()
    assert not at.exception
    saved = storage.load_settings("wiener-cafe")
    assert (saved["country"], saved["station"], saved["currency"], saved["temp_unit"]) == ("AT", "Vienna", "€", "C")

    demo = run_app().button(key="login_demo_btn").click().run()
    assert demo.selectbox(key="country__guest_demo").value == "AT"


def test_german_is_the_default_language():
    storage.save_settings("someone-else", {"name": "x", "password_hash": storage.hash_password(PASSWORD)})
    at = run_app(language=None)
    assert not at.exception
    assert at.selectbox(key="language_selector").value == "German"
    labels = [b.label for b in at.button]
    assert "Anmelden" in labels and "Demo ausprobieren (ohne Konto)" in labels


def _geo(admin1):
    return lambda query, country="US": {"lat": 48.8, "lon": 9.2, "name": query, "admin1": admin1,
                                        "country_code": country, "matched": True}


def test_state_holidays_are_picked_from_the_city(account, monkeypatch):
    import core
    storage.save_settings(account, {**storage.load_settings(account), "country": "DE", "station": "Stuttgart"})
    monkeypatch.setattr(core, "resolve_location_info", _geo("Baden-Wurttemberg"))
    at = sign_in(run_app(), "Harbor Bistro", PASSWORD)
    assert not at.exception
    assert any("DE-BW" in c.value for c in at.caption)


def test_a_state_from_another_country_is_never_used(account, monkeypatch):
    import core
    # "Victoria" is an Australian state; a city with that admin1 in Germany must not become DE-VIC
    storage.save_settings(account, {**storage.load_settings(account), "country": "DE", "station": "Somewhere"})
    monkeypatch.setattr(core, "resolve_location_info", _geo("Victoria"))
    at = sign_in(run_app(), "Harbor Bistro", PASSWORD)
    assert not at.exception
    assert not any("DE-VIC" in c.value for c in at.caption)


def test_model_menu_and_custom_blend_work(account):
    at = sign_in(run_app(), "Harbor Bistro", PASSWORD)
    at = at.selectbox(key=f"model_{account}").set_value("custom").run()
    at = at.multiselect(key=f"members_{account}").set_value(["ridge", "extra_trees"]).run()
    assert not at.exception
    at = at.selectbox(key=f"model_{account}").set_value("classic").run()
    at = at.checkbox(key=f"opt_direct_{account}").uncheck().run()
    assert not at.exception


def test_accuracy_tab_shows_schedule_outcomes_and_unusual_days(account):
    at = sign_in(run_app(), "Harbor Bistro", PASSWORD)
    assert not at.exception
    labels = [m.label for m in at.metric]
    for label in ("Short-staffed shifts", "Overstaffed shifts", "Wages for extra staff", "Typical miss with TableCast"):
        assert label in labels


def test_recipe_upload_fills_the_prep_list(account, monkeypatch):
    recipe = b"dish;share;ingredient;amount;unit;pack\nBurger;40%;Buns;1;pcs;12\nBurger;40%;Beef;0.2;kg;\n"
    real_uploader = streamlit.file_uploader
    monkeypatch.setattr(streamlit, "file_uploader", lambda *a, **k: FakeUpload(recipe)
                        if str(k.get("key", "")).startswith("prep_upload") else real_uploader(*a, **k))
    at = sign_in(run_app(), "Harbor Bistro", PASSWORD)
    assert not at.exception
    items = at.session_state[f"prep_items_list_{account}"]
    assert [(i["Item"], round(i["Per Guest"], 3)) for i in items] == [("Buns", 0.4), ("Beef", 0.08)]
    assert items[0]["Pack"] == 12.0 and (items[1]["Pack"] is None or items[1]["Pack"] != items[1]["Pack"])  # empty
    assert any("Loaded 2 items" in s.value for s in at.success)


def test_staff_roles_are_editable_and_saved(account):
    at = sign_in(run_app(), "Harbor Bistro", PASSWORD)
    assert not at.exception
    assert any("Waiters" in str(c) for df in at.dataframe for c in df.value.columns)
    at.session_state[f"lunch_hours_{account}"] = 4.0
    at = at.button(key="save_location_button").click().run()
    assert not at.exception
    saved = storage.load_settings(account)
    assert [r["name"] for r in saved["staff_roles"]] == ["Waiters", "Chefs"]
    assert saved["shift_hours"]["lunch"] == 4.0 and saved["staff_plan_for"] == "expected"

"""End-to-end checks of the Streamlit app (sign-in, demo lockdown, saving) with AppTest."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import storage
from conftest import make_visits

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PASSWORD = "test-password-123"


def run_app():
    return AppTest.from_file(APP, default_timeout=180).run()


@pytest.fixture
def account():
    """A saved restaurant with a password, so the app requires sign-in."""
    storage.save_visits("harbor-bistro", make_visits())
    storage.save_settings("harbor-bistro", {
        "name": "Harbor Bistro", "country": "AT", "station": "Vienna",
        "recipients": "team@example.com", "password_hash": storage.hash_password(PASSWORD),
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

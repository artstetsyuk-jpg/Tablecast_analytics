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


def run_app():
    return AppTest.from_file(APP, default_timeout=180).run()


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

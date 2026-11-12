import pandas as pd
import pytest

import mailer
import reports
import storage


def test_slugify():
    assert storage.slugify("Café Zürich #2") == "cafe-zurich-2"
    assert storage.slugify("   ") == "default"


def test_password_hash_and_verify():
    storage.save_location_password("Harbor Bistro", "correct horse")
    assert storage.verify_location_password("harbor-bistro", "correct horse")
    assert not storage.verify_location_password("harbor-bistro", "wrong")
    assert not storage.verify_location_password("harbor-bistro", "")
    assert storage.has_any_location_passwords() is True


def test_login_lockout_after_repeated_failures():
    now = 1_000_000.0
    for _ in range(storage.MAX_FAILED_LOGINS - 1):
        storage.record_failed_login("bistro", now=now)
    assert storage.login_locked_seconds("bistro", now=now) == 0
    storage.record_failed_login("bistro", now=now)
    assert storage.login_locked_seconds("bistro", now=now) == storage.LOGIN_LOCK_SECONDS
    # Lock expires on its own...
    later = now + storage.LOGIN_LOCK_SECONDS + 1
    assert storage.login_locked_seconds("bistro", now=later) == 0
    # ...and a successful sign-in clears the counter.
    storage.record_failed_login("bistro", now=later)
    storage.clear_failed_logins("bistro")
    assert storage.login_locked_seconds("bistro", now=later) == 0
    assert storage.login_locked_seconds("other-place", now=now) == 0


def test_reset_code_is_single_use_and_expires():
    now = 2_000_000.0
    code = storage.create_reset_code("bistro", now=now)
    assert code and len(code) == 6 and code.isdigit()
    assert not storage.use_reset_code("bistro", "999999" if code != "999999" else "111111", now=now)
    assert storage.use_reset_code("bistro", code, now=now)
    assert not storage.use_reset_code("bistro", code, now=now)                 # used up

    later = now + storage.RESET_RESEND_SECONDS + 1
    code = storage.create_reset_code("bistro", now=later)
    expired = later + storage.RESET_CODE_MINUTES * 60 + 1
    assert not storage.use_reset_code("bistro", code, now=expired)            # too late


def test_reset_code_limits_guesses_and_resends():
    now = 3_000_000.0
    code = storage.create_reset_code("bistro", now=now)
    assert storage.create_reset_code("bistro", now=now + 5) is None          # no re-send within a minute
    wrong = "000000" if code != "000000" else "111111"
    for _ in range(storage.RESET_MAX_ATTEMPTS):
        assert not storage.use_reset_code("bistro", wrong, now=now)
    assert not storage.use_reset_code("bistro", code, now=now)                # locked after too many guesses


def test_save_visits_merges_and_new_values_win(visits):
    storage.save_visits("bistro", visits.head(80))
    update = visits.iloc[70:].copy()
    update.loc[update.index[0], "visitors"] = 999.0
    merged = storage.save_visits("bistro", update)
    assert len(merged) == len(visits)
    assert merged.loc[merged["visit_date"] == update["visit_date"].iloc[0], "visitors"].item() == 999.0


def test_forecast_log_round_trip_and_track_record():
    plan = [
        {"date": pd.Timestamp("2025-03-03"), "expected": 100, "low": 90, "high": 110, "is_closed": False},
        {"date": pd.Timestamp("2025-03-04"), "expected": 0, "low": 0, "high": 0, "is_closed": True},
    ]
    assert storage.log_forecast("bistro", pd.Timestamp("2025-03-01").date(), plan) == 1
    actuals = pd.DataFrame({"visit_date": [pd.Timestamp("2025-03-03")], "visitors": [105.0]})
    compared = storage.compare_log_to_actuals(storage.load_log("bistro"), actuals)
    metrics = storage.track_record_metrics(compared)
    assert metrics["n"] == 1 and metrics["coverage"] == 100.0
    assert metrics["wape"] == pytest.approx(5 / 105 * 100)


def test_calendar_export_has_stable_ids():
    plan = [{"date": pd.Timestamp("2025-03-03"), "expected": 100, "low": 90, "high": 110, "is_closed": False,
             "servers": 7, "lunch": 40, "dinner": 60, "revenue": 2600.0}]
    first, second = reports.build_ics(plan, "Bistro"), reports.build_ics(plan, "Bistro")
    uid = [line for line in first.decode().splitlines() if line.startswith("UID:")]
    assert uid and uid == [line for line in second.decode().splitlines() if line.startswith("UID:")]


def test_pdf_is_generated():
    pdf = reports.build_pdf("Report – Café", ["line"], ["A", "B"], [["1", "2"]])
    assert pdf.startswith(b"%PDF")


def test_parse_recipients():
    assert mailer.parse_recipients("a@x.com, bad, b@y.org; c@z") == ["a@x.com", "b@y.org"]

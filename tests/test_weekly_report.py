from datetime import date, timedelta

import pandas as pd

import storage
import weekly_report
from conftest import make_visits


def _save_location(slug="bistro", **settings):
    storage.save_visits(slug, make_visits())
    storage.save_settings(slug, {"name": "Bistro", "country": "AT", "station": "Vienna", "recipients": "", **settings})
    storage.save_events(slug, {(date.today() + timedelta(days=2)).isoformat(): 40})


def test_build_location_report():
    _save_location(subdiv="", weekend_lunch_share=55, dinner_spend=30.0)
    pdf, subject, body, plan = weekly_report.build_location_report("bistro", storage.load_settings("bistro"), date.today())
    assert pdf.startswith(b"%PDF")
    assert "Bistro" in subject
    assert len(plan) == 7
    event_day = pd.Timestamp(date.today() + timedelta(days=2))
    assert any(row["on_event"] for row in plan if row["date"] == event_day)


def test_dry_run_writes_a_pdf_per_location(isolated):
    _save_location("bistro")
    assert weekly_report.main(["--dry-run"]) == 0
    assert list((isolated / "out").glob("bistro_*.pdf"))


def test_stale_data_is_reported_not_crashed():
    storage.save_visits("old", make_visits(end="2022-09-30"))
    storage.save_settings("old", {"name": "Old"})
    assert weekly_report.main(["--dry-run", "--location", "old"]) == 1

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


def test_report_forecasts_sales_from_a_sales_column():
    v = make_visits(days=120)
    storage.save_visits("bistro", v.assign(revenue=v["visitors"] * 30.0))
    storage.save_settings("bistro", {"name": "Bistro", "country": "AT", "station": "Vienna", "avg_spend": 5.0,
                                     "dinner_spend": 5.0})
    _, _, _, plan = weekly_report.build_location_report("bistro", storage.load_settings("bistro"), date.today())
    open_days = [r for r in plan if not r["is_closed"]]
    per_guest = sum(r["revenue"] for r in open_days) / sum(r["expected"] for r in open_days)
    assert 25 < per_guest < 35   # from the sales model (~30 per guest), not the typed-in 5.00


def test_report_for_a_sales_only_file_estimates_guests():
    v = make_visits(days=120)
    storage.save_visits("bistro", v.assign(visitors=v["visitors"] * 30.0))
    storage.save_settings("bistro", {"name": "Bistro", "country": "AT", "station": "Vienna", "avg_spend": 30.0,
                                     "dinner_spend": 30.0, "data_format": {"unit": "sales"}})
    _, _, _, plan = weekly_report.build_location_report("bistro", storage.load_settings("bistro"), date.today())
    open_days = [r for r in plan if not r["is_closed"]]
    assert all(40 < r["expected"] < 400 for r in open_days)            # guests, not euros
    assert all(abs(r["revenue"] - r["expected"] * 30) < 60 for r in open_days)


def test_dry_run_writes_a_pdf_per_location(isolated):
    _save_location("bistro")
    assert weekly_report.main(["--dry-run"]) == 0
    assert list((isolated / "out").glob("bistro_*.pdf"))


def test_stale_data_is_reported_not_crashed():
    storage.save_visits("old", make_visits(end="2022-09-30"))
    storage.save_settings("old", {"name": "Old"})
    assert weekly_report.main(["--dry-run", "--location", "old"]) == 1

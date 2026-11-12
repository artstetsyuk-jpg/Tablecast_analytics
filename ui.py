"""Presentation helpers: the stylesheet and small HTML building blocks for st.markdown.

Every piece of user-provided text (restaurant names, city names...) goes through `esc()`
before it is placed inside HTML.
"""
import html
import re
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets"

# Chart palette (kept in sync with assets/style.css and .streamlit/config.toml)
INK = "#1C1917"
MUTED = "#78716C"
LINE = "#E7E5E4"
ACCENT = "#C2410C"
ACCENT_LIGHT = "#FDBA74"
ACCENT_DARK = "#7C2D12"
BASELINE = "#A8A29E"

_LEADING_SYMBOLS = re.compile(r"^[^\w(\"'«]+")


def esc(text):
    return html.escape(str(text), quote=True)


def plain(text):
    """Drop leading emoji/symbols from a translated label: '📅 7-Day Planner' -> '7-Day Planner'."""
    return _LEADING_SYMBOLS.sub("", str(text)).strip()


def load_css():
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def md(markup, sidebar=False):
    (st.sidebar if sidebar else st).markdown(markup, unsafe_allow_html=True)


def sidebar_brand(name):
    return (
        f'<div class="tc-brand"><div class="tc-logo">{esc(name[:1])}</div>'
        f'<div class="tc-brand-name">{esc(name)}</div></div>'
    )


def section_label(text):
    return f'<div class="tc-section">{esc(text)}</div>'


def login_header(name, subtitle):
    return (
        f'<div class="tc-login-head"><div class="tc-logo">{esc(name[:1])}</div>'
        f'<div class="tc-login-title">{esc(name)}</div>'
        f'<div class="tc-login-sub">{esc(subtitle)}</div></div>'
    )


def page_header(eyebrow, title, subtitle, chips):
    """chips: [(text, kind)] with kind in {'', 'good', 'warn', 'accent'}."""
    chip_html = "".join(f'<span class="tc-chip {kind}">{esc(text)}</span>' for text, kind in chips)
    return (
        '<div class="tc-header"><div>'
        f'<div class="tc-eyebrow">{esc(eyebrow)}</div>'
        f'<h1 class="tc-title">{esc(title)}</h1>'
        f'<div class="tc-sub">{esc(subtitle)}</div>'
        f'</div><div class="tc-chips">{chip_html}</div></div>'
    )


def note(text):
    return f'<div class="tc-note">{esc(text)}</div>'


def kpi_cards(cards):
    """cards: [{'label', 'value', 'unit'?, 'sub'?, 'pill'?: (text, kind)}]"""
    parts = ['<div class="tc-kpis">']
    for c in cards:
        unit = f'<span class="tc-kpi-unit">{esc(c["unit"])}</span>' if c.get("unit") else ""
        sub = esc(c.get("sub", ""))
        if c.get("pill"):
            text, kind = c["pill"]
            sub = f'<span class="tc-pill {kind}">{esc(text)}</span> {sub}'
        parts.append(
            f'<div class="tc-kpi"><div class="tc-kpi-label">{esc(c["label"])}</div>'
            f'<div class="tc-kpi-value">{esc(c["value"])}{unit}</div>'
            f'<div class="tc-kpi-sub">{sub}</div></div>'
        )
    parts.append("</div>")
    return "".join(parts)


def day_cards(days):
    """days: [{'name', 'date', 'weather', 'guests', 'range', 'tag', 'tag_kind', 'closed', 'event'}]"""
    parts = ['<div class="tc-days">']
    for d in days:
        cls = "tc-day" + (" closed" if d["closed"] else "") + (" event" if d["event"] else "")
        parts.append(
            f'<div class="{cls}">'
            f'<div class="tc-day-head"><span class="tc-day-name">{esc(d["name"])}</span>'
            f'<span class="tc-day-date">{esc(d["date"])}</span></div>'
            f'<div class="tc-day-wx">{esc(d["weather"])}</div>'
            f'<div class="tc-day-guests">{esc(d["guests"])}</div>'
            f'<div class="tc-day-range">{esc(d["range"])}</div>'
            f'<span class="tc-tag {d["tag_kind"]}">{esc(d["tag"])}</span>'
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def empty_state(title, body, steps):
    step_html = "".join(
        f'<div class="tc-step"><div class="tc-step-num">{i}</div>{esc(s)}</div>'
        for i, s in enumerate(steps, start=1)
    )
    return (
        f'<div class="tc-empty"><h2>{esc(title)}</h2><p>{esc(body)}</p>'
        f'<div class="tc-steps">{step_html}</div></div>'
    )


def footer(left, right):
    return f'<div class="tc-footer"><span>{esc(left)}</span><span>{esc(right)}</span></div>'

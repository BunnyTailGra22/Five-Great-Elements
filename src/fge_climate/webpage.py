"""Build the four-measure web page from the processed daily dataset.

The page is generated rather than hand-written so it stays in step with the
daily sync. Everything it draws comes from ``data/processed`` and is embedded in
the page as JSON, which keeps the artifact self-contained.

Colour slots come from the validated categorical palette, assigned in the
documented slot order so that neighbouring panels always clear the adjacent-pair
gate in both light and dark mode.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, Station
from .indices import MEAN_MIN_COVERAGE, SUM_MIN_COVERAGE
from .trends import trend

TEMPLATE_PATH = Path(__file__).parent / "templates" / "measures.html"
PLACEHOLDER = "/*__PAYLOAD__*/"

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

MEASURES = [
    {
        "key": "TxSoil0cm",
        "zh": "地溫 0 公分",
        "en": "Soil temperature at 0 cm",
        "unit": "°C",
        "agg": "mean",
        "slot": 1,
        "why": (
            "Bud break and root activity respond to the soil, not the air. This is the "
            "measure most directly tied to phenology, and the one with the shortest record."
        ),
    },
    {
        "key": "Precp",
        "zh": "日降水量",
        "en": "Daily precipitation",
        "unit": "mm",
        "agg": "sum",
        "slot": 2,
        "why": (
            "The wettest measure on the mountain and the longest clean record here. "
            "Timing matters more than total: when the rain arrives sets the flush."
        ),
    },
    {
        "key": "SunShine",
        "zh": "日照時數",
        "en": "Sunshine duration",
        "unit": "hour",
        "agg": "mean",
        "slot": 3,
        "why": (
            "Drives photosynthesis and leaf temperature. The instrument was absent for "
            "two decades, so this measure has the largest hole in it."
        ),
    },
    {
        "key": "WS",
        "zh": "風速",
        "en": "Wind speed",
        "unit": "m/s",
        "agg": "mean",
        "slot": 4,
        "why": (
            "Controls evaporative demand and physical stress on new growth. Also the "
            "measure whose record is most disturbed by changes to the station itself."
        ),
    },
]

# Confirmed and visually evident discontinuities, surfaced on the long-record chart.
KNOWN_BREAKS: dict[str, list[dict]] = {
    "WS": [
        {"year": 2001, "label": "step up", "confirmed": False},
        {"year": 2007, "label": "confirmed break", "confirmed": True},
    ],
}


PROSE_UNIT = {"hour": "h"}


def _measure_findings(measure: dict, entry: dict, window: tuple[int, int]) -> list[dict]:
    """Compact, fully computed findings for one measure on the overview page.

    Derived from the payload that was just built, so the wording tracks the data
    rather than being written once and going stale.
    """
    unit = PROSE_UNIT.get(measure["unit"], measure["unit"])
    clim = [c for c in entry["climatology"] if c.get("mean") is not None]
    if not clim:
        return []

    high = max(clim, key=lambda c: c["mean"])
    low = min(clim, key=lambda c: c["mean"])
    decimals = 0 if measure["agg"] == "sum" else 1
    blocks = entry["blocks"]
    out: list[dict] = []

    shape = (
        f"{high['label']} is the peak month at {high['mean']:,.{decimals}f} {unit} and "
        f"{low['label']} the trough at {low['mean']:,.{decimals}f}"
    )
    if measure["agg"] == "sum":
        ratio = high["mean"] / low["mean"] if low["mean"] else 0
        shape += f" — a {ratio:.1f}x difference across the year."
    else:
        shape += f", a spread of {high['mean'] - low['mean']:,.1f} {unit}."
    out.append({"lead": "Shape of the year", "text": shape})

    # How much record actually exists, stated as gaps rather than a raw span.
    span = f"{blocks[0][0]}–{blocks[-1][1]}" if blocks else "none"
    fragments = len(blocks)
    record = f"{entry['record_years']} usable years within {span}"
    if fragments > 1:
        record += f", broken into {fragments} runs by years below the coverage gate"
    else:
        record += ", unbroken"
    out.append({"lead": "How much record", "text": record + "."})

    trend = entry.get("segment_trend")
    if entry.get("breaks"):
        years = ", ".join(str(b["year"]) for b in entry["breaks"])
        out.append({
            "lead": "Treat trends with care",
            "text": f"Step changes at {years} come from the station, not the weather; "
                    f"only the segment after the last one is comparable.",
        })
    elif trend and trend.get("slope_per_decade") is not None:
        p_value = trend.get("p_value")
        verdict = (
            "significant" if p_value is not None and p_value < 0.05 else "not significant"
        )
        caution = " — far too short to read as climate" if trend["n"] < 12 else ""
        out.append({
            "lead": "Trend so far",
            "text": f"{trend['slope_per_decade']:+,.2f} {unit}/decade over "
                    f"{trend['start']}–{trend['end']} ({trend['n']} years, {verdict}){caution}.",
        })
    else:
        out.append({
            "lead": "Trend so far",
            "text": "The record is too short or too broken for a trend to mean anything.",
        })

    extremes = entry.get("extremes") or {}
    if extremes.get("daily_max") is not None:
        out.append({
            "lead": f"Extreme in {window[0]}–{window[1]}",
            "text": f"{extremes['daily_max']:g} {unit} on {extremes['daily_max_date']}.",
        })
    return out[:4]


def _gate(measure: dict) -> float:
    return SUM_MIN_COVERAGE if measure["agg"] == "sum" else MEAN_MIN_COVERAGE


def _usable_blocks(years: list[int]) -> list[list[int]]:
    """Collapse a sorted year list into contiguous [start, end] runs."""
    blocks: list[list[int]] = []
    for year in years:
        if blocks and year == blocks[-1][1] + 1:
            blocks[-1][1] = year
        else:
            blocks.append([year, year])
    return blocks


def _annual_series(daily: pd.DataFrame, measure: dict) -> list[dict]:
    key, gate = measure["key"], _gate(measure)
    rows = []
    for year, group in daily.groupby("year"):
        series = group[key]
        coverage = float(series.notna().mean()) if len(group) else 0.0
        if coverage >= gate:
            value = float(series.sum()) if measure["agg"] == "sum" else float(series.mean())
        else:
            value = None
        rows.append({"year": int(year), "value": value, "coverage": round(coverage, 4)})
    return rows


def _climatology(daily: pd.DataFrame, measure: dict, window: tuple[int, int]) -> list[dict]:
    """Monthly cycle over the common window: mean plus the year-to-year envelope."""
    key = measure["key"]
    block = daily[(daily["year"] >= window[0]) & (daily["year"] <= window[1])]

    per_year: list[dict] = []
    for (year, month), group in block.groupby(["year", "month"]):
        series = group[key]
        days = pd.Period(f"{int(year)}-{int(month):02d}").days_in_month
        if series.notna().sum() < days * MEAN_MIN_COVERAGE:
            continue
        value = float(series.sum()) if measure["agg"] == "sum" else float(series.mean())
        per_year.append({"year": int(year), "month": int(month), "value": value})

    frame = pd.DataFrame(per_year)
    out = []
    for month in range(1, 13):
        values = frame[frame["month"] == month]["value"] if not frame.empty else pd.Series(dtype=float)
        if values.empty:
            out.append({"month": month, "label": MONTH_LABELS[month - 1], "mean": None})
            continue
        out.append(
            {
                "month": month,
                "label": MONTH_LABELS[month - 1],
                "mean": round(float(values.mean()), 2),
                "lo": round(float(values.min()), 2),
                "hi": round(float(values.max()), 2),
                "n": int(len(values)),
            }
        )
    return out


def _daily_extremes(daily: pd.DataFrame, measure: dict, window: tuple[int, int]) -> dict:
    key = measure["key"]
    block = daily[(daily["year"] >= window[0]) & (daily["year"] <= window[1])]
    series = block[key].dropna()
    if series.empty:
        return {}
    peak_row = block.loc[block[key].idxmax()]
    return {
        "daily_mean": round(float(series.mean()), 2),
        "daily_max": round(float(series.max()), 2),
        "daily_max_date": str(pd.Timestamp(peak_row["date"]).date()),
        "n_days": int(len(series)),
    }


def build_payload(config: Config, station: Station, daily: pd.DataFrame) -> dict:
    """Assemble everything the page draws."""
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])

    last_complete = int(config.end_year - 1)

    # The window where all four instruments are simultaneously reporting. Soil
    # temperature and sunshine both begin in 2018, so this is what bounds it.
    starts = []
    for measure in MEASURES:
        gate = _gate(measure)
        good = [
            int(y)
            for y, group in daily.groupby("year")
            if group[measure["key"]].notna().mean() >= gate and y <= last_complete
        ]
        blocks = _usable_blocks(sorted(good))
        starts.append(blocks[-1][0] if blocks else last_complete)
    common_start = max(starts)
    window = (common_start, last_complete)

    measures = []
    for measure in MEASURES:
        gate = _gate(measure)
        annual = _annual_series(daily, measure)
        usable_years = sorted(
            r["year"] for r in annual if r["value"] is not None and r["year"] <= last_complete
        )
        blocks = _usable_blocks(usable_years)

        frame = pd.DataFrame([r for r in annual if r["value"] is not None and r["year"] <= last_complete])
        window_values = [
            r["value"] for r in annual if r["value"] is not None and window[0] <= r["year"] <= window[1]
        ]

        # Trend only within the most recent unbroken run of years, and only when
        # that run is long enough to mean anything.
        segment_trend = None
        if blocks:
            seg_start, seg_end = blocks[-1]
            segment = frame[(frame["year"] >= seg_start) & (frame["year"] <= seg_end)]
            if len(segment) >= 8:
                result = trend(segment["year"], segment["value"])
                segment_trend = {
                    "start": seg_start,
                    "end": seg_end,
                    "slope_per_decade": None
                    if result["slope_per_decade"] is None or np.isnan(result["slope_per_decade"])
                    else round(result["slope_per_decade"], 3),
                    "p_value": None
                    if np.isnan(result["p_value"])
                    else round(result["p_value"], 3),
                    "n": result["n"],
                }

        measures.append(
            {
                **{k: measure[k] for k in ("key", "zh", "en", "unit", "agg", "slot", "why")},
                "gate": gate,
                "climatology": _climatology(daily, measure, window),
                "annual": annual,
                "blocks": blocks,
                "breaks": KNOWN_BREAKS.get(measure["key"], []),
                "extremes": _daily_extremes(daily, measure, window),
                "window_mean": round(float(np.mean(window_values)), 2) if window_values else None,
                "window_min": round(float(np.min(window_values)), 2) if window_values else None,
                "window_max": round(float(np.max(window_values)), 2) if window_values else None,
                "segment_trend": segment_trend,
                "record_years": len(usable_years),
            }
        )
        measures[-1]["findings"] = _measure_findings(measure, measures[-1], window)

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "station": {
            "id": station.id,
            "name_zh": station.name_zh,
            "name_en": station.name_en,
            "lat": station.lat,
            "lon": station.lon,
            "elevation_m": station.elevation_m,
            "established": station.established,
            "address_zh": "新北市石碇區格頭村",
        },
        "area": {
            "name_zh": config.study_area.name_zh,
            "name_en": config.study_area.name_en,
            "summit_elevation_m": config.study_area.summit_elevation_m,
        },
        "window": {"start": window[0], "end": window[1]},
        "record": {
            "start": int(daily["year"].min()),
            "end": str(pd.Timestamp(daily["date"].max()).date()),
        },
        "monthLabels": MONTH_LABELS,
        "measures": measures,
    }


TITLE_PATTERN = re.compile(r"<title>(.*?)</title>\s*", re.S)

# Standalone documents are for GitHub Pages, which serves the file directly.
# The Artifact publisher instead wraps a fragment in its own skeleton, so the
# two builds differ only by this wrapper and the print variant.
STANDALONE = """<!doctype html>
<html lang="en" data-variant="{variant}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{description}">
<title>{title}</title>
</head>
<body>
{content}</body>
</html>
"""


def write_page(
    payload: dict,
    out_path: Path,
    template_path: Path | None = None,
    standalone: bool = False,
    variant: str = "print",
    description: str = "",
) -> Path:
    template = (template_path or TEMPLATE_PATH).read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise ValueError(f"Template is missing the {PLACEHOLDER} placeholder")

    # separators keep the embedded JSON compact; ensure_ascii=False keeps the
    # Chinese station and measure names readable in the source.
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Guard against the JSON terminating the enclosing <script> element.
    blob = blob.replace("</", "<\\/")
    rendered = template.replace(PLACEHOLDER, blob)

    if standalone:
        match = TITLE_PATTERN.search(rendered)
        title = match.group(1).strip() if match else "Erge Mountain climate"
        # The title moves into <head>; leaving it in <body> would be invalid.
        rendered = TITLE_PATTERN.sub("", rendered, count=1)
        rendered = STANDALONE.format(
            variant=variant,
            title=title,
            description=description or title,
            content=rendered,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    return out_path

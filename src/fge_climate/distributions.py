"""Daily-resolution distribution analysis for the post-2018 window.

The annual indices elsewhere in this package compress each year to a single
number. This module goes the other way and works on the raw daily series, which
is where the shape of the data lives: how skewed a month is, how much of the
year is dry, how wide the spread around a monthly mean actually is.

2018 is the natural floor. The soil probe and the sunshine recorder both begin
then, and every artificial break the homogeneity tests found (2002, 2007, 2016)
falls before it -- so this window needs no adjustment before it can be pooled.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, Station
from .webpage import MEASURES, MONTH_LABELS, write_page

TEMPLATE_PATH = Path(__file__).parent / "templates" / "distributions.html"

START_YEAR = 2018

# Whiskers follow Tukey: the furthest observation within 1.5 IQR of the box.
WHISKER_IQR = 1.5

# Per-month outliers carried into the page. Beyond this the dots overplot and
# add nothing, so the count is reported instead.
MAX_OUTLIERS_PER_MONTH = 40

# Fraction of a month's days that must be present for a heat-map cell to be drawn.
CELL_MIN_COVERAGE = 0.90

HEATMAP_CLASSES = 6

# Frequency binning. Daily rainfall is ~48% zeros and runs to 216 mm, so equal
# width bins would be one spike; the standard rainfall classes are used instead.
PRECIP_CLASS_EDGES = [0.0, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, np.inf]
# Kept short so all eight fit on one axis without dropping every other label.
PRECIP_CLASS_LABELS = [
    "<0.1", "0.1–1", "1–5", "5–10", "10–25", "25–50", "50–100", "≥100",
]

BIN_WIDTHS = {"TxSoil0cm": 2.0, "SunShine": 1.0, "WS": 0.25}

# A day counts as wet at or above this. Daily rainfall is ~48% zeros, which
# collapses a box plot to median 0 with everything else an outlier, so the
# monthly boxes for rainfall describe wet days only -- the intensity when it
# does rain. The dry fraction is reported alongside so nothing is hidden.
WET_DAY_MM = 0.1


# Threshold above which a rainfall day counts as heavy, for the concentration
# statistic ("what share of the year's rain arrives on how few days").
HEAVY_DAY_MM = 50.0

# A day with less than this much sun is effectively overcast.
DULL_DAY_HOURS = 1.0

# Soil cold-exposure threshold, the quantity dormancy models care about.
COLD_SOIL_C = 10.0

# Daily-mean wind above this is a windy day at this sheltered site.
WINDY_DAY_MS = 2.5


def _pct(part: float, whole: float) -> float:
    return round(100 * part / whole, 1) if whole else 0.0


# Units are written for axis labels; prose wants the short form.
PROSE_UNIT = {"hour": "h"}


def _u(unit: str) -> str:
    return PROSE_UNIT.get(unit, unit)


def _month_list(months: list[int]) -> str:
    """Join month names. A range is wrong here: cold months wrap the year end,
    so [1, 2, 3, 12] must not read as "Jan-Dec"."""
    names = [MONTH_LABELS[m - 1] for m in months]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _extreme(frame: pd.DataFrame, key: str) -> tuple[float, str]:
    """Largest observed daily value and the date it fell on."""
    valid = frame[["date", key]].dropna(subset=[key])
    if valid.empty:
        return float("nan"), ""
    row = valid.loc[valid[key].idxmax()]
    return float(row[key]), str(pd.Timestamp(row["date"]).date())


def _key_findings(measure: dict, monthly: list[dict], frame: pd.DataFrame) -> list[dict]:
    """Three or four compact, fully computed statements per measure.

    Everything here is derived from the pooled window, so the wording stays
    true when the dataset is re-synced -- nothing is written by hand.
    """
    key, unit = measure["key"], _u(measure["unit"])
    filled = [m for m in monthly if m.get("n")]
    if not filled:
        return []

    values = frame[key].dropna()
    hottest = max(filled, key=lambda m: m["median"])
    coolest = min(filled, key=lambda m: m["median"])
    widest = max(filled, key=lambda m: m["iqr"])
    tightest = min(filled, key=lambda m: m["iqr"])
    peak, peak_date = _extreme(frame, key)

    out: list[dict] = []

    if key == "TxSoil0cm":
        cold = frame.loc[frame[key] < COLD_SOIL_C]
        cold_months = sorted({int(m) for m in cold["month"]}) if len(cold) else []
        out.append({
            "lead": "Seasonal march",
            "text": f"Median soil rises from {coolest['median']:.1f} {unit} in {coolest['label']} "
                    f"to {hottest['median']:.1f} in {hottest['label']}, a swing of "
                    f"{hottest['median'] - coolest['median']:.1f} {unit}.",
        })
        out.append({
            "lead": "Summer runs steady",
            "text": f"Spread narrows to {tightest['iqr']:.1f} {unit} in {tightest['label']} "
                    f"against {widest['iqr']:.1f} in {widest['label']} — summer soil varies "
                    f"far less day to day than winter soil.",
        })
        if len(cold):
            out.append({
                "lead": "Cold exposure",
                "text": f"{len(cold)} days below {COLD_SOIL_C:.0f} {unit} "
                        f"({_pct(len(cold), len(values))}% of the record), only in "
                        f"{_month_list(cold_months)}; coldest {values.min():.1f} {unit}.",
            })

    elif key == "Precp":
        dry = float((values < WET_DAY_MM).sum())
        heavy = values[values >= HEAVY_DAY_MM]
        wettest_often = min(filled, key=lambda m: m["dry_pct"])
        out.append({
            "lead": "Dry more often than not",
            "text": f"{_pct(dry, len(values))}% of days record no rain, yet the site takes "
                    f"{values.sum() / (len(values) / 365.25):,.0f} {unit} a year.",
        })
        out.append({
            "lead": "Frequency and intensity peak apart",
            "text": f"It rains most often in {wettest_often['label']} "
                    f"({wettest_often['dry_pct']:.0f}% dry) but hardest in {hottest['label']} "
                    f"(median wet day {hottest['median']:.1f} {unit} against "
                    f"{coolest['median']:.1f} in {coolest['label']}).",
        })
        out.append({
            "lead": "A few days carry the year",
            "text": f"The {len(heavy)} days at or above {HEAVY_DAY_MM:.0f} {unit} are "
                    f"{_pct(len(heavy), len(values))}% of the record but deliver "
                    f"{_pct(float(heavy.sum()), float(values.sum()))}% of all rainfall.",
        })

    elif key == "SunShine":
        dull = float((values < DULL_DAY_HOURS).sum())
        out.append({
            "lead": "A grey winter",
            "text": f"{int(dull)} days ({_pct(dull, len(values))}%) saw under "
                    f"{DULL_DAY_HOURS:.0f} hour of sun; {coolest['label']} is dullest, "
                    f"median {coolest['median']:.1f} {unit}.",
        })
        out.append({
            "lead": "Summer is dependable",
            "text": f"{hottest['label']} runs a median {hottest['median']:.1f} {unit} with an "
                    f"IQR of {hottest['iqr']:.1f} — the sun is not just stronger, it is "
                    f"more reliable.",
        })
        out.append({
            "lead": "Range",
            "text": f"Monthly medians span {coolest['median']:.1f}–{hottest['median']:.1f} "
                    f"{unit}; the best single day reached {peak:.1f}.",
        })

    elif key == "WS":
        windy = float((values >= WINDY_DAY_MS).sum())
        medians = [m["median"] for m in filled]
        out.append({
            "lead": "Almost no seasonal cycle",
            "text": f"Monthly medians span only {max(medians) - min(medians):.2f} {unit} "
                    f"({min(medians):.2f}–{max(medians):.2f}) — unusual among the four "
                    f"measures, which are otherwise strongly seasonal.",
        })
        out.append({
            "lead": "A sheltered site",
            "text": f"Half of all days sit between {values.quantile(.25):.2f} and "
                    f"{values.quantile(.75):.2f} {unit}; only {int(windy)} days "
                    f"({_pct(windy, len(values))}%) reach {WINDY_DAY_MS:.1f}.",
        })

    if peak == peak or key != "TxSoil0cm":
        out.append({
            "lead": "Record day",
            "text": f"{peak:g} {unit} on {peak_date}.",
        })
    return out[:4]


def _describe(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {
        "n": int(values.size),
        "min": round(float(values.min()), 2),
        "q1": round(float(q1), 2),
        "median": round(float(median), 2),
        "q3": round(float(q3), 2),
        "max": round(float(values.max()), 2),
        "mean": round(float(values.mean()), 2),
        "sd": round(float(values.std(ddof=1)), 2) if values.size > 1 else 0.0,
        "p05": round(float(np.percentile(values, 5)), 2),
        "p95": round(float(np.percentile(values, 95)), 2),
    }


def _box(values: np.ndarray) -> dict:
    """Tukey box statistics plus the outliers that fall outside the whiskers."""
    stats = _describe(values)
    if not stats["n"]:
        return stats

    iqr = stats["q3"] - stats["q1"]
    lo_fence = stats["q1"] - WHISKER_IQR * iqr
    hi_fence = stats["q3"] + WHISKER_IQR * iqr

    inside = values[(values >= lo_fence) & (values <= hi_fence)]
    stats["lo_whisker"] = round(float(inside.min()), 2) if inside.size else stats["q1"]
    stats["hi_whisker"] = round(float(inside.max()), 2) if inside.size else stats["q3"]
    stats["iqr"] = round(float(iqr), 2)

    outliers = values[(values < lo_fence) | (values > hi_fence)]
    stats["outlier_count"] = int(outliers.size)
    if outliers.size:
        # Keep the most extreme, measured from the middle of the box.
        centre = stats["median"]
        order = np.argsort(-np.abs(outliers - centre))
        kept = outliers[order][:MAX_OUTLIERS_PER_MONTH]
        stats["outliers"] = [round(float(v), 2) for v in np.sort(kept)]
    else:
        stats["outliers"] = []
    return stats


def _frequency(values: np.ndarray, measure: dict) -> dict:
    """Histogram of daily values, binned appropriately for the measure."""
    key = measure["key"]

    if key == "Precp":
        counts, _ = np.histogram(values, bins=PRECIP_CLASS_EDGES)
        total = int(counts.sum()) or 1
        bins = [
            {
                "label": label,
                "lo": PRECIP_CLASS_EDGES[i],
                "count": int(count),
                "pct": round(100 * count / total, 2),
            }
            for i, (label, count) in enumerate(zip(PRECIP_CLASS_LABELS, counts))
        ]
        return {
            "mode": "classes",
            "bins": bins,
            "note": (
                "Rainfall classes, not equal-width bins: the first class is dry days, "
                "nearly half the record, which would swamp a linear histogram."
            ),
        }

    width = BIN_WIDTHS[key]
    lo = np.floor(values.min() / width) * width
    hi = np.ceil(values.max() / width) * width
    edges = np.arange(lo, hi + width / 2, width)
    counts, _ = np.histogram(values, bins=edges)
    total = int(counts.sum()) or 1

    decimals = 0 if width >= 1 else 2
    bins = [
        {
            "label": f"{edges[i]:.{decimals}f}",
            "lo": round(float(edges[i]), 3),
            "hi": round(float(edges[i + 1]), 3),
            "count": int(count),
            "pct": round(100 * count / total, 2),
        }
        for i, count in enumerate(counts)
    ]
    return {
        "mode": "bins",
        "bins": bins,
        "width": width,
        "note": f"Equal-width bins of {width:g} {measure['unit']}.",
    }


def _heatmap(daily: pd.DataFrame, measure: dict, end_year: int) -> dict:
    """Year x month grid of the monthly aggregate."""
    key, agg = measure["key"], measure["agg"]
    rows = []
    values: list[float] = []

    for year in range(START_YEAR, end_year + 1):
        cells = []
        for month in range(1, 13):
            group = daily[(daily["year"] == year) & (daily["month"] == month)]
            days_in_month = pd.Period(f"{year}-{month:02d}").days_in_month
            valid = group[key].dropna()
            coverage = len(valid) / days_in_month if days_in_month else 0.0

            if coverage >= CELL_MIN_COVERAGE:
                value = float(valid.sum()) if agg == "sum" else float(valid.mean())
                values.append(value)
                cells.append({"month": month, "value": round(value, 2), "coverage": round(coverage, 3)})
            else:
                cells.append({"month": month, "value": None, "coverage": round(coverage, 3)})
        rows.append({"year": year, "cells": cells})

    lo = min(values) if values else 0.0
    hi = max(values) if values else 1.0
    step = (hi - lo) / HEATMAP_CLASSES if hi > lo else 1.0
    edges = [round(lo + step * i, 2) for i in range(HEATMAP_CLASSES + 1)]

    return {
        "rows": rows,
        "min": round(lo, 2),
        "max": round(hi, 2),
        "edges": edges,
        "agg": "total" if agg == "sum" else "mean",
    }


def build_payload(config: Config, station: Station, daily: pd.DataFrame) -> dict:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])

    # Box plots and histograms pool complete years only, so every month is
    # represented by the same number of seasons. The heat map keeps the running
    # year, flagged by its own coverage.
    last_complete = int(config.end_year - 1)
    pooled = daily[(daily["year"] >= START_YEAR) & (daily["year"] <= last_complete)]
    heat_end = int(daily["year"].max())

    measures = []
    for measure in MEASURES:
        key = measure["key"]
        series = pooled[key].dropna()

        wet_only = key == "Precp"

        monthly = []
        for month in range(1, 13):
            observed = pooled.loc[pooled["month"] == month, key].dropna().to_numpy(dtype=float)
            if wet_only:
                values = observed[observed >= WET_DAY_MM]
                dry_pct = round(100 * float((observed < WET_DAY_MM).mean()), 1) if observed.size else None
            else:
                values = observed
                dry_pct = None
            monthly.append(
                {
                    "month": month,
                    "label": MONTH_LABELS[month - 1],
                    "days_observed": int(observed.size),
                    "dry_pct": dry_pct,
                    **_box(values),
                }
            )

        measures.append(
            {
                **{k: measure[k] for k in ("key", "zh", "en", "unit", "agg", "slot")},
                "overall": _describe(series.to_numpy(dtype=float)),
                "boxBasis": "wet days only (≥ 0.1 mm)" if wet_only else "all observed days",
                "logScale": wet_only,
                "monthly": monthly,
                "frequency": _frequency(series.to_numpy(dtype=float), measure),
                "findings": _key_findings(measure, monthly, pooled),
                "heatmap": _heatmap(daily, measure, heat_end),
            }
        )

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "station": {
            "id": station.id,
            "name_zh": station.name_zh,
            "elevation_m": station.elevation_m,
        },
        "area": {"name_zh": config.study_area.name_zh, "name_en": config.study_area.name_en},
        "window": {"start": START_YEAR, "end": last_complete},
        "heatmapEnd": heat_end,
        "totalObservations": int(
            sum(pooled[m["key"]].notna().sum() for m in MEASURES)
        ),
        "days": int(len(pooled)),
        "monthLabels": MONTH_LABELS,
        "measures": measures,
    }


def write_distributions(
    config: Config,
    station: Station,
    daily: pd.DataFrame,
    out_path: Path,
    nav: list[dict] | None = None,
    **kwargs,
) -> Path:
    payload = build_payload(config, station, daily)
    if nav:
        payload["nav"] = nav
        payload["navNote"] = f"Station {station.id} · {station.name_zh}"
    return write_page(payload, out_path, TEMPLATE_PATH, **kwargs)

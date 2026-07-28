"""Generate the baseline climate report from the processed datasets.

The report is regenerated on every sync, so its wording is derived from the
statistics rather than written by hand: significance language is driven by the
Mann-Kendall p-value, direction by the Theil-Sen slope, and whether a trend is
presented as a finding at all is driven by the homogeneity tests.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import Config, Station
from .homogeneity import (
    DERIVED_FROM,
    homogeneity_report,
    marginal_breaks,
    propagate_derived_breaks,
    propagate_dtr_breaks,
    usable_segments,
)
from .trends import period_comparison, trend, trend_table

# index -> (label, unit, decimals)
INDEX_LABELS: dict[str, tuple[str, str, int]] = {
    "tmean": ("Mean temperature", "°C", 2),
    "tmax_mean": ("Mean daily maximum", "°C", 2),
    "tmin_mean": ("Mean daily minimum", "°C", 2),
    "dtr": ("Diurnal temperature range", "°C", 2),
    "gdd10": ("Growing degree days (base 10 °C)", "°C·d", 0),
    "gdd10_doy100": ("Date reaching 100 °C·d", "DOY", 1),
    "gdd10_doy200": ("Date reaching 200 °C·d", "DOY", 1),
    "warm_season_start_doy": ("Warm season start (Tmean ≥ 20 °C)", "DOY", 1),
    "warm_season_end_doy": ("Warm season end", "DOY", 1),
    "warm_season_length": ("Warm season length", "days", 1),
    "chill_days_tmin10": ("Chill days (Tmin ≤ 10 °C)", "days", 1),
    "cold_days_tmean12": ("Cold days (Tmean ≤ 12 °C)", "days", 1),
    "last_cold_day_doy": ("Last spring cold day", "DOY", 1),
    "summer_days_tmax30": ("Hot days (Tmax ≥ 30 °C)", "days", 1),
    "hot_nights_tmin25": ("Warm nights (Tmin ≥ 25 °C)", "days", 1),
    "precp_total": ("Annual rainfall", "mm", 0),
    "precp_wet_season": ("Wet-season rainfall (Apr–Sep)", "mm", 0),
    "precp_dry_season": ("Dry-season rainfall (Oct–Mar)", "mm", 0),
    "rain_days_01": ("Rain days (≥ 0.1 mm)", "days", 1),
    "rain_days_1": ("Rain days (≥ 1 mm)", "days", 1),
    "rain_days_10": ("Heavy rain days (≥ 10 mm)", "days", 1),
    "rain_days_50": ("Very heavy rain days (≥ 50 mm)", "days", 1),
    "sdii": ("Rainfall intensity (mm per wet day)", "mm/d", 2),
    "rx1day": ("Wettest day", "mm", 1),
    "rx5day": ("Wettest 5 days", "mm", 1),
    "max_dry_spell": ("Longest dry spell", "days", 1),
    "sunshine_total": ("Annual sunshine", "hours", 0),
    "sunshine_daily_mean": ("Mean daily sunshine", "hours", 2),
    "ws_mean": ("Mean wind speed", "m/s", 2),
    "tsoil0_mean": ("Mean 0 cm soil temperature", "°C", 2),
}

TREND_INDICES = list(INDEX_LABELS)

# Indices tested for artificial breaks. Restricted to those a reference station
# can also supply, since the difference test needs both sides.
HOMOGENEITY_INDICES = [
    "tmean",
    "tmax_mean",
    "tmin_mean",
    "dtr",
    "gdd10",
    "warm_season_length",
    "chill_days_tmin10",
    "summer_days_tmax30",
    "precp_total",
    "rain_days_1",
    "sdii",
    "ws_mean",
]

# Indices that carry the headline story, in the order they are discussed.
HEADLINE = [
    "tmean",
    "tmin_mean",
    "tmax_mean",
    "dtr",
    "gdd10",
    "warm_season_length",
    "chill_days_tmin10",
    "summer_days_tmax30",
    "precp_total",
    "rain_days_1",
    "sdii",
    "max_dry_spell",
    "ws_mean",
]

# The climate normals worth quoting for a phenology study.
NORMALS = [
    "tmean",
    "tmax_mean",
    "tmin_mean",
    "gdd10",
    "gdd10_doy100",
    "gdd10_doy200",
    "warm_season_start_doy",
    "warm_season_length",
    "chill_days_tmin10",
    "last_cold_day_doy",
    "precp_total",
    "rain_days_1",
    "sunshine_daily_mean",
    "tsoil0_mean",
]


def _isnan(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _fmt(value, decimals: int) -> str:
    if _isnan(value):
        return "–"
    return f"{value:,.{decimals}f}"


def _signed(value, decimals: int) -> str:
    if _isnan(value):
        return "–"
    return f"{value:+,.{max(decimals, 1)}f}"


def _doy_to_date(doy) -> str:
    if _isnan(doy):
        return "–"
    stamp = pd.Timestamp("2001-01-01") + pd.Timedelta(days=float(doy) - 1)
    return stamp.strftime("%-d %b")


def _p(value) -> str:
    return "–" if _isnan(value) else f"{value:.3f}"


def _significance(p) -> str:
    if _isnan(p):
        return "not tested"
    if p < 0.01:
        return "significant, p < 0.01"
    if p < 0.05:
        return "significant, p < 0.05"
    if p < 0.10:
        return "marginal, p < 0.10"
    return "not significant"


def _break_map(breaks: pd.DataFrame) -> dict[str, list[int]]:
    """index -> break years confirmed against at least one reference station."""
    if breaks.empty:
        return {}
    return {
        index: usable_segments(breaks, index)
        for index in breaks["index"].unique()
        if usable_segments(breaks, index)
    }


def _segment_trends(
    annual: pd.DataFrame, break_map: dict[str, list[int]]
) -> dict[str, dict]:
    """Trend within the homogeneous segment that follows the last known break."""
    out: dict[str, dict] = {}
    for index in TREND_INDICES:
        if index not in annual.columns:
            continue
        breaks = break_map.get(index, [])
        start = max(breaks) if breaks else None
        if start is None:
            continue
        segment = annual[annual["year"] >= start]
        result = trend(segment["year"], segment[index])
        result["segment_start"] = start
        out[index] = result
    return out


def _findings(
    trends: pd.DataFrame,
    break_map: dict[str, list[int]],
    segment_trends: dict[str, dict],
    inferred: dict[str, list[int]],
    inherited: dict[str, list[int]],
) -> tuple[list[str], list[str]]:
    """Split headline indices into interpretable and break-contaminated."""
    lookup = trends.set_index("index")
    clean: list[str] = []
    contaminated: list[str] = []

    for name in HEADLINE:
        if name not in lookup.index:
            continue
        row = lookup.loc[name]
        label, unit, decimals = INDEX_LABELS[name]
        slope = row["slope_per_decade"]
        if _isnan(slope):
            continue

        breaks = break_map.get(name, [])
        span = f"{int(row['first_year'])}–{int(row['last_year'])}"

        if breaks:
            years = ", ".join(str(y) for y in breaks)
            segment = segment_trends.get(name)
            note = ""
            if segment and not _isnan(segment.get("slope_per_decade")):
                note = (
                    f" Within the post-{segment['segment_start']} segment alone "
                    f"({segment['n']} years) the slope is "
                    f"{_signed(segment['slope_per_decade'], decimals)} {unit}/decade "
                    f"({_significance(segment['p_value'])})."
                )
            if name in inherited:
                basis = (
                    f"it counts days from {INDEX_LABELS[DERIVED_FROM[name]][0]}, which "
                    f"breaks at {years}"
                )
            elif name in inferred:
                basis = (
                    f"a step change at {years}, attributed from the confirmed break in "
                    "diurnal temperature range"
                )
            else:
                basis = (
                    f"a step change at {years} survives differencing against the "
                    "reference station(s)"
                )
            contaminated.append(
                f"- **{label}** — the {span} slope of "
                f"{_signed(slope, decimals)} {unit}/decade is **not usable**: {basis}, so it "
                f"reflects the station, not the climate.{note}"
            )
            continue

        direction = "rising" if slope > 0 else ("falling" if slope < 0 else "flat")
        total = row.get("total_change")
        detail = f"{abs(slope):,.{max(decimals, 1)}f} {unit}/decade"
        if name.endswith("_doy") and not _isnan(total):
            detail += f" ({'later' if slope > 0 else 'earlier'}, {abs(total):,.0f} days over {span})"
        elif not _isnan(total):
            detail += f" ({abs(total):,.{decimals}f} {unit} over {span})"

        clean.append(
            f"- **{label}** — {direction}, {detail}; {_significance(row['p_value'])}, "
            f"n = {int(row['n'])} years."
        )

    return clean, contaminated


def write_report(
    config: Config,
    station: Station,
    annual: pd.DataFrame,
    seasonal: pd.DataFrame,
    coverage: pd.DataFrame,
    references: dict[str, pd.DataFrame],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # The current year is incomplete; exclude it from every trend fit.
    complete = annual[annual["year"] < config.end_year].copy()
    span_start = int(complete["year"].min())
    span_end = int(complete["year"].max())

    trends = trend_table(complete, TREND_INDICES)
    breaks = homogeneity_report(complete, references, HOMOGENEITY_INDICES)
    break_map = _break_map(breaks)
    inferred = propagate_dtr_breaks(breaks, break_map)
    for index, years in inferred.items():
        break_map[index] = sorted(set(break_map.get(index, [])) | set(years))

    # Threshold counts inherit breaks from the variable they are counted from.
    inherited = propagate_derived_breaks(break_map)
    for index, years in inherited.items():
        break_map[index] = sorted(set(break_map.get(index, [])) | set(years))
    segment_trends = _segment_trends(complete, break_map)

    cautions = {
        index: marginal_breaks(breaks, index)
        for index in HOMOGENEITY_INDICES
        if marginal_breaks(breaks, index) and index not in break_map
    }

    split = span_start + (span_end - span_start + 1) // 2
    periods = period_comparison(complete, TREND_INDICES, split)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    distance_km = _haversine(
        station.lat, station.lon, config.study_area.summit_lat, config.study_area.summit_lon
    )

    L: list[str] = []
    L.append(f"# Climate baseline for {config.study_area.name_en} ({config.study_area.name_zh})")
    L.append("")
    L.append(
        f"*Auto-generated by `fge-climate analyze` on {generated}. "
        "Do not edit by hand — rerun the pipeline instead.*"
    )
    L.append("")

    # --- Station ---------------------------------------------------------
    L.append("## Station")
    L.append("")
    L.append("| Field | Value |")
    L.append("| --- | --- |")
    L.append(f"| Station | `{station.id}` {station.name_zh} ({station.name_en}) |")
    L.append(f"| Position | {station.lat:.6f}°N, {station.lon:.6f}°E |")
    L.append(f"| Elevation | {station.elevation_m:,.0f} m |")
    L.append(
        f"| Distance to {config.study_area.name_en} summit | {distance_km:.1f} km "
        f"({config.study_area.summit_elevation_m - station.elevation_m:,.0f} m below the summit) |"
    )
    L.append(f"| Record established | {station.established} |")
    L.append(f"| Analysis window | {span_start}–{span_end} ({span_end - span_start + 1} years) |")
    for ref in config.references:
        L.append(
            f"| Reference station | `{ref.id}` {ref.name_zh} ({ref.elevation_m:,.0f} m, "
            f"{_haversine(station.lat, station.lon, ref.lat, ref.lon):.1f} km) |"
        )
    L.append("")

    # --- Headline --------------------------------------------------------
    L.append("## Headline")
    L.append("")
    if break_map:
        affected = ", ".join(INDEX_LABELS[i][0] for i in break_map)
        L.append(
            f"**The raw record is not homogeneous.** Step changes that survive differencing "
            f"against a neighbouring station were detected in {len(break_map)} of "
            f"{len(HOMOGENEITY_INDICES)} tested indices ({affected}). Long-term trends in "
            "those variables measure changes in the instrument or its exposure, not the "
            "climate, and must not be fed into a phenology model before homogenisation."
        )
        if inferred:
            attributed = ", ".join(
                f"{INDEX_LABELS[i][0]} ({', '.join(str(y) for y in years)})"
                for i, years in inferred.items()
            )
            L.append("")
            L.append(
                f"Attributed by inference from the diurnal-range break: {attributed}. "
                "See the homogeneity section for how attribution works."
            )
        if inherited:
            carried = ", ".join(
                f"{INDEX_LABELS[i][0]} ({', '.join(str(y) for y in years)})"
                for i, years in inherited.items()
            )
            L.append("")
            L.append(f"Inherited from a broken base variable: {carried}.")
        if cautions:
            flagged = ", ".join(
                f"{INDEX_LABELS[i][0]} ({', '.join(str(y) for y in years)})"
                for i, years in cautions.items()
            )
            L.append("")
            L.append(
                f"Marginal breaks (0.05 ≤ p < 0.10), not disqualifying but worth "
                f"watching as the record lengthens: {flagged}."
            )
    else:
        L.append(
            "No significant artificial breaks were detected against the reference "
            "station(s); full-record trends can be read at face value."
        )
    L.append("")

    # --- Findings --------------------------------------------------------
    clean, contaminated = _findings(trends, break_map, segment_trends, inferred, inherited)
    L.append("## What has changed")
    L.append("")
    L.append(
        "Theil–Sen slopes with Mann–Kendall significance. Years failing the coverage "
        "gate are excluded, so `n` differs between indices."
    )
    L.append("")
    L.append("### Interpretable as climate")
    L.append("")
    L.extend(clean or ["- *(none — every headline index shows an artificial break)*"])
    L.append("")
    if contaminated:
        L.append("### Contaminated by station changes")
        L.append("")
        L.extend(contaminated)
        L.append("")

    # --- Homogeneity -----------------------------------------------------
    L.append("## Homogeneity tests")
    L.append("")
    L.append(
        "Pettitt changepoint test. *Direct* runs on the station's own series and will "
        "flag real regional climate shifts as well as artificial ones. *vs reference* runs "
        "on the difference against a neighbouring station, which cancels regional weather — "
        "a break surviving there is a property of this station."
    )
    L.append("")
    L.append("| Index | Test | Reference | n | Break year | Shift | p |")
    L.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for _, row in breaks.iterrows():
        label, unit, decimals = INDEX_LABELS[row["index"]]
        mark = " **\\***" if row["significant"] else ""
        L.append(
            f"| {label} | {row['test']} | {row['reference']} | {int(row['n'])} | "
            f"{'–' if _isnan(row['break_year']) else int(row['break_year'])}{mark} | "
            f"{_signed(row.get('shift'), decimals)} {unit} | {_p(row['p_value'])} |"
        )
    L.append("")
    L.append("**\\*** marks breaks significant at p < 0.05.")
    L.append("")

    # --- Normals ---------------------------------------------------------
    normal_start = max(break_map[i][-1] for i in break_map) if break_map else span_start
    normal_start = max(normal_start, span_end - 29)
    normals = complete[complete["year"] >= normal_start]
    L.append(f"## Site climatology ({normal_start}–{span_end})")
    L.append("")
    L.append(
        "Computed over the most recent homogeneous segment, which is the period a "
        "phenology model should be calibrated against."
    )
    L.append("")
    L.append("| Index | Unit | Mean | Min | Max | n |")
    L.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for index in NORMALS:
        if index not in normals.columns:
            continue
        series = normals[index].dropna()
        if series.empty:
            continue
        label, unit, decimals = INDEX_LABELS[index]
        if index.endswith("_doy"):
            L.append(
                f"| {label} | date | {_doy_to_date(series.mean())} | "
                f"{_doy_to_date(series.min())} | {_doy_to_date(series.max())} | {len(series)} |"
            )
        else:
            L.append(
                f"| {label} | {unit} | {_fmt(series.mean(), decimals)} | "
                f"{_fmt(series.min(), decimals)} | {_fmt(series.max(), decimals)} | {len(series)} |"
            )
    L.append("")

    # --- Trend table -----------------------------------------------------
    L.append("## Full trend table")
    L.append("")
    L.append("| Index | Unit | n | Mean | Slope/decade | 95% CI | p | Break? |")
    L.append("| --- | --- | ---: | ---: | ---: | :---: | ---: | :---: |")
    for _, row in trends.iterrows():
        label, unit, decimals = INDEX_LABELS[row["index"]]
        ci = (
            "–"
            if _isnan(row["slope_lo"])
            else f"{_signed(row['slope_lo'], decimals)} … {_signed(row['slope_hi'], decimals)}"
        )
        mark = " **\\***" if row.get("significant") else ""
        flagged = break_map.get(row["index"])
        flag = "⚠ " + ", ".join(str(y) for y in flagged) if flagged else ""
        L.append(
            f"| {label} | {unit} | {int(row['n'])} | {_fmt(row.get('mean'), decimals)} | "
            f"{_signed(row['slope_per_decade'], decimals)}{mark} | {ci} | "
            f"{_p(row['p_value'])} | {flag} |"
        )
    L.append("")
    L.append("**\\*** significant at p < 0.05. ⚠ marks an index with a confirmed artificial break.")
    L.append("")

    # --- Seasonal --------------------------------------------------------
    L.append("## Seasonal temperature")
    L.append("")
    L.append("| Season | n | Mean (°C) | Slope/decade (°C) | p |")
    L.append("| --- | ---: | ---: | ---: | ---: |")
    for season in ("DJF", "MAM", "JJA", "SON"):
        block = seasonal[(seasonal["season"] == season) & (seasonal["year"] < config.end_year)]
        stats_row = trend_table(block, ["tmean"])
        if stats_row.empty:
            continue
        row = stats_row.iloc[0]
        mark = " **\\***" if row["significant"] else ""
        L.append(
            f"| {season} | {int(row['n'])} | {_fmt(row.get('mean'), 2)} | "
            f"{_signed(row['slope_per_decade'], 2)}{mark} | {_p(row['p_value'])} |"
        )
    L.append("")

    # --- Phenological timing --------------------------------------------
    L.append("## Phenological timing, decade by decade")
    L.append("")
    L.append(
        "Thermal-forcing dates, which are what field phenology observations need to be "
        "matched against."
    )
    L.append("")
    timing = ["gdd10_doy100", "gdd10_doy200", "warm_season_start_doy", "last_cold_day_doy"]
    L.append("| Decade | " + " | ".join(INDEX_LABELS[c][0] for c in timing) + " | n |")
    L.append("| --- | " + " | ".join(["---"] * len(timing)) + " | ---: |")
    decades = complete.assign(decade=(complete["year"] // 10) * 10)
    for decade, group in decades.groupby("decade"):
        cells = [_doy_to_date(group[c].mean()) for c in timing]
        L.append(
            f"| {int(decade)}s | " + " | ".join(cells) + f" | {int(group['gdd10_doy200'].notna().sum())} |"
        )
    L.append("")

    # --- Early vs late ---------------------------------------------------
    if not periods.empty:
        L.append(f"## First half vs second half ({span_start}–{split - 1} vs {split}–{span_end})")
        L.append("")
        L.append(
            "Shown for completeness. Where an index is flagged with a break above, this "
            "comparison mostly measures that break."
        )
        L.append("")
        L.append("| Index | Unit | Early mean | Late mean | Change | p | Break? |")
        L.append("| --- | --- | ---: | ---: | ---: | ---: | :---: |")
        for _, row in periods.iterrows():
            label, unit, decimals = INDEX_LABELS[row["index"]]
            mark = " **\\***" if row["significant"] else ""
            flag = "⚠" if break_map.get(row["index"]) else ""
            L.append(
                f"| {label} | {unit} | {_fmt(row['early_mean'], decimals)} | "
                f"{_fmt(row['late_mean'], decimals)} | {_signed(row['delta'], decimals)}{mark} | "
                f"{_p(row['p_value'])} | {flag} |"
            )
        L.append("")

    # --- Coverage --------------------------------------------------------
    L.append("## Data coverage")
    L.append("")
    L.append(
        "Fraction of days per year with a valid value after QC. Low values are real gaps "
        "in the station record, not pipeline failures. Years below the coverage gate are "
        "dropped from the indices rather than silently averaged."
    )
    L.append("")
    L.append("| Year | Days | Tmean | Precp | Sunshine | Wind | Soil 0 cm |")
    L.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in coverage.iterrows():
        L.append(
            f"| {int(row['year'])} | {int(row['days'])} | {row['Tmean']:.0%} | "
            f"{row['Precp']:.0%} | {row['SunShine']:.0%} | {row['WS']:.0%} | {row['TxSoil0cm']:.0%} |"
        )
    L.append("")

    path = out_dir / "climate_baseline.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")

    breaks.to_csv(out_dir / "homogeneity_tests.csv", index=False)
    trends.to_csv(out_dir / "trends.csv", index=False)
    return path


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))

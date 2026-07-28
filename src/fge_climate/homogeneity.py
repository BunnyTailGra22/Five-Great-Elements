"""Detect artificial breaks in the station record.

A long station series is only usable for trend work if the instrument, its
exposure and its site stayed comparable. Station 82A750 does not obviously
satisfy that: the raw record contains step changes that no regional climate
signal can explain.

Two tests are applied:

* **Pettitt** on the station's own annual series, which locates the single most
  likely changepoint and tests whether it is significant.
* **Pettitt on a difference series** (target minus a nearby reference station).
  Regional climate is common to both stations and cancels in the difference, so
  a break that survives is a property of the target station's instrumentation
  or siting rather than of the weather. This is the test that carries weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def pettitt(values: np.ndarray) -> tuple[int, float, float]:
    """Return (index of changepoint, K statistic, approximate p-value)."""
    n = len(values)
    if n < 8:
        return -1, np.nan, np.nan

    ranks = rankdata(values)
    # U_t for t = 1..n-1, computed from ranks rather than the O(n^2) double sum.
    u = np.array([2.0 * np.sum(ranks[:t]) - t * (n + 1) for t in range(1, n)])
    k_index = int(np.argmax(np.abs(u)))
    k_stat = float(np.abs(u[k_index]))

    p = 2.0 * np.exp(-6.0 * k_stat**2 / (n**3 + n**2))
    return k_index, k_stat, float(min(p, 1.0))


def changepoint(years: pd.Series, values: pd.Series) -> dict:
    """Locate the most likely break year in an annual series."""
    mask = values.notna()
    y = years[mask].to_numpy()
    v = values[mask].to_numpy(dtype=float)

    if len(v) < 8:
        return {"n": int(len(v)), "break_year": None, "p_value": np.nan, "significant": False}

    idx, k_stat, p = pettitt(v)
    # The break falls between y[idx] and y[idx + 1]; report the first year of the
    # later segment, which is the year the new regime starts.
    break_year = int(y[idx + 1])
    before = float(np.mean(v[: idx + 1]))
    after = float(np.mean(v[idx + 1 :]))

    return {
        "n": int(len(v)),
        "break_year": break_year,
        "k_stat": k_stat,
        "p_value": p,
        "significant": bool(p < 0.05),
        "mean_before": before,
        "mean_after": after,
        "shift": after - before,
    }


def difference_series(
    target: pd.DataFrame, reference: pd.DataFrame, column: str
) -> pd.DataFrame:
    """Annual target-minus-reference series for one index."""
    left = target[["year", column]].rename(columns={column: "target"})
    right = reference[["year", column]].rename(columns={column: "reference"})
    merged = left.merge(right, on="year", how="inner").dropna()
    merged["difference"] = merged["target"] - merged["reference"]
    return merged


def homogeneity_report(
    target: pd.DataFrame,
    references: dict[str, pd.DataFrame],
    columns: list[str],
) -> pd.DataFrame:
    """Run the direct and difference-series break tests across indices."""
    rows: list[dict] = []

    for column in columns:
        if column not in target.columns:
            continue

        direct = changepoint(target["year"], target[column])
        rows.append({"index": column, "test": "direct", "reference": "—", **direct})

        for ref_name, ref_frame in references.items():
            if column not in ref_frame.columns:
                continue
            diff = difference_series(target, ref_frame, column)
            if len(diff) < 8:
                continue
            result = changepoint(diff["year"], diff["difference"])
            rows.append(
                {"index": column, "test": "vs reference", "reference": ref_name, **result}
            )

    return pd.DataFrame(rows)


def usable_segments(breaks: pd.DataFrame, index: str) -> list[int]:
    """Break years that the difference tests agree on for one index."""
    relevant = breaks[
        (breaks["index"] == index)
        & (breaks["test"] == "vs reference")
        & (breaks["significant"])
    ]
    return sorted({int(y) for y in relevant["break_year"].dropna()})


def marginal_breaks(breaks: pd.DataFrame, index: str, lo: float = 0.05, hi: float = 0.10) -> list[int]:
    """Difference-test breaks that fall short of significance but warrant caution."""
    relevant = breaks[
        (breaks["index"] == index)
        & (breaks["test"] == "vs reference")
        & (breaks["p_value"] >= lo)
        & (breaks["p_value"] < hi)
    ]
    return sorted({int(y) for y in relevant["break_year"].dropna()})


# Threshold-counting indices inherit any break in the daily variable they count.
DERIVED_FROM: dict[str, str] = {
    "summer_days_tmax30": "tmax_mean",
    "hot_nights_tmin25": "tmin_mean",
    "chill_days_tmin10": "tmin_mean",
    "cold_days_tmean12": "tmean",
    "gdd10": "tmean",
    "warm_season_length": "tmean",
    "warm_season_start_doy": "tmean",
    "warm_season_end_doy": "tmean",
    "gdd10_doy100": "tmean",
    "gdd10_doy200": "tmean",
    "last_cold_day_doy": "tmin_mean",
}


def propagate_derived_breaks(confirmed: dict[str, list[int]]) -> dict[str, list[int]]:
    """Carry a break in a base variable through to the indices counted from it.

    A shift in the daily maximum moves every threshold count built on it, so
    "days above 30 C" is no more trustworthy than the Tmax series underneath it.
    Returns the inherited breaks only.
    """
    inherited: dict[str, list[int]] = {}
    for derived, base in DERIVED_FROM.items():
        base_years = confirmed.get(base)
        if not base_years:
            continue
        existing = set(confirmed.get(derived, []))
        new_years = sorted(set(base_years) - existing)
        if new_years:
            inherited[derived] = new_years
    return inherited


def propagate_dtr_breaks(
    breaks: pd.DataFrame, confirmed: dict[str, list[int]], window: int = 2
) -> dict[str, list[int]]:
    """Attribute a confirmed DTR break to whichever of Tmax/Tmin caused it.

    Diurnal range is the difference of two temperatures, so a break in it must
    originate in at least one of them. The difference series for DTR is far less
    noisy than for Tmax or Tmin individually -- year-to-year swings in the
    overall temperature level cancel out -- so Pettitt routinely resolves a break
    in DTR that it misses in the component that actually shifted. When that
    happens the component is flagged by attribution: take the component whose own
    difference test puts a break at the same year (within ``window``) and shows
    the larger absolute shift.

    Returns the inferred breaks only, keyed by index.
    """
    dtr_years = confirmed.get("dtr", [])
    if not dtr_years:
        return {}

    difference_tests = breaks[breaks["test"] == "vs reference"]
    inferred: dict[str, list[int]] = {}

    for dtr_year in dtr_years:
        candidates: list[tuple[float, str, int]] = []
        for component in ("tmax_mean", "tmin_mean"):
            rows = difference_tests[difference_tests["index"] == component]
            for _, row in rows.iterrows():
                year = row.get("break_year")
                shift = row.get("shift")
                if year is None or pd.isna(year) or pd.isna(shift):
                    continue
                if abs(int(year) - dtr_year) <= window:
                    candidates.append((abs(float(shift)), component, int(year)))

        if not candidates:
            continue

        _, component, year = max(candidates)
        # Skip if that component already failed its own test at this year.
        if year in confirmed.get(component, []):
            continue
        inferred.setdefault(component, [])
        if year not in inferred[component]:
            inferred[component].append(year)

    return {k: sorted(v) for k, v in inferred.items()}

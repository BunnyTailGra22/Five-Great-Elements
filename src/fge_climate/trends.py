"""Trend estimation for annual index series.

Uses the non-parametric pair that is standard in climatology: a Theil-Sen slope
for magnitude and a Mann-Kendall test for significance. Both tolerate the
non-normal, outlier-heavy distributions typical of rainfall indices, and both
handle the gaps left by coverage gating without interpolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mann_kendall(values: np.ndarray) -> tuple[float, float, float]:
    """Return (S, tau, two-sided p) using the normal approximation with tie correction."""
    n = len(values)
    if n < 4:
        return np.nan, np.nan, np.nan

    signs = np.sign(values[None, :] - values[:, None])
    s = float(np.sum(np.triu(signs, k=1)))

    # Variance with correction for tied ranks.
    _, tie_counts = np.unique(values, return_counts=True)
    tie_term = np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5))
    variance = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if variance <= 0:
        return s, np.nan, np.nan

    if s > 0:
        z = (s - 1) / np.sqrt(variance)
    elif s < 0:
        z = (s + 1) / np.sqrt(variance)
    else:
        z = 0.0

    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    return s, float(tau), float(p)


def trend(years: pd.Series, values: pd.Series) -> dict:
    """Theil-Sen slope plus Mann-Kendall significance for one series."""
    mask = values.notna() & years.notna()
    y = years[mask].to_numpy(dtype=float)
    v = values[mask].to_numpy(dtype=float)

    result: dict = {"n": int(len(v))}
    if len(v) < 8:
        result.update(
            {
                "slope_per_decade": np.nan,
                "slope_lo": np.nan,
                "slope_hi": np.nan,
                "tau": np.nan,
                "p_value": np.nan,
                "significant": False,
                "first_year": int(y[0]) if len(y) else None,
                "last_year": int(y[-1]) if len(y) else None,
            }
        )
        return result

    slope, intercept, lo, hi = stats.theilslopes(v, y, alpha=0.95)
    _, tau, p = mann_kendall(v)

    result.update(
        {
            "slope_per_decade": float(slope) * 10.0,
            "slope_lo": float(lo) * 10.0,
            "slope_hi": float(hi) * 10.0,
            "intercept": float(intercept),
            "tau": tau,
            "p_value": p,
            "significant": bool(p < 0.05) if not np.isnan(p) else False,
            "first_year": int(y[0]),
            "last_year": int(y[-1]),
            "mean": float(np.mean(v)),
            "total_change": float(slope) * (y[-1] - y[0]),
        }
    )
    return result


def trend_table(annual: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Run :func:`trend` across many index columns."""
    rows = []
    for column in columns:
        if column not in annual.columns:
            continue
        row = {"index": column}
        row.update(trend(annual["year"], annual[column]))
        rows.append(row)
    return pd.DataFrame(rows)


def period_comparison(
    annual: pd.DataFrame, columns: list[str], split_year: int
) -> pd.DataFrame:
    """Compare index means before and from ``split_year``, with a Mann-Whitney test."""
    early = annual[annual["year"] < split_year]
    late = annual[annual["year"] >= split_year]

    rows = []
    for column in columns:
        if column not in annual.columns:
            continue
        a = early[column].dropna()
        b = late[column].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        rows.append(
            {
                "index": column,
                "early_n": len(a),
                "late_n": len(b),
                "early_mean": float(a.mean()),
                "late_mean": float(b.mean()),
                "delta": float(b.mean() - a.mean()),
                "p_value": float(p),
                "significant": bool(p < 0.05),
            }
        )
    return pd.DataFrame(rows)

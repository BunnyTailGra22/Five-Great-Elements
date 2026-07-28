"""Annual climate indices chosen for their bearing on plant phenology.

Index choices are adapted to a subtropical montane site. The standard ETCCDI
growing-season-length index (mean temperature above 10 C) saturates at 365 days
here and carries no signal, so a 20 C "warm season" threshold is used instead,
alongside thermal-accumulation timing, which is what actually paces bud break
and flushing at this latitude.

Every index is gated on data coverage: an index is emitted only when enough of
the contributing days are present, so a year with a three-month hole cannot
masquerade as a cool or dry year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Coverage floors. Sums (rainfall, degree-days) are biased low by any missing
# day, so they are held to a stricter standard than means.
MEAN_MIN_COVERAGE = 0.90
SUM_MIN_COVERAGE = 0.95

GDD_BASE = 10.0
GDD_THRESHOLDS = (100.0, 200.0)
WARM_SEASON_THRESHOLD = 20.0
RUN_LENGTH = 6

SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


def _gated(series: pd.Series, minimum: float, value):
    """Return ``value`` only if ``series`` clears the coverage floor."""
    if len(series) == 0:
        return np.nan
    return value if series.notna().mean() >= minimum else np.nan


def _max_run(flags: pd.Series) -> int:
    """Longest run of True in a boolean series (NaN counts as False)."""
    best = current = 0
    for flag in flags.fillna(False).to_numpy():
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def _first_run_start(flags: np.ndarray, run_length: int, offset: int = 0) -> float:
    """Index of the first run of ``run_length`` consecutive True values."""
    count = 0
    for i in range(offset, len(flags)):
        count = count + 1 if flags[i] else 0
        if count >= run_length:
            return i - run_length + 1
    return np.nan


def gdd_threshold_doy(group: pd.DataFrame, threshold: float) -> float:
    """Day of year on which cumulative GDD from Jan 1 first reaches ``threshold``."""
    first_half = group[group["doy"] <= 181]
    if first_half["Tmean"].notna().mean() < SUM_MIN_COVERAGE:
        return np.nan

    daily_gdd = (group["Tmean"] - GDD_BASE).clip(lower=0).fillna(0.0)
    cumulative = daily_gdd.cumsum()
    reached = group.loc[cumulative >= threshold, "doy"]
    return float(reached.iloc[0]) if len(reached) else np.nan


def warm_season(group: pd.DataFrame) -> tuple[float, float, float]:
    """Start DOY, end DOY and length of the season above ``WARM_SEASON_THRESHOLD``."""
    tmean = group["Tmean"]
    if tmean.notna().mean() < MEAN_MIN_COVERAGE:
        return np.nan, np.nan, np.nan

    warm = (tmean >= WARM_SEASON_THRESHOLD).fillna(False).to_numpy()
    doy = group["doy"].to_numpy()

    start_idx = _first_run_start(warm, RUN_LENGTH)
    if np.isnan(start_idx):
        return np.nan, np.nan, np.nan

    # End: first sustained cool spell after midsummer.
    midsummer = int(np.searchsorted(doy, 182))
    end_idx = _first_run_start(~warm, RUN_LENGTH, offset=midsummer)
    if np.isnan(end_idx):
        return float(doy[int(start_idx)]), np.nan, np.nan

    start_doy = float(doy[int(start_idx)])
    end_doy = float(doy[int(end_idx)])
    return start_doy, end_doy, end_doy - start_doy


def annual_indices(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute one row of phenology-relevant indices per calendar year."""
    rows: list[dict] = []

    for year, group in daily.groupby("year"):
        group = group.sort_values("doy")
        tmean, tmax, tmin = group["Tmean"], group["TxMaxAbs"], group["TxMinAbs"]
        precp, sun, wind = group["Precp"], group["SunShine"], group["WS"]
        soil = group["TxSoil0cm"]

        row: dict = {
            "year": int(year),
            "days": len(group),
            "cov_tmean": round(float(tmean.notna().mean()), 4),
            "cov_precp": round(float(precp.notna().mean()), 4),
            "cov_sunshine": round(float(sun.notna().mean()), 4),
            "cov_soil0cm": round(float(soil.notna().mean()), 4),
        }

        # --- Thermal means -------------------------------------------------
        row["tmean"] = _gated(tmean, MEAN_MIN_COVERAGE, tmean.mean())
        row["tmax_mean"] = _gated(tmax, MEAN_MIN_COVERAGE, tmax.mean())
        row["tmin_mean"] = _gated(tmin, MEAN_MIN_COVERAGE, tmin.mean())
        row["dtr"] = _gated(tmax, MEAN_MIN_COVERAGE, (tmax - tmin).mean())

        # --- Thermal accumulation and its timing ---------------------------
        row["gdd10"] = _gated(
            tmean, SUM_MIN_COVERAGE, float((tmean - GDD_BASE).clip(lower=0).sum())
        )
        for threshold in GDD_THRESHOLDS:
            row[f"gdd10_doy{int(threshold)}"] = gdd_threshold_doy(group, threshold)

        start, end, length = warm_season(group)
        row["warm_season_start_doy"] = start
        row["warm_season_end_doy"] = end
        row["warm_season_length"] = length

        # --- Cold accumulation (dormancy / chilling) -----------------------
        row["chill_days_tmin10"] = _gated(tmin, MEAN_MIN_COVERAGE, float((tmin <= 10).sum()))
        row["cold_days_tmean12"] = _gated(tmean, MEAN_MIN_COVERAGE, float((tmean <= 12).sum()))
        spring = group[group["doy"] <= 181]
        cold_spring = spring.loc[spring["TxMinAbs"] <= 10, "doy"]
        row["last_cold_day_doy"] = (
            float(cold_spring.iloc[-1])
            if len(cold_spring) and spring["TxMinAbs"].notna().mean() >= SUM_MIN_COVERAGE
            else np.nan
        )

        # --- Heat extremes -------------------------------------------------
        row["summer_days_tmax30"] = _gated(tmax, MEAN_MIN_COVERAGE, float((tmax >= 30).sum()))
        row["hot_nights_tmin25"] = _gated(tmin, MEAN_MIN_COVERAGE, float((tmin >= 25).sum()))

        # --- Precipitation -------------------------------------------------
        row["precp_total"] = _gated(precp, SUM_MIN_COVERAGE, float(precp.sum()))
        for threshold, label in ((0.1, "01"), (1.0, "1"), (10.0, "10"), (50.0, "50")):
            row[f"rain_days_{label}"] = _gated(
                precp, SUM_MIN_COVERAGE, float((precp >= threshold).sum())
            )
        row["rx1day"] = _gated(precp, MEAN_MIN_COVERAGE, float(precp.max()))
        row["rx5day"] = _gated(
            precp, MEAN_MIN_COVERAGE, float(precp.rolling(5, min_periods=5).sum().max())
        )
        wet_days = precp[precp >= 1.0]
        row["sdii"] = _gated(
            precp, SUM_MIN_COVERAGE, float(wet_days.mean()) if len(wet_days) else np.nan
        )
        row["max_dry_spell"] = _gated(precp, SUM_MIN_COVERAGE, float(_max_run(precp < 1.0)))

        wet_season = group[group["month"].between(4, 9)]["Precp"]
        dry_season = group[~group["month"].between(4, 9)]["Precp"]
        row["precp_wet_season"] = _gated(wet_season, SUM_MIN_COVERAGE, float(wet_season.sum()))
        row["precp_dry_season"] = _gated(dry_season, SUM_MIN_COVERAGE, float(dry_season.sum()))

        # --- Sunshine, wind, soil -----------------------------------------
        row["sunshine_total"] = _gated(sun, SUM_MIN_COVERAGE, float(sun.sum()))
        row["sunshine_daily_mean"] = _gated(sun, MEAN_MIN_COVERAGE, float(sun.mean()))
        row["ws_mean"] = _gated(wind, MEAN_MIN_COVERAGE, float(wind.mean()))
        row["tsoil0_mean"] = _gated(soil, MEAN_MIN_COVERAGE, float(soil.mean()))

        rows.append(row)

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def seasonal_means(daily: pd.DataFrame) -> pd.DataFrame:
    """Seasonal mean temperature and rainfall totals.

    December is assigned to the following year's winter so DJF stays contiguous.
    """
    frame = daily.copy()
    frame["season_year"] = np.where(frame["month"] == 12, frame["year"] + 1, frame["year"])
    frame["season"] = frame["month"].map(
        {m: name for name, months in SEASONS.items() for m in months}
    )

    rows = []
    for (season_year, season), group in frame.groupby(["season_year", "season"]):
        expected = {"DJF": 90, "MAM": 92, "JJA": 92, "SON": 91}[season]
        tmean, precp = group["Tmean"], group["Precp"]
        rows.append(
            {
                "year": int(season_year),
                "season": season,
                "days": len(group),
                "tmean": tmean.mean()
                if len(group) >= expected * MEAN_MIN_COVERAGE
                and tmean.notna().mean() >= MEAN_MIN_COVERAGE
                else np.nan,
                "precp_total": float(precp.sum())
                if len(group) >= expected * SUM_MIN_COVERAGE
                and precp.notna().mean() >= SUM_MIN_COVERAGE
                else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values(["year", "season"]).reset_index(drop=True)


def monthly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Monthly means and totals with a coverage column."""
    rows = []
    for (year, month), group in daily.groupby(["year", "month"]):
        days = pd.Period(f"{year}-{month:02d}").days_in_month
        tmean, precp = group["Tmean"], group["Precp"]
        rows.append(
            {
                "year": int(year),
                "month": int(month),
                "days_present": len(group),
                "days_in_month": days,
                "coverage": round(float(tmean.notna().sum() / days), 4),
                "tmean": _gated(tmean, MEAN_MIN_COVERAGE, tmean.mean()),
                "tmax_mean": _gated(group["TxMaxAbs"], MEAN_MIN_COVERAGE, group["TxMaxAbs"].mean()),
                "tmin_mean": _gated(group["TxMinAbs"], MEAN_MIN_COVERAGE, group["TxMinAbs"].mean()),
                "precp_total": _gated(precp, SUM_MIN_COVERAGE, float(precp.sum())),
                "rain_days_1": _gated(precp, SUM_MIN_COVERAGE, float((precp >= 1.0).sum())),
                "sunshine_total": _gated(
                    group["SunShine"], SUM_MIN_COVERAGE, float(group["SunShine"].sum())
                ),
                "tsoil0_mean": _gated(
                    group["TxSoil0cm"], MEAN_MIN_COVERAGE, float(group["TxSoil0cm"].mean())
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["year", "month"]).reset_index(drop=True)

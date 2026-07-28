import numpy as np
import pandas as pd

from fge_climate.indices import annual_indices, monthly_summary
from fge_climate.qc import add_derived


def synthetic_year(year=2001, tmean=20.0, precp=5.0, missing_days=0):
    """A full year with a flat seasonal cycle, optionally punched with gaps."""
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "Tx": np.full(len(dates), tmean),
            "TxMaxAbs": np.full(len(dates), tmean + 5),
            "TxMinAbs": np.full(len(dates), tmean - 5),
            "Precp": np.full(len(dates), precp),
            "SunShine": np.full(len(dates), 5.0),
            "WS": np.full(len(dates), 1.5),
            "TxSoil0cm": np.full(len(dates), tmean),
        }
    )
    if missing_days:
        frame.loc[: missing_days - 1, ["Tx", "TxMaxAbs", "TxMinAbs", "Precp"]] = np.nan
    return add_derived(frame)


def test_gdd_and_rainfall_totals():
    annual = annual_indices(synthetic_year(tmean=20.0, precp=5.0))
    row = annual.iloc[0]

    # 365 days at 10 degrees above base.
    assert row["gdd10"] == 3650.0
    assert row["precp_total"] == 365 * 5.0
    assert row["rain_days_1"] == 365
    assert row["max_dry_spell"] == 0


def test_sums_are_suppressed_when_coverage_fails():
    # 30 missing days of 365 is 91.8% coverage: below the 95% sum gate but above
    # the 90% mean gate, so totals drop out while means survive.
    annual = annual_indices(synthetic_year(missing_days=30))
    row = annual.iloc[0]

    assert np.isnan(row["precp_total"])
    assert np.isnan(row["gdd10"])
    assert not np.isnan(row["tmean"])


def test_means_are_suppressed_when_coverage_fails_badly():
    annual = annual_indices(synthetic_year(missing_days=200))
    row = annual.iloc[0]

    assert np.isnan(row["tmean"])
    assert np.isnan(row["precp_total"])


def test_dry_spell_counts_longest_run():
    frame = synthetic_year(precp=0.0)
    # A single wet day splits the year into two dry runs.
    frame.loc[frame["doy"] == 100, "Precp"] = 20.0
    annual = annual_indices(frame)

    assert annual.iloc[0]["max_dry_spell"] == 265  # days 101-365


def test_warm_season_detected_only_above_threshold():
    warm = annual_indices(synthetic_year(tmean=25.0)).iloc[0]
    cold = annual_indices(synthetic_year(tmean=15.0)).iloc[0]

    assert warm["warm_season_start_doy"] == 1
    assert np.isnan(cold["warm_season_start_doy"])


def test_gdd_threshold_timing():
    # 11 C mean gives 1 degree-day per day, so 100 C.d lands on day 100.
    annual = annual_indices(synthetic_year(tmean=11.0))
    assert annual.iloc[0]["gdd10_doy100"] == 100


def test_monthly_summary_reports_coverage():
    monthly = monthly_summary(synthetic_year())
    assert len(monthly) == 12
    assert monthly["coverage"].max() == 1.0
    assert monthly.loc[0, "precp_total"] == 31 * 5.0

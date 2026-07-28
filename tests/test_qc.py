import numpy as np
import pandas as pd

from fge_climate.qc import add_derived, clean_daily


def make_frame(**columns):
    n = max(len(v) for v in columns.values())
    base = {"date": pd.date_range("2020-01-01", periods=n, freq="D")}
    base.update(columns)
    return pd.DataFrame(base)


def test_sentinels_are_masked_not_averaged():
    frame = make_frame(Tx=[20.0, -99.5, 22.0], Precp=[0.0, -9999.5, 5.0], RH=[80.0, -9995.0, 90.0])
    cleaned, report = clean_daily(frame)

    assert cleaned["Tx"].isna().sum() == 1
    assert cleaned["Tx"].mean() == 21.0  # not dragged down by -99.5
    assert cleaned["Precp"].isna().sum() == 1
    assert cleaned["RH"].isna().sum() == 1
    assert report.sentinel_masked["Tx"] == 1
    assert report.sentinel_masked["Precp"] == 1


def test_impossible_values_are_masked():
    frame = make_frame(RH=[80.0, 124.0, 90.0], Precp=[0.0, -3.0, 5.0])
    cleaned, report = clean_daily(frame)

    assert cleaned["RH"].isna().sum() == 1
    assert report.range_masked["RH"] == 1
    # A negative rainfall that is not a sentinel is still impossible.
    assert cleaned["Precp"].isna().sum() == 1


def test_plausible_values_survive():
    frame = make_frame(Tx=[12.0, 20.0, 31.0], Precp=[0.0, 120.0, 3.5], RH=[55.0, 100.0, 88.0])
    cleaned, report = clean_daily(frame)

    assert cleaned["Tx"].notna().all()
    assert cleaned["Precp"].notna().all()
    assert report.sentinel_masked == {}
    assert report.range_masked == {}


def test_tmean_prefers_station_mean_over_midpoint():
    frame = make_frame(
        Tx=[20.0, np.nan], TxMaxAbs=[25.0, 26.0], TxMinAbs=[15.0, 16.0]
    )
    cleaned, _ = clean_daily(frame)
    derived = add_derived(cleaned)

    assert derived.loc[0, "Tmean"] == 20.0
    assert derived.loc[0, "Tmean_source"] == "station_mean"
    # Falls back to the midpoint only where the station mean is missing.
    assert derived.loc[1, "Tmean"] == 21.0
    assert derived.loc[1, "Tmean_source"] == "minmax_midpoint"

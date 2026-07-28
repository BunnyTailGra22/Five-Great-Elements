import numpy as np
import pandas as pd

from fge_climate.homogeneity import (
    changepoint,
    difference_series,
    pettitt,
    propagate_derived_breaks,
    propagate_dtr_breaks,
)
from fge_climate.trends import mann_kendall, trend


def test_mann_kendall_detects_monotonic_increase():
    s, tau, p = mann_kendall(np.arange(30, dtype=float))
    assert tau == 1.0
    assert p < 0.01


def test_mann_kendall_flat_series_is_not_significant():
    rng = np.random.default_rng(0)
    _, _, p = mann_kendall(rng.normal(size=40))
    assert p > 0.05


def test_theil_sen_recovers_a_known_slope():
    years = pd.Series(range(1990, 2020))
    values = pd.Series(0.03 * (years - 1990) + 10.0)
    result = trend(years, values)

    assert np.isclose(result["slope_per_decade"], 0.3, atol=1e-6)
    assert result["p_value"] < 0.01


def test_theil_sen_is_robust_to_an_outlier():
    years = pd.Series(range(1990, 2020))
    values = pd.Series(0.03 * (years - 1990) + 10.0)
    values.iloc[15] = 500.0  # one wild value
    result = trend(years, values)

    assert np.isclose(result["slope_per_decade"], 0.3, atol=0.05)


def test_short_series_returns_no_slope():
    result = trend(pd.Series(range(2015, 2020)), pd.Series([1.0, 2, 3, 4, 5]))
    assert np.isnan(result["slope_per_decade"])


def test_pettitt_locates_a_step():
    values = np.concatenate([np.full(15, 10.0), np.full(15, 14.0)])
    index, _, p = pettitt(values)

    assert index == 14  # last element of the first segment
    assert p < 0.05


def test_changepoint_reports_break_year_and_shift():
    years = pd.Series(range(1990, 2020))
    values = pd.Series(np.concatenate([np.full(15, 10.0), np.full(15, 14.0)]))
    result = changepoint(years, values)

    assert result["break_year"] == 2005
    assert result["significant"]
    assert np.isclose(result["shift"], 4.0)


def test_difference_series_cancels_a_shared_climate_signal():
    years = list(range(1990, 2020))
    warming = np.linspace(0, 3, 30)
    target = pd.DataFrame({"year": years, "tmean": 20 + warming})
    reference = pd.DataFrame({"year": years, "tmean": 18 + warming})

    diff = difference_series(target, reference, "tmean")
    # Shared warming cancels; only the constant offset remains.
    assert np.allclose(diff["difference"], 2.0)
    assert not changepoint(diff["year"], diff["difference"])["significant"]


def test_difference_series_exposes_a_station_only_step():
    years = list(range(1990, 2020))
    warming = np.linspace(0, 3, 30)
    step = np.where(np.array(years) >= 2005, -1.5, 0.0)
    target = pd.DataFrame({"year": years, "tmean": 20 + warming + step})
    reference = pd.DataFrame({"year": years, "tmean": 18 + warming})

    diff = difference_series(target, reference, "tmean")
    result = changepoint(diff["year"], diff["difference"])

    assert result["break_year"] == 2005
    assert result["significant"]


def test_dtr_break_is_attributed_to_the_larger_component_shift():
    breaks = pd.DataFrame(
        [
            {"index": "tmax_mean", "test": "vs reference", "break_year": 2002, "shift": -1.33},
            {"index": "tmin_mean", "test": "vs reference", "break_year": 2003, "shift": -0.19},
        ]
    )
    inferred = propagate_dtr_breaks(breaks, {"dtr": [2002]})

    assert inferred == {"tmax_mean": [2002]}


def test_no_dtr_break_means_no_attribution():
    breaks = pd.DataFrame(
        [{"index": "tmax_mean", "test": "vs reference", "break_year": 2002, "shift": -1.33}]
    )
    assert propagate_dtr_breaks(breaks, {}) == {}


def test_threshold_counts_inherit_their_base_variable_break():
    inherited = propagate_derived_breaks({"tmax_mean": [2002]})

    assert inherited["summer_days_tmax30"] == [2002]
    # Indices built on other variables are untouched.
    assert "chill_days_tmin10" not in inherited

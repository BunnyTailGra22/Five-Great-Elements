import numpy as np
import pandas as pd
import pytest

from fge_climate.distributions import (
    PRECIP_CLASS_LABELS,
    WET_DAY_MM,
    _box,
    _frequency,
    _heatmap,
    build_payload,
)
from fge_climate.qc import add_derived
from tests.test_webpage import STATION, make_config


def make_daily(start="2018-01-01", end="2025-12-31", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    doy = dates.dayofyear.to_numpy()
    season = np.sin((doy - 100) / 365 * 2 * np.pi)

    # Roughly half the days dry, the rest exponentially distributed.
    wet = rng.random(len(dates)) > 0.5
    rain = np.where(wet, rng.exponential(12.0, len(dates)).round(1), 0.0)

    frame = pd.DataFrame(
        {
            "date": dates,
            "Tx": 20 + 5 * season,
            "TxMaxAbs": 25 + 5 * season,
            "TxMinAbs": 15 + 5 * season,
            "Precp": rain,
            "SunShine": np.clip(6 + 3 * season + rng.normal(0, 1, len(dates)), 0, 12).round(1),
            "WS": np.clip(1.3 + rng.normal(0, 0.3, len(dates)), 0.1, None).round(1),
            "TxSoil0cm": 21 + 6 * season,
        }
    )
    return add_derived(frame)


def test_box_matches_known_quartiles():
    stats = _box(np.arange(1, 101, dtype=float))  # 1..100

    assert stats["n"] == 100
    assert stats["median"] == pytest.approx(50.5)
    assert stats["q1"] == pytest.approx(25.75)
    assert stats["q3"] == pytest.approx(75.25)
    assert stats["outlier_count"] == 0


def test_box_separates_outliers_from_whiskers():
    values = np.concatenate([np.arange(10, 21, dtype=float), [500.0]])
    stats = _box(values)

    assert stats["outlier_count"] == 1
    assert stats["outliers"] == [500.0]
    # The whisker stops at the last value inside the fence, not at the outlier.
    assert stats["hi_whisker"] == 20.0


def test_rainfall_uses_classes_and_counts_dry_days():
    values = np.array([0.0, 0.0, 0.05, 0.5, 3.0, 7.0, 15.0, 30.0, 70.0, 150.0])
    freq = _frequency(values, {"key": "Precp", "unit": "mm"})

    assert freq["mode"] == "classes"
    assert [b["label"] for b in freq["bins"]] == PRECIP_CLASS_LABELS
    assert freq["bins"][0]["count"] == 3       # 0, 0, 0.05 are all dry
    assert freq["bins"][-1]["count"] == 1      # 150 mm
    assert sum(b["count"] for b in freq["bins"]) == len(values)


def test_continuous_measures_use_equal_width_bins():
    values = np.array([0.2, 1.5, 3.4, 5.9, 7.0, 9.9])
    freq = _frequency(values, {"key": "SunShine", "unit": "hour"})

    assert freq["mode"] == "bins"
    assert freq["width"] == 1.0
    widths = {round(b["hi"] - b["lo"], 6) for b in freq["bins"]}
    assert widths == {1.0}
    assert sum(b["count"] for b in freq["bins"]) == len(values)


def test_rainfall_boxes_describe_wet_days_only():
    payload = build_payload(make_config(2026), STATION, make_daily())
    precp = next(m for m in payload["measures"] if m["key"] == "Precp")

    assert precp["logScale"] is True
    assert "wet days" in precp["boxBasis"]
    for month in precp["monthly"]:
        # Every value behind the box must clear the wet-day threshold.
        assert month["min"] >= WET_DAY_MM
        assert 0 < month["dry_pct"] < 100
        # Wet days are a subset of the days observed that month.
        assert month["n"] < month["days_observed"]


def test_other_measures_keep_every_observed_day():
    payload = build_payload(make_config(2026), STATION, make_daily())
    wind = next(m for m in payload["measures"] if m["key"] == "WS")

    assert wind.get("logScale") is False
    assert wind["monthly"][0]["dry_pct"] is None
    assert wind["monthly"][0]["n"] == wind["monthly"][0]["days_observed"]


def test_pooling_uses_complete_years_only():
    # 2026 is the running year, so it must not enter the pooled statistics.
    payload = build_payload(make_config(2026), STATION, make_daily(end="2026-06-30"))

    assert payload["window"] == {"start": 2018, "end": 2025}
    assert payload["days"] == 2922        # 2018-2025 inclusive, two leap years
    assert payload["heatmapEnd"] == 2026


def test_heatmap_blanks_months_below_coverage():
    daily = make_daily()
    # Wipe most of March 2020.
    mask = (daily["date"] >= "2020-03-01") & (daily["date"] <= "2020-03-25")
    daily.loc[mask, "WS"] = np.nan

    heat = _heatmap(daily, {"key": "WS", "agg": "mean"}, 2025)
    row = next(r for r in heat["rows"] if r["year"] == 2020)
    march = next(c for c in row["cells"] if c["month"] == 3)

    assert march["value"] is None
    assert march["coverage"] < 0.9
    # A well-covered month is still reported.
    assert next(c for c in row["cells"] if c["month"] == 4)["value"] is not None


def test_heatmap_edges_span_the_observed_range():
    payload = build_payload(make_config(2026), STATION, make_daily())
    soil = next(m for m in payload["measures"] if m["key"] == "TxSoil0cm")
    heat = soil["heatmap"]

    assert heat["edges"][0] == pytest.approx(heat["min"], abs=0.01)
    assert heat["edges"][-1] == pytest.approx(heat["max"], abs=0.01)
    assert heat["agg"] == "mean"


def test_findings_are_computed_for_every_measure():
    payload = build_payload(make_config(2026), STATION, make_daily())

    for measure in payload["measures"]:
        findings = measure["findings"]
        assert 1 <= len(findings) <= 4
        for item in findings:
            assert item["lead"] and item["text"]
            assert "{" not in item["text"]
        # Every measure closes on its record day.
        assert findings[-1]["lead"] == "Record day"


def test_cold_month_list_does_not_read_as_a_range():
    from fge_climate.distributions import _month_list

    # Winter wraps the year end, so Jan-Dec would be nonsense.
    assert _month_list([1, 2, 3, 12]) == "Jan, Feb, Mar and Dec"
    assert _month_list([7]) == "Jul"

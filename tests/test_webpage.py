import json

import numpy as np
import pandas as pd
import pytest

from fge_climate.config import Config, Station, StudyArea
from fge_climate.qc import add_derived
from fge_climate.webpage import PLACEHOLDER, build_payload, write_page

STATION = Station(
    id="82A750", name_zh="茶改北部分場", name_en="TRES Northern Branch",
    station_type="農業站", elevation_m=401, lat=24.955778, lon=121.631222,
    established="1987-05-01", role="primary",
)
AREA = StudyArea(
    name_zh="二格山", name_en="Erge Mountain",
    summit_lat=24.9436, summit_lon=121.6206, summit_elevation_m=678,
)


def make_config(end_year):
    return Config(stations=[STATION], study_area=AREA, start_year=2015, end_year=end_year)


def make_daily(start="2015-01-01", end="2020-12-31", soil_from=2018):
    dates = pd.date_range(start, end, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "Tx": 20.0, "TxMaxAbs": 25.0, "TxMinAbs": 15.0,
            "Precp": 5.0, "SunShine": 5.0, "WS": 1.5,
            "TxSoil0cm": np.where(dates.year >= soil_from, 20.0, np.nan),
        }
    )
    return add_derived(frame)


def test_payload_covers_the_four_measures():
    payload = build_payload(make_config(2021), STATION, make_daily())
    keys = [m["key"] for m in payload["measures"]]

    assert keys == ["TxSoil0cm", "Precp", "SunShine", "WS"]
    assert {m["slot"] for m in payload["measures"]} == {1, 2, 3, 4}
    for m in payload["measures"]:
        assert len(m["climatology"]) == 12


def test_common_window_starts_where_the_last_instrument_does():
    # Soil temperature begins in 2018, so the all-four window cannot start earlier.
    payload = build_payload(make_config(2021), STATION, make_daily(soil_from=2018))
    assert payload["window"]["start"] == 2018
    assert payload["window"]["end"] == 2020  # end_year - 1: the last complete year


def test_years_below_the_gate_carry_no_value():
    daily = make_daily()
    # Knock out half of 2019's rainfall, dropping it below the 95% sum gate.
    mask = (daily["date"] >= "2019-01-01") & (daily["date"] <= "2019-06-30")
    daily.loc[mask, "Precp"] = np.nan

    payload = build_payload(make_config(2021), STATION, daily)
    precp = next(m for m in payload["measures"] if m["key"] == "Precp")
    row = next(r for r in precp["annual"] if r["year"] == 2019)

    assert row["value"] is None
    assert row["coverage"] < 0.95
    # The gap must also break the contiguous block, so the chart draws a gap.
    assert not any(start <= 2019 <= end for start, end in precp["blocks"])


def test_sums_and_means_use_the_right_aggregation():
    payload = build_payload(make_config(2021), STATION, make_daily())
    precp = next(m for m in payload["measures"] if m["key"] == "Precp")
    wind = next(m for m in payload["measures"] if m["key"] == "WS")

    assert precp["agg"] == "sum"
    assert precp["window_mean"] == pytest.approx(365 * 5.0, rel=0.01)
    assert wind["agg"] == "mean"
    assert wind["window_mean"] == pytest.approx(1.5, rel=0.01)


def test_page_embeds_valid_json_and_closes_no_script_tag(tmp_path):
    payload = build_payload(make_config(2021), STATION, make_daily())
    template = tmp_path / "t.html"
    template.write_text(f'<script type="application/json">{PLACEHOLDER}</script>', encoding="utf-8")

    out = write_page(payload, tmp_path / "index.html", template)
    html = out.read_text(encoding="utf-8")

    assert PLACEHOLDER not in html
    # A literal </script> inside the blob would terminate the element early.
    blob = html.split(">", 1)[1].rsplit("</script>", 1)[0]
    assert "</" not in blob
    assert json.loads(blob.replace("<\\/", "</"))["station"]["id"] == "82A750"


def test_missing_placeholder_is_an_error(tmp_path):
    template = tmp_path / "t.html"
    template.write_text("<p>no placeholder</p>", encoding="utf-8")

    with pytest.raises(ValueError, match="placeholder"):
        write_page({}, tmp_path / "index.html", template)

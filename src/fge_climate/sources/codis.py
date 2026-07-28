"""Direct CODiS client, used to top up the most recent days.

The mirror lags official CODiS by up to a day or two. When a run needs the
freshest possible tail, this client pulls daily records straight from
``https://codis.cwa.gov.tw/api/station`` (the endpoint the StationData page
itself calls) one calendar month at a time.

This is an opt-in path (``sync --with-codis``); the mirror remains the source
of truth for the historical record.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any

import pandas as pd
import requests

API_URL = "https://codis.cwa.gov.tw/api/station"
STATION_LIST_URL = "https://codis.cwa.gov.tw/api/station_list"

REQUEST_HEADERS = {
    "accept": "application/json, text/javascript, */*; q=0.01",
    "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
    "x-requested-with": "XMLHttpRequest",
    "referer": "https://codis.cwa.gov.tw/StationData",
    "user-agent": "five-great-elements-climate-sync/1.0",
}

# Flattened CODiS field -> canonical project column.
DAILY_FIELD_MAP = {
    "AirTemperature.Mean": "Tx",
    "AirTemperature.Maximum": "TxMaxAbs",
    "AirTemperature.Minimum": "TxMinAbs",
    "RelativeHumidity.Mean": "RH",
    "WindSpeed.Mean": "WS",
    "Precipitation.Accumulation": "Precp",
    "SunshineDuration.Total": "SunShine",
    "GlobalSolarRadiation.Accumulation": "GloblRad",
    "SoilTemperatureAt0cm.Mean": "TxSoil0cm",
    "SoilTemperatureAt5cm.Mean": "TxSoil5cm",
    "SoilTemperatureAt10cm.Mean": "TxSoil10cm",
    "SoilTemperatureAt20cm.Mean": "TxSoil20cm",
}


def _flatten(prefix: str, value: dict, out: dict) -> None:
    for key, item in value.items():
        name = f"{prefix}.{key}"
        if isinstance(item, dict):
            _flatten(name, item, out)
        else:
            out[name] = item


def fetch_daily_month(
    station_id: str,
    station_type: str,
    month_start: date,
    month_end: date,
    session: requests.Session | None = None,
    timeout: int = 60,
    max_retries: int = 4,
) -> pd.DataFrame:
    """Fetch one month of daily observations as a canonical frame."""
    session = session or requests.Session()
    session.headers.update(REQUEST_HEADERS)

    payload = {
        "date": f"{month_start.isoformat()}T00:00:00.000+08:00",
        "type": "report_month",  # daily rows for one month
        "stn_ID": station_id,
        "stn_type": station_type,
        "start": f"{month_start.isoformat()}T00:00:00",
        "end": f"{month_end.isoformat()}T00:00:00",
    }

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = session.post(API_URL, data=payload, timeout=timeout)
            response.raise_for_status()
            return _to_frame(response.json())
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def _to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    data = payload.get("data") or []
    if not data:
        return pd.DataFrame()
    entries = data[0].get("dts") or []
    if not entries:
        return pd.DataFrame()

    rows: list[dict] = []
    for entry in entries:
        if "DataDate" not in entry:
            continue
        flat: dict[str, Any] = {}
        for key, value in entry.items():
            if key == "DataDate":
                continue
            if isinstance(value, dict):
                _flatten(key, value, flat)
            else:
                flat[key] = value

        row: dict[str, Any] = {"date": entry["DataDate"]}
        for source_field, canonical in DAILY_FIELD_MAP.items():
            if source_field in flat:
                row[canonical] = flat[source_field]
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame[frame["date"].notna()]
    frame = frame.drop_duplicates(subset=["date"], keep="first")
    return frame.sort_values("date").reset_index(drop=True)


def fetch_daily_range(
    station_id: str,
    station_type: str,
    start: date,
    end: date,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Fetch daily observations across a date range, month by month."""
    frames: list[pd.DataFrame] = []
    cursor = start.replace(day=1)
    session = requests.Session()

    while cursor <= end:
        month_end = _last_day_of_month(cursor)
        if verbose:
            print(f"  CODiS {station_id} {cursor:%Y-%m}")
        frame = fetch_daily_month(
            station_id, station_type, cursor, month_end, session=session, **kwargs
        )
        if not frame.empty:
            frames.append(frame)
        cursor = (month_end + pd.Timedelta(days=1)).date()

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    mask = (combined["date"] >= pd.Timestamp(start)) & (combined["date"] <= pd.Timestamp(end))
    return combined[mask].reset_index(drop=True)


def _last_day_of_month(day: date) -> date:
    period = pd.Period(day, freq="M")
    return period.end_time.date()

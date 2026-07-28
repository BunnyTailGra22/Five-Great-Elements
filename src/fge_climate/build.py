"""Assemble raw yearly station files into one tidy, QC'd daily table."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .qc import QCReport, add_derived, clean_daily

# Canonical observation columns carried through to the processed dataset.
KEEP_COLUMNS = [
    "Tx",
    "TxMaxAbs",
    "TxMinAbs",
    "RH",
    "WS",
    "Precp",
    "SunShine",
    "GloblRad",
    "TxSoil0cm",
    "TxSoil5cm",
    "TxSoil10cm",
    "TxSoil20cm",
]

YEAR_PATTERN = re.compile(r"_(\d{4})_daily\.csv$")


def read_raw_year(path: Path) -> pd.DataFrame:
    """Read one raw daily file, keeping only the canonical columns."""
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig", low_memory=False)
    if frame.empty:
        return pd.DataFrame()

    # The first column holds the date and is unnamed in the upstream files.
    frame = frame.rename(columns={frame.columns[0]: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame[frame["date"].notna()]
    if frame.empty:
        return pd.DataFrame()

    columns = ["date"] + [c for c in KEEP_COLUMNS if c in frame.columns]
    out = frame[columns].copy()
    for column in KEEP_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out[["date"] + KEEP_COLUMNS]


def build_daily(raw_dir: Path, start_year: int, end_year: int) -> tuple[pd.DataFrame, QCReport]:
    """Combine yearly raw files into a gap-explicit, cleaned daily table."""
    frames: list[pd.DataFrame] = []
    for path in sorted(raw_dir.glob("*_daily.csv")):
        match = YEAR_PATTERN.search(path.name)
        if not match:
            continue
        year = int(match.group(1))
        if not start_year <= year <= end_year:
            continue
        frame = read_raw_year(path)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No usable daily files in {raw_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date")

    # Make absent days explicit rows so completeness is measurable rather than
    # inferred from row counts.
    full_index = pd.date_range(
        f"{start_year}-01-01", combined["date"].max(), freq="D", name="date"
    )
    combined = combined.set_index("date").reindex(full_index).reset_index()

    cleaned, report = clean_daily(combined)
    return add_derived(cleaned), report


def merge_codis_tail(daily: pd.DataFrame, tail: pd.DataFrame) -> pd.DataFrame:
    """Overlay fresher CODiS rows onto the mirror-derived daily table."""
    if tail.empty:
        return daily

    tail = tail.copy()
    tail["date"] = pd.to_datetime(tail["date"]).dt.normalize()
    for column in KEEP_COLUMNS:
        if column not in tail.columns:
            tail[column] = pd.NA
    tail = tail[["date"] + KEEP_COLUMNS]

    cleaned_tail, _ = clean_daily(tail)

    base = daily.drop(columns=["Tmean", "Tmean_source", "year", "month", "doy"], errors="ignore")
    # CODiS wins on overlapping dates; it is the authoritative upstream.
    combined = pd.concat([base, cleaned_tail], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    return add_derived(combined.reset_index(drop=True))


def coverage_table(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-year fraction of days with a valid value, by variable."""
    rows = []
    for year, group in daily.groupby("year"):
        days_in_year = len(group)
        row = {"year": int(year), "days": days_in_year}
        for column in ["Tmean", "TxMaxAbs", "TxMinAbs", "Precp", "SunShine", "WS", "RH", "TxSoil0cm"]:
            row[column] = round(float(group[column].notna().mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(daily: pd.DataFrame, report: QCReport, out_dir: Path, station_id: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_path = out_dir / f"{station_id}_daily.csv"
    daily.to_csv(daily_path, index=False, float_format="%.3f")

    coverage = coverage_table(daily)
    coverage_path = out_dir / f"{station_id}_coverage.csv"
    coverage.to_csv(coverage_path, index=False)

    qc_path = out_dir / f"{station_id}_qc.json"
    qc_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

    return {"daily": daily_path, "coverage": coverage_path, "qc": qc_path}

"""Quality control for CODiS daily records.

Two classes of bad value show up in the raw station files:

1. Missing-data sentinels encoded as large negatives. Station 82A750 uses
   ``-99.5`` for most fields, ``-9999.5`` for precipitation and ``-9995`` for
   relative humidity. Read naively these are numbers, and they silently wreck
   any mean or trend, so they are masked before anything else happens.
2. Physically impossible values that survive the sentinel filter, e.g.
   relative humidity of 124%.

Everything masked here is counted, so the QC report says exactly how much of
the record was dropped and why.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Any value at or below this is a missing-data sentinel, never a real reading.
# The station sits at 401 m in subtropical Taiwan; -90 C is unreachable.
SENTINEL_THRESHOLD = -90.0

# Physically plausible ranges for this station. Values outside are masked.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "Tx": (-10.0, 45.0),
    "TxMaxAbs": (-10.0, 45.0),
    "TxMinAbs": (-15.0, 40.0),
    "RH": (0.0, 100.0),
    "WS": (0.0, 60.0),
    "Precp": (0.0, 1500.0),
    "SunShine": (0.0, 15.0),
    "GloblRad": (0.0, 40.0),
    "TxSoil0cm": (-10.0, 60.0),
    "TxSoil5cm": (-10.0, 60.0),
    "TxSoil10cm": (-10.0, 60.0),
    "TxSoil20cm": (-10.0, 60.0),
}


@dataclass
class QCReport:
    sentinel_masked: dict[str, int]
    range_masked: dict[str, int]
    non_numeric: dict[str, int]
    valid_counts: dict[str, int]
    total_rows: int

    def to_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "sentinel_masked": self.sentinel_masked,
            "range_masked": self.range_masked,
            "non_numeric": self.non_numeric,
            "valid_counts": self.valid_counts,
        }


def clean_daily(frame: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, QCReport]:
    """Coerce observation columns to numeric and mask sentinels/impossible values."""
    out = frame.copy()
    columns = columns or [c for c in VALID_RANGES if c in out.columns]

    sentinel_masked: dict[str, int] = {}
    range_masked: dict[str, int] = {}
    non_numeric: dict[str, int] = {}
    valid_counts: dict[str, int] = {}

    for column in columns:
        if column not in out.columns:
            continue

        original = out[column]
        numeric = pd.to_numeric(original, errors="coerce")

        # Text that is not a recognised number (CWA trace/instrument codes).
        non_numeric[column] = int((numeric.isna() & original.notna() & (original != "")).sum())

        sentinel = numeric <= SENTINEL_THRESHOLD
        sentinel_masked[column] = int(sentinel.sum())
        numeric = numeric.mask(sentinel)

        low, high = VALID_RANGES.get(column, (-np.inf, np.inf))
        out_of_range = numeric.notna() & ((numeric < low) | (numeric > high))
        range_masked[column] = int(out_of_range.sum())
        numeric = numeric.mask(out_of_range)

        out[column] = numeric
        valid_counts[column] = int(numeric.notna().sum())

    report = QCReport(
        sentinel_masked={k: v for k, v in sentinel_masked.items() if v},
        range_masked={k: v for k, v in range_masked.items() if v},
        non_numeric={k: v for k, v in non_numeric.items() if v},
        valid_counts=valid_counts,
        total_rows=len(out),
    )
    return out, report


def add_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Add fields the indices depend on.

    ``Tmean`` prefers the station's own mean temperature (``Tx``, computed by
    CWA from the full sub-daily series) and only falls back to the midpoint of
    max and min when the mean is absent.
    """
    out = frame.copy()

    midpoint = (out["TxMaxAbs"] + out["TxMinAbs"]) / 2.0
    out["Tmean"] = out["Tx"].where(out["Tx"].notna(), midpoint)
    out["Tmean_source"] = np.where(
        out["Tx"].notna(), "station_mean", np.where(midpoint.notna(), "minmax_midpoint", "missing")
    )

    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["doy"] = out["date"].dt.dayofyear
    return out

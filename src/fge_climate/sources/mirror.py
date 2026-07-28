"""Bulk history from the Raingel/historical_weather CODiS mirror.

The mirror republishes official CWA CODiS station files as
``{station}/{station}_{year}_daily.csv``. It is the primary source here because
it already carries the full multi-decade rebuild and can be fetched without
hammering the CODiS service.

Sync is incremental: each year's ETag is recorded in a manifest, so unchanged
years come back as HTTP 304 and cost nothing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

RAW_BASE = "https://raw.githubusercontent.com/Raingel/historical_weather/main/data"
USER_AGENT = "five-great-elements-climate-sync/1.0"

MANIFEST_NAME = "_manifest.json"


@dataclass
class SyncResult:
    updated: list[int] = field(default_factory=list)
    unchanged: list[int] = field(default_factory=list)
    missing: list[int] = field(default_factory=list)
    failed: dict[int, str] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"updated={len(self.updated)} unchanged={len(self.unchanged)} "
            f"missing={len(self.missing)} failed={len(self.failed)}"
        )


def _manifest_path(dest_dir: Path) -> Path:
    return dest_dir / MANIFEST_NAME


def load_manifest(dest_dir: Path) -> dict:
    path = _manifest_path(dest_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_manifest(dest_dir: Path, manifest: dict) -> None:
    _manifest_path(dest_dir).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def year_url(station_id: str, year: int, granularity: str = "daily") -> str:
    suffix = "" if granularity == "hourly" else f"_{granularity}"
    return f"{RAW_BASE}/{station_id}/{station_id}_{year}{suffix}.csv"


def sync_station(
    station_id: str,
    years: list[int],
    dest_dir: Path,
    granularity: str = "daily",
    timeout: int = 60,
    max_retries: int = 4,
    session: requests.Session | None = None,
    verbose: bool = True,
) -> SyncResult:
    """Download each year's file into ``dest_dir``, skipping unchanged years."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(dest_dir)
    entries = manifest.setdefault(granularity, {})

    session = session or requests.Session()
    session.headers.update({"user-agent": USER_AGENT})
    result = SyncResult()

    for year in years:
        url = year_url(station_id, year, granularity)
        target = dest_dir / Path(url).name
        prior = entries.get(str(year), {})

        headers = {}
        # Only trust a stored ETag if the file it described is still on disk.
        if prior.get("etag") and target.exists():
            headers["If-None-Match"] = prior["etag"]

        try:
            response = _get_with_retries(
                session, url, headers=headers, timeout=timeout, max_retries=max_retries
            )
        except requests.RequestException as exc:
            result.failed[year] = str(exc)
            if verbose:
                print(f"  [FAIL] {year}: {exc}")
            continue

        if response.status_code == 304:
            result.unchanged.append(year)
            continue

        if response.status_code == 404:
            result.missing.append(year)
            if verbose:
                print(f"  [MISS] {year}: not published upstream")
            continue

        if response.status_code != 200:
            result.failed[year] = f"HTTP {response.status_code}"
            if verbose:
                print(f"  [FAIL] {year}: HTTP {response.status_code}")
            continue

        target.write_bytes(response.content)
        entries[str(year)] = {
            "etag": response.headers.get("ETag"),
            "bytes": len(response.content),
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "url": url,
        }
        result.updated.append(year)
        if verbose:
            print(f"  [ OK ] {year}: {len(response.content):,} bytes")

    save_manifest(dest_dir, manifest)
    return result


def _get_with_retries(
    session: requests.Session,
    url: str,
    headers: dict,
    timeout: int,
    max_retries: int,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, headers=headers, timeout=timeout)
            # Retry transient server-side failures; pass everything else through.
            if response.status_code in (429, 500, 502, 503, 504):
                raise requests.RequestException(f"HTTP {response.status_code}")
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(2**attempt)
    assert last_error is not None
    raise last_error

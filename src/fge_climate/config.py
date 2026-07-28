"""Project paths and station configuration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "stations.yml"

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = REPO_ROOT / "reports"


@dataclass(frozen=True)
class Station:
    id: str
    name_zh: str
    name_en: str
    station_type: str
    elevation_m: float
    lat: float
    lon: float
    established: str
    role: str

    @property
    def raw_dir(self) -> Path:
        return RAW_DIR / self.id


@dataclass(frozen=True)
class StudyArea:
    name_zh: str
    name_en: str
    summit_lat: float
    summit_lon: float
    summit_elevation_m: float


@dataclass(frozen=True)
class Config:
    stations: list[Station]
    study_area: StudyArea
    start_year: int
    end_year: int

    @property
    def primary(self) -> Station:
        for station in self.stations:
            if station.role == "primary":
                return station
        return self.stations[0]

    @property
    def references(self) -> list[Station]:
        return [s for s in self.stations if s.role == "reference"]

    @property
    def years(self) -> list[int]:
        return list(range(self.start_year, self.end_year + 1))


def load_config(path: Path | None = None) -> Config:
    raw = yaml.safe_load((path or CONFIG_PATH).read_text(encoding="utf-8"))

    stations = [
        Station(
            id=str(s["id"]),
            name_zh=s["name_zh"],
            name_en=s["name_en"],
            station_type=s["station_type"],
            elevation_m=float(s["elevation_m"]),
            lat=float(s["lat"]),
            lon=float(s["lon"]),
            established=str(s["established"]),
            role=s.get("role", "secondary"),
        )
        for s in raw["stations"]
    ]

    area = raw["study_area"]
    ingest = raw.get("ingest", {})
    end_year = ingest.get("end_year") or date.today().year

    return Config(
        stations=stations,
        study_area=StudyArea(
            name_zh=area["name_zh"],
            name_en=area["name_en"],
            summit_lat=float(area["summit_lat"]),
            summit_lon=float(area["summit_lon"]),
            summit_elevation_m=float(area["summit_elevation_m"]),
        ),
        start_year=int(ingest.get("start_year", 1990)),
        end_year=int(end_year),
    )

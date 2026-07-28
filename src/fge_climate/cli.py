"""Command line entry point for the Erge Mountain climate dataset."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

import pandas as pd

from . import report as report_mod
from .build import build_daily, merge_codis_tail, write_outputs
from .config import PROCESSED_DIR, REPO_ROOT, REPORTS_DIR, load_config
from .indices import annual_indices, monthly_summary, seasonal_means
from .sources import codis, mirror
from .webpage import build_payload, write_page


def cmd_sync(args: argparse.Namespace) -> int:
    config = load_config()
    exit_code = 0

    for station in config.stations:
        print(
            f"Syncing {station.id} ({station.name_zh}, {station.role}) "
            f"{config.start_year}-{config.end_year}"
        )
        result = mirror.sync_station(
            station.id, config.years, station.raw_dir, granularity="daily", verbose=args.verbose
        )
        print(f"  {result.summary()}")

        if not result.updated and not result.unchanged:
            print(f"ERROR: sync produced nothing usable for {station.id}", file=sys.stderr)
            if station.role == "primary":
                exit_code = 1

    return exit_code


def cmd_build(args: argparse.Namespace) -> int:
    config = load_config()
    station = config.primary

    daily, qc = build_daily(station.raw_dir, config.start_year, config.end_year)
    print(f"Built {len(daily):,} daily rows ({daily['date'].min():%Y-%m-%d} to {daily['date'].max():%Y-%m-%d})")

    if args.with_codis:
        start = (daily["date"].max() - timedelta(days=args.codis_days)).date()
        print(f"Topping up from CODiS since {start}")
        try:
            tail = codis.fetch_daily_range(station.id, station.station_type, start, date.today())
            daily = merge_codis_tail(daily, tail)
            print(f"  merged {len(tail):,} CODiS rows")
        except Exception as exc:  # noqa: BLE001 - the mirror data stays valid either way
            print(f"  WARNING: CODiS top-up failed, keeping mirror data only: {exc}")

    paths = write_outputs(daily, qc, PROCESSED_DIR, station.id)

    monthly = monthly_summary(daily)
    monthly.to_csv(PROCESSED_DIR / f"{station.id}_monthly.csv", index=False, float_format="%.3f")

    annual = annual_indices(daily)
    annual.to_csv(PROCESSED_DIR / f"{station.id}_annual.csv", index=False, float_format="%.3f")

    seasonal = seasonal_means(daily)
    seasonal.to_csv(PROCESSED_DIR / f"{station.id}_seasonal.csv", index=False, float_format="%.3f")

    for name, path in paths.items():
        print(f"  wrote {path.relative_to(path.parents[2])}")
    print(f"  wrote {len(monthly):,} monthly, {len(annual):,} annual, {len(seasonal):,} seasonal rows")

    # Reference stations only need annual indices, for the homogeneity tests.
    for reference in config.references:
        try:
            ref_daily, _ = build_daily(reference.raw_dir, config.start_year, config.end_year)
        except FileNotFoundError:
            print(f"  WARNING: no raw data for reference {reference.id}, skipping")
            continue
        ref_annual = annual_indices(ref_daily)
        ref_annual.to_csv(
            PROCESSED_DIR / f"{reference.id}_annual.csv", index=False, float_format="%.3f"
        )
        print(f"  wrote reference {reference.id} ({reference.name_zh}): {len(ref_annual)} years")

    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    config = load_config()
    station = config.primary

    annual = pd.read_csv(PROCESSED_DIR / f"{station.id}_annual.csv")
    seasonal = pd.read_csv(PROCESSED_DIR / f"{station.id}_seasonal.csv")
    coverage = pd.read_csv(PROCESSED_DIR / f"{station.id}_coverage.csv")

    references: dict[str, pd.DataFrame] = {}
    for reference in config.references:
        path = PROCESSED_DIR / f"{reference.id}_annual.csv"
        if path.exists():
            references[f"{reference.id} {reference.name_zh}"] = pd.read_csv(path)
        else:
            print(f"  WARNING: reference {reference.id} not built, homogeneity test reduced")

    path = report_mod.write_report(
        config, station, annual, seasonal, coverage, references, REPORTS_DIR
    )
    print(f"Wrote {path}")
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    config = load_config()
    station = config.primary

    daily = pd.read_csv(PROCESSED_DIR / f"{station.id}_daily.csv", parse_dates=["date"])
    payload = build_payload(config, station, daily)
    path = write_page(payload, REPO_ROOT / "web" / "index.html")

    window = payload["window"]
    print(f"Wrote {path} ({window['start']}-{window['end']} common window)")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    for step in (cmd_sync, cmd_build, cmd_analyze, cmd_web):
        code = step(args)
        if code != 0:
            return code
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fge-climate", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="log every file fetched")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("sync", help="download raw station files").set_defaults(func=cmd_sync)

    build_p = sub.add_parser("build", help="build processed datasets from raw files")
    build_p.add_argument(
        "--with-codis",
        action="store_true",
        help="top up the recent tail directly from the CODiS API",
    )
    build_p.add_argument(
        "--codis-days", type=int, default=45, help="days of tail to re-fetch (default: 45)"
    )
    build_p.set_defaults(func=cmd_build)

    sub.add_parser("analyze", help="regenerate the baseline climate report").set_defaults(
        func=cmd_analyze
    )

    sub.add_parser("web", help="regenerate the four-measure web page").set_defaults(func=cmd_web)

    all_p = sub.add_parser("all", help="sync, build, analyze and regenerate the page")
    all_p.add_argument("--with-codis", action="store_true")
    all_p.add_argument("--codis-days", type=int, default=45)
    all_p.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

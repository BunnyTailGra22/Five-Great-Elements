# Erge Mountain (二格山) phenology — climate base

Long-term climate dataset for the Erge Mountain phenology study: an automated
sync of the CWA station record, a QC'd daily dataset, phenology-relevant annual
indices, and a regenerated findings report.

The current deliverable is step one of the phenology project — the climate
foundation and a first read of what has changed. No phenology observations are
ingested yet.

## The station

| | |
| --- | --- |
| Station | `82A750` 茶改北部分場 (Tea Research and Extension Station, Northern Branch) |
| Position | 24.955778°N, 121.631222°E, 401 m |
| Address | 新北市石碇區格頭村北宜路五段12號 |
| Record | daily since 1990 (established 1987-05-01; the 1987–89 files are empty upstream) |

It sits in Getou village at the northern foot of Erge Mountain — **1.7 km from
the summit, 277 m below it**. As a tea-research station it also records soil
temperature, which most CWA stations do not.

Two neighbouring stations are synced purely as homogeneity references:
`466930` 竹子湖 (607 m, 24.5 km, complete record since 1943) and `C0A530` 坪林
(300 m, 8.2 km).

## Read this before using the trends

**The raw record is not homogeneous.** Testing against the reference stations
found step changes that regional weather cannot explain:

| Variable | Break | Evidence |
| --- | --- | --- |
| Diurnal temperature range | 2002 | −1.29 °C vs 竹子湖, p = 0.001 |
| Mean daily maximum | 2002 | −1.33 °C vs 竹子湖; attributed from the DTR break |
| Mean wind speed | 2007 | −1.17 m/s vs 竹子湖, p < 0.001 |
| Rainfall intensity | 2016 | +4.25 mm/d vs 坪林, p = 0.031 |

These breaks invert conclusions rather than merely blurring them. Read naively,
the full record says the site's hot days are *falling* 6.2 days/decade; once the
2002 break is respected, the post-2002 segment gives *+8.9* days/decade. Any
phenology model calibrated on the raw series would be fitting an instrument
change.

The pipeline detects this automatically and refuses to present affected trends
as climate. See `reports/climate_baseline.md` for the full treatment.

## Usage

```bash
pip install -r requirements.txt
export PYTHONPATH=src

python -m fge_climate.cli sync      # download raw yearly files (incremental)
python -m fge_climate.cli build     # QC + tidy + indices
python -m fge_climate.cli analyze   # regenerate the report
python -m fge_climate.cli web       # regenerate both web/ pages
python -m fge_climate.cli all       # all four
```

`build --with-codis` additionally pulls the last 45 days straight from the CODiS
API, for when the mirror's lag matters. It is off by default and falls back to
mirror data if the API is unreachable.

## How the sync works

Bulk history comes from the [Raingel/historical_weather](https://github.com/Raingel/historical_weather)
mirror of CWA CODiS, which is already rebuilt from official downloads. Each
year's ETag is recorded in `data/raw/<station>/_manifest.json`, so a re-run
re-fetches only what changed upstream — a no-op sync is 111 conditional requests
(3 stations x 37 years) that all return 304. `data/raw/` is gitignored, so the
daily commit carries only the processed outputs; CI caches the raw tree to keep
the sync incremental there too.

`.github/workflows/climate-sync.yml` runs this daily at 04:10 Taipei time, runs
the tests, rebuilds, regenerates the report and both web pages, and commits any
change.

## The four-measure page

`web/index.html` charts the four measures the phenology work depends on —
`TxSoil0cm` 地溫, `Precp` 日降水量, `SunShine` 日照時數 and `WS` 風速 — each with
its seasonal cycle, its full record, and a per-year coverage strip. It is
generated from `data/processed/`, so it never drifts from the dataset.

Two things the page is built to make unavoidable:

- **The four measures only overlap for 2019–2025.** The soil probe and sunshine
  recorder both start in 2018, and rainfall's 2018 misses the coverage gate. Every
  "typical year" figure on the page is taken over that common window.
- **Gaps are drawn as gaps.** A year below its coverage gate is left empty rather
  than interpolated, and wind's station breaks are marked on the record itself.

Colours come from the validated categorical palette in slot order, so adjacent
panels clear the colourblind-separation gate in both light and dark mode; every
chart also ships a table view.

## Daily distributions since 2018

`web/distributions.html` is the companion analysis at daily resolution —
11,461 observations over 2,922 days — with three views per measure: a **box
plot by month**, a **frequency** histogram, and a **heat map by month**
(year × month). Both pages come from `fge-climate web`.

2018 is the right floor for pooling, and not only because the soil probe and
sunshine recorder start there: every artificial break in the record (2002, 2007,
2016) falls *before* it, so this window needs no homogeneity adjustment.

Two analytic choices worth knowing:

- **Rainfall boxes describe wet days only** (≥ 0.1 mm), on a log axis. Daily
  rainfall is 47.5% zeros, which collapses a linear box plot to a median of 0
  with everything else an outlier. The dry-day fraction is reported per month
  alongside, so nothing is hidden.
- **Frequency uses rainfall classes** rather than equal-width bins for the same
  reason; the other three measures use equal-width bins.

Box plots and histograms pool complete years only, so every month carries the
same number of seasons. The heat map keeps the running year and hatches its
unfinished months.

## Layout

```
config/stations.yml          station registry and study-area definition
src/fge_climate/
  sources/mirror.py          incremental fetch from the CODiS mirror
  sources/codis.py           direct CODiS API client (recent tail)
  qc.py                      sentinel masking and range checks
  build.py                   raw yearly files -> tidy daily table
  indices.py                 phenology-relevant annual indices
  trends.py                  Theil-Sen slopes, Mann-Kendall tests
  homogeneity.py             Pettitt changepoints, break attribution
  report.py                  report generation
  webpage.py                 payload for the four-measure page
  distributions.py           daily box/frequency/heat-map statistics
  templates/measures.html    page template (charts are inline SVG, no libraries)
  templates/distributions.html   daily-distribution page template
data/raw/<station>/          verbatim upstream files + sync manifest
data/processed/              daily, monthly, seasonal, annual, coverage, QC
reports/climate_baseline.md  generated findings
web/index.html               generated four-measure page
web/distributions.html       generated daily-distribution page
```

## Method notes

**Missing data.** The station encodes missing values as large negatives
(`-99.5`, `-9999.5`, `-9995`), which read as ordinary numbers. 724 sentinel
temperatures and 8 impossible humidity values (>100%) were masked; averaging
them in would have shifted annual means by whole degrees.

**Coverage gating.** Absent days are materialised as explicit rows so gaps are
measurable. An annual index is emitted only if its inputs clear a coverage floor
— 95% for sums (rainfall, degree-days), 90% for means — so a year with a
three-month hole cannot pose as a cool or dry year. This drops 1990, 1998, 2000,
2006 and 2012–2015 from most indices.

**Index choice.** The standard ETCCDI growing-season-length index (mean
temperature above 10 °C) saturates at 365 days at this site and carries no
signal, so a 20 °C warm-season threshold is used instead, alongside GDD
accumulation timing — the quantities that actually pace bud break and flushing
in the subtropics.

**Statistics.** Theil–Sen slopes with Mann–Kendall significance throughout, both
robust to the skewed, outlier-heavy distributions of rainfall indices. Breaks
are located with a Pettitt test on the target-minus-reference difference series,
which cancels regional climate so that a surviving break belongs to the station.

## Known limits

- Sunshine has no usable record for 1995–2015; soil temperature begins in 2018.
  Neither supports a long-term trend.
- Breaks are detected but **not yet corrected**. The next step is to adjust the
  pre-break segments against the reference stations, which would restore the
  full record for trend work.
- `sources/codis.py` is written against the documented CODiS contract but has
  not been exercised here — the sandbox this was built in blocks
  `codis.cwa.gov.tw`. The mirror path is fully tested.

## Data source and citation

Observations are CWA (中央氣象署) CODiS station data, obtained via the
Raingel/historical_weather rebuild:

> Ou, J.-H., et al., 2023. Application-oriented deep learning model for early
> warning of rice blast in Taiwan. *Ecological Informatics* 73, 101950.
> https://doi.org/10.1016/j.ecoinf.2022.101950

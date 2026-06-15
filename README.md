## Streetscope: Measuring Gender Representation in Public Spaces

Analyzing gender representation in Indian cities using GoPro wearable camera footage.

## Key Findings

### Summary

| City | Images | Pedestrians | Prop. Female | Sex Ratio |
|------|--------|-------------|--------------|-----------|
| Mumbai | 2,251 | 12,382 | 19.3% | 239 |
| Navi Mumbai | 1,239 | 4,951 | 18.2% | 222 |

Women are significantly underrepresented in public spaces, comprising only 18-19% of pedestrians. The pedestrian sex ratio (females per 1000 males) is far below census baselines (Mumbai: 838, Navi Mumbai: 910).

### By Mode

| Mode | Mumbai | Navi Mumbai |
|------|--------|-------------|
| Pedestrians | 19.3% | 18.2% |
| Two-wheelers | 8.4% | 5.7% |

Women are far less represented among two-wheeler riders than pedestrians.

### By Road Type

| City | Primary | Secondary | Tertiary | Residential |
|------|---------|-----------|----------|-------------|
| Mumbai | 14.7% | 14.1% | 16.9% | 17.1% |
| Navi Mumbai | 17.9% | 15.5% | 16.4% | 14.9% |

## Pipeline

The project is a linear pipeline from sampling design to published results:

```
geo-sampling → itineraries (allocator) → GoPro collection → EXIF/GPS/frame extraction
  → face-frame extraction → human annotation (Label Studio)
  → parse annotations → assign GPS → enrich with road type → analysis_data
  → figures + tables + maps
```

### Frozen vs. fair-game

The **collection half is frozen**: it has been run once in the field and on the raw
footage, and its outputs (extracted frames, EXIF, GPS index, annotations) are treated as
immutable inputs. The **analysis half is fair game**: it transforms those frozen inputs
into the published dataset and outputs and can be re-run freely.

| Stage | Script(s) | Status |
|-------|-----------|--------|
| Geo-sampling / itineraries (allocator) | `scripts/gen_data/<city>/run_*_sampler.py` | **frozen** |
| Video processing: EXIF, GPS, frame extraction | `00_process_videos.py` | **frozen** |
| Face-detection frame extraction | `01_extract_face_frames.py` | **frozen** |
| Coverage QA / frame compression | `02_plot_video_coverage.py`, `03_compress_frames.py` | **frozen** |
| GPS lookup index / EXIF recovery | `04_build_gps_index.py`, `rebuild_csvs_from_exif.py` | **frozen** |
| Human annotation (Label Studio) | external | **frozen** |
| Parse annotations | `05_parse_annotations.py` | fair game |
| Assign GPS to frames (ts/lat/lon) | `06_assign_frame_gps.py` | fair game |
| Enrich with road type (itinerary + OSM) | `07_enrich_with_geo.py` | fair game |
| Build analysis dataset | `08_build_analysis_data.py` | fair game |
| EDA / publication figures+tables / maps | `09_eda.py`, `10_analysis.py`, `11_make_maps.py` | fair game |
| Design-based inference (geoinference) | `inference.py` | fair game |
| LLM annotation validation | `12_validate_annotations.py` | fair game |

### Run the analysis half

The analysis half starts from the frozen GPS index and annotations and never re-touches
videos or frames:

```bash
# One city, end to end (analysis_data + its figures/tables/maps), using cached GPS index:
make analyze CITY=mumbai

# Regenerate figures/tables/maps across cities from existing analysis_data:
make figures CITIES=mumbai,navi_mumbai

# Tables only:
make tables CITIES=mumbai,navi_mumbai
```

Equivalent direct invocation:

```bash
.venv/bin/python scripts/run_pipeline.py --city mumbai --skip-rebuild-gps
```

### Re-running collection (frozen — only if raw footage changes)

```bash
# Re-process videos and re-extract frames (requires video_dir in cities.yaml):
.venv/bin/python scripts/run_pipeline.py --city mumbai --process-videos
```

## Inference

All design-based estimates and standard errors come from the `geoinference` library
(packaged in [`../geoinference`](../geoinference)) via `scripts/inference.py`. Frames collected
along one video session are spatially and temporally correlated, so estimates cluster by
`base_video_id` and use cluster-robust standard errors (Horvitz–Thompson linearization,
t_{G-1} interval). Two estimands are reported:

- **Person-weighted ratio** — `sum(women) / sum(people)` (`geoinference` `ratio`).
- **Image-level mean** — `mean(women_i / people_i)` (`geoinference` `photo_mean`).

`inference.summarize(df)` returns both, with cluster-robust 95% CIs and the ICC / design
effect, from a single `geoinference.estimate()` call.

## Setup

```bash
# System tools (macOS)
brew install ffmpeg exiftool

# Python environment (uv-managed .venv) + all deps incl. editable geoinference:
make install
```

`make install` installs the editable `../geoinference` plus pandas, numpy, geopandas,
shapely, folium, contextily, matplotlib, statsmodels, scikit-learn, pyarrow, pyyaml,
tqdm, pillow, osmnx, and the linters (black, isort, flake8).

### Optional: OSM ground-truth road type

`07_enrich_with_geo.py` can attach an OSM "ground truth" road type alongside the
itinerary road type. It is skipped automatically unless the city's PBF is present.
Download the region from [Geofabrik](https://download.geofabrik.de/) into `data/osm/`
using the filename in `cities.yaml`, e.g.:

```
data/osm/maharashtra-latest.osm.pbf   # mumbai, navi_mumbai
data/osm/karnataka-latest.osm.pbf     # bangalore
```

When present, `tableS1_road_type.tex` gains an OSM ground-truth section.

## Directory Structure

```
scripts/
  gen_data/{city}/           Sampling-design scripts (frozen)
  00_process_videos.py ...   Pipeline scripts (see table above)
  inference.py               geoinference-backed design-based inference
  run_pipeline.py            Analysis-half orchestrator
data/
  {city}/
    exif/                    Raw EXIF text files (frozen)
    gps_index/               GPS lookup parquet files (frozen)
    labelstudio/             Label Studio JSON exports (frozen)
    face_frame_metadata.csv  Face-detection frame log
    analysis_data.parquet    Final analysis dataset (primary annotator)
    analysis_data_long.parquet  Long format (all annotators)
  annotation_task/           Extracted frames for annotation
  osm/                       Optional OSM PBFs for ground-truth road type
sampling/
  {city}/itineraries/        Itinerary road-type classifications
figs/                        Generated figures and maps
tabs/                        Generated LaTeX tables
cities.yaml                  Per-city config (video_dir, osm_file, timezone)
Makefile                     Analysis-half workflow targets
```

## Outputs

### Data
- `data/{city}/analysis_data.parquet` - Primary analysis dataset
- `data/{city}/analysis_data_long.parquet` - Long format (all annotators)

### Figures (`figs/`)
- `fig2_distribution.pdf` - Proportion female distribution
- `fig3_multipanel.pdf` - Multi-panel summary (mode, road type, POI, time)
- `fig4_weekday_weekend.pdf` - Weekday vs weekend comparison
- `fig5_pedestrian_loess.pdf` - LOESS: pedestrian women vs crowd size
- `map_locations.pdf` - Data collection locations
- `map_sexratio.pdf` - Female share by location
- `eda_*.pdf` - Exploratory analysis plots

### Tables (`tabs/`)
- `table1_city_summary.tex` - City-level summary
- `tableS1_road_type.tex` - By road type (itinerary + optional OSM ground truth)
- `tableS2_temporal.tex` - Temporal patterns
- `tableS3_poi_infrastructure.tex` - POI and infrastructure

## Data Coverage

| Metric | Mumbai | Navi Mumbai |
|--------|--------|-------------|
| GPS coverage | 97.4% | 99.4% |
| Itinerary match | 66.3% | 72.1% |
| Hour range (IST) | 7-19 | 9-18 |

Bangalore, Delhi, and Hyderabad have sampling itineraries and partially processed
footage (face frames extracted); annotation/analysis for them is in progress.

## License

MIT

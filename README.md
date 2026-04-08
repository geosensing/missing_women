## Missing Women: Measuring Gender Representation in Public Spaces

Analyzing gender representation in Mumbai and Navi Mumbai using GoPro wearable camera footage.

## Key Findings

### Summary

| City | Images | People | Prop. Female | Sex Ratio |
|------|--------|--------|--------------|-----------|
| Mumbai | 2,251 | 16,908 | 16.4% | 196 |
| Navi Mumbai | 1,239 | 6,855 | 14.7% | 172 |

Women are significantly underrepresented in public spaces, comprising only 15-16% of visible people. The pedestrian sex ratio (females per 1000 males) is far below census baselines (Mumbai: 838, Navi Mumbai: 910).

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

```
Raw Videos → EXIF Extraction → GPS Parsing → GPS Index →
Annotations → GPS Assignment → Geo Enrichment → Analysis Data →
EDA + Analysis + Maps
```

### Run Full Pipeline

```bash
# Run E2E for all cities
python scripts/run_pipeline.py --city all --skip-osm

# Run for single city
python scripts/run_pipeline.py --city mumbai --skip-osm

# Skip visualization steps
python scripts/run_pipeline.py --city all --skip-osm --skip-viz
```

### Pipeline Scripts

| Script | Description |
|--------|-------------|
| `00_extract_exif.py` | Extract EXIF metadata from video files |
| `01_parse_exif.py` | Parse EXIF text to GPS timeseries CSV |
| `04_build_gps_index.py` | Build GPS lookup index (parquet) |
| `05_parse_annotations.py` | Parse Label Studio JSON exports |
| `06_assign_frame_gps.py` | Match frames to GPS coordinates |
| `07_enrich_with_geo.py` | Add road type from itineraries |
| `08_build_analysis_data.py` | Build final analysis dataset |
| `09_eda.py` | Exploratory data analysis plots |
| `10_analysis.py` | Publication tables and figures |
| `11_make_maps.py` | Interactive and static maps |
| `run_pipeline.py` | Run full E2E pipeline |

## Setup

```bash
# System tools (macOS)
brew install ffmpeg exiftool

# Python environment
pip install pandas numpy geopandas folium contextily statsmodels scikit-learn pyyaml tqdm
```

## Directory Structure

```
data/
  {city}/
    exif/                    Raw EXIF text files
    exif_metadata/           Parsed GPS timeseries CSVs
    gps_index/               GPS lookup parquet files
    labelstudio/             Label Studio JSON exports
    analysis_data.parquet    Final analysis dataset
sampling/
  {city}/
    itineraries/             Road type classifications
figs/                        Generated figures and maps
tabs/                        Generated LaTeX tables
scripts/                     Pipeline scripts
cities.yaml                  City configuration
```

## Outputs

### Data
- `data/{city}/analysis_data.parquet` - Primary analysis dataset
- `data/{city}/analysis_data_long.parquet` - Long format (all annotators)

### Figures (`figs/`)
- `fig2_distribution.pdf` - Proportion female distribution
- `fig3_multipanel.pdf` - Multi-panel summary
- `fig4_weekday_weekend.pdf` - Weekday vs weekend comparison
- `fig5_pedestrian_loess.pdf` - LOESS: pedestrian women vs crowd size
- `map_locations.pdf/html` - Data collection locations
- `map_sex_ratio.pdf/html` - Sex ratio by location
- `eda_*.pdf` - Exploratory analysis plots

### Tables (`tabs/`)
- `table1_city_summary.tex` - City-level summary
- `tableS1_road_type.tex` - By road type
- `tableS2_temporal.tex` - Temporal patterns
- `tableS3_poi_infrastructure.tex` - POI and infrastructure

## Data Coverage

| Metric | Mumbai | Navi Mumbai |
|--------|--------|-------------|
| GPS coverage | 97.4% | 99.4% |
| Itinerary match | 66.3% | 72.1% |
| Hour range (IST) | 7-19 | 9-18 |

## License

MIT

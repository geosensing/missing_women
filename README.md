## Streetscope: Measuring Street-level Conditions From Wearable Camera Transects

Analyzing gender representation and infrastructure quality in Mumbai and Navi Mumbai using GoPro wearable camera footage.

## Key Findings

### Proportion Women

| City | Images | Unweighted | Weighted | Ped. Sex Ratio |
|------|--------|------------|----------|----------------|
| Mumbai | 2,740 | 0.147 | 0.160 | 238.8 |
| Navi Mumbai | 389 | 0.131 | 0.131 | 219.7 |

Women are significantly underrepresented in public spaces, comprising only 13-16% of visible people. The pedestrian sex ratio (females per 1000 males) is far below census baselines (Mumbai: 838, Navi Mumbai: 910).

### By Mode

| City | Pedestrians (unweighted) | Pedestrians (weighted) | Two-wheelers (unweighted) | Two-wheelers (weighted) |
|------|--------------------------|------------------------|---------------------------|------------------------|
| Mumbai | 0.185 | 0.193 | 0.071 | 0.083 |
| Navi Mumbai | 0.186 | 0.180 | 0.045 | 0.050 |

Proportion of women is consistently lower among two-wheeler riders than pedestrians.

### By Road Type

| City | Primary | Secondary | Tertiary | Residential |
|------|---------|-----------|----------|-------------|
| Mumbai (prop women) | 0.164 | 0.127 | 0.164 | 0.162 |
| Mumbai (sex ratio) | 196 | 146 | 197 | 193 |
| Navi Mumbai (prop women) | 0.090 | 0.110 | 0.145 | 0.179 |
| Navi Mumbai (sex ratio) | 99 | 123 | 169 | 219 |

In Navi Mumbai, there is a clear gradient: proportion of women increases from primary roads (9%) to residential roads (18%). Mumbai shows less variation across road types.

### Temporal Patterns

**Data coverage**: ~7 AM - 7 PM IST. Thursday overrepresented (40%).

**Weekday vs Weekend:**

| City | Weekday | Weekend | Significant? |
|------|---------|---------|--------------|
| Mumbai | 0.163 (n=2100) | 0.145 (n=458) | Yes (p=0.048) |
| Navi Mumbai | 0.138 (n=835) | 0.168 (n=129) | No |

Mumbai shows significantly higher proportion of women on weekdays. Navi Mumbai shows the opposite pattern but the difference is not statistically significant.

**Data Collection by Time Period (equal 4-hour windows):**

| Time Period (IST) | Mumbai | Navi Mumbai |
|-------------------|--------|-------------|
| Morning (7-11) | 0.21 | 0.39 |
| Midday (11-15) | 0.41 | 0.47 |
| Evening (15-19) | 0.37 | 0.13 |

Columns sum to 1.0 within each city. Navi Mumbai data is concentrated in morning/midday hours with sparse evening coverage.

See `scripts/analysis/temporal_analysis.ipynb` for detailed temporal analysis including distribution plots and POI × time interactions.

### By Points of Interest (POI)

| POI | Yes | No | Significant? |
|-----|-----|-----|--------------|
| Bus Station | 0.168 (n=137) | 0.156 (n=3385) | Yes (p=0.04) |
| Railway Station | 0.141 (n=7) | 0.156 (n=3515) | No |
| Street Vendor | 0.208 (n=238) | 0.150 (n=3283) | Yes (p<0.001) |

Street vendors are associated with significantly higher proportion of women. Bus stations show a small positive effect.

### Infrastructure

**Footpath:**

| City | Paved | Paved-Blocked | No Sidewalk | N |
|------|-------|---------------|-------------|---|
| Mumbai | 1,150 (59%) | 324 (17%) | 459 (24%) | 1,938 |
| Navi Mumbai | 217 (74%) | 17 (6%) | 60 (20%) | 294 |

**Litter:**

| City | Yes | Construction Debris | No | N |
|------|-----|---------------------|-----|---|
| Mumbai | 256 (78%) | 57 (17%) | 16 (5%) | 329 |
| Navi Mumbai | 54 (93%) | 3 (5%) | 1 (2%) | 58 |

**Lane Markings:**

| City | Yes | No | N |
|------|-----|-----|---|
| Mumbai | 1,602 (99%) | 12 (1%) | 1,614 |
| Navi Mumbai | 318 (100%) | 0 | 318 |

**Potholes:**

| City | Yes | No | N |
|------|-----|-----|---|
| Mumbai | 30 (54%) | 26 (46%) | 56 |
| Navi Mumbai | 3 (75%) | 1 (25%) | 4 |

Note: Infrastructure annotations are sparse - not all images have all fields.

## Data Pipeline

```
GoPro Videos
     |
scripts/extract_exif.py         Extract EXIF metadata from videos
     |
scripts/extract_gps_timeseries.py   Parse GPS data from EXIF files
     |
scripts/extract_frames.py       Extract frames at regular intervals
     |
scripts/compress_frames.py      Compress frames for annotation
     |
Label Studio                    Human annotation of images
     |
scripts/analysis/*.ipynb        Analysis and visualization
```

## Setup

### System Tools

```bash
# macOS
brew install ffmpeg exiftool

# Ubuntu
sudo apt-get install ffmpeg exiftool
```

### Python Packages

```bash
pip install pandas Pillow tqdm
pip install opencv-python  # optional fallback for video info
```

## Usage

```bash
# 1. Extract EXIF metadata from videos
python scripts/extract_exif.py \
    --input /path/to/videos \
    --exif-output exif/ \
    --metadata-csv video_metadata.csv

# 2. Extract GPS timeseries from EXIF files
python scripts/extract_gps_timeseries.py \
    --input exif/ \
    --output gps_timeseries.csv

# 3. Extract frames every 10 seconds
python scripts/extract_frames.py \
    --input /path/to/videos \
    --output frames/ \
    --log frame_log.txt \
    --report extraction_report.csv \
    --every-seconds 10

# 4. Compress frames for annotation
python scripts/compress_frames.py \
    --input frames/ \
    --output annotation_frames/ \
    --resolution 1280x720 \
    --quality 75

# 5. Run analysis notebooks
jupyter notebook scripts/analysis/mumbai_annotations.ipynb
jupyter notebook scripts/analysis/navi_mumbai_annotations.ipynb
```

## Data

- **Mumbai**: 2,740 annotated images from GoPro transects
- **Navi Mumbai**: 389 annotated images from GoPro transects
- **Annotation fields**: men_count, women_count, men_twowheeler, women_twowheeler, footpath, lane_markings, potholes, litter, bus_station, railway_station, street_vendor

## Directory Structure

```
annotation_frames/      Compressed frames for Label Studio
data/                   Data files (CSVs)
exif/                   EXIF metadata files
frames/                 Extracted video frames
labelstudio/            Label Studio JSON exports
output/maps/            HTML visualizations (maps, heatmaps)
scripts/
  analysis/             Jupyter notebooks for analysis
  extract_exif.py       Extract EXIF from videos
  extract_gps_timeseries.py   Parse GPS from EXIF
  extract_frames.py     Extract frames from videos
  compress_frames.py    Compress frames for annotation
```

## Output Files

| File | Description |
|------|-------------|
| `output/maps/mumbai_annotations_map.html` | Interactive map of Mumbai annotations |
| `output/maps/navi_mumbai_annotations_map.html` | Interactive map of Navi Mumbai annotations |
| `output/maps/mumbai_sex_ratio_heatmap.html` | Mumbai sex ratio heatmap |
| `output/maps/navi_mumbai_sex_ratio_heatmap.html` | Navi Mumbai sex ratio heatmap |
| `output/maps/mumbai_heatmap.html` | GPS point density heatmap |
| `output/maps/mumbai_route_map.html` | Route visualization |
| `data/mumbai_annotations_with_exif.csv` | Merged annotation and GPS data |

## License

MIT

## Streetscope: Measuring Street-level Conditions From Wearable Camera Transects

Analyzing gender representation and infrastructure quality in Mumbai and Navi Mumbai using GoPro wearable camera footage.

## Key Findings

### Proportion Women

| City | Images | Prop. Women (weighted) | Pedestrian Sex Ratio |
|------|--------|------------------------|----------------------|
| Mumbai | 2,740 | 0.160 | 238.8 |
| Navi Mumbai | 389 | 0.131 | 219.7 |

Women are significantly underrepresented in public spaces, comprising only 13-16% of visible people. The pedestrian sex ratio (females per 1000 males) is far below census baselines (Mumbai: 838, Navi Mumbai: 910).

### By Mode

| City | Pedestrians (weighted) | Two-wheelers (weighted) |
|------|------------------------|-------------------------|
| Mumbai | 0.193 | 0.083 |
| Navi Mumbai | 0.180 | 0.050 |

Proportion of women is consistently lower among two-wheeler riders than pedestrians.

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
data/                   Analysis outputs (maps, CSVs)
exif/                   EXIF metadata files
frames/                 Extracted video frames
labelstudio/            Label Studio JSON exports
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
| `data/mumbai_annotations_map.html` | Interactive map of annotated locations |
| `data/mumbai_heatmap.html` | Sex ratio heatmap |
| `data/mumbai_route_map.html` | Route visualization |
| `data/mumbai_annotations_with_exif.csv` | Merged annotation and GPS data |

## License

MIT

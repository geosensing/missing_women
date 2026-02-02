## Missing Women

Scripts and data for sampling segments and drawing routes.

## Dependencies

### System Tools

- `ffmpeg` - Video frame extraction
- `ffprobe` - Video metadata analysis
- `exiftool` - EXIF metadata extraction

Install on macOS:
```bash
brew install ffmpeg exiftool
```

Install on Ubuntu:
```bash
sudo apt-get install ffmpeg exiftool
```

### Python Packages

```bash
pip install pandas Pillow tqdm
```

Optional (fallback for video info):
```bash
pip install opencv-python
```

## Scripts

### 1. Extract EXIF Metadata

Extract EXIF metadata from video files in parallel.

```bash
python scripts/extract_exif.py \
    --input /path/to/videos \
    --exif-output exif_files/ \
    --metadata-csv video_metadata.csv
```

**Arguments:**
| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--input` | `-i` | Yes | Input folder containing videos |
| `--exif-output` | `-o` | Yes | Output folder for EXIF text files |
| `--metadata-csv` | `-m` | Yes | Path for video metadata CSV file |
| `--workers` | `-w` | No | Number of parallel workers (default: CPU count) |
| `--extensions` | `-e` | No | Comma-separated extensions (default: `.mp4,.mov,.avi,.mkv`) |
| `--timeout` | `-t` | No | Timeout in seconds for exiftool (default: 180) |
| `--verbose` | `-v` | No | Enable verbose logging |

**Outputs:**
- `exif_files/` - Directory containing `*_exif.txt` files for each video
- `video_metadata.csv` - CSV with video metadata (duration, fps, resolution, codec, camera model)

### 2. Extract GPS Timeseries

Parse GPS data from EXIF text files (GoPro format).

```bash
python scripts/extract_gps_timeseries.py \
    --input exif_files/ \
    --output gps_timeseries.csv
```

**Arguments:**
| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--input` | `-i` | Yes | Directory containing EXIF text files |
| `--output` | `-o` | Yes | Output CSV file for GPS timeseries |
| `--pattern` | `-p` | No | File pattern to match (default: `*_exif.txt`) |
| `--verbose` | `-v` | No | Enable verbose logging |

**Outputs:**
- `gps_timeseries.csv` - CSV with columns: `video_id`, `gps_datetime`, `gps_latitude`, `gps_longitude`, `gps_altitude`, `gps_speed`, `gps_speed_3d`, `gps_measure_mode`, `gpsdop`

### 3. Extract Frames

Extract frames from videos at regular time intervals using ffmpeg.

```bash
python scripts/extract_frames.py \
    --input /path/to/videos \
    --output frames/ \
    --log frame_log.txt \
    --report extraction_report.csv \
    --every-seconds 10
```

**Arguments:**
| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--input` | `-i` | Yes | Input folder containing videos |
| `--output` | `-o` | Yes | Output folder for frames |
| `--log` | `-l` | Yes | Path for metadata log file |
| `--report` | `-r` | Yes | Path for CSV report |
| `--every-seconds` | `-s` | No | Extract one frame every N seconds (default: 10) |
| `--quality` | `-q` | No | JPEG quality 1-100 (default: 95) |
| `--extensions` | `-e` | No | Comma-separated extensions (default: `.mp4,.mov,.avi,.mkv`) |
| `--overwrite` | | No | Re-extract even if already done |

**Outputs:**
- `frames/` - Directory containing `{video_id}_frame{NNNNNN}.jpg` files
- `frame_log.txt` - Pipe-delimited log with frame metadata (video_id, source_folder, video_name, frame_filename, frame_number, timestamp, etc.)
- `extraction_report.csv` - CSV report with per-video statistics (expected vs extracted frames, status)

### 4. Compress Frames

Compress high-resolution frames for annotation tasks.

```bash
python scripts/compress_frames.py \
    --input frames/ \
    --output annotation_frames/ \
    --resolution 1280x720 \
    --quality 75
```

**Arguments:**
| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--input` | `-i` | Yes | Input directory containing high-resolution frames |
| `--output` | `-o` | Yes | Output directory for compressed frames |
| `--resolution` | `-r` | No | Target resolution WIDTHxHEIGHT (default: 1280x720) |
| `--quality` | `-q` | No | JPEG quality 1-100 (default: 75) |
| `--workers` | `-w` | No | Number of worker threads (default: CPU count) |
| `--pattern` | `-p` | No | File pattern to match (default: `*.jpg`) |
| `--verbose` | `-v` | No | Enable verbose logging |

**Outputs:**
- `annotation_frames/` - Directory containing compressed JPEG files

## Full Pipeline Example

```bash
# 1. Extract EXIF metadata from videos
python scripts/extract_exif.py \
    --input /path/to/videos \
    --exif-output exif_files/ \
    --metadata-csv video_metadata.csv

# 2. Extract GPS timeseries from EXIF files
python scripts/extract_gps_timeseries.py \
    --input exif_files/ \
    --output gps_timeseries.csv

# 3. Extract frames every 5 seconds
python scripts/extract_frames.py \
    --input /path/to/videos \
    --output frames/ \
    --log frame_log.txt \
    --report extraction_report.csv \
    --every-seconds 5

# 4. Compress frames for annotation
python scripts/compress_frames.py \
    --input frames/ \
    --output annotation_frames/ \
    --resolution 1280x720 \
    --quality 75
```

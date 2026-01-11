## Missing Women

Scripts and data for sampling segments and drawing routes.

## Scripts

### Video Processing Pipeline

**1. Extract EXIF Metadata**
```bash
python extract_exif.py --input "/path/to/videos" --exif-output "exif_files/" --metadata-csv "video_metadata.csv"
```

**2. Extract GPS Timeseries**
```bash
python extract_gps_timeseries.py --input "exif_files/" --output "gps_timeseries.csv"
```

**3. Extract Frames**
```bash
python extract_frames.py --input "/path/to/videos" --output "frames/" --log "frame_extraction.log" --interval 300
```

### Recent Fixes
- Fixed path calculation bugs for sibling directories
- Switched frame extraction to logging-based approach for thread safety
- Improved parallel processing reliability

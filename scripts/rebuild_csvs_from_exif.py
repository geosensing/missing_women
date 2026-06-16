#!/usr/bin/env python3
"""
Rebuild the GPS-index input CSVs from raw EXIF .txt files for a city.

Parses every ``data/{city}/exif/*_exif.txt`` and writes the two files that
``04_build_gps_index.py`` globs:

- ``data/{city}/exif_metadata/{city}_exif_video_metadata.csv``
- ``data/{city}/exif_metadata/{city}_gps_timeseries.csv``

It also (re)writes ``data/{city}/frame_metadata.csv`` from the extracted frames
when the frames directory is available.

Use this for cities whose ``exif_metadata/`` parquet inputs were never built
(e.g. bangalore, delhi). The itinerary-named exif files carry the GPS that links
back to the itinerary-named annotation frames.

Usage:
    python scripts/rebuild_csvs_from_exif.py --city bangalore
"""

import argparse
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm


def parse_video_metadata_from_exif(exif_content: str, exif_file: Path) -> Optional[Dict]:
    """Parse video metadata from EXIF content."""
    video_id = exif_file.stem.replace("_exif", "")

    metadata = {
        "video_duration_sec": None,
        "video_fps": None,
        "video_resolution_wxh": None,
        "video_codec": None,
        "camera_model": None,
        "recording_datetime": None,
        "file_name": None,
        "directory": None,
        "file_size_bytes": None,
    }

    for line in exif_content.strip().split("\n"):
        if ": " not in line:
            continue

        tag_part, value = line.split(": ", 1)
        tag = tag_part.strip().lower()
        value = value.strip()

        if "[file]" in tag and "file name" in tag:
            metadata["file_name"] = value

        if "[file]" in tag and "directory" in tag:
            metadata["directory"] = value

        if "[file]" in tag and "file size" in tag:
            match = re.search(r"([\d.]+)\s*(gb|mb|kb|bytes)", value.lower())
            if match:
                size, unit = match.groups()
                size = float(size)
                if unit == "gb":
                    size *= 1024 * 1024 * 1024
                elif unit == "mb":
                    size *= 1024 * 1024
                elif unit == "kb":
                    size *= 1024
                metadata["file_size_bytes"] = int(size)

        if "duration" in tag and ("quicktime" in tag or "composite" in tag):
            try:
                if ":" in value:
                    parts = value.split(":")
                    if len(parts) == 3:
                        h, m, s = parts
                        metadata["video_duration_sec"] = (
                            int(h) * 3600 + int(m) * 60 + float(s)
                        )
                    elif len(parts) == 2:
                        m, s = parts
                        metadata["video_duration_sec"] = int(m) * 60 + float(s)
            except (ValueError, IndexError):
                pass

        elif "frame rate" in tag or "video frame rate" in tag:
            try:
                if "/" in value:
                    num, den = value.split("/")
                    metadata["video_fps"] = float(num) / float(den)
                else:
                    metadata["video_fps"] = float(value.split()[0])
            except (ValueError, IndexError):
                pass

        elif "image size" in tag or "video frame size" in tag:
            if "x" in value:
                metadata["video_resolution_wxh"] = value.strip()

        elif "compressor id" in tag or "codec" in tag:
            metadata["video_codec"] = value.strip()

        elif "camera model" in tag or "device name" in tag:
            if metadata["camera_model"] is None:
                metadata["camera_model"] = value.strip()

        elif "create date" in tag and "quicktime" in tag:
            metadata["recording_datetime"] = value.strip()

    parts = video_id.rsplit("_", 2)
    if len(parts) >= 2:
        source_folder = "_".join(parts[:-2]) if len(parts) > 2 else parts[0]
        video_name = parts[-2] if len(parts) > 2 else parts[0]
    else:
        source_folder = video_id
        video_name = video_id

    return {
        "video_id": video_id,
        "source_folder": metadata.get("directory", "").split("/")[-1] if metadata.get("directory") else source_folder,
        "video_name": video_name,
        "original_video_filename": metadata.get("file_name", f"{video_name}.MP4"),
        "unique_video_filename": f"{video_id}.MP4",
        "file_size_bytes": metadata.get("file_size_bytes"),
        "file_size_mb": metadata["file_size_bytes"] / (1024 * 1024) if metadata.get("file_size_bytes") else None,
        "video_duration_sec": metadata.get("video_duration_sec"),
        "video_fps": metadata.get("video_fps"),
        "video_resolution_wxh": metadata.get("video_resolution_wxh"),
        "video_codec": metadata.get("video_codec"),
        "camera_model": metadata.get("camera_model"),
        "recording_datetime": metadata.get("recording_datetime"),
        "exif_filename": exif_file.name,
    }


def parse_gps_timeseries(exif_content: str, video_id: str) -> List[Dict]:
    """Parse GPS data from EXIF content."""
    gps_data = []

    gps_pattern = (
        r"\[GoPro\]\s+GPS Measure Mode\s+:\s+(.+?)\n"
        r"\[GoPro\]\s+GPS Latitude\s+:\s+(.+?)\n"
        r"\[GoPro\]\s+GPS Longitude\s+:\s+(.+?)\n"
        r"\[GoPro\]\s+GPS Altitude\s+:\s+(.+?)\n"
        r"\[GoPro\]\s+GPS Speed\s+:\s+(.+?)\n"
        r"\[GoPro\]\s+GPS Speed 3D\s+:\s+(.+?)\n"
        r"\[GoPro\]\s+GPS Date Time\s+:\s+(.+?)\n"
        r"(?:\[GoPro\]\s+GPSDOP\s+:\s+(.+?)\n)?"
    )

    matches = re.findall(gps_pattern, exif_content)

    for match in matches:
        measure_mode, latitude, longitude, altitude, speed, speed_3d, date_time = match[:7]
        gpsdop = match[7] if len(match) > 7 else None

        latitude = latitude.strip()
        longitude = longitude.strip()
        altitude = altitude.strip()
        speed = speed.strip()
        speed_3d = speed_3d.strip()
        date_time = date_time.strip()

        if not latitude or not longitude or latitude == "-" or longitude == "-":
            continue

        gps_point = {
            "video_id": video_id,
            "gps_datetime": date_time,
            "gps_latitude": latitude,
            "gps_longitude": longitude,
            "gps_altitude": altitude,
            "gps_speed": speed,
            "gps_speed_3d": speed_3d,
            "gps_measure_mode": measure_mode.strip(),
            "gpsdop": gpsdop.strip() if gpsdop else None,
        }

        gps_data.append(gps_point)

    return gps_data


def build_frame_metadata(frames_dir: Path, exif_files: List[Path]) -> List[Dict]:
    """Build frame metadata from existing frame files."""
    frame_meta = []

    video_id_to_folder = {}
    for exif_file in exif_files:
        video_id = exif_file.stem.replace("_exif", "")
        content = exif_file.read_text(encoding="utf-8", errors="ignore")
        for line in content.split("\n"):
            if "[File]" in line and "Directory" in line and ": " in line:
                directory = line.split(": ", 1)[1].strip()
                folder = directory.split("/")[-1]
                video_id_to_folder[video_id] = folder
                break

    frame_files = sorted(frames_dir.glob("*.jpg"))

    for frame_file in frame_files:
        name = frame_file.stem
        match = re.match(r"(.+)_frame(\d+)$", name)
        if not match:
            continue

        frame_id = match.group(1)
        frame_num = int(match.group(2))

        parts = frame_id.rsplit("_", 1)
        if len(parts) == 2:
            source_folder = parts[0]
            video_name = parts[1]
        else:
            source_folder = frame_id
            video_name = frame_id

        frame_meta.append({
            "video_id": frame_id,
            "source_folder": source_folder,
            "video_name": video_name,
            "original_video_filename": f"{video_name}.MP4",
            "frame_filename": frame_file.name,
            "frame_number": frame_num,
            "timestamp_sec": frame_num * 120,
            "timestamp_display": str(timedelta(seconds=frame_num * 120)),
        })

    return frame_meta


def rebuild_csvs_from_exif(city: str, project_root: Path) -> int:
    """Parse ``data/{city}/exif/*_exif.txt`` into the GPS-index input CSVs."""
    exif_dir = project_root / "data" / city / "exif"
    frames_dir = project_root / "data" / "annotation_task" / f"{city}_frames"
    exif_metadata_dir = project_root / "data" / city / "exif_metadata"
    output_dir = project_root / "data" / city

    if not exif_dir.exists():
        print(f"Error: EXIF directory not found: {exif_dir}")
        return 1

    exif_metadata_dir.mkdir(parents=True, exist_ok=True)

    exif_files = sorted(exif_dir.glob("*_exif.txt"))
    print(f"Found {len(exif_files)} EXIF files in {exif_dir}")

    all_video_meta = []
    all_gps = []

    for exif_file in tqdm(exif_files, desc="Parsing EXIF files"):
        content = exif_file.read_text(encoding="utf-8", errors="ignore")
        video_id = exif_file.stem.replace("_exif", "")

        meta = parse_video_metadata_from_exif(content, exif_file)
        if meta:
            all_video_meta.append(meta)

        gps_points = parse_gps_timeseries(content, video_id)
        all_gps.extend(gps_points)

    # Names/suffixes here must match the globs in 04_build_gps_index.py:
    # *_exif_video_metadata.csv and *_gps_timeseries*.parquet. The GPS timeseries is
    # written as Parquet (~10x smaller than CSV) and is a regenerable scratch file
    # (gitignored); the tiny video metadata stays CSV.
    exif_csv = exif_metadata_dir / f"{city}_exif_video_metadata.csv"
    gps_parquet = exif_metadata_dir / f"{city}_gps_timeseries.parquet"
    gps_csv_stale = exif_metadata_dir / f"{city}_gps_timeseries.csv"
    frame_csv = output_dir / "frame_metadata.csv"

    if all_video_meta:
        pd.DataFrame(all_video_meta).to_csv(exif_csv, index=False)
        print(f"Saved {len(all_video_meta)} video metadata to {exif_csv}")

    if all_gps:
        gps_df = pd.DataFrame(all_gps)
        gps_df = gps_df.sort_values(["video_id", "gps_datetime"])
        gps_df.to_parquet(gps_parquet, index=False)
        print(f"Saved {len(all_gps)} GPS points to {gps_parquet}")
        # Remove any stale CSV from the old format so 04's glob can't double-count.
        if gps_csv_stale.exists():
            gps_csv_stale.unlink()
            print(f"Removed stale {gps_csv_stale}")

    if frames_dir.exists():
        all_frames = build_frame_metadata(frames_dir, exif_files)
        if all_frames:
            pd.DataFrame(all_frames).to_csv(frame_csv, index=False)
            print(f"Saved {len(all_frames)} frame records to {frame_csv}")
    else:
        print(f"Frames directory not found ({frames_dir}); skipping frame_metadata.csv")

    print("\nDone!")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", required=True, help="City to process (e.g. bangalore, delhi)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    return rebuild_csvs_from_exif(args.city, project_root)


if __name__ == "__main__":
    sys.exit(main())

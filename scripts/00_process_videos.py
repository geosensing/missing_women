#!/usr/bin/env python3
"""
Consolidated video processing pipeline.

Processes each video completely before moving to the next:
1. Extract EXIF (if not exists) → save .txt
2. Parse GPS from EXIF → collect points
3. Extract frames (if not exist) → save .jpg files

After all videos:
4. Write exif_metadata.csv
5. Write gps_timeseries.csv
6. Write frame_metadata.csv

Usage:
    python scripts/00_process_videos.py \
        -i "/Volumes/Samsung USB/delhi" \
        -o data/delhi \
        --frames-dir data/annotation_task/delhi_frames \
        --every-seconds 120 \
        --quality 95
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


def generate_video_id(video_path: Path) -> str:
    """Generate unique video ID from path (with hash for EXIF deduplication)."""
    path_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
    return f"{video_path.parent.name}_{video_path.stem}_{path_hash}"


def generate_frame_id(video_path: Path) -> str:
    """Generate frame ID from path (no hash, matches original 02_extract_frames.py)."""
    return f"{video_path.parent.name}_{video_path.stem}"


def find_videos(input_dir: Path, extensions: List[str]) -> List[Path]:
    """Find all video files in directory."""
    videos = []
    for ext in extensions:
        videos.extend(input_dir.rglob(f"*{ext}"))
        videos.extend(input_dir.rglob(f"*{ext.upper()}"))

    videos = [v for v in videos if not v.name.startswith("._")]
    return sorted(set(videos))


def run_exiftool(video_path: Path, timeout: int = 180) -> Optional[str]:
    """Run exiftool on a video file."""
    try:
        cmd = [
            "exiftool",
            "-G",
            "-ee",
            "-api",
            "LargeFileSupport=1",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        return None


def parse_video_metadata(exif_content: str, video_path: Path, video_id: str) -> Dict:
    """Parse video metadata from EXIF content."""
    metadata = {
        "video_duration_sec": None,
        "video_fps": None,
        "video_resolution_wxh": None,
        "video_codec": None,
        "camera_model": None,
        "recording_datetime": None,
    }

    for line in exif_content.strip().split("\n"):
        if ": " not in line:
            continue

        tag_part, value = line.split(": ", 1)
        tag = tag_part.strip().lower()

        if "duration" in tag and ("quicktime" in tag or "composite" in tag):
            try:
                if ":" in value:
                    parts = value.split(":")
                    if len(parts) == 3:
                        h, m, s = parts
                        metadata["video_duration_sec"] = (
                            int(h) * 3600 + int(m) * 60 + float(s)
                        )
            except (ValueError, IndexError):
                pass

        elif "frame rate" in tag or "video frame rate" in tag:
            try:
                metadata["video_fps"] = float(value.split()[0])
            except (ValueError, IndexError):
                pass

        elif "image size" in tag or "video frame size" in tag:
            if "x" in value:
                metadata["video_resolution_wxh"] = value.strip()

        elif "compressor id" in tag or "codec" in tag:
            metadata["video_codec"] = value.strip()

        elif "camera model" in tag or "device name" in tag:
            metadata["camera_model"] = value.strip()

        elif "create date" in tag and "quicktime" in tag:
            metadata["recording_datetime"] = value.strip()

    try:
        stat = video_path.stat()
        file_size_bytes = stat.st_size
        file_created_at = datetime.fromtimestamp(stat.st_ctime)
        file_modified_at = datetime.fromtimestamp(stat.st_mtime)
    except OSError:
        file_size_bytes = None
        file_created_at = None
        file_modified_at = None

    return {
        "video_id": video_id,
        "source_folder": video_path.parent.name,
        "video_name": video_path.stem,
        "original_video_filename": video_path.name,
        "unique_video_filename": f"{video_path.parent.name}_{video_path.name}_{hashlib.md5(str(video_path).encode()).hexdigest()[:8]}",
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_bytes / (1024 * 1024) if file_size_bytes else None,
        "file_created_at": file_created_at,
        "file_modified_at": file_modified_at,
        "video_duration_sec": metadata.get("video_duration_sec"),
        "video_fps": metadata.get("video_fps"),
        "video_resolution_wxh": metadata.get("video_resolution_wxh"),
        "video_codec": metadata.get("video_codec"),
        "camera_model": metadata.get("camera_model"),
        "recording_datetime": metadata.get("recording_datetime"),
        "exif_filename": f"{video_id}_exif.txt",
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
        measure_mode, latitude, longitude, altitude, speed, speed_3d, date_time = match[
            :7
        ]
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


def parse_fraction(s: str) -> float:
    """Parse a fraction string to float."""
    try:
        return float(Fraction(s))
    except Exception:
        return 0.0


def ffprobe_video_info(video_path: Path) -> Tuple[float, float]:
    """Get video duration and fps using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration:stream=avg_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and res.stdout.strip():
        try:
            data = json.loads(res.stdout)
            duration = float(data.get("format", {}).get("duration") or 0.0)
            streams = data.get("streams") or []
            fps = (
                parse_fraction(streams[0].get("avg_frame_rate", "0/0"))
                if streams
                else 0.0
            )
            if duration > 0 and fps > 0:
                return duration, fps
        except Exception:
            pass

    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0.0, 0.0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        cap.release()
        duration = (frame_count / fps) if fps > 0 else 0.0
        return duration, fps
    except ImportError:
        return 0.0, 0.0


def jpeg_quality_to_ffmpeg_qv(quality: int) -> int:
    """Convert JPEG quality (1-100) to ffmpeg -q:v (1=best, 31=worst)."""
    q = max(1, min(100, int(quality)))
    qv = int((100 - q) / 3) + 1
    return max(1, min(31, qv))


def expected_frames_for_interval(duration_sec: float, every_seconds: float) -> int:
    """Calculate expected number of frames for time-based extraction."""
    if every_seconds <= 0:
        raise ValueError("every_seconds must be > 0")
    if duration_sec <= 0:
        return 1
    return int(duration_sec // every_seconds) + 1


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    frame_id: str,
    every_seconds: float,
    quality: int,
    scale_width: Optional[int] = None,
) -> List[Dict]:
    """Extract frames from a video using ffmpeg."""
    frames_dir.mkdir(parents=True, exist_ok=True)

    duration_sec, fps = ffprobe_video_info(video_path)
    if duration_sec <= 0:
        return []

    expected = expected_frames_for_interval(duration_sec, every_seconds)
    qv = jpeg_quality_to_ffmpeg_qv(quality)

    vf_string = f"scale={scale_width}:-1" if scale_width else None

    timestamps = [i * every_seconds for i in range(expected)]
    frame_meta = []

    for i, ts in enumerate(timestamps):
        output_file = frames_dir / f"{frame_id}_frame{i:06d}.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-ss",
            str(ts),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
        ]
        if vf_string:
            cmd.extend(["-vf", vf_string])
        cmd.extend(["-q:v", str(qv), str(output_file)])

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and output_file.exists():
            frame_num_est = int(round(ts * fps)) if fps > 0 else i
            time_display = str(timedelta(seconds=ts))

            frame_meta.append(
                {
                    "video_id": frame_id,
                    "source_folder": video_path.parent.name,
                    "video_name": video_path.stem,
                    "original_video_filename": video_path.name,
                    "frame_filename": output_file.name,
                    "frame_number": frame_num_est,
                    "timestamp_sec": ts,
                    "timestamp_display": time_display,
                    "video_fps": fps,
                    "video_duration_sec": duration_sec,
                }
            )

    return frame_meta


def build_frame_metadata_from_existing(
    existing_frames: List[Path],
    frame_id: str,
    video_path: Path,
    every_seconds: float,
) -> List[Dict]:
    """Build frame metadata from existing frame files."""
    duration_sec, fps = ffprobe_video_info(video_path)

    frame_meta = []
    sorted_frames = sorted(existing_frames, key=lambda p: p.name)

    for i, frame_path in enumerate(sorted_frames):
        ts = i * every_seconds
        frame_num_est = int(round(ts * fps)) if fps > 0 else i
        time_display = str(timedelta(seconds=ts))

        frame_meta.append(
            {
                "video_id": frame_id,
                "source_folder": video_path.parent.name,
                "video_name": video_path.stem,
                "original_video_filename": video_path.name,
                "frame_filename": frame_path.name,
                "frame_number": frame_num_est,
                "timestamp_sec": ts,
                "timestamp_display": time_display,
                "video_fps": fps,
                "video_duration_sec": duration_sec,
            }
        )

    return frame_meta


def process_video(
    video_path: Path,
    output_dir: Path,
    frames_dir: Path,
    every_seconds: float,
    quality: int,
    timeout: int = 180,
    scale_width: Optional[int] = None,
) -> Tuple[Optional[Dict], List[Dict], List[Dict]]:
    """Process one video completely: EXIF, GPS, frames."""
    video_id = generate_video_id(video_path)
    frame_id = generate_frame_id(video_path)
    exif_dir = output_dir / "exif"
    exif_dir.mkdir(parents=True, exist_ok=True)
    exif_file = exif_dir / f"{video_id}_exif.txt"

    if exif_file.exists():
        exif_content = exif_file.read_text(encoding="utf-8")
    else:
        exif_content = run_exiftool(video_path, timeout)
        if exif_content is None:
            return None, [], []
        exif_file.write_text(exif_content, encoding="utf-8")

    video_meta = parse_video_metadata(exif_content, video_path, video_id)
    gps_points = parse_gps_timeseries(exif_content, video_id)

    existing = list(frames_dir.glob(f"{frame_id}_frame*.jpg"))
    if not existing:
        frame_meta = extract_frames(
            video_path, frames_dir, frame_id, every_seconds, quality, scale_width
        )
    else:
        frame_meta = build_frame_metadata_from_existing(
            existing, frame_id, video_path, every_seconds
        )

    return video_meta, gps_points, frame_meta


def check_dependencies() -> bool:
    """Check that required tools are available."""
    try:
        subprocess.run(["exiftool", "-ver"], capture_output=True, check=True)
    except FileNotFoundError:
        print("Error: exiftool not found. Install with: brew install exiftool")
        return False

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("Error: ffmpeg not found. Install with: brew install ffmpeg")
        return False

    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("Error: ffprobe not found. Install with: brew install ffmpeg")
        return False

    return True


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process videos: extract EXIF, GPS, and frames"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input folder containing videos"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output folder for all data"
    )
    parser.add_argument(
        "--frames-dir",
        "-f",
        default=None,
        help="Output directory for frames (default: <output>/frames)",
    )
    parser.add_argument(
        "--every-seconds",
        "-s",
        type=float,
        default=120,
        help="Extract one frame every N seconds (default: 120)",
    )
    parser.add_argument(
        "--quality",
        "-q",
        type=int,
        default=95,
        help="JPEG quality 1-100 (default: 95)",
    )
    parser.add_argument(
        "--extensions",
        "-e",
        default=".mp4,.mov,.avi,.mkv",
        help="Comma-separated video extensions",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=180,
        help="Timeout per video for exiftool (seconds)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=None,
        help="Scale output width in pixels (height scales proportionally)",
    )

    args = parser.parse_args()

    if not check_dependencies():
        return 1

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    frames_dir = Path(args.frames_dir) if args.frames_dir else output_dir / "frames"
    extensions = [ext.strip() for ext in args.extensions.split(",")]

    if not input_dir.exists():
        print(f"Error: Input folder does not exist: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(input_dir, extensions)
    print(f"Found {len(videos)} video files")
    print(f"Frames output: {frames_dir}")

    if not videos:
        print("No video files found")
        return 0

    all_video_meta = []
    all_gps = []
    all_frames = []

    for video in tqdm(videos, desc="Processing videos"):
        meta, gps, frames = process_video(
            video,
            output_dir,
            frames_dir,
            args.every_seconds,
            args.quality,
            args.timeout,
            args.scale,
        )
        if meta:
            all_video_meta.append(meta)
        all_gps.extend(gps)
        all_frames.extend(frames)

    exif_csv = output_dir / "exif_metadata.csv"
    gps_csv = output_dir / "gps_timeseries.csv"
    frame_csv = output_dir / "frame_metadata.csv"

    if all_video_meta:
        pd.DataFrame(all_video_meta).to_csv(exif_csv, index=False)
        print(f"Saved {len(all_video_meta)} video metadata to {exif_csv}")

    if all_gps:
        gps_df = pd.DataFrame(all_gps)
        gps_df = gps_df.sort_values(["video_id", "gps_datetime"])
        gps_df.to_csv(gps_csv, index=False)
        print(f"Saved {len(all_gps)} GPS points to {gps_csv}")
    else:
        pd.DataFrame().to_csv(gps_csv, index=False)
        print("No GPS data found")

    if all_frames:
        pd.DataFrame(all_frames).to_csv(frame_csv, index=False)
        print(f"Saved {len(all_frames)} frame records to {frame_csv}")

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Videos processed: {len(all_video_meta)}")
    print(f"GPS points: {len(all_gps)}")
    print(f"Frames: {len(all_frames)}")
    print(f"Output directory: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

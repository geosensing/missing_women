#!/usr/bin/env python3
"""
Extract frames from videos at GoPro face detection timestamps.

Parses EXIF metadata for face detection events and extracts those frames.
Filters by confidence level to keep only high-quality detections.

Inputs:
    - EXIF directory: Text files with GoPro metadata (from 00_process_videos.py)
    - Video directory: Original MP4 files organized by day folder

Outputs:
    - Face frames: JPEG images at face detection timestamps
    - Metadata CSV: Log of all extracted frames with confidence scores

Usage:
    python scripts/01_extract_face_frames.py \\
        -e data/delhi/exif \\
        -v "/Volumes/Samsung USB/delhi" \\
        -o data/annotation_task/delhi_face_frames \\
        -l data/delhi/face_frame_metadata.csv \\
        --min-confidence 90
"""

import argparse
import csv
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def check_dependencies() -> bool:
    """Check that ffmpeg is available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, check=True
        )
        return True
    except FileNotFoundError:
        print("Error: ffmpeg not found. Install with: brew install ffmpeg")
        return False


def parse_exif_for_faces(exif_path: Path) -> list[dict]:
    """
    Parse EXIF file for face detection entries.

    Returns list of dicts with: index, timestamp_sec, face_count, max_confidence
    """
    face_pattern = re.compile(
        r"\[GoPro\]\s+Face Detected\s+:\s+"
        r"(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)"
    )
    timestamp_pattern = re.compile(r"\[GoPro\]\s+Time Stamp\s+:\s+([\d.]+)")

    face_events = {}
    current_face_info = None

    with open(exif_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            face_match = face_pattern.search(line)
            if face_match:
                _, face_count, index, confidence = face_match.groups()
                index = int(index)
                face_count = int(face_count)
                confidence = int(confidence)

                if index not in face_events:
                    face_events[index] = {
                        "index": index,
                        "face_count": face_count,
                        "max_confidence": confidence,
                        "timestamp_sec": None,
                    }
                else:
                    face_events[index]["face_count"] = max(
                        face_events[index]["face_count"], face_count
                    )
                    face_events[index]["max_confidence"] = max(
                        face_events[index]["max_confidence"], confidence
                    )
                current_face_info = index
                continue

            if current_face_info is not None:
                ts_match = timestamp_pattern.search(line)
                if ts_match:
                    timestamp = float(ts_match.group(1))
                    if face_events[current_face_info]["timestamp_sec"] is None:
                        face_events[current_face_info]["timestamp_sec"] = timestamp
                    current_face_info = None

    valid_events = [e for e in face_events.values() if e["timestamp_sec"] is not None]
    return sorted(valid_events, key=lambda x: x["index"])


def parse_exif_filename(exif_path: Path) -> tuple[str, str, str]:
    """
    Parse EXIF filename to extract day folder, video name, and frame_id.

    EXIF filename format: day_1_4_28_2026_11.15_5856a1d1_exif.txt
                      or: day_3_4_30_2026_Itinerary_37.1_d7470e01_exif.txt
    Returns: (day_folder, video_name, frame_id)

    frame_id is without hash to avoid duplicates from same video.
    """
    name = exif_path.stem
    if not name.endswith("_exif"):
        raise ValueError(f"Invalid EXIF filename format: {exif_path.name}")

    name = name[:-5]  # Remove _exif

    # Extract day_folder pattern from the start
    match = re.match(r"(day_\d+_\d+_\d+_\d+)_(.+)_([a-f0-9]{8})$", name)
    if not match:
        raise ValueError(f"Cannot parse EXIF filename: {exif_path.name}")

    day_folder = match.group(1)
    video_name = match.group(2)  # Everything between day_folder and hash
    # hash_id = match.group(3)  # Not used - we use frame_id without hash

    frame_id = f"{day_folder}_{video_name}"

    return day_folder, video_name, frame_id


def extract_frame(
    video_path: Path,
    timestamp_sec: float,
    output_path: Path,
    scale_width: int = 1020,
) -> bool:
    """
    Extract a single frame from video at given timestamp using ffmpeg.

    Uses -ss before -i for fast seeking.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        str(timestamp_sec),
        "-i",
        str(video_path),
        "-vframes",
        "1",
        "-vf",
        f"scale={scale_width}:-1",
        "-q:v",
        "2",
        "-y",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def process_exif_file(
    exif_path: Path,
    video_dir: Path,
    output_dir: Path,
    scale_width: int,
    min_confidence: int,
) -> list[dict]:
    """Process a single EXIF file and extract face frames."""
    results = []

    try:
        day_folder, video_name, frame_id = parse_exif_filename(exif_path)
    except ValueError as e:
        print(f"  Warning: {e}")
        return results

    video_path = video_dir / day_folder / f"{video_name}.MP4"
    if not video_path.exists():
        video_path = video_dir / day_folder / f"{video_name}.mp4"
        if not video_path.exists():
            print(f"  Warning: Video not found: {video_path}")
            return results

    face_events = parse_exif_for_faces(exif_path)

    if min_confidence > 0:
        face_events = [e for e in face_events if e["max_confidence"] >= min_confidence]

    for i, event in enumerate(face_events):
        frame_name = f"{frame_id}_face{i:06d}.jpg"
        output_path = output_dir / frame_name

        if output_path.exists():
            results.append(
                {
                    "video_id": frame_id,
                    "frame_file": frame_name,
                    "timestamp_sec": event["timestamp_sec"],
                    "face_count": event["face_count"],
                    "max_confidence": event["max_confidence"],
                    "status": "skipped",
                }
            )
            continue

        success = extract_frame(video_path, event["timestamp_sec"], output_path, scale_width)

        results.append(
            {
                "video_id": frame_id,
                "frame_file": frame_name,
                "timestamp_sec": event["timestamp_sec"],
                "face_count": event["face_count"],
                "max_confidence": event["max_confidence"],
                "status": "ok" if success else "error",
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames at face detection timestamps from GoPro videos"
    )
    parser.add_argument(
        "-e", "--exif-dir", required=True, help="Directory with EXIF txt files"
    )
    parser.add_argument(
        "-v", "--video-dir", required=True, help="Directory with source videos"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output directory for face frames"
    )
    parser.add_argument(
        "-l", "--log", help="Path for metadata CSV (default: derived from exif-dir)"
    )
    parser.add_argument(
        "-s", "--scale", type=int, default=1020, help="Output width (default: 1020)"
    )
    parser.add_argument(
        "-c",
        "--min-confidence",
        type=int,
        default=90,
        help="Minimum confidence filter (default: 90)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )

    args = parser.parse_args()

    if not check_dependencies():
        return 1

    exif_dir = Path(args.exif_dir)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output)

    if not exif_dir.exists():
        print(f"Error: EXIF directory does not exist: {exif_dir}")
        return 1

    if not video_dir.exists():
        print(f"Error: Video directory does not exist: {video_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    exif_files = sorted(exif_dir.glob("*_exif.txt"))
    print(f"Found {len(exif_files)} EXIF files (using {args.workers} workers)")

    all_results = []
    results_lock = threading.Lock()
    completed = [0]
    total_files = len(exif_files)

    def process_one(exif_path: Path) -> list[dict]:
        return process_exif_file(
            exif_path, video_dir, output_dir, args.scale, args.min_confidence
        )

    if args.workers == 1:
        for i, exif_path in enumerate(exif_files, 1):
            print(f"[{i}/{total_files}] Processing: {exif_path.name}")
            results = process_one(exif_path)
            all_results.extend(results)
            ok_count = sum(1 for r in results if r["status"] == "ok")
            skip_count = sum(1 for r in results if r["status"] == "skipped")
            err_count = sum(1 for r in results if r["status"] == "error")
            print(f"    Extracted: {ok_count}, Skipped: {skip_count}, Errors: {err_count}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_exif = {executor.submit(process_one, ep): ep for ep in exif_files}
            for future in as_completed(future_to_exif):
                exif_path = future_to_exif[future]
                results = future.result()
                with results_lock:
                    all_results.extend(results)
                    completed[0] += 1
                    ok_count = sum(1 for r in results if r["status"] == "ok")
                    skip_count = sum(1 for r in results if r["status"] == "skipped")
                    err_count = sum(1 for r in results if r["status"] == "error")
                    print(f"[{completed[0]}/{total_files}] {exif_path.name}: "
                          f"ok={ok_count}, skip={skip_count}, err={err_count}")

    if args.log:
        csv_path = Path(args.log)
    else:
        csv_path = exif_dir.parent / "face_frame_metadata.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "frame_file",
                "timestamp_sec",
                "face_count",
                "max_confidence",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    total = len(all_results)
    ok_count = sum(1 for r in all_results if r["status"] == "ok")
    skip_count = sum(1 for r in all_results if r["status"] == "skipped")
    err_count = sum(1 for r in all_results if r["status"] == "error")
    print(f"Total face events: {total}")
    print(f"  Extracted: {ok_count}")
    print(f"  Skipped:   {skip_count}")
    print(f"  Errors:    {err_count}")
    print(f"\nCSV log saved to: {csv_path}")
    print(f"Frames saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

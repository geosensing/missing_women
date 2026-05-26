#!/usr/bin/env python3
"""
Extract frames from videos at timestamps where GoPro detected faces.

Parses GoPro EXIF metadata to find face detection events and extracts
those specific frames using ffmpeg.
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


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
    Parse EXIF filename to extract day folder, video name, and video_id.

    EXIF filename format: day_1_4_28_2026_11.15_5856a1d1_exif.txt
    Returns: (day_folder, video_name, video_id)
    """
    name = exif_path.stem
    if not name.endswith("_exif"):
        raise ValueError(f"Invalid EXIF filename format: {exif_path.name}")

    name = name[:-5]

    parts = name.rsplit("_", 2)
    if len(parts) < 3:
        raise ValueError(f"Cannot parse EXIF filename: {exif_path.name}")

    day_video_part, video_name, hash_id = parts[0], parts[1], parts[2]

    match = re.match(r"(day_\d+_\d+_\d+_\d+)", day_video_part)
    if not match:
        raise ValueError(f"Cannot extract day folder from: {exif_path.name}")

    day_folder = match.group(1)
    video_id = f"{day_folder}_{video_name}_{hash_id}"

    return day_folder, video_name, video_id


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
        day_folder, video_name, video_id = parse_exif_filename(exif_path)
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

    for event in face_events:
        timestamp_ms = int(event["timestamp_sec"] * 1000)
        frame_name = f"{video_id}_face_{event['index']:03d}_{timestamp_ms:08d}.jpg"
        output_path = output_dir / frame_name

        if output_path.exists():
            results.append(
                {
                    "video_id": video_id,
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
                "video_id": video_id,
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
        "--exif-dir", required=True, help="Directory with EXIF txt files"
    )
    parser.add_argument(
        "--video-dir", required=True, help="Directory with source videos"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for face frames"
    )
    parser.add_argument(
        "--scale", type=int, default=1020, help="Output width (default: 1020)"
    )
    parser.add_argument(
        "--min-confidence",
        type=int,
        default=0,
        help="Minimum confidence filter (default: 0, i.e., all)",
    )

    args = parser.parse_args()

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
    print(f"Found {len(exif_files)} EXIF files")

    all_results = []

    for i, exif_path in enumerate(exif_files, 1):
        print(f"[{i}/{len(exif_files)}] Processing: {exif_path.name}")

        results = process_exif_file(
            exif_path, video_dir, output_dir, args.scale, args.min_confidence
        )
        all_results.extend(results)

        ok_count = sum(1 for r in results if r["status"] == "ok")
        skip_count = sum(1 for r in results if r["status"] == "skipped")
        err_count = sum(1 for r in results if r["status"] == "error")
        print(f"    Extracted: {ok_count}, Skipped: {skip_count}, Errors: {err_count}")

    csv_path = output_dir / "face_frames_log.csv"
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

    return 1 if err_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

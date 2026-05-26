#!/usr/bin/env python3
"""
Extract frames from videos using ffmpeg with time-based sampling.

Features:
- Time-based extraction (every X seconds) using ffmpeg fps filter
- Per-video output subfolder to prevent filename collisions
- Manifest files to track completion and allow re-extraction with different params
- Expected vs extracted validation with configurable tolerance
- CSV report with per-video statistics
"""

import argparse
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parse_fraction(s: str) -> float:
    try:
        return float(Fraction(s))
    except Exception:
        return 0.0


def ffprobe_video_info(video_path: Path) -> tuple[float, float]:
    """Get video duration and fps using ffprobe, with OpenCV fallback."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration:stream=avg_frame_rate",
        "-of", "json",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and res.stdout.strip():
        try:
            data = json.loads(res.stdout)
            duration = float(data.get("format", {}).get("duration") or 0.0)
            streams = data.get("streams") or []
            fps = parse_fraction(streams[0].get("avg_frame_rate", "0/0")) if streams else 0.0
            if duration > 0 and fps > 0:
                return duration, fps
        except Exception:
            pass

    # Fallback: OpenCV
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


def expected_frames_for_interval(duration_sec: float, every_seconds: float) -> int:
    """Calculate expected number of frames for time-based extraction."""
    if every_seconds <= 0:
        raise ValueError("every_seconds must be > 0")
    if duration_sec <= 0:
        return 1
    return int(duration_sec // every_seconds) + 1


def jpeg_quality_to_ffmpeg_qv(quality: int) -> int:
    """Convert JPEG quality (1-100) to ffmpeg -q:v (1=best, 31=worst)."""
    q = max(1, min(100, int(quality)))
    qv = int((100 - q) / 3) + 1
    return max(1, min(31, qv))


def extract_frames_ffmpeg(
    video_path: Path,
    output_folder: Path,
    every_seconds: float,
    quality: int = 95,
    tolerance_pct: float = 0.05,
    tolerance_abs: int = 1,
    overwrite: bool = False,
    scale_width: int | None = None,
) -> dict:
    """
    Extract frames from a video using fast seeking.
    Uses -ss before -i for keyframe seeking (much faster than fps filter).
    """
    video_name = video_path.stem
    source_folder = video_path.parent.name
    video_id = f"{source_folder}_{video_name}"

    output_folder.mkdir(parents=True, exist_ok=True)

    # Check if already extracted by counting existing frames with this prefix
    existing_frames = list(output_folder.glob(f"{video_id}_frame*.jpg"))
    if existing_frames and not overwrite:
        duration_sec, fps = ffprobe_video_info(video_path)
        return {
            "video_id": video_id,
            "video_path": str(video_path),
            "status": "skipped",
            "expected_frames": expected_frames_for_interval(duration_sec, every_seconds) if duration_sec > 0 else len(existing_frames),
            "extracted_frames": len(existing_frames),
            "duration_sec": duration_sec,
            "fps": fps,
        }

    # Clean up if overwriting
    if overwrite:
        for f in existing_frames:
            f.unlink()

    # Get video info
    duration_sec, fps = ffprobe_video_info(video_path)
    if duration_sec <= 0:
        return {
            "video_id": video_id,
            "video_path": str(video_path),
            "status": "error",
            "error": "Could not determine video duration",
            "expected_frames": 0,
            "extracted_frames": 0,
            "duration_sec": 0,
            "fps": 0,
        }

    expected = expected_frames_for_interval(duration_sec, every_seconds)
    qv = jpeg_quality_to_ffmpeg_qv(quality)

    # Build video filter
    vf_string = f"scale={scale_width}:-1" if scale_width else None

    # Fast seek-based extraction: jump to each timestamp directly
    timestamps = [i * every_seconds for i in range(expected)]
    extracted = 0
    errors = []

    for i, ts in enumerate(timestamps):
        output_file = output_folder / f"{video_id}_frame{i:06d}.jpg"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", str(ts),
            "-i", str(video_path),
            "-frames:v", "1",
        ]
        if vf_string:
            cmd.extend(["-vf", vf_string])
        cmd.extend(["-q:v", str(qv), str(output_file)])

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and output_file.exists():
            extracted += 1
        else:
            errors.append(f"ts={ts}: {res.stderr.strip()}")

    if extracted == 0:
        return {
            "video_id": video_id,
            "video_path": str(video_path),
            "status": "error",
            "error": f"No frames extracted. Errors: {errors[:3]}",
            "expected_frames": expected,
            "extracted_frames": 0,
            "duration_sec": duration_sec,
            "fps": fps,
        }

    # Count extracted frames
    frame_files = list(output_folder.glob(f"{video_id}_frame*.jpg"))
    extracted = len(frame_files)

    # Validate count
    allowed = max(int(tolerance_abs), int(round(expected * tolerance_pct)))
    status = "ok"
    error_msg = None

    if abs(extracted - expected) > allowed:
        status = "warning"
        error_msg = f"Expected ~{expected} frames but got {extracted} (allowed diff: {allowed})"

    return {
        "video_id": video_id,
        "video_path": str(video_path),
        "status": status,
        "error": error_msg,
        "expected_frames": expected,
        "extracted_frames": extracted,
        "duration_sec": duration_sec,
        "fps": fps,
    }


def find_videos(input_folder: Path, extensions: list[str]) -> list[Path]:
    """Find all video files in folder, excluding macOS metadata files."""
    all_files = []
    for ext in extensions:
        all_files.extend(input_folder.rglob(f"*{ext}"))
        all_files.extend(input_folder.rglob(f"*{ext.upper()}"))

    # Filter out macOS metadata and deduplicate
    all_files = list(set(f for f in all_files if not f.name.startswith("._")))
    all_files.sort()
    return all_files


def write_log_entries(
    log_file_path: Path,
    video_path: Path,
    video_id: str,
    output_folder: Path,
    every_seconds: float,
    fps: float,
    duration_sec: float,
):
    """Write frame metadata to log file."""
    frame_files = sorted(output_folder.glob(f"{video_id}_frame*.jpg"))
    source_folder = video_path.parent.name
    video_name = video_path.stem

    lines = []
    for i, frame_path in enumerate(frame_files):
        ts = i * every_seconds
        frame_num_est = int(round(ts * fps)) if fps > 0 else i
        time_display = str(timedelta(seconds=ts))

        lines.append(
            f"{video_id}|{source_folder}|{video_name}|{video_path.name}|"
            f"{frame_path.name}|{frame_num_est}|{ts}|{time_display}|{fps}|{duration_sec}\n"
        )

    with open(log_file_path, "a", encoding="utf-8") as f:
        f.writelines(lines)


def process_videos(
    input_folder: Path,
    output_folder: Path,
    every_seconds: float,
    log_file_path: Path,
    quality: int = 95,
    extensions: list[str] | None = None,
    overwrite: bool = False,
    scale_width: int | None = None,
    workers: int = 1,
) -> list[dict]:
    """Process all videos in folder with optional parallel processing."""
    if extensions is None:
        extensions = [".mp4", ".mov", ".avi", ".mkv"]

    output_folder.mkdir(parents=True, exist_ok=True)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    videos = find_videos(input_folder, extensions)
    print(f"Found {len(videos)} video files (using {workers} worker(s))\n")

    results = []
    total_frames = 0
    ok_count = 0
    skipped_count = 0
    error_count = 0
    log_lock = threading.Lock()

    def process_one(video_path: Path) -> dict:
        return extract_frames_ffmpeg(
            video_path,
            output_folder,
            every_seconds,
            quality=quality,
            overwrite=overwrite,
            scale_width=scale_width,
        )

    pbar = tqdm(total=len(videos), desc="Extracting frames", unit="video")

    if workers == 1:
        for video_path in videos:
            pbar.set_postfix_str(f"{video_path.name[:30]}...")
            result = process_one(video_path)
            results.append(result)

            total_frames += result["extracted_frames"]
            if result["status"] in ("ok", "warning"):
                ok_count += 1
            elif result["status"] == "skipped":
                skipped_count += 1
            else:
                error_count += 1

            pbar.set_description(
                f"Frames: {total_frames} | OK: {ok_count} | Skip: {skipped_count} | Err: {error_count}"
            )

            if result["status"] in ("ok", "warning") and result["extracted_frames"] > 0:
                write_log_entries(
                    log_file_path,
                    video_path,
                    result["video_id"],
                    output_folder,
                    every_seconds,
                    result["fps"],
                    result["duration_sec"],
                )

            if result["status"] == "error":
                tqdm.write(f"ERROR: {video_path.name} - {result.get('error', 'Unknown error')}")

            pbar.update(1)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_video = {executor.submit(process_one, v): v for v in videos}
            for future in as_completed(future_to_video):
                video_path = future_to_video[future]
                result = future.result()
                results.append(result)

                total_frames += result["extracted_frames"]
                if result["status"] in ("ok", "warning"):
                    ok_count += 1
                elif result["status"] == "skipped":
                    skipped_count += 1
                else:
                    error_count += 1

                pbar.set_description(
                    f"Frames: {total_frames} | OK: {ok_count} | Skip: {skipped_count} | Err: {error_count}"
                )

                if result["status"] in ("ok", "warning") and result["extracted_frames"] > 0:
                    with log_lock:
                        write_log_entries(
                            log_file_path,
                            video_path,
                            result["video_id"],
                            output_folder,
                            every_seconds,
                            result["fps"],
                            result["duration_sec"],
                        )

                if result["status"] == "error":
                    tqdm.write(f"ERROR: {video_path.name} - {result.get('error', 'Unknown error')}")

                pbar.update(1)

    pbar.close()
    return results


def print_report(results: list[dict]):
    """Print summary report to console."""
    print("\n" + "=" * 80)
    print("EXTRACTION REPORT")
    print("=" * 80)

    df = pd.DataFrame(results)

    # Summary stats
    total = len(df)
    ok_count = len(df[df["status"] == "ok"])
    warning_count = len(df[df["status"] == "warning"])
    skipped_count = len(df[df["status"] == "skipped"])
    error_count = len(df[df["status"] == "error"])

    print(f"\nSummary: {total} videos")
    print(f"  OK:       {ok_count}")
    print(f"  Warnings: {warning_count}")
    print(f"  Skipped:  {skipped_count}")
    print(f"  Errors:   {error_count}")

    # Per-video details
    print("\nPer-video details:")
    print("-" * 80)

    display_cols = ["video_path", "duration_sec", "fps", "expected_frames", "extracted_frames", "status"]
    display_df = df[display_cols].copy()
    display_df["video_path"] = display_df["video_path"].apply(lambda x: Path(x).name)
    display_df["duration_sec"] = display_df["duration_sec"].apply(lambda x: f"{x:.1f}")
    display_df["fps"] = display_df["fps"].apply(lambda x: f"{x:.2f}")
    display_df.columns = ["video", "duration", "fps", "expected", "extracted", "status"]

    print(display_df.to_string(index=False))

    # Totals
    total_expected = df["expected_frames"].sum()
    total_extracted = df["extracted_frames"].sum()
    total_duration = df["duration_sec"].sum()

    print("-" * 80)
    print(f"TOTALS: {total_duration:.1f}s duration, {total_expected} expected, {total_extracted} extracted")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from videos using ffmpeg with time-based sampling"
    )
    parser.add_argument("--input", "-i", required=True, help="Input folder containing videos")
    parser.add_argument("--output", "-o", required=True, help="Output folder for frames")
    parser.add_argument("--every-seconds", "-s", type=float, default=60, help="Extract one frame every N seconds (default: 60)")
    parser.add_argument("--log", "-l", required=True, help="Path for metadata log file")
    parser.add_argument("--report", "-r", required=True, help="Path for CSV report")
    parser.add_argument("--quality", "-q", type=int, default=95, help="JPEG quality 1-100 (default: 95)")
    parser.add_argument("--scale", type=int, default=None, help="Scale output width in pixels (height scales proportionally)")
    parser.add_argument("--extensions", "-e", default=".mp4,.mov,.avi,.mkv", help="Comma-separated extensions")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if already done")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Number of parallel workers (default: 1)")

    args = parser.parse_args()

    input_folder = Path(args.input)
    output_folder = Path(args.output)
    log_file_path = Path(args.log)
    report_path = Path(args.report)
    extensions = args.extensions.split(",")

    if not input_folder.exists():
        print(f"Error: Input folder does not exist: {input_folder}")
        return 1

    results = process_videos(
        input_folder,
        output_folder,
        args.every_seconds,
        log_file_path,
        quality=args.quality,
        extensions=extensions,
        overwrite=args.overwrite,
        scale_width=args.scale,
        workers=args.workers,
    )

    # Save CSV report
    df = pd.DataFrame(results)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_path, index=False)
    print(f"\nCSV report saved to: {report_path}")

    # Print report
    print_report(results)

    # Exit with error if any failures
    if any(r["status"] == "error" for r in results):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

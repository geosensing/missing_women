"""
Assign GPS coordinates to frames via temporal interpolation.

For each annotated frame:
1. Look up video recording_datetime (or GPS first timestamp if clock drift)
2. Compute frame_datetime = start_time + timestamp_sec
3. Find bracketing GPS points in timeseries
4. Linear interpolation if gap < 30 sec

This module does NOT persist output - the DataFrame is passed to the next step.

Inputs:
    annotations DataFrame (from 05_parse_annotations)
    data/{city}/gps_index/video_metadata.parquet
    data/{city}/gps_index/gps_timeseries.parquet

Returns:
    DataFrame with added columns: gps_lat, gps_lon, gps_alt, gps_time_diff_sec
"""

import re
from datetime import timedelta
from pathlib import Path

import pandas as pd

MAX_GPS_GAP_SEC = 30.0

# Default frame-timing convention (seconds of video time per frame_number unit).
# 1/30 means frame_number is a 30 fps video-frame index. Per-city values come from
# cities.yaml (see compute_frame_datetime / assign_frame_gps).
DEFAULT_INTERVAL_SEC = 1.0 / 30.0
DEFAULT_ANCHOR = "gps"


def compute_frame_datetime(
    row: pd.Series,
    video_meta: pd.DataFrame,
    interval_sec: float = DEFAULT_INTERVAL_SEC,
    anchor: str = DEFAULT_ANCHOR,
) -> pd.Timestamp | None:
    """
    Compute absolute datetime for a frame.

    The in-video offset is ``timestamp_sec`` when the filename carries one, else
    ``frame_number * interval_sec``. ``interval_sec`` encodes the per-batch frame
    convention: 1/30 for a 30 fps video-frame index (mumbai, navi_mumbai) or the
    extraction interval in seconds for sampled frames (e.g. 120 for bangalore,
    delhi, sampled every 2 min).

    ``anchor`` picks the wall-clock origin: ``"recording"`` uses the camera clock
    (correct when it is GPS-aligned, e.g. mumbai, where GPS logging starts well
    before the recording), ``"gps"`` uses the first GPS timestamp (when the camera
    clock is offset from GPS, e.g. navi_mumbai, bangalore, delhi). Each falls back
    to the other when its preferred field is missing.
    """
    base_video_id = row.get("base_video_id")
    timestamp_sec = row.get("timestamp_sec")
    frame_number = row.get("frame_number")

    if pd.isna(base_video_id):
        return None

    video_row = video_meta[video_meta["base_video_id"] == base_video_id]
    if video_row.empty:
        # Escape special regex chars in base_video_id for safe contains match
        escaped = re.escape(str(base_video_id))
        video_row = video_meta[video_meta["video_id"].str.contains(escaped, na=False, regex=True)]
    if video_row.empty:
        return None

    video_row = video_row.iloc[0]

    gps_first = video_row.get("gps_first_datetime")
    recording = video_row.get("recording_datetime")

    if anchor == "recording":
        start_time = recording if pd.notna(recording) else gps_first
    else:
        start_time = gps_first if pd.notna(gps_first) else recording
    if pd.isna(start_time):
        return None

    if pd.notna(timestamp_sec):
        offset_sec = timestamp_sec
    elif pd.notna(frame_number):
        offset_sec = frame_number * interval_sec
    else:
        return None

    return start_time + timedelta(seconds=offset_sec)


def interpolate_gps(frame_dt: pd.Timestamp, video_gps: pd.DataFrame) -> dict:
    """
    Interpolate GPS coordinates at frame_dt.

    Uses linear interpolation between bracketing GPS points.
    Returns None if gap > MAX_GPS_GAP_SEC.
    """
    result = {
        "gps_lat": None,
        "gps_lon": None,
        "gps_alt": None,
        "gps_time_diff_sec": None,
    }

    if video_gps.empty or pd.isna(frame_dt):
        return result

    before = video_gps[video_gps["gps_datetime"] <= frame_dt]
    after = video_gps[video_gps["gps_datetime"] >= frame_dt]

    if before.empty and after.empty:
        return result

    if before.empty:
        closest = after.iloc[0]
        diff = (closest["gps_datetime"] - frame_dt).total_seconds()
        if abs(diff) <= MAX_GPS_GAP_SEC:
            result["gps_lat"] = closest["lat"]
            result["gps_lon"] = closest["lon"]
            result["gps_alt"] = closest["alt"]
            result["gps_time_diff_sec"] = diff
        return result

    if after.empty:
        closest = before.iloc[-1]
        diff = (frame_dt - closest["gps_datetime"]).total_seconds()
        if abs(diff) <= MAX_GPS_GAP_SEC:
            result["gps_lat"] = closest["lat"]
            result["gps_lon"] = closest["lon"]
            result["gps_alt"] = closest["alt"]
            result["gps_time_diff_sec"] = -diff
        return result

    pt_before = before.iloc[-1]
    pt_after = after.iloc[0]

    t_before = pt_before["gps_datetime"]
    t_after = pt_after["gps_datetime"]

    gap = (t_after - t_before).total_seconds()
    if gap > MAX_GPS_GAP_SEC:
        return result

    if gap == 0:
        result["gps_lat"] = pt_before["lat"]
        result["gps_lon"] = pt_before["lon"]
        result["gps_alt"] = pt_before["alt"]
        result["gps_time_diff_sec"] = 0.0
        return result

    t_frame = (frame_dt - t_before).total_seconds()
    ratio = t_frame / gap

    result["gps_lat"] = pt_before["lat"] + ratio * (pt_after["lat"] - pt_before["lat"])
    result["gps_lon"] = pt_before["lon"] + ratio * (pt_after["lon"] - pt_before["lon"])

    if pd.notna(pt_before["alt"]) and pd.notna(pt_after["alt"]):
        result["gps_alt"] = pt_before["alt"] + ratio * (pt_after["alt"] - pt_before["alt"])

    result["gps_time_diff_sec"] = min(t_frame, gap - t_frame)

    return result


def assign_frame_gps(
    annotations: pd.DataFrame,
    video_meta: pd.DataFrame,
    gps_df: pd.DataFrame,
    interval_sec: float = DEFAULT_INTERVAL_SEC,
    anchor: str = DEFAULT_ANCHOR,
) -> pd.DataFrame:
    """
    Assign GPS coordinates and a wall-clock datetime to all annotated frames.

    ``frame_datetime`` is computed for every frame with a resolvable video and
    anchor (independent of whether a GPS fix is found), so the temporal fields
    downstream are populated even where GPS coverage is incomplete.

    Args:
        annotations: Parsed annotation DataFrame
        video_meta: Video metadata with recording times
        gps_df: GPS timeseries sorted by (video_id, gps_datetime)
        interval_sec: Seconds of video time per frame_number unit (per-batch).
        anchor: "recording" or "gps" wall-clock origin (per-batch).

    Returns:
        annotations with added GPS columns
    """
    result = annotations.copy()
    result["frame_datetime"] = None
    result["gps_lat"] = None
    result["gps_lon"] = None
    result["gps_alt"] = None
    result["gps_time_diff_sec"] = None

    # Create lookup from base_video_id -> video_id
    # base_video_id should be unique now (e.g., "1_itinerary_1_1", "day13_itinerary_1")
    video_id_map = dict(zip(video_meta["base_video_id"], video_meta["video_id"]))

    gps_by_video = {vid: grp for vid, grp in gps_df.groupby("video_id")}

    print(f"Assigning GPS to {len(result)} frames...")
    n_matched = 0

    for idx, row in result.iterrows():
        frame_dt = compute_frame_datetime(row, video_meta, interval_sec, anchor)
        if frame_dt is None:
            continue

        result.at[idx, "frame_datetime"] = frame_dt

        base_video_id = row["base_video_id"]
        video_id = video_id_map.get(base_video_id)

        if video_id is None:
            for vid in video_id_map.values():
                if base_video_id in vid:
                    video_id = vid
                    break

        if video_id is None or video_id not in gps_by_video:
            continue

        video_gps = gps_by_video[video_id]
        gps_result = interpolate_gps(frame_dt, video_gps)

        for col, val in gps_result.items():
            result.at[idx, col] = val

        if gps_result["gps_lat"] is not None:
            n_matched += 1

    result["frame_datetime"] = pd.to_datetime(result["frame_datetime"])

    print(
        f"  Matched GPS for {n_matched}/{len(result)} frames ({100 * n_matched / len(result):.1f}%)"
    )

    return result


if __name__ == "__main__":
    import argparse
    import importlib.util

    def import_from_path(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, help="City to process")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    scripts_dir = Path(__file__).parent

    parse_annotations_mod = import_from_path(
        "parse_annotations", scripts_dir / "05_parse_annotations.py"
    )
    parse_all_annotations = parse_annotations_mod.parse_all_annotations

    labelstudio_dir = project_root / "data" / args.city / "labelstudio"
    gps_index_dir = project_root / "data" / args.city / "gps_index"

    annotations = parse_all_annotations(labelstudio_dir, args.city)

    video_meta = pd.read_parquet(gps_index_dir / "video_metadata.parquet")
    gps_df = pd.read_parquet(gps_index_dir / "gps_timeseries.parquet")

    result = assign_frame_gps(annotations, video_meta, gps_df)

    print("\n=== Summary ===")
    print(f"Total frames: {len(result)}")
    print(f"With GPS: {result['gps_lat'].notna().sum()}")
    print(f"Mean GPS time diff: {result['gps_time_diff_sec'].mean():.2f} sec")

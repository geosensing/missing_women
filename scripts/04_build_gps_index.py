"""
Build GPS lookup data structures for frame-level GPS assignment.

Consolidates EXIF metadata from batch extractions into efficient lookup structures.
GPS timeseries is converted from DMS to decimal degrees and sorted for merge_asof.

Why persist this?
    GPS timeseries is ~1M rows. Building once avoids re-processing when
    iterating on annotation parsing or derived fields.

Inputs:
    data/{city}/exif_metadata/*_exif_video_metadata.csv
    data/{city}/exif_metadata/*_gps_timeseries*.csv

Outputs (persisted):
    data/{city}/gps_index/video_metadata.parquet
        video_id, base_video_id, recording_datetime, gps_first_datetime,
        use_gps_time, video_duration_sec

    data/{city}/gps_index/gps_timeseries.parquet
        video_id, gps_datetime, lat, lon, alt
        Sorted by (video_id, gps_datetime) for efficient merge_asof
"""

import re
from pathlib import Path

import pandas as pd


def dms_to_decimal(dms_str: str) -> float | None:
    """
    Convert DMS string to decimal degrees.

    Examples:
        "19 deg 6' 39.28\" N" → 19.110911
        "73 deg 0' 22.27\" E" → 73.006186
    """
    if pd.isna(dms_str) or not dms_str:
        return None

    pattern = r"""(\d+)\s*deg\s*(\d+)'\s*([\d.]+)"\s*([NSEW])"""
    match = re.match(pattern, str(dms_str).strip())
    if not match:
        return None

    deg, minutes, sec, direction = match.groups()
    decimal = float(deg) + float(minutes) / 60 + float(sec) / 3600

    if direction in ("S", "W"):
        decimal = -decimal

    return decimal


def parse_altitude(alt_str: str) -> float | None:
    """Parse altitude string like '11.815 m' to float."""
    if pd.isna(alt_str) or not alt_str:
        return None
    match = re.match(r"([-\d.]+)\s*m", str(alt_str).strip())
    return float(match.group(1)) if match else None


def load_video_metadata(exif_dir: Path) -> pd.DataFrame:
    """Load and consolidate video metadata from all batch CSVs."""
    dfs = []
    for csv_path in exif_dir.glob("*_exif_video_metadata.csv"):
        df = pd.read_csv(csv_path)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No video metadata CSVs in {exif_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["video_id"])

    combined["recording_datetime"] = pd.to_datetime(
        combined["recording_datetime"], format="%Y:%m:%d %H:%M:%S", errors="coerce"
    )

    result = combined[
        ["video_id", "video_name", "recording_datetime", "video_duration_sec"]
    ].copy()

    # base_video_id strips the trailing hash to match annotation filenames
    # e.g., "1_itinerary_1_1_8e5f0fc4" -> "1_itinerary_1_1"
    # e.g., "day13_itinerary_1_33e296da" -> "day13_itinerary_1"
    result["base_video_id"] = result["video_id"].str.replace(r"_[a-f0-9]{8}$", "", regex=True)

    return result


def load_gps_timeseries(exif_dir: Path) -> pd.DataFrame:
    """Load and consolidate GPS timeseries from all batch CSVs."""
    dfs = []
    for csv_path in exif_dir.glob("*_gps_timeseries*.csv"):
        df = pd.read_csv(csv_path)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No GPS timeseries CSVs in {exif_dir}")

    combined = pd.concat(dfs, ignore_index=True)

    combined["gps_datetime"] = pd.to_datetime(
        combined["gps_datetime"], format="%Y:%m:%d %H:%M:%S.%f", errors="coerce"
    )

    combined["lat"] = combined["gps_latitude"].apply(dms_to_decimal)
    combined["lon"] = combined["gps_longitude"].apply(dms_to_decimal)
    combined["alt"] = combined["gps_altitude"].apply(parse_altitude)

    result = combined[["video_id", "gps_datetime", "lat", "lon", "alt"]].copy()
    result = result.dropna(subset=["gps_datetime", "lat", "lon"])
    result = result.sort_values(["video_id", "gps_datetime"]).reset_index(drop=True)

    return result


def detect_clock_offset(
    video_df: pd.DataFrame, gps_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Detect GoPro clock drift by comparing recording_datetime to first GPS timestamp.

    GoPro cameras sometimes have incorrect system clocks. GPS time is accurate.
    If the difference exceeds 1 hour, flag the video to use GPS time instead.
    """
    gps_first = (
        gps_df.groupby("video_id")["gps_datetime"].min().reset_index()
    )
    gps_first.columns = ["video_id", "gps_first_datetime"]

    merged = video_df.merge(gps_first, on="video_id", how="left")

    merged["clock_offset_sec"] = (
        merged["gps_first_datetime"] - merged["recording_datetime"]
    ).dt.total_seconds()

    merged["use_gps_time"] = merged["clock_offset_sec"].abs() > 3600

    return merged


def build_gps_index(
    exif_dir: Path, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry point: build and persist GPS lookup structures.

    Returns:
        (video_metadata_df, gps_timeseries_df)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading video metadata...")
    video_df = load_video_metadata(exif_dir)
    print(f"  Found {len(video_df)} videos")

    print("Loading GPS timeseries...")
    gps_df = load_gps_timeseries(exif_dir)
    print(f"  Found {len(gps_df):,} GPS points")

    print("Detecting clock offsets...")
    video_df = detect_clock_offset(video_df, gps_df)
    n_offset = video_df["use_gps_time"].sum()
    print(f"  {n_offset} videos with clock drift >1 hour")

    video_path = output_dir / "video_metadata.parquet"
    gps_path = output_dir / "gps_timeseries.parquet"

    video_df.to_parquet(video_path, index=False)
    gps_df.to_parquet(gps_path, index=False)

    print(f"\nSaved: {video_path}")
    print(f"Saved: {gps_path}")

    return video_df, gps_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, help="City to process")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    exif_dir = project_root / "data" / args.city / "exif_metadata"
    output_dir = project_root / "data" / args.city / "gps_index"

    video_df, gps_df = build_gps_index(exif_dir, output_dir)

    print("\n=== Summary ===")
    print(f"Videos: {len(video_df)}")
    print(f"GPS points: {len(gps_df):,}")
    print(f"Videos with GPS: {gps_df['video_id'].nunique()}")

    if "use_gps_time" in video_df.columns:
        print(f"Videos using GPS time (clock drift): {video_df['use_gps_time'].sum()}")

"""
Run the full analysis data pipeline for one or all cities.

Chains all pipeline steps, using in-memory DataFrames where possible
to minimize unnecessary I/O.

Pipeline steps:
- 00: Process videos (EXIF extraction, GPS parsing, frame extraction)
- 04: Build GPS index (PERSISTED - expensive to rebuild)
- 05: Parse annotations (IN-MEMORY)
- 06: Assign frame GPS (IN-MEMORY)
- 07: Enrich with geo (IN-MEMORY)
- 08: Build analysis data (PERSISTED - final dataset)
- 09: EDA (generates plots)
- 10: Analysis (generates tables and figures)
- 11: Maps (generates interactive and static maps)
- 14: Descriptive patterns (generates tabs/descriptive_patterns.md)

(13_interrater_reliability.py runs separately via `make irr`.)

Usage:
    # Run full E2E pipeline for all cities:
    python scripts/run_pipeline.py --city all --skip-osm

    # Run for single city:
    python scripts/run_pipeline.py --city mumbai --skip-osm

    # Skip visualization steps:
    python scripts/run_pipeline.py --city all --skip-osm --skip-viz

    # Skip video processing (use cached data):
    python scripts/run_pipeline.py --city mumbai --skip-process-videos --skip-rebuild-gps

"""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


def import_from_path(name: str, path: Path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scripts_dir = Path(__file__).parent

build_gps_index_mod = import_from_path("build_gps_index", scripts_dir / "04_build_gps_index.py")
parse_annotations_mod = import_from_path(
    "parse_annotations", scripts_dir / "05_parse_annotations.py"
)
assign_frame_gps_mod = import_from_path("assign_frame_gps", scripts_dir / "06_assign_frame_gps.py")
enrich_with_geo_mod = import_from_path("enrich_with_geo", scripts_dir / "07_enrich_with_geo.py")
build_analysis_data_mod = import_from_path(
    "build_analysis_data", scripts_dir / "08_build_analysis_data.py"
)

build_gps_index = build_gps_index_mod.build_gps_index
parse_all_annotations = parse_annotations_mod.parse_all_annotations
assign_frame_gps = assign_frame_gps_mod.assign_frame_gps
enrich_with_geo = enrich_with_geo_mod.enrich_with_geo
build_analysis_data = build_analysis_data_mod.build_analysis_data


def load_city_config(project_root: Path, city: str) -> dict:
    """Load city configuration from cities.yaml."""
    config_path = project_root / "cities.yaml"
    with open(config_path) as f:
        cities_config = yaml.safe_load(f)

    city_config = cities_config.get(city)
    if not city_config:
        available = ", ".join(cities_config.keys())
        raise ValueError(f"Unknown city: {city}. Available: {available}")

    return city_config


def run_process_videos(
    project_root: Path,
    city: str,
    city_config: dict,
    every_seconds: float = 120,
    quality: int = 95,
    frames_dir: Path | None = None,
) -> None:
    """
    Run step 00: Process videos (EXIF extraction, GPS parsing, frame extraction).

    Requires video_dir in city config.
    """
    video_dir = city_config.get("video_dir")
    if not video_dir:
        raise ValueError(f"No video_dir configured for {city} in cities.yaml")

    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    output_dir = project_root / "data" / city
    if frames_dir is None:
        frames_dir = project_root / "data" / "annotation_task" / f"{city}_frames"

    script_path = scripts_dir / "00_process_videos.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--input",
        str(video_dir),
        "--output",
        str(output_dir),
        "--frames-dir",
        str(frames_dir),
        "--every-seconds",
        str(every_seconds),
        "--quality",
        str(quality),
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_extract_face_frames(
    project_root: Path,
    city: str,
    city_config: dict,
    face_frames_dir: Path | None = None,
    scale_width: int = 1020,
    min_confidence: int = 90,
) -> None:
    """
    Run step 01: Extract frames at face detection timestamps.

    Requires video_dir in city config and EXIF files from step 00.
    """
    video_dir = city_config.get("video_dir")
    if not video_dir:
        raise ValueError(f"No video_dir configured for {city} in cities.yaml")

    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    exif_dir = project_root / "data" / city / "exif"
    if not exif_dir.exists():
        raise FileNotFoundError(f"EXIF directory not found: {exif_dir}. Run step 00 first.")

    if face_frames_dir is None:
        face_frames_dir = project_root / "data" / "annotation_task" / f"{city}_face_frames"

    script_path = scripts_dir / "01_extract_face_frames.py"

    log_path = project_root / "data" / city / "face_frame_metadata.csv"

    cmd = [
        sys.executable,
        str(script_path),
        "--exif-dir",
        str(exif_dir),
        "--video-dir",
        str(video_dir),
        "--output",
        str(face_frames_dir),
        "--log",
        str(log_path),
        "--scale",
        str(scale_width),
        "--min-confidence",
        str(min_confidence),
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_pipeline(
    project_root: Path,
    city: str,
    rebuild_gps: bool = True,
    skip_osm: bool = False,
    process_videos: bool = False,
    every_seconds: float = 120,
    quality: int = 95,
    frames_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Run the full analysis data pipeline for a specific city.

    Args:
        project_root: Root directory of the project
        city: City to process (e.g., 'mumbai', 'navi_mumbai', 'bangalore')
        rebuild_gps: If True, rebuild GPS index even if it exists
        skip_osm: If True, skip OSM enrichment
        process_videos: If True, run step 00 (process videos)
        every_seconds: Frame extraction interval
        quality: JPEG quality for frame extraction
        frames_dir: Output directory for frames (default: data/annotation_task/{city}_frames)

    Returns:
        Final analysis DataFrame
    """
    city_config = load_city_config(project_root, city)

    exif_metadata_dir = project_root / "data" / city / "exif_metadata"
    gps_index_dir = project_root / "data" / city / "gps_index"
    labelstudio_dir = project_root / "data" / city / "labelstudio"
    sampling_dir = project_root / "sampling" / city
    output_dir = project_root / "data" / city

    video_meta_path = gps_index_dir / "video_metadata.parquet"
    gps_path = gps_index_dir / "gps_timeseries.parquet"

    print("=" * 60)
    print(f"PIPELINE FOR: {city.upper()}")
    print("=" * 60)

    if process_videos:
        print("\n" + "=" * 60)
        print("STEP 00: Process Videos (EXIF, GPS, Frames)")
        print("=" * 60)
        run_process_videos(project_root, city, city_config, every_seconds, quality, frames_dir)
        rebuild_gps = True

        print("\n" + "=" * 60)
        print("STEP 01: Extract Face Frames")
        print("=" * 60)
        run_extract_face_frames(project_root, city, city_config)

    print("\n" + "=" * 60)
    print("STEP 04: Build GPS Index")
    print("=" * 60)

    if rebuild_gps or not video_meta_path.exists() or not gps_path.exists():
        video_meta, gps_df = build_gps_index(exif_metadata_dir, gps_index_dir)
    else:
        print(f"Loading cached: {video_meta_path}")
        video_meta = pd.read_parquet(video_meta_path)
        print(f"Loading cached: {gps_path}")
        gps_df = pd.read_parquet(gps_path)
        print(f"  {len(video_meta)} videos, {len(gps_df):,} GPS points")

    print("\n" + "=" * 60)
    print("STEP 05: Parse Annotations")
    print("=" * 60)

    annotations = parse_all_annotations(labelstudio_dir, city)

    print("\n" + "=" * 60)
    print("STEP 06: Assign Frame GPS")
    print("=" * 60)

    interval_sec = float(city_config.get("frame_interval_sec", 1.0 / 30.0))
    anchor = city_config.get("frame_time_anchor", "gps")
    with_gps = assign_frame_gps(annotations, video_meta, gps_df, interval_sec, anchor)

    print("\n" + "=" * 60)
    print("STEP 07: Enrich with Geographic Data")
    print("=" * 60)

    with_geo = enrich_with_geo(with_gps, sampling_dir, city, use_osm=not skip_osm)

    print("\n" + "=" * 60)
    print("STEP 08: Build Analysis Dataset")
    print("=" * 60)

    primary, long = build_analysis_data(with_geo, output_dir)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nCity: {city}")
    print(f"Output: {output_dir}")
    print(f"Primary rows: {len(primary):,}")
    print(f"Long rows: {len(long):,}")
    print(f"Columns: {len(primary.columns)}")

    return primary


def print_summary(df: pd.DataFrame) -> None:
    """Print summary statistics for verification."""
    print("\n=== VERIFICATION SUMMARY ===")

    print(f"\n1. Row count: {len(df):,}")

    gps_rate = df["gps_valid"].mean()
    print(f"\n2. GPS coverage: {100 * gps_rate:.1f}%")

    if "region" in df.columns:
        for region in df["region"].unique():
            mask = df["region"] == region
            region_gps = df.loc[mask, "gps_valid"].mean()
            print(f"   - {region}: {100 * region_gps:.1f}%")

    if "gps_time_diff_sec" in df.columns:
        diff = df["gps_time_diff_sec"].dropna()
        print(f"\n3. GPS time diff: mean={diff.mean():.2f}s, max={diff.max():.2f}s")

    if "frame_hour" in df.columns:
        hours = df["frame_hour"].dropna()
        print(f"\n4. Hour range (IST): {int(hours.min())}-{int(hours.max())}")

    if "itinerary_road_type" in df.columns:
        match_rate = df["itinerary_road_type"].notna().mean()
        print(f"\n5. Itinerary match rate: {100 * match_rate:.1f}%")

    if "gps_lat" in df.columns:
        lat = df["gps_lat"].dropna()
        lon = df["gps_lon"].dropna()
        print(
            f"\n6. Lat/lon ranges: {lat.min():.2f}-{lat.max():.2f}°N, "
            f"{lon.min():.2f}-{lon.max():.2f}°E"
        )

    if "prop_female" in df.columns:
        prop = df["prop_female"].dropna()
        print(f"\n7. Mean prop_female: {prop.mean():.3f} (n={len(prop):,})")


def get_all_cities(project_root: Path) -> list[str]:
    """Get list of all configured cities from cities.yaml."""
    config_path = project_root / "cities.yaml"
    with open(config_path) as f:
        cities_config = yaml.safe_load(f)
    return list(cities_config.keys())


def run_visualization_scripts(project_root: Path, cities: list[str]) -> None:
    """Run EDA, analysis, and maps scripts."""
    scripts_dir = project_root / "scripts"
    cities_arg = ",".join(cities)

    print("\n" + "=" * 60)
    print("STEP 09: EDA")
    print("=" * 60)
    cmd = [sys.executable, str(scripts_dir / "09_eda.py"), "--cities", cities_arg]
    subprocess.run(cmd, check=True)

    print("\n" + "=" * 60)
    print("STEP 10: Analysis (Tables & Figures)")
    print("=" * 60)
    cmd = [sys.executable, str(scripts_dir / "10_analysis.py"), "--cities", cities_arg]
    subprocess.run(cmd, check=True)

    print("\n" + "=" * 60)
    print("STEP 11: Maps")
    print("=" * 60)
    cmd = [sys.executable, str(scripts_dir / "11_make_maps.py"), "--cities", cities_arg]
    subprocess.run(cmd, check=True)

    print("\n" + "=" * 60)
    print("STEP 14: Descriptive patterns")
    print("=" * 60)
    cmd = [
        sys.executable,
        str(scripts_dir / "14_descriptive_patterns.py"),
        "--cities",
        cities_arg,
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run analysis data pipeline")
    parser.add_argument(
        "--city",
        required=True,
        help="City to process (e.g., mumbai, navi_mumbai) or 'all' for all cities",
    )
    parser.add_argument(
        "--skip-rebuild-gps",
        action="store_true",
        help="Skip rebuilding GPS index, use cached files if they exist",
    )
    parser.add_argument(
        "--skip-osm",
        action="store_true",
        help="Skip OSM enrichment step",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization steps (EDA, analysis, maps)",
    )
    parser.add_argument(
        "--process-videos",
        action="store_true",
        help="Run step 00: Process videos - EXIF, GPS, frames (requires video_dir in cities.yaml)",
    )
    parser.add_argument(
        "--every-seconds",
        type=float,
        default=120,
        help="Frame extraction interval in seconds (default: 120)",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default=None,
        help="Output directory for frames (default: data/annotation_task/{city}_frames)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality for frame extraction (default: 95)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    if args.city.lower() == "all":
        cities = get_all_cities(project_root)
        cities = [
            c
            for c in cities
            if (project_root / "data" / c / "labelstudio").exists()
            and any((project_root / "data" / c / "labelstudio").iterdir())
        ]
        if not cities:
            print("No cities with annotation data found")
            return
    else:
        cities = [args.city]

    results = {}
    for city in cities:
        try:
            result = run_pipeline(
                project_root,
                city=city,
                rebuild_gps=not args.skip_rebuild_gps,
                skip_osm=args.skip_osm,
                process_videos=args.process_videos,
                every_seconds=args.every_seconds,
                quality=args.quality,
                frames_dir=Path(args.frames_dir) if args.frames_dir else None,
            )
            print_summary(result)
            results[city] = result
        except Exception as e:
            print(f"ERROR processing {city}: {e}")
            continue

    if not args.skip_viz and results:
        run_visualization_scripts(project_root, list(results.keys()))

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

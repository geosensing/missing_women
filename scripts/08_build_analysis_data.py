"""
Add derived fields and produce final analysis dataset.

Converts counts to numeric, adds temporal features, computes proportions.

Inputs:
    annotations DataFrame (from 07_enrich_with_geo)

Outputs (persisted):
    data/{city}/analysis_data.parquet
        Primary annotator only, for core analysis
    data/{city}/analysis_data_long.parquet
        All annotations, for inter-rater reliability analysis
"""

from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np


IST_OFFSET = timedelta(hours=5, minutes=30)


def convert_count(value) -> int | None:
    """
    Convert count string to integer.

    Handles: "5", ">10", "10+", None, etc.
    ">10" and "10+" are mapped to 11.
    """
    if pd.isna(value):
        return None

    s = str(value).strip()

    if s in (">10", "10+", ">10 ", " >10"):
        return 11

    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def convert_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all count columns to numeric."""
    result = df.copy()

    count_cols = [
        "men_count",
        "women_count",
        "men_twowheeler",
        "women_twowheeler",
    ]

    for col in count_cols:
        if col in result.columns:
            result[col] = result[col].apply(convert_count).fillna(0).astype(int)

    return result


def add_temporal_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features from frame_datetime.

    Converts UTC to IST (UTC+5:30) for analysis.
    """
    result = df.copy()

    if "frame_datetime" not in result.columns:
        return result

    result["frame_datetime"] = pd.to_datetime(result["frame_datetime"])

    result["frame_datetime_ist"] = result["frame_datetime"] + IST_OFFSET

    result["frame_hour"] = result["frame_datetime_ist"].dt.hour
    result["frame_dayofweek"] = result["frame_datetime_ist"].dt.dayofweek
    result["is_weekend"] = result["frame_dayofweek"].isin([5, 6])
    result["frame_date"] = result["frame_datetime_ist"].dt.date

    return result


def add_computed_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived count fields."""
    result = df.copy()

    if "men_count" in result.columns and "women_count" in result.columns:
        men = pd.to_numeric(result["men_count"], errors="coerce")
        women = pd.to_numeric(result["women_count"], errors="coerce")
        result["total_pedestrians"] = men + women

        total_nonzero = result["total_pedestrians"].replace(0, np.nan)
        result["prop_female"] = women / total_nonzero

    if "men_twowheeler" in result.columns and "women_twowheeler" in result.columns:
        men_tw = pd.to_numeric(result["men_twowheeler"], errors="coerce")
        women_tw = pd.to_numeric(result["women_twowheeler"], errors="coerce")
        result["total_twowheeler"] = men_tw + women_tw

        total_tw_nonzero = result["total_twowheeler"].replace(0, np.nan)
        result["prop_female_twowheeler"] = women_tw / total_tw_nonzero

    if "total_pedestrians" in result.columns and "total_twowheeler" in result.columns:
        men = pd.to_numeric(result["men_count"], errors="coerce")
        women = pd.to_numeric(result["women_count"], errors="coerce")
        men_tw = pd.to_numeric(result["men_twowheeler"], errors="coerce")
        women_tw = pd.to_numeric(result["women_twowheeler"], errors="coerce")

        result["total_people"] = result["total_pedestrians"] + result["total_twowheeler"]
        result["total_women"] = women + women_tw
        result["total_men"] = men + men_tw

        total_nonzero = result["total_people"].replace(0, np.nan)
        result["prop_women"] = result["total_women"] / total_nonzero

    return result


def normalize_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize categorical fields for consistency."""
    result = df.copy()

    bool_cols = ["footpath", "lane_markings", "potholes", "litter",
                 "bus_station", "railway_station", "street_vendor"]

    yes_values = {"Yes", "yes", "YES", "1", "True", "true"}
    no_values = {"No", "no", "NO", "0", "False", "false", "No sidewalk"}

    for col in bool_cols:
        if col not in result.columns:
            continue

        def normalize_bool(val):
            if pd.isna(val):
                return None
            s = str(val).strip()
            if s in yes_values:
                return True
            if s in no_values:
                return False
            return None

        result[col] = result[col].apply(normalize_bool)

    return result


def fill_infrastructure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill infrastructure observation columns with False (not observed = absent)."""
    result = df.copy()
    infra_cols = ["potholes", "litter", "footpath"]
    for col in infra_cols:
        if col in result.columns:
            result[col] = result[col].map(lambda x: False if pd.isna(x) else x).astype(bool)
    return result


def get_primary_annotator(df: pd.DataFrame) -> str:
    """Return the annotator with most annotations."""
    return df["annotator"].value_counts().idxmax()


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order columns for final output.

    All Label Studio annotation fields are preserved (appended at end).
    """
    column_order = [
        "task_id",
        "annotation_id",
        "annotator",
        "image",
        "region",
        "base_video_id",
        "frame_number",
        "frame_datetime",
        "frame_datetime_ist",
        "frame_hour",
        "frame_dayofweek",
        "is_weekend",
        "frame_date",
        "gps_lat",
        "gps_lon",
        "gps_alt",
        "gps_time_diff_sec",
        "osm_highway",
        "osm_road_name",
        "osm_surface",
        "osm_distance_m",
        "itinerary_road_type",
        "itinerary_distance_m",
        "men_count",
        "women_count",
        "men_twowheeler",
        "women_twowheeler",
        "total_pedestrians",
        "total_twowheeler",
        "total_people",
        "total_women",
        "total_men",
        "prop_female",
        "prop_female_twowheeler",
        "prop_women",
        "footpath",
        "lane_markings",
        "potholes",
        "litter",
        "bus_station",
        "railway_station",
        "street_vendor",
        "land_use",
        "street_width",
        "additional_notes",
    ]

    existing = [c for c in column_order if c in df.columns]
    extra = [c for c in df.columns if c not in column_order]

    return df[existing + extra]


def build_analysis_data(
    annotations: pd.DataFrame, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build final analysis datasets and save to parquet.

    Produces two outputs:
    - analysis_data.parquet: Primary annotator only (for core analysis)
    - analysis_data_long.parquet: All annotations (for inter-rater reliability)

    Args:
        annotations: DataFrame from pipeline step 07
        output_dir: Directory to save parquet files

    Returns:
        Tuple of (primary_df, long_df)
    """
    print("Converting counts to numeric...")
    result = convert_counts(annotations)

    print("Adding temporal fields...")
    result = add_temporal_fields(result)

    print("Adding computed fields...")
    result = add_computed_fields(result)

    print("Normalizing categorical fields...")
    result = normalize_categorical(result)

    print("Filling infrastructure columns with 0...")
    result = fill_infrastructure_columns(result)

    print("Selecting output columns...")
    result = select_output_columns(result)

    output_dir.mkdir(parents=True, exist_ok=True)

    long_path = output_dir / "analysis_data_long.parquet"
    result.to_parquet(long_path, index=False)
    print(f"\nSaved long format: {long_path} ({len(result)} rows)")

    primary_annotator = get_primary_annotator(result)
    print(f"Primary annotator: {primary_annotator}")
    primary = result[result["annotator"] == primary_annotator].copy()
    primary = primary.drop_duplicates(subset=["image"])

    primary_path = output_dir / "analysis_data.parquet"
    primary.to_parquet(primary_path, index=False)
    print(f"Saved primary format: {primary_path} ({len(primary)} rows)")

    return primary, result


if __name__ == "__main__":
    import argparse
    import importlib.util
    import yaml

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

    parse_annotations_mod = import_from_path("parse_annotations", scripts_dir / "05_parse_annotations.py")
    assign_frame_gps_mod = import_from_path("assign_frame_gps", scripts_dir / "06_assign_frame_gps.py")
    enrich_with_geo_mod = import_from_path("enrich_with_geo", scripts_dir / "07_enrich_with_geo.py")
    parse_all_annotations = parse_annotations_mod.parse_all_annotations
    assign_frame_gps = assign_frame_gps_mod.assign_frame_gps
    enrich_with_geo = enrich_with_geo_mod.enrich_with_geo

    with open(project_root / "cities.yaml") as f:
        city_config = yaml.safe_load(f)[args.city]

    labelstudio_dir = project_root / "data" / args.city / "labelstudio"
    gps_index_dir = project_root / "data" / args.city / "gps_index"
    sampling_dir = project_root / "sampling" / args.city
    osm_path = project_root / "data" / "osm" / city_config["osm_file"]
    output_dir = project_root / "data" / args.city

    annotations = parse_all_annotations(labelstudio_dir, args.city)
    video_meta = pd.read_parquet(gps_index_dir / "video_metadata.parquet")
    gps_df = pd.read_parquet(gps_index_dir / "gps_timeseries.parquet")

    with_gps = assign_frame_gps(annotations, video_meta, gps_df)
    with_geo = enrich_with_geo(with_gps, sampling_dir, args.city, osm_path)

    primary, long = build_analysis_data(with_geo, output_dir)

    print("\n=== Summary ===")
    print(f"Primary rows: {len(primary)}")
    print(f"Long rows: {len(long)}")
    print(f"Columns: {len(primary.columns)}")
    print(f"\nGPS coverage: {primary['gps_lat'].notna().sum()}/{len(primary)} "
          f"({100*primary['gps_lat'].notna().mean():.1f}%)")

    if "frame_hour" in primary.columns:
        print(f"\nHour range (IST): {primary['frame_hour'].min()}-{primary['frame_hour'].max()}")

    if "prop_female" in primary.columns:
        print(f"\nMean prop_female: {primary['prop_female'].mean():.3f}")

    if "itinerary_road_type" in primary.columns:
        print(f"\nRoad types (itinerary):")
        print(primary["itinerary_road_type"].value_counts().head(10))

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

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from analysis_config import (
    TOPCODE_MINIMUM,
    canonical_video_id,
    collection_day_id,
    valid_gps_mask,
)

IST_OFFSET = timedelta(hours=5, minutes=30)


def convert_count(value) -> int | None:
    """
    Convert count string to integer.

    Handles: "5", ">10", "10+", None, etc.
    ">10" and "10+" are mapped to their known minimum, 11. Separate indicator
    columns preserve which values are interval-censored for sensitivity analysis.
    """
    if pd.isna(value):
        return None

    s = str(value).strip()

    if s in (">10", "10+", ">10 ", " >10"):
        return TOPCODE_MINIMUM

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
            raw = result[col].astype("string").str.strip()
            result[f"{col}_topcoded"] = raw.isin({">10", "10+"})
            result[col] = result[col].apply(convert_count).fillna(0).astype(int)

    return result


def add_analysis_identifiers(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Add stable video aliases, fieldwork-day clusters, and GPS QA flags."""
    result = df.copy()
    result["canonical_video_id"] = result["base_video_id"].map(canonical_video_id)
    result["collection_day"] = result["base_video_id"].map(
        lambda value: collection_day_id(value, city)
    )
    if result["collection_day"].isna().any():
        examples = result.loc[result["collection_day"].isna(), "base_video_id"].drop_duplicates()
        raise ValueError(
            f"Could not derive collection day from video IDs: {examples.head().tolist()}"
        )
    result["gps_valid"] = valid_gps_mask(result, city).astype(bool)
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
    result["is_weekend"] = (
        result["frame_dayofweek"].isin([5, 6]).where(result["frame_dayofweek"].notna())
    ).astype("boolean")
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
    """Normalize categorical fields while retaining explicit unknown responses."""
    result = df.copy()

    bool_cols = [
        "footpath",
        "lane_markings",
        "potholes",
        "litter",
        "bus_station",
        "railway_station",
        "street_vendor",
    ]

    yes_values = {"Yes", "yes", "YES", "1", "True", "true", "Paved", "Paved - Blocked"}
    no_values = {"No", "no", "NO", "0", "False", "false", "No sidewalk"}

    infrastructure_maps = {
        "footpath": {
            "Paved": True,
            "Paved - Blocked": True,
            "No sidewalk": False,
            "Not visible": pd.NA,
        },
        "potholes": {"Yes": True, "No": False, "N/A": pd.NA},
        "litter": {"Yes": True, "Construction debris": True, "No": False},
    }

    for col in bool_cols:
        if col not in result.columns:
            continue

        def normalize_bool(val):
            if pd.isna(val):
                return False if col in infrastructure_maps else pd.NA
            s = str(val).strip()
            if col in infrastructure_maps:
                return infrastructure_maps[col].get(s, pd.NA)
            if s in yes_values:
                return True
            if s in no_values:
                return False
            return pd.NA

        result[col] = result[col].apply(normalize_bool).astype("boolean")

    return result


def fill_infrastructure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce nullable booleans after taxonomy-aware normalization."""
    result = df.copy()
    infra_cols = ["potholes", "litter", "footpath"]
    for col in infra_cols:
        if col in result.columns:
            result[col] = result[col].astype("boolean")
    return result


def annotator_order(df: pd.DataFrame) -> list[str]:
    """Annotators ranked by annotation count (descending), name as deterministic tie-break."""
    counts = df.groupby("annotator").size()
    return sorted(counts.index, key=lambda a: (-counts[a], a))


def get_primary_annotator(df: pd.DataFrame) -> str:
    """Return the annotator with most annotations (deterministic tie-break)."""
    return annotator_order(df)[0]


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
        "canonical_video_id",
        "collection_day",
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
        "gps_valid",
        "gps_out_of_bounds",
        "osm_highway",
        "osm_road_name",
        "osm_surface",
        "osm_distance_m",
        "itinerary_road_type",
        "itinerary_distance_m",
        "men_count",
        "men_count_topcoded",
        "women_count",
        "women_count_topcoded",
        "men_twowheeler",
        "men_twowheeler_topcoded",
        "women_twowheeler",
        "women_twowheeler_topcoded",
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


def keep_latest_annotation(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per (image, annotator), keeping the most recent annotation.

    An image can carry more than one annotation by the same person — a genuine
    re-annotation after review, or simply the same annotation appearing in overlapping
    Label Studio exports. Keeping the latest by ``updated_at`` makes post-review edits win
    and de-duplicates re-exported annotations. Falls back to a plain de-dup when the
    timestamp is absent (older parses).
    """
    if "updated_at" not in df.columns:
        return df.drop_duplicates(["image", "annotator"]).reset_index(drop=True)
    ordered = df.sort_values("updated_at", na_position="first")
    return ordered.drop_duplicates(["image", "annotator"], keep="last").reset_index(drop=True)


def build_analysis_data(
    annotations: pd.DataFrame, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build final analysis datasets and save to parquet.

    Persists ``analysis_data.parquet`` (primary annotator only, deduped to one row per
    image) for core analysis. The full multi-annotator frame is returned in memory but
    not saved — it is trivially recreatable from the committed Label Studio JSON, and
    ``13_interrater_reliability.py`` rebuilds it on demand.

    Args:
        annotations: DataFrame from pipeline step 07
        output_dir: Directory to save parquet files

    Returns:
        Tuple of (primary_df, all_annotations_df)
    """
    print("Keeping latest annotation per (image, annotator)...")
    before = len(annotations)
    annotations = keep_latest_annotation(annotations)
    print(f"  {before} -> {len(annotations)} annotations after de-dup")

    print("Converting counts to numeric...")
    result = convert_counts(annotations)

    print("Adding temporal fields...")
    result = add_temporal_fields(result)

    city = output_dir.name
    print("Adding analysis identifiers and GPS QA flags...")
    result = add_analysis_identifiers(result, city)

    print("Adding computed fields...")
    result = add_computed_fields(result)

    print("Normalizing categorical fields...")
    result = normalize_categorical(result)

    print("Finalizing infrastructure columns...")
    result = fill_infrastructure_columns(result)

    print("Selecting output columns...")
    result = select_output_columns(result)

    output_dir.mkdir(parents=True, exist_ok=True)

    # One row per image: the primary (most prolific) annotator's row when they covered
    # the image, else fall back to the next-most-prolific annotator who did.
    order = annotator_order(result)
    primary_annotator = order[0]
    print(f"Primary annotator: {primary_annotator}")
    rank = {a: i for i, a in enumerate(order)}
    primary = (
        result.assign(_rank=result["annotator"].map(rank))
        .sort_values("_rank", kind="stable")
        .drop_duplicates(subset=["image"])
        .drop(columns="_rank")
        .copy()
    )
    n_fallback = int((primary["annotator"] != primary_annotator).sum())
    if n_fallback:
        print(f"  {n_fallback} images covered only by reviewer annotators (fallback used)")

    primary_path = output_dir / "analysis_data.parquet"
    primary.to_parquet(primary_path, index=False)
    print(f"Saved primary format: {primary_path} ({len(primary)} rows)")

    return primary, result


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
    parser.add_argument("--osm", action="store_true", help="Download + match OSM road type")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    scripts_dir = Path(__file__).parent

    parse_annotations_mod = import_from_path(
        "parse_annotations", scripts_dir / "05_parse_annotations.py"
    )
    assign_frame_gps_mod = import_from_path(
        "assign_frame_gps", scripts_dir / "06_assign_frame_gps.py"
    )
    enrich_with_geo_mod = import_from_path("enrich_with_geo", scripts_dir / "07_enrich_with_geo.py")
    parse_all_annotations = parse_annotations_mod.parse_all_annotations
    assign_frame_gps = assign_frame_gps_mod.assign_frame_gps
    enrich_with_geo = enrich_with_geo_mod.enrich_with_geo

    labelstudio_dir = project_root / "data" / args.city / "labelstudio"
    gps_index_dir = project_root / "data" / args.city / "gps_index"
    sampling_dir = project_root / "sampling" / args.city
    output_dir = project_root / "data" / args.city

    annotations = parse_all_annotations(labelstudio_dir, args.city)
    video_meta = pd.read_parquet(gps_index_dir / "video_metadata.parquet")
    gps_df = pd.read_parquet(gps_index_dir / "gps_timeseries.parquet")

    with_gps = assign_frame_gps(annotations, video_meta, gps_df)
    with_geo = enrich_with_geo(with_gps, sampling_dir, args.city, use_osm=args.osm)

    primary, long = build_analysis_data(with_geo, output_dir)

    print("\n=== Summary ===")
    print(f"Primary rows: {len(primary)}")
    print(f"Long rows: {len(long)}")
    print(f"Columns: {len(primary.columns)}")
    print(
        f"\nGPS coverage: {primary['gps_lat'].notna().sum()}/{len(primary)} "
        f"({100 * primary['gps_lat'].notna().mean():.1f}%)"
    )

    if "frame_hour" in primary.columns:
        print(f"\nHour range (IST): {primary['frame_hour'].min()}-{primary['frame_hour'].max()}")

    if "prop_female" in primary.columns:
        print(f"\nMean prop_female: {primary['prop_female'].mean():.3f}")

    if "itinerary_road_type" in primary.columns:
        print("\nRoad types (itinerary):")
        print(primary["itinerary_road_type"].value_counts().head(10))

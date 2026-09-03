"""Shared analysis definitions for Streetscope."""

from __future__ import annotations

import re

import pandas as pd

TOPCODE_MINIMUM = 11
TOPCODE_SENSITIVITY_VALUES = (11, 15, 20, 30)
PHYSICAL_FRAME_COLUMNS = ("region", "canonical_video_id", "frame_number")

CITY_BOUNDS = {
    "mumbai": {"lat": (18.85, 19.30), "lon": (72.75, 73.05)},
    "navi_mumbai": {"lat": (18.90, 19.25), "lon": (72.80, 73.15)},
    "bangalore": {"lat": (12.60, 13.30), "lon": (77.30, 77.90)},
    "delhi": {"lat": (28.40, 28.90), "lon": (76.70, 77.50)},
}


def canonical_video_id(value: object) -> object:
    """Canonicalize known aliases without conflating distinct camera chapters."""
    if pd.isna(value):
        return pd.NA
    return re.sub(r"^day(?=\d+_)", "", str(value), flags=re.IGNORECASE)


def keep_latest_annotation(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest row for each physical frame and annotator.

    Physical-frame keys reconcile Label Studio filename aliases while preserving
    distinct annotators for reliability analysis. Rows without a complete physical key
    fall back to image URL, and malformed rows without either key remain untouched.
    """
    result = df.copy()
    if "canonical_video_id" not in result.columns and "base_video_id" in result.columns:
        result["canonical_video_id"] = result["base_video_id"].map(canonical_video_id)

    physical_key = [*PHYSICAL_FRAME_COLUMNS, "annotator"]
    image_key = ["image", "annotator"]

    physical_complete = pd.Series(False, index=result.index)
    if set(physical_key).issubset(result.columns):
        physical_complete = result[physical_key].notna().all(axis=1)

    image_complete = pd.Series(False, index=result.index)
    if set(image_key).issubset(result.columns):
        image_complete = result[image_key].notna().all(axis=1)

    def latest(rows: pd.DataFrame, key: list[str]) -> pd.DataFrame:
        if "updated_at" in rows.columns:
            rows = rows.sort_values("updated_at", na_position="first", kind="stable")
        return rows.drop_duplicates(key, keep="last")

    physical = latest(result.loc[physical_complete], physical_key)
    fallback_mask = ~physical_complete & image_complete
    fallback = latest(result.loc[fallback_mask], image_key)
    malformed = result.loc[~physical_complete & ~image_complete]
    return pd.concat([physical, fallback, malformed]).sort_index().reset_index(drop=True)


def collection_day_id(base_video_id: object, city: str) -> object:
    """Derive the fieldwork-day cluster encoded in a camera filename."""
    if pd.isna(base_video_id):
        return pd.NA
    match = re.match(r"^(?:day_?|)(\d+)(?:_|$)", str(base_video_id), flags=re.IGNORECASE)
    if match is None:
        return pd.NA
    return f"{city}:day-{int(match.group(1))}"


def valid_gps_mask(df: pd.DataFrame, city: str) -> pd.Series:
    """Identify finite coordinates inside the study city's QA bounds."""
    bounds = CITY_BOUNDS[city]
    lat = pd.to_numeric(df["gps_lat"], errors="coerce")
    lon = pd.to_numeric(df["gps_lon"], errors="coerce")
    return lat.between(*bounds["lat"]) & lon.between(*bounds["lon"])

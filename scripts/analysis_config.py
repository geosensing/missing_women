"""Shared analysis definitions for Streetscope."""

from __future__ import annotations

import re

import pandas as pd

TOPCODE_MINIMUM = 11
TOPCODE_SENSITIVITY_VALUES = (11, 15, 20, 30)

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

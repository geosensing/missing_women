"""
Parse Label Studio JSON exports into flat DataFrame.

Handles multiple filename formats and normalizes video_id prefixes.
Filters incomplete/skip annotations.

This module does NOT persist output - the parsed DataFrame is passed
to the next pipeline step or returned directly.

Inputs:
    data/{city}/labelstudio/*.json

Returns:
    DataFrame with columns:
        task_id, annotation_id, image, region,
        base_video_id, frame_number, timestamp_sec,
        men_count, women_count, ... (annotation fields)
"""

import json
import re
from pathlib import Path

import pandas as pd


def extract_frame_info(image_path: str) -> dict:
    """
    Extract video_id and frame metadata from image filename.

    Handles multiple naming conventions:
    - gs://bucket/annotation_frames_new/13_itinerary_1_frame000003.jpg
    - upload/221408/c594e56e-3_itinerary_7_frame05100_t000250_170.jpg
    - upload/226573/af8d896c-day13_itinerary_1_frame000000.jpg

    Returns:
        dict with base_video_id, frame_number, timestamp_sec (if available)
    """
    filename = image_path.split("/")[-1]
    filename = re.sub(r"^[a-f0-9]+-", "", filename)
    filename = filename.replace(".jpg", "")

    result = {"base_video_id": None, "frame_number": None, "timestamp_sec": None}

    pattern_with_time = r"^(.+)_frame(\d+)_t(\d+)_(\d+)$"
    match = re.match(pattern_with_time, filename)
    if match:
        result["base_video_id"] = match.group(1)
        result["frame_number"] = int(match.group(2))
        result["timestamp_sec"] = int(match.group(3)) + int(match.group(4)) / 1000
        return result

    pattern_simple = r"^(.+)_frame(\d+)$"
    match = re.match(pattern_simple, filename)
    if match:
        result["base_video_id"] = match.group(1)
        result["frame_number"] = int(match.group(2))
        return result

    return result


def parse_annotation_result(result_list: list) -> dict:
    """
    Parse Label Studio annotation result into flat dict.

    Each result item has structure:
        {
            "from_name": "men_count",
            "value": {"taxonomy": [["5"]]}
        }
    """
    parsed = {}
    for item in result_list:
        field_name = item.get("from_name", "")
        value = item.get("value", {})

        if "taxonomy" in value:
            taxonomy = value["taxonomy"]
            if taxonomy and len(taxonomy) > 0:
                parsed[field_name] = taxonomy[0][0] if len(taxonomy[0]) > 0 else None
        elif "text" in value:
            parsed[field_name] = value["text"]
        elif "choices" in value:
            parsed[field_name] = value["choices"][0] if value["choices"] else None

    return parsed


def parse_labelstudio_export(json_path: Path, region: str) -> pd.DataFrame:
    """Parse a single Label Studio JSON export file.

    Parses ALL non-cancelled annotations (not just first) to support
    inter-rater reliability analysis.
    """
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for task in data:
        task_id = task.get("id")
        image = task.get("data", {}).get("image", "")
        annotations = task.get("annotations", [])

        if not annotations:
            continue

        for annotation in annotations:
            if annotation.get("was_cancelled", False):
                continue

            annotation_id = annotation.get("id")
            annotator = annotation.get("completed_by", {}).get("email", None)
            result = annotation.get("result", [])
            parsed_fields = parse_annotation_result(result)
            frame_info = extract_frame_info(image)

            row = {
                "task_id": task_id,
                "annotation_id": annotation_id,
                "image": image,
                "region": region,
                "annotator": annotator,
                # Kept so step 08 can dedup to the most recent annotation when an image
                # is re-annotated or appears in overlapping Label Studio exports.
                "created_at": annotation.get("created_at"),
                "updated_at": annotation.get("updated_at"),
                **frame_info,
                **parsed_fields,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def filter_skip_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove empty / skipped annotations only.

    A frame showing an empty street is a real observation: the taxonomy has no
    explicit "0", so when there are no pedestrians the annotator simply leaves
    ``men_count``/``women_count`` blank (imputed to 0 downstream in step 08).
    Dropping those frames would discard genuine zero-count observations — and any
    frame with only a two-wheeler rider, which still has ``total_people`` > 0.

    So we keep any row that carries at least one annotation field, and drop only
    rows with no annotation content at all (true skips).
    """
    meta_cols = {
        "task_id",
        "annotation_id",
        "annotator",
        "image",
        "region",
        "base_video_id",
        "frame_number",
        "timestamp_sec",
    }
    annotation_cols = [c for c in df.columns if c not in meta_cols]

    if not annotation_cols:
        return df

    mask = df[annotation_cols].notna().any(axis=1)
    return df[mask].copy()


def parse_all_annotations(labelstudio_dir: Path, city: str) -> pd.DataFrame:
    """
    Parse all Label Studio JSON exports in directory.

    Args:
        labelstudio_dir: Directory containing JSON exports
        city: City name to use as region

    Returns:
        Combined DataFrame with all annotations
    """
    dfs = []
    for json_path in labelstudio_dir.glob("*.json"):
        print(f"Parsing {json_path.name} ({city})...")
        df = parse_labelstudio_export(json_path, region=city)
        print(f"  Found {len(df)} annotations")
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No JSON files in {labelstudio_dir}")

    dfs = [df.dropna(axis=1, how="all") for df in dfs if not df.empty]
    combined = pd.concat(dfs, ignore_index=True, join="outer")

    print("\nFiltering incomplete annotations...")
    n_before = len(combined)
    combined = filter_skip_rows(combined)
    n_after = len(combined)
    print(f"  Removed {n_before - n_after} incomplete rows")

    return combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, help="City to process")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    labelstudio_dir = project_root / "data" / args.city / "labelstudio"

    df = parse_all_annotations(labelstudio_dir, args.city)

    print("\n=== Summary ===")
    print(f"Total annotations: {len(df)}")
    print(f"Unique videos: {df['base_video_id'].nunique()}")
    print("\nRegion breakdown:")
    print(df["region"].value_counts())
    print(f"\nSample columns: {list(df.columns)}")

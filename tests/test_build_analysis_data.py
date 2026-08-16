from pathlib import Path

import pandas as pd


def test_topcodes_are_minimum_imputed_and_flagged(load_script):
    build = load_script("08_build_analysis_data.py")
    source = pd.DataFrame(
        {
            "men_count": [">10", "10+", "4", None],
            "women_count": ["2", None, 1, None],
        }
    )
    result = build.convert_counts(source)
    assert result["men_count"].tolist() == [11, 11, 4, 0]
    assert result["men_count_topcoded"].tolist() == [True, True, False, False]


def test_infrastructure_taxonomy_preserves_explicit_unknowns(load_script):
    build = load_script("08_build_analysis_data.py")
    source = pd.DataFrame(
        {
            "footpath": [None, "Paved", "Paved - Blocked", "No sidewalk", "Not visible"],
            "potholes": [None, "Yes", "No", "N/A", None],
            "litter": [None, "Yes", "No", "Construction debris", None],
        }
    )
    result = build.fill_infrastructure_columns(build.normalize_categorical(source))
    assert result["footpath"].tolist()[:4] == [False, True, True, False]
    assert pd.isna(result.loc[4, "footpath"])
    assert pd.isna(result.loc[3, "potholes"])
    assert result.loc[3, "litter"]


def test_timestamp_metadata_does_not_make_empty_annotation_nonempty(load_script):
    parser = load_script("05_parse_annotations.py")
    source = pd.DataFrame(
        {
            "task_id": [1, 2],
            "image": ["a.jpg", "b.jpg"],
            "created_at": ["2026-01-01", "2026-01-01"],
            "updated_at": ["2026-01-02", "2026-01-02"],
            "women_count": [None, "1"],
        }
    )
    result = parser.filter_skip_rows(source)
    assert result["task_id"].tolist() == [2]


def test_physical_frame_aliases_are_deduplicated_within_annotator(load_script):
    parser = load_script("05_parse_annotations.py")
    source = pd.DataFrame(
        {
            "region": ["navi_mumbai"] * 3,
            "base_video_id": ["day13_itinerary_1", "13_itinerary_1", "13_itinerary_1"],
            "frame_number": [5, 5, 5],
            "annotator": ["primary", "primary", "reviewer"],
            "updated_at": ["2026-02-01", "2026-02-02", "2026-02-03"],
            "annotation_id": [1, 2, 3],
        }
    )
    result = parser.keep_latest_annotation(source)
    assert result["annotation_id"].tolist() == [2, 3]
    assert result["canonical_video_id"].tolist() == ["13_itinerary_1", "13_itinerary_1"]


def test_keep_latest_annotation_uses_physical_frame_not_image_url(load_script):
    config = load_script("analysis_config.py")
    source = pd.DataFrame(
        {
            "region": ["navi_mumbai"] * 3,
            "base_video_id": ["day13_itinerary_1", "13_itinerary_1", "13_itinerary_1"],
            "frame_number": [5, 5, 5],
            "annotator": ["primary", "primary", "reviewer"],
            "image": ["alias-a.jpg", "alias-b.jpg", "alias-b.jpg"],
            "updated_at": ["2026-02-01", "2026-02-02", "2026-02-03"],
            "annotation_id": [1, 2, 3],
        }
    )
    result = config.keep_latest_annotation(source)
    assert result["annotation_id"].tolist() == [2, 3]


def test_primary_selection_uses_physical_frame_key(load_script, tmp_path):
    build = load_script("08_build_analysis_data.py")
    source = pd.DataFrame(
        {
            "region": ["navi_mumbai"] * 3,
            "base_video_id": ["day13_itinerary_1", "13_itinerary_1", "13_itinerary_1"],
            "frame_number": [5, 5, 6],
            "annotator": ["primary", "reviewer", "primary"],
            "image": ["alias-a.jpg", "alias-b.jpg", "frame-6.jpg"],
            "updated_at": ["2026-02-01", "2026-02-02", "2026-02-03"],
            "men_count": [2, 9, 1],
            "women_count": [1, 9, 0],
            "men_twowheeler": [0, 0, 0],
            "women_twowheeler": [0, 0, 0],
            "gps_lat": [19.1, 19.1, 19.1],
            "gps_lon": [73.0, 73.0, 73.0],
        }
    )
    primary, _ = build.build_analysis_data(source, tmp_path / "navi_mumbai")
    assert len(primary) == 2
    assert primary.loc[primary["frame_number"] == 5, "image"].item() == "alias-a.jpg"


def test_reliability_pairs_filename_aliases_as_one_physical_frame(load_script):
    irr = load_script("13_interrater_reliability.py")
    source = pd.DataFrame(
        {
            "region": ["navi_mumbai"] * 3,
            "base_video_id": ["day13_itinerary_1", "day13_itinerary_1", "13_itinerary_1"],
            "canonical_video_id": ["13_itinerary_1"] * 3,
            "frame_number": [5, 6, 5],
            "annotator": ["primary", "primary", "reviewer"],
            "image": ["alias-a.jpg", "frame-6.jpg", "alias-b.jpg"],
            "updated_at": ["2026-02-01", "2026-02-01", "2026-02-02"],
            "women_count": [1, 0, 2],
            "men_count": [2, 1, 2],
            "total_pedestrians": [3, 1, 4],
            "prop_female": [1 / 3, 0, 1 / 2],
        }
    )
    pairs, primary = irr.build_pairs(source)
    assert primary == "primary"
    assert len(pairs) == 1
    assert pairs.loc[0, "women_count_p"] == 1
    assert pairs.loc[0, "women_count_r"] == 2


def test_navi_mumbai_committed_exports_preserve_long_form_reviewer_rows(load_script):
    parser = load_script("05_parse_annotations.py")
    labelstudio = Path(__file__).resolve().parents[1] / "data/navi_mumbai/labelstudio"
    result = parser.parse_all_annotations(labelstudio, "navi_mumbai")
    physical_frame = ["region", "canonical_video_id", "frame_number"]

    assert len(result) == 2085
    assert not result.duplicated([*physical_frame, "annotator"]).any()
    assert result[physical_frame].drop_duplicates().shape[0] == 1999
    assert (result.groupby(physical_frame)["annotator"].nunique() > 1).sum() == 86

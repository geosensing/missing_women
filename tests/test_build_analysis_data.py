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

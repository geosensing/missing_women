import math

import pandas as pd
from inference import summarize


def test_primary_estimands_and_collection_day_clusters():
    data = pd.DataFrame(
        {
            "women_count": [1, 0, 1, 2],
            "total_pedestrians": [1, 1, 2, 2],
            "collection_day": ["a", "a", "b", "b"],
        }
    )
    result = summarize(data)
    assert math.isclose(result["weighted"], 4 / 6)
    assert math.isclose(result["unweighted"], 0.625)
    assert result["n_obs"] == 4
    assert result["n_clusters"] == 2
    assert 0 <= result["weighted_ci_lower"] <= result["weighted"]
    assert result["weighted"] <= result["weighted_ci_upper"] <= 1


def test_zero_person_rows_are_excluded_and_one_cluster_has_no_interval():
    data = pd.DataFrame(
        {
            "women_count": [0, 1],
            "total_pedestrians": [0, 2],
            "collection_day": ["a", "a"],
        }
    )
    result = summarize(data)
    assert result["weighted"] == 0.5
    assert result["n_obs"] == 1
    assert math.isnan(result["weighted_ci_lower"])

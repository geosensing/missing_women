import math

import pandas as pd


def test_topcode_sensitivity_uses_pedestrian_fields_directly(load_script):
    analysis = load_script("10_make_publication_outputs.py")
    data = pd.DataFrame(
        {
            "city": ["alpha", "alpha"],
            "women_count": [11, 1],
            "men_count": [1, 3],
            "women_count_topcoded": [True, False],
            "men_count_topcoded": [False, False],
            "collection_day": ["day-1", "day-2"],
        }
    )

    estimates = analysis.compute_topcode_sensitivity(data, ["alpha"])

    assert math.isclose(estimates[11]["alpha"], 12 / 16)
    assert math.isclose(estimates[30]["alpha"], 31 / 35)


def test_topcode_sensitivity_replaces_each_censored_sex_count(load_script):
    analysis = load_script("10_make_publication_outputs.py")
    data = pd.DataFrame(
        {
            "city": ["alpha"],
            "women_count": [11],
            "men_count": [11],
            "women_count_topcoded": [True],
            "men_count_topcoded": [True],
            "collection_day": ["day-1"],
        }
    )

    estimates = analysis.compute_topcode_sensitivity(data, ["alpha"])

    assert math.isclose(estimates[11]["alpha"], 0.5)
    assert math.isclose(estimates[30]["alpha"], 0.5)

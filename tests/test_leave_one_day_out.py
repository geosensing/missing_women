import math

import pandas as pd


def _frame(rows):
    return pd.DataFrame.from_records(
        rows, columns=["women_count", "total_pedestrians", "collection_day", "frame_dayofweek"]
    )


def test_results_are_ordered_by_share_and_exclude_the_named_day(load_script):
    analysis = load_script("10_analysis.py")
    data = _frame(
        [
            [1, 10, "a", 3],  # Thu, the dominant day
            [1, 10, "b", 3],
            [2, 4, "c", 0],  # Mon
            [1, 2, "d", 1],  # Tue
        ]
    )

    baseline, results = analysis.leave_one_day_out(data)

    assert math.isclose(baseline, 5 / 26)
    assert [day for day, _, _ in results] == ["Thu", "Mon", "Tue"]
    assert math.isclose(results[0][1], 20 / 26)
    # Dropping Thu leaves the Mon and Tue rows: 3 women over 6 pedestrians.
    assert math.isclose(results[0][2]["weighted"], 0.5)
    assert results[0][2]["n_clusters"] == 2


def test_baseline_ignores_frames_with_an_unknown_day(load_script):
    """The baseline must be the known-day subset, or the comparison also drops
    frames whose timestamp is missing and overstates the movement."""
    analysis = load_script("10_analysis.py")
    data = _frame(
        [
            [1, 2, "a", 3],
            [1, 2, "b", 0],
            [0, 10, "c", None],  # no timestamp: must not enter the baseline
        ]
    )

    baseline, results = analysis.leave_one_day_out(data)

    assert math.isclose(baseline, 0.5)
    assert all(not math.isnan(s["weighted"]) for _, _, s in results)


def test_no_usable_rows_yields_an_empty_result(load_script):
    analysis = load_script("10_analysis.py")
    data = _frame([[0, 0, "a", 3]])

    baseline, results = analysis.leave_one_day_out(data)

    assert math.isnan(baseline)
    assert results == []

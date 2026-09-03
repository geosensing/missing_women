import math

import pandas as pd
from inference import effective_clusters, summarize


def test_equal_sized_clusters_give_back_the_raw_count():
    clusters = pd.Series(["a", "a", "b", "b", "c", "c"])

    assert math.isclose(effective_clusters(clusters), 3.0)


def test_one_dominant_cluster_collapses_the_effective_count():
    """The point of the diagnostic: nine collection days are not nine clusters
    when one of them contributed most of the frames."""
    clusters = pd.Series(["a"] * 90 + list("bcdefghi"))

    eff = effective_clusters(clusters)

    assert clusters.nunique() == 9
    assert eff < 2.0


def test_effective_count_never_exceeds_the_raw_count():
    for sizes in ([5, 5, 5], [10, 1, 1], [7, 3, 2, 1], [1, 1]):
        clusters = pd.Series([f"c{i}" for i, n in enumerate(sizes) for _ in range(n)])
        assert effective_clusters(clusters) <= clusters.nunique() + 1e-9


def test_summarize_reports_it_alongside_the_raw_count():
    data = pd.DataFrame(
        {
            "women_count": [1, 0, 1, 2, 1],
            "total_pedestrians": [1, 1, 2, 2, 3],
            "collection_day": ["a", "a", "a", "a", "b"],
        }
    )

    result = summarize(data)

    assert result["n_clusters"] == 2
    assert result["n_clusters_eff"] < result["n_clusters"]


def test_empty_input_is_not_a_crash():
    assert math.isnan(effective_clusters(pd.Series([], dtype=object)))

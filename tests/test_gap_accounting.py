import math


def test_published_rates_are_mutually_consistent(load_script):
    """Goel reports the mobility rate and the trip rate separately. The trip rate
    is over all respondents, so it must equal the mobility rate times the trips
    made by mobile people. If the two published numbers disagreed, the implied
    conditional factor would be nonsense and this catches it."""
    acc = load_script("15_gap_accounting.py")

    implied = acc.out_of_home_factor() * acc.conditional_trip_factor()

    assert math.isclose(implied, acc.trip_factor())
    assert 0.7 < acc.conditional_trip_factor() < 1.0


def test_extensive_margin_dominates_the_intensive_one(load_script):
    """The claim rests on this ordering: not leaving home explains far more than
    making fewer trips once out."""
    acc = load_script("15_gap_accounting.py")

    assert acc.out_of_home_factor() < acc.conditional_trip_factor()


def test_the_three_parts_sum_to_the_gap(load_script):
    acc = load_script("15_gap_accounting.py")
    parts = acc.account(observed=236.0, residential=832.0)

    total = parts["gap_home"] + parts["gap_trips"] + parts["gap_residual"]
    assert math.isclose(total, parts["total_gap"])
    assert math.isclose(parts["share_home"] + parts["share_trips"] + parts["share_residual"], 1.0)


def test_shortfall_is_the_unadjusted_headline(load_script):
    """The mobility factors must not touch the headline number."""
    acc = load_script("15_gap_accounting.py")
    parts = acc.account(observed=236.0, residential=832.0)

    assert math.isclose(parts["shortfall"], 1 - 236.0 / 832.0)


def test_an_observation_at_the_trip_scaled_ratio_leaves_no_residual(load_script):
    """A city whose street presence matched published mobility exactly would add
    nothing beyond the survey, and the residual should say so."""
    acc = load_script("15_gap_accounting.py")
    residential = 832.0
    trip_scaled = residential * acc.trip_factor()

    parts = acc.account(observed=trip_scaled, residential=residential)

    assert math.isclose(parts["gap_residual"], 0.0, abs_tol=1e-9)
    assert math.isclose(parts["share_home"] + parts["share_trips"], 1.0)


def test_every_configured_city_carries_a_source_string(load_script):
    acc = load_script("15_gap_accounting.py")
    for city, (ratio, source) in acc.RESIDENTIAL_SEX_RATIO.items():
        assert 500 < ratio < 1100, city
        assert "Census" in source, city

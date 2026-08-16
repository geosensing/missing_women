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


def test_c14_workbook_matches_the_validated_download(load_script):
    acc = load_script("15_gap_accounting.py")

    assert acc.file_md5(acc.C14_PATH) == acc.C14_MD5
    assert acc.C14_CATALOG_URL.endswith("/1640")


def test_exact_age_20_plus_benchmarks_are_derived_from_c14(load_script):
    acc = load_script("15_gap_accounting.py")
    benchmarks = acc.adult_residential_benchmarks()
    expected_counts = {
        "mumbai": (4_631_682, 3_916_598),
        "navi_mumbai": (409_867, 337_732),
        "bangalore": (3_071_612, 2_802_891),
        "delhi": (3_779_669, 3_384_018),
    }

    for city, (men, women) in expected_counts.items():
        benchmark = benchmarks[city]
        assert benchmark["age_min"] == 20
        assert benchmark["men"] == men
        assert benchmark["women"] == women
        assert math.isclose(benchmark["ratio"], women / men * 1000)
        assert 800 < benchmark["ratio"] < 950


def test_c14_city_codes_resolve_to_the_expected_municipal_units(load_script):
    acc = load_script("15_gap_accounting.py")
    benchmarks = acc.adult_residential_benchmarks()

    for city, (state_code, town_code, area_name) in acc.C14_CITY_GEOGRAPHIES.items():
        benchmark = benchmarks[city]
        assert benchmark["state_code"] == state_code
        assert benchmark["town_code"] == town_code
        assert benchmark["area_name"] == area_name

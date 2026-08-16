from __future__ import annotations

import pytest


def test_clock_timestamp_suffix_is_converted_to_elapsed_seconds(load_script):
    parser = load_script("05_parse_annotations.py")

    parsed = parser.extract_frame_info(
        "upload/221408/c594e56e-3_itinerary_7_frame05100_t000250_170.jpg"
    )

    assert parsed == {
        "base_video_id": "3_itinerary_7",
        "frame_number": 5100,
        "timestamp_sec": pytest.approx(170.17),
    }


def test_index_named_frame_has_no_explicit_timestamp(load_script):
    parser = load_script("05_parse_annotations.py")

    parsed = parser.extract_frame_info(
        "gs://bucket/annotation_frames/day13_itinerary_1_frame000007.jpg"
    )

    assert parsed == {
        "base_video_id": "day13_itinerary_1",
        "frame_number": 7,
        "timestamp_sec": None,
    }

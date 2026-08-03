import pandas as pd
from analysis_config import canonical_video_id, collection_day_id, valid_gps_mask


def test_collection_day_id_handles_all_filename_families():
    assert collection_day_id("11_itinerary_3_1", "mumbai") == "mumbai:day-11"
    assert collection_day_id("day13_itinerary_2_1", "navi_mumbai") == "navi_mumbai:day-13"
    assert collection_day_id("13_itinerary_2_1", "navi_mumbai") == "navi_mumbai:day-13"
    assert collection_day_id("day_7_4_01_2026_Itinerary_67.2", "bangalore") == ("bangalore:day-7")


def test_canonical_video_id_reconciles_navi_alias_only():
    assert canonical_video_id("day13_itinerary_2_1") == "13_itinerary_2_1"
    assert canonical_video_id("day_1_4_28_2026_11.4") == "day_1_4_28_2026_11.4"


def test_valid_gps_mask_rejects_missing_and_out_of_city_coordinates():
    data = pd.DataFrame(
        {
            "gps_lat": [19.0, None, 0.0],
            "gps_lon": [72.9, 72.9, 0.0],
        }
    )
    assert valid_gps_mask(data, "mumbai").tolist() == [True, False, False]

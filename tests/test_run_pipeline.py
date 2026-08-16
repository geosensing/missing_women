from __future__ import annotations

import pandas as pd
import pytest


def test_video_processing_uses_configured_frame_interval(load_script, monkeypatch, tmp_path):
    pipeline = load_script("run_pipeline.py")
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    commands = []

    def record_command(command, check):
        commands.append(command)
        assert check is True

    monkeypatch.setattr(pipeline.subprocess, "run", record_command)
    pipeline.run_process_videos(
        tmp_path,
        "mumbai",
        {"video_dir": str(video_dir), "frame_interval_sec": 10},
    )

    assert len(commands) == 1
    command = commands[0]
    interval_position = command.index("--every-seconds") + 1
    assert command[interval_position] == "10.0"
    assert all("01_extract_face_frames.py" not in part for part in command)


@pytest.mark.parametrize(
    "config",
    [{}, {"frame_interval_sec": 0}, {"frame_interval_sec": "bad"}],
)
def test_frame_interval_is_required_and_positive(load_script, config):
    pipeline = load_script("run_pipeline.py")

    with pytest.raises(ValueError):
        pipeline.configured_frame_interval("mumbai", config)


def test_gps_index_loads_current_processor_outputs(load_script, tmp_path):
    gps_index = load_script("04_build_gps_index.py")
    pd.DataFrame(
        {
            "video_id": ["day1_trip_1234abcd"],
            "video_name": ["trip"],
            "recording_datetime": ["2026:04:28 09:00:00"],
            "video_duration_sec": [60.0],
        }
    ).to_csv(tmp_path / "exif_metadata.csv", index=False)
    pd.DataFrame(
        {
            "video_id": ["day1_trip_1234abcd"],
            "gps_datetime": ["2026:04:28 09:00:01.000"],
            "gps_latitude": ["28 deg 36' 0.00\" N"],
            "gps_longitude": ["77 deg 12' 0.00\" E"],
            "gps_altitude": ["216.0 m"],
        }
    ).to_csv(tmp_path / "gps_timeseries.csv.gz", index=False, compression="gzip")

    videos = gps_index.load_video_metadata(tmp_path)
    points = gps_index.load_gps_timeseries(tmp_path)

    assert videos["base_video_id"].tolist() == ["day1_trip"]
    assert points[["lat", "lon", "alt"]].iloc[0].tolist() == [28.6, 77.2, 216.0]


def test_gps_rebuild_prefers_current_processor_outputs(load_script, tmp_path):
    pipeline = load_script("run_pipeline.py")
    output_dir = tmp_path / "data" / "mumbai"
    legacy_dir = output_dir / "exif_metadata"
    legacy_dir.mkdir(parents=True)
    (output_dir / "exif_metadata.csv").touch()
    (output_dir / "gps_timeseries.csv.gz").touch()

    assert pipeline.gps_source_directory(output_dir, legacy_dir) == output_dir

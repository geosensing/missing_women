import pandas as pd


def test_reliability_outputs_share_one_set_of_estimates(load_script, tmp_path, monkeypatch):
    irr = load_script("13_interrater_reliability.py")
    monkeypatch.setattr(irr, "TABS", tmp_path)
    continuous = {
        "women_count": {"n": 4, "icc": 0.81, "r": 0.82, "mad": 0.19},
        "men_count": {"n": 4, "icc": 0.91, "r": 0.92, "mad": 0.29},
        "total_pedestrians": {"n": 4, "icc": 0.93, "r": 0.94, "mad": 0.39},
        "prop_female": {"n": 3, "icc": 0.84, "r": 0.85, "mad": 0.04},
    }
    binary = {
        name: {"n": 4, "agree": 0.75, "kappa": value}
        for name, value in {
            "footpath": 0.61,
            "lane_markings": float("nan"),
            "potholes": 0.41,
            "litter": 0.51,
            "bus_station": 0.62,
            "railway_station": 0.49,
            "street_vendor": 0.31,
        }.items()
    }
    overlaps = {"mumbai": 1, "navi_mumbai": 1, "bangalore": 1, "delhi": 1}

    irr.write_outputs(continuous, binary, overlaps)

    table = (tmp_path / "tableS4_irr.tex").read_text()
    macros = (tmp_path / "irr_macros.tex").read_text()
    assert "Footpath & 4 & 75.0\\% & \\multicolumn{2}{c}{0.61}" in table
    assert "when either rater's classifications have no variance" in table
    assert r"\newcommand{\IRRWomenCountICC}{0.81}" in macros
    assert r"\newcommand{\IRROverlaps}{4}" in macros


def test_reliability_matches_malformed_ids_by_image(load_script):
    irr = load_script("13_interrater_reliability.py")
    source = pd.DataFrame(
        {
            "region": ["mumbai"] * 4,
            "canonical_video_id": [pd.NA] * 4,
            "frame_number": [pd.NA] * 4,
            "annotator": ["primary", "reviewer", "primary", "reviewer"],
            "image": ["shared.jpg", "shared.jpg", "primary-only.jpg", "reviewer-only.jpg"],
        }
    )

    pairs, primary = irr.build_pairs(source)

    assert primary == "primary"
    assert len(pairs) == 1

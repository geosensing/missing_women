# Streetscope — analysis-half workflow.
#
# These targets cover the FAIR-GAME half of the pipeline only: parse annotations
# -> assign GPS -> enrich -> build analysis_data -> figures/tables/maps. They never
# re-run the frozen collection half (video processing / frame extraction); that is
# done once, by hand, via scripts/00_process_videos.py and 01_extract_face_frames.py.

PY ?= .venv/bin/python
CITY ?= mumbai                       # single city for run_pipeline
CITIES ?= mumbai,navi_mumbai         # comma-separated for figure/table scripts

.PHONY: help install analyze figures tables lint clean

help:
	@echo "make install              uv sync all deps (incl. editable ../geoinference) into .venv"
	@echo "make analyze CITY=mumbai  run the analysis half for one city (uses cached GPS index)"
	@echo "make figures CITIES=...    regenerate EDA + publication figures/tables + maps"
	@echo "make tables  CITIES=...    regenerate publication tables only"
	@echo "make lint                 ruff format + ruff check --fix over scripts/"
	@echo "make clean                remove generated figures and tables"

install:
	uv sync

# Build analysis_data for one city, then its figures/tables/maps. Skips the GPS
# index rebuild (cached) and never processes videos.
analyze:
	$(PY) scripts/run_pipeline.py --city $(CITY) --skip-rebuild-gps

figures:
	$(PY) scripts/09_eda.py --cities $(CITIES)
	$(PY) scripts/10_analysis.py --cities $(CITIES)
	$(PY) scripts/11_make_maps.py --cities $(CITIES)

tables:
	$(PY) scripts/10_analysis.py --cities $(CITIES)

lint:
	uv run ruff format scripts/
	uv run ruff check --fix scripts/

clean:
	rm -f figs/*.pdf figs/*.png figs/*.html tabs/*.tex

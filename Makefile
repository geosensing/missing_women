# Streetscope — analysis-half workflow.
#
# These targets cover the FAIR-GAME half of the pipeline only: parse annotations
# -> assign GPS -> enrich -> build analysis_data -> figures/tables/maps. They never
# re-run the frozen collection half (video processing / fixed-interval frame
# extraction); that is done once via scripts/00_process_videos.py.

PY ?= .venv/bin/python
CITY ?= mumbai                       # single city for run_pipeline
CITIES ?= mumbai,navi_mumbai,bangalore,delhi  # comma-separated for figure/table scripts
SOURCE_DATE_EPOCH ?= 1786838400
export SOURCE_DATE_EPOCH
PYTHONHASHSEED ?= 0
export PYTHONHASHSEED

.PHONY: help install data analyze figures tables irr analysis paper paper-clean test lint ci clean

help:
	@echo "make install              sync the locked analysis environment"
	@echo "make data                 (re)build analysis_data.parquet for all cities (gitignored, derived)"
	@echo "make analyze CITY=mumbai  run the analysis half for one city (uses cached GPS index)"
	@echo "make figures CITIES=...    regenerate EDA + publication figures/tables + maps"
	@echo "make tables  CITIES=...    regenerate publication tables only"
	@echo "make irr     CITIES=...    inter-rater reliability table + console summary"
	@echo "make analysis              rebuild data and all manuscript figures/tables"
	@echo "make paper                 rebuild the analysis and compile ms/ms.pdf"
	@echo "make paper-clean           remove LaTeX build files"
	@echo "make test                 run the analysis regression tests"
	@echo "make lint                 ruff format + ruff check --fix over scripts/"
	@echo "make ci                   run formatting, lint, and tests without modifying files"
	@echo "make clean                remove generated figures and tables"

install:
	uv sync

# Rebuild the (gitignored) analysis_data.parquet for every city from committed inputs
# (GPS index + annotations + osm_roads.parquet). Skips GPS-index rebuild and viz.
data:
	$(PY) scripts/run_pipeline.py --city all --skip-rebuild-gps --skip-viz

# Build analysis_data for one city, then its figures/tables/maps. Skips the GPS
# index rebuild (cached) and never processes videos.
analyze:
	$(PY) scripts/run_pipeline.py --city $(CITY) --skip-rebuild-gps

figures:
	$(PY) scripts/09_eda.py --cities $(CITIES)
	$(PY) scripts/10_make_publication_outputs.py --cities $(CITIES)
	$(PY) scripts/11_make_maps.py --cities $(CITIES)
	$(PY) scripts/14_descriptive_patterns.py --cities $(CITIES)
	$(PY) scripts/15_gap_accounting.py --cities $(CITIES)

tables:
	$(PY) scripts/10_make_publication_outputs.py --cities $(CITIES)
	$(PY) scripts/15_gap_accounting.py --cities $(CITIES)

irr:
	$(PY) scripts/13_interrater_reliability.py --cities $(CITIES)

analysis: data figures irr

paper: analysis
	latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=ms ms/ms.tex

paper-clean:
	latexmk -C -outdir=ms ms/ms.tex

test:
	uv run pytest

lint:
	uv run ruff format scripts/
	uv run ruff check --fix scripts/

ci:
	uv run ruff format --check scripts/ tests/
	uv run ruff check scripts/ tests/
	uv run pytest

clean:
	rm -f figs/*.pdf figs/*.png figs/*.html tabs/*.tex tabs/descriptive_patterns.md

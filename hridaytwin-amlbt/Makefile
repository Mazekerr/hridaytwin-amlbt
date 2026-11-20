PY ?= python

.PHONY: install run quick test figures report all clean

install:            ## install Python dependencies
	$(PY) -m pip install -r requirements.txt

run:                ## full pipeline (~4 min): data -> twin -> models -> validation -> console -> results
	$(PY) run_pipeline.py

quick:              ## reduced pipeline (~1 min)
	$(PY) run_pipeline.py --quick

test:               ## unit tests
	$(PY) -m pytest

figures:            ## figures for the IEEE report (needs a full run first)
	$(PY) report/make_figures.py

report: figures     ## compile the IEEE report PDF (needs LaTeX)
	cd report && latexmk -pdf -interaction=nonstopmode main.tex && cp main.pdf HridayTwin_AMLBT_Report.pdf

all: run test figures

clean:
	rm -rf data/*.csv model/*.joblib model/oof_predictions.csv dashboard/data.json
	cd report && latexmk -C main.tex || true

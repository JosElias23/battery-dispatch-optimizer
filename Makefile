# Reproduce every number in the README, in order.
#   make all   runs the full pipeline

PYTHON ?= python

.PHONY: help install test experiment ablation montecarlo figures all clean

help:
	@echo "install     install the package and dev dependencies"
	@echo "test        run the test suite (66 tests)"
	@echo "experiment  fit forecasters on 2023, dispatch 2024, write metrics"
	@echo "ablation    quantify the revenue invented by relaxing complementarity"
	@echo "montecarlo  revenue distribution under resampled forecast error"
	@echo "erroranalysis  why the Monte Carlo is a conservative bound"
	@echo "figures     regenerate every figure in the README"
	@echo "all         test -> experiment -> ablation -> montecarlo -> figures"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

experiment:
	$(PYTHON) scripts/run_experiment.py

ablation:
	$(PYTHON) scripts/ablate_complementarity.py

erroranalysis:
	$(PYTHON) scripts/analyse_forecast_error.py

montecarlo:
	$(PYTHON) scripts/run_monte_carlo.py

figures:
	$(PYTHON) scripts/make_figures.py

all: test experiment ablation montecarlo erroranalysis figures

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

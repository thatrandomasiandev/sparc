.PHONY: help install test lint smoke-exp3 paper clean

help:
	@echo "install | test | lint | smoke-exp3 | paper | clean"

install:
	python -m pip install -e ".[dev]"

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q

lint:
	ruff check plr tests experiment_1.py experiment_2.py experiment_3.py aggregate_exp3.py

smoke-exp3:
	python experiment_3.py --seed 0 --d 4 --n-users 3 --n-queries 4 --n-particles 12 --n-candidates 6 \
	  --out experiments/runs/exp3_smoke.jsonl
	python aggregate_exp3.py experiments/runs/exp3_smoke.jsonl

paper:
	$(MAKE) -C paper pdf

clean:
	rm -rf .pytest_cache .ruff_cache
	$(MAKE) -C paper clean

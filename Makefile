.PHONY: install dev-install smoke test lint format clean

install:
	pip install -U pip
	pip install -r requirements.txt
	pip install -e .

dev-install:
	pip install -U pip
	pip install -r requirements.txt
	pip install -e ".[dev]"

smoke:
	python scripts/smoke_test_env.py

test:
	pytest -q

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

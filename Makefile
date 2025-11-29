.PHONY: help install install-dev test lint format clean build docs

help:  ## Show this help message
	@echo "JSF-Core Development Commands"
	@echo "=============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in production mode
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run tests with pytest
	pytest tests/ -v --cov=jsf --cov-report=html --cov-report=term

test-fast:  ## Run tests without coverage
	pytest tests/ -v

lint:  ## Run linting checks
	ruff check src/ tests/
	mypy src/

format:  ## Format code with black and ruff
	black src/ tests/
	ruff check --fix src/ tests/

clean:  ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build distribution packages
	python -m build

docs:  ## Generate documentation (placeholder)
	@echo "Documentation generation not yet implemented"

check:  ## Run all checks (lint + test)
	$(MAKE) lint
	$(MAKE) test

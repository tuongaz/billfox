.PHONY: test lint format typecheck build publish

test:
	python3 -m pytest tests/ -v

lint:
	python3 -m ruff check src/ tests/

format:
	python3 -m ruff format src/ tests/

typecheck:
	python3 -m mypy src/billfox/

build:
	rm -rf dist
	uv build --sdist --wheel

publish: build
	uv publish --token $(PYPI_TOKEN)

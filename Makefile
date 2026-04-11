.PHONY: test lint format typecheck

test:
	python3 -m pytest tests/ -v

lint:
	python3 -m ruff check src/ tests/

format:
	python3 -m ruff format src/ tests/

typecheck:
	python3 -m mypy src/billfox/

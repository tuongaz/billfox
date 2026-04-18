.PHONY: test lint format typecheck build publish release run

run:
	uv run --all-extras billfox

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

release:
	@python3 -c "\
	import re, pathlib; \
	p = pathlib.Path('src/billfox/_version.py'); \
	text = p.read_text(); \
	major, minor, patch = re.search(r'(\d+)\.(\d+)\.(\d+)', text).groups(); \
	new_ver = f'{major}.{minor}.{int(patch)+1}'; \
	p.write_text(re.sub(r'\"[\d.]+\"', f'\"{new_ver}\"', text)); \
	print(f'Bumped version to {new_ver}')"
	git add src/billfox/_version.py
	git commit -m "chore: bump version to $$(python3 -c "import re, pathlib; print(re.search(r'[\d.]+', pathlib.Path('src/billfox/_version.py').read_text()).group())")"
	$(MAKE) publish

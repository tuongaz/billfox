# Contributing to billfox

Thank you for your interest in contributing to billfox! This guide covers everything you need to get started.

## Development Setup

### Prerequisites

- Python 3.11 or later
- Git

### Clone and Install

```bash
git clone https://github.com/billfox-ai/billfox.git
cd billfox

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode with all extras
pip install -e ".[dev]"
```

The `dev` extra installs all optional dependencies plus testing and linting tools.

## Running Tests

```bash
# Run the full test suite
make test

# Run a specific test file
python3 -m pytest tests/test_source_local.py -v

# Run with coverage
python3 -m pytest tests/ --cov=billfox --cov-report=term-missing
```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [mypy](https://mypy-lang.org/) for type checking.

```bash
make lint       # Check for lint errors
make format     # Auto-format code
make typecheck  # Run mypy in strict mode
```

### Style Guidelines

- **Type annotations** on all public functions and methods.
- **Docstrings** on all public classes and functions (Google style).
- **`from __future__ import annotations`** in all source files (except CLI modules -- typer requires runtime annotations).
- **Protocols** live in `_base.py` files and use `@runtime_checkable`.
- **Lazy imports** for optional dependencies with clear install instructions on `ImportError`.

## Project Structure

```
src/billfox/
  __init__.py          # Re-exports Pipeline, Document, ExtractionResult, SearchResult
  _types.py            # Core frozen dataclasses
  _version.py          # Version string
  pipeline.py          # Pipeline compositor
  source/              # Document loading
  preprocess/           # Image preprocessing (resize, YOLO, chain)
  extract/             # OCR / text extraction
  parse/               # LLM structured parsing
  embed/               # Text embeddings
  store/               # SQLite storage + hybrid search
  cli/                 # Typer CLI application
tests/                 # pytest test suite
docs/                  # mkdocs-material documentation
```

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`.
2. **Implement** your changes with tests.
3. **Verify** all checks pass:
   ```bash
   make lint && make typecheck && make test
   ```
4. **Commit** with a clear, descriptive message.
5. **Submit** a pull request with:
   - A summary of the changes and motivation.
   - Any relevant issue numbers.
   - Confirmation that tests pass.

## Adding a New Module

When adding a new extractor, preprocessor, or store backend:

1. Create a `_base.py` protocol if one doesn't exist for the stage.
2. Implement the protocol in a new file within the appropriate subpackage.
3. Re-export the class in the subpackage's `__init__.py`.
4. Add the new optional dependencies to the relevant extra in `pyproject.toml`.
5. Write unit tests with mocked external dependencies.
6. Add a documentation page under `docs/`.

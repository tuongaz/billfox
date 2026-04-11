# Contributing to Billfox

## Development Setup

```bash
# Clone the repository
git clone https://github.com/billfox/billfox.git
cd billfox

# Install in development mode
pip install -e ".[dev]"
```

## Running Tests

```bash
make test
```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [mypy](https://mypy-lang.org/) for type checking.

```bash
make lint       # Check for lint errors
make format     # Auto-format code
make typecheck  # Run type checker
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Ensure all checks pass: `make lint && make typecheck && make test`
4. Submit a pull request with a clear description

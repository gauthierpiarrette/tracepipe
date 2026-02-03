# Contributing to TracePipe

Thank you for your interest in contributing to TracePipe!

## Quick Start

```bash
# Clone the repository
git clone https://github.com/tracepipe/tracepipe.git
cd tracepipe

# Install with uv (recommended)
uv sync --all-extras

# Or with pip
pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -v

# Run linting
uv run task lint
```

## Development Workflow

1. **Create a branch**: `git checkout -b feature/your-feature-name`
2. **Make changes**: Write code, add tests, update docs
3. **Run tests**: `uv run pytest tests/ -v`
4. **Run linting**: `uv run task lint`
5. **Submit PR**: Push and create a Pull Request

## Code Style

- Follow PEP 8
- Use type hints for public functions
- Maximum line length: 100 characters
- Use Black for formatting, Ruff for linting

## Testing

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Aim for high coverage on new code

## Documentation

For detailed contribution guidelines, see the [full documentation](https://tracepipe.github.io/tracepipe/contributing/).

## Code of Conduct

Please be respectful and inclusive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

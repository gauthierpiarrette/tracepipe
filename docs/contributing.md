# Contributing

Thank you for your interest in contributing to TracePipe!

## Development Setup

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/tracepipe/tracepipe.git
cd tracepipe

# Install with uv (recommended)
uv sync --all-extras

# Or with pip
pip install -e ".[dev]"
```

### Verify Setup

```bash
# Run tests
uv run pytest tests/ -v

# Run linting
uv run task lint

# Run full checks
uv run task check
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write code following the existing style
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_api.py -v

# Run with coverage
uv run pytest tests/ --cov=tracepipe --cov-report=term-missing
```

### 4. Run Linting

```bash
# Check linting
uv run task lint

# Auto-fix formatting
uv run task format
```

### 5. Submit PR

- Push your branch
- Create a Pull Request
- Fill out the PR template

## Code Style

### Python Style

- Follow PEP 8
- Use type hints for all public functions
- Maximum line length: 100 characters
- Use Black for formatting
- Use Ruff for linting

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """Short description of the function.

    Longer description if needed. Can span multiple lines.

    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to 10.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is empty.

    Example:
        >>> example_function("test")
        True
    """
```

### Testing

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for setup
- Aim for high coverage on new code

## Project Structure

```
tracepipe/
├── tracepipe/              # Main package
│   ├── __init__.py         # Public API exports
│   ├── api.py              # Core API functions
│   ├── contracts.py        # Contract builder
│   ├── convenience.py      # check/trace/why functions
│   ├── core.py             # Configuration and context
│   ├── instrumentation/    # Pandas monkey-patching
│   │   ├── pandas_inst.py  # Main instrumentation
│   │   ├── filter_capture.py
│   │   └── ...
│   └── storage/            # Lineage storage
│       ├── lineage_store.py
│       └── row_identity.py
├── tests/                  # Test suite
├── examples/               # Example scripts
├── docs/                   # Documentation
└── benchmarks/             # Performance benchmarks
```

## Adding Features

### 1. New Pandas Operation Support

To add tracking for a new pandas operation:

1. Identify the operation type (filter, transform, merge, etc.)
2. Add wrapper in appropriate `tracepipe/instrumentation/` module
3. Register in `pandas_inst.py`
4. Add tests in `tests/`
5. Document in README and docs

### 2. New Contract Expectation

To add a new contract expectation:

1. Add method to `ContractBuilder` in `contracts.py`
2. Implement validation logic
3. Add tests in `tests/test_contracts.py`
4. Document in `docs/guide/contracts.md`

## Running Benchmarks

```bash
cd benchmarks
python run_all.py
```

## Documentation

### Local Preview

```bash
# Install mkdocs
pip install mkdocs mkdocs-material mkdocstrings[python]

# Serve locally
mkdocs serve
```

### Building Docs

```bash
mkdocs build
```

## Release Process

Releases are managed by maintainers:

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag
4. GitHub Actions builds and publishes to PyPI

## Getting Help

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Tag maintainers for urgent issues

## Code of Conduct

Please be respectful and inclusive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

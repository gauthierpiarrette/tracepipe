# Installation

## Requirements

- Python 3.9 or higher
- pandas 1.5.0 or higher

## Basic Installation

Install TracePipe from PyPI:

```bash
pip install tracepipe
```

## Optional Dependencies

TracePipe has optional dependencies for additional features:

=== "Arrow/Parquet Support"

    ```bash
    pip install tracepipe[arrow]
    ```

    Enables:

    - `pd.read_parquet()` tracking
    - `df.to_parquet()` with automatic column stripping
    - Optimized Arrow-based serialization

=== "Memory Profiling"

    ```bash
    pip install tracepipe[memory]
    ```

    Enables:

    - Memory usage statistics in `tp.debug.inspect().stats()`
    - Process memory tracking

=== "All Features"

    ```bash
    pip install tracepipe[all]
    ```

    Installs all optional dependencies.

## Development Installation

For contributing to TracePipe:

```bash
git clone https://github.com/tracepipe/tracepipe.git
cd tracepipe

# Using uv (recommended)
uv sync --all-extras

# Or using pip
pip install -e ".[dev]"
```

## Verify Installation

```python
import tracepipe as tp
print(tp.__version__)  # Should print version number

# Quick test
import pandas as pd
tp.enable()
df = pd.DataFrame({"a": [1, 2, 3]})
df = df.dropna()
print(tp.check(df))
tp.disable()
```

## Supported Pandas Versions

TracePipe is tested against:

| pandas Version | Status |
|----------------|--------|
| 1.5.x | ✅ Supported |
| 2.0.x | ✅ Supported |
| 2.1.x | ✅ Supported |
| 2.2.x | ✅ Supported |

## Troubleshooting

### Import Error

If you get an import error, ensure pandas is installed:

```bash
pip install pandas>=1.5.0
```

### Version Conflicts

If you have version conflicts, try creating a fresh virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows
pip install tracepipe
```

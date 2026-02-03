# TracePipe

**Row-level data lineage for pandas pipelines.**

TracePipe automatically tracks what happens to every row and cell in your DataFrame — drops, transformations, merges, and value changes. Zero code changes required.

[![PyPI version](https://img.shields.io/pypi/v/tracepipe.svg)](https://pypi.org/project/tracepipe/)
[![Python 3.9+](https://img.shields.io/pypi/pyversions/tracepipe.svg)](https://pypi.org/project/tracepipe/)
[![CI](https://github.com/gauthierpiarrette/tracepipe/actions/workflows/ci.yml/badge.svg)](https://github.com/gauthierpiarrette/tracepipe/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/gauthierpiarrette/tracepipe/branch/main/graph/badge.svg)](https://codecov.io/gh/gauthierpiarrette/tracepipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://gauthierpiarrette.github.io/tracepipe/)

---

## The Problem

Data pipelines are black boxes. When something goes wrong, you're left asking:

- **"Where did row X go?"** — Dropped somewhere, but which step?
- **"Why is this value wrong?"** — It was fine in the source, what changed it?
- **"How did these rows get merged?"** — Which parent records combined?
- **"Why are there nulls here?"** — When did they appear?

```python
df = pd.read_csv("customers.csv")
df = df.dropna()                          # Some rows disappear
df = df.merge(regions, on="zip")          # New rows appear, some vanish
df["income"] = df["income"].fillna(0)     # Values change silently
df = df[df["age"] >= 18]                  # More rows gone
# What actually happened to customer C-789?
```

Traditional debugging means `print()` statements, manual diffs, and guesswork. **TracePipe gives you the complete audit trail.**

---

## The Solution

```python
import tracepipe as tp
import pandas as pd

tp.enable(mode="debug", watch=["income", "score"])

df = pd.read_csv("customers.csv")
df = df.dropna()
df["income"] = df["income"].fillna(0)
df = df.merge(segments, on="customer_id")
df = df[df["age"] >= 18]

# Pipeline health check
print(tp.check(df))
```
```
TracePipe Check: [OK] Pipeline healthy
  Mode: debug

Retention: 847/1000 (84.7%)
Dropped: 153 rows
  • DataFrame.dropna: 42
  • DataFrame.__getitem__[mask]: 111

Value changes: 23 cells modified
  • DataFrame.fillna: 23 (income)
```

```python
# Why did this customer's income change?
print(tp.why(df, col="income", where={"customer_id": "C-789"}))
```
```
Cell History: row 42, column 'income'
  Current value: 0.0
  [i] Was null at step 1 (later recovered)
      by: DataFrame.fillna

  History (1 change):
    None -> 0.0
      by: DataFrame.fillna
```

**One import. Complete audit trail.**

---

## Installation

```bash
pip install tracepipe
```

---

## Quick Start

### 1. Enable tracking

```python
import tracepipe as tp

tp.enable(mode="debug", watch=["price", "quantity"])  # Track specific columns
```

### 2. Run your pipeline normally

```python
df = pd.DataFrame({
    "product": ["A", "B", "C", "D"],
    "price": [10.0, None, 30.0, 40.0],
    "quantity": [5, 10, 0, 8]
})

df = df.dropna()                    # Drops row B
df = df[df["quantity"] > 0]         # Drops row C
df["total"] = df["price"] * df["quantity"]
```

### 3. Inspect the lineage

```python
# Health check - see drops AND changes
print(tp.check(df))
```
```
TracePipe Check: [OK] Pipeline healthy
  Mode: debug

Retention: 2/4 (50.0%)
Dropped: 2 rows
  • DataFrame.dropna: 1
  • DataFrame.__getitem__[mask]: 1

Value changes: 2 cells
  • DataFrame.__setitem__[total]: 2
```

```python
# Trace a specific row's full journey
print(tp.trace(df, where={"product": "A"}))
```
```
Row 0 Journey:
  Status: [OK] Alive

  Events: 1
    [MODIFIED] DataFrame.__setitem__[total]: total
```

```python
# Explain why a specific cell has its current value
print(tp.why(df, col="total", row=0))
```
```
Cell History: row 0, column 'total'
  Current value: 50.0

  History (1 change):
    None -> 50.0
      by: DataFrame.__setitem__[total]
```

---

## Key Features

### 🔍 Zero-Code Instrumentation

TracePipe monkey-patches pandas at runtime. Your existing code works unchanged:

```python
tp.enable()
# Your existing pipeline runs exactly as before
# TracePipe silently records everything
tp.disable()
```

### 📊 Rich Provenance Data

Track everything that happens in your pipeline:

| Question | Answer |
|----------|--------|
| Which rows were dropped? | `tp.check(df)` shows retention by operation |
| Why did this value change? | `tp.why(df, col="amount", row=5)` shows before/after |
| What's this row's history? | `tp.trace(df, row=0)` shows full journey |
| Where did these rows merge from? | Merge parent tracking in debug mode |
| Which rows grouped together? | `tp.debug.inspect().explain_group("A")` |
| When did nulls appear? | `tp.why()` flags null introduction |

### 🎯 Business-Key Lookups

Find rows by their values, not internal IDs:

```python
# Find by business key
tp.trace(df, where={"customer_id": "C-12345"})
tp.trace(df, where={"email": "alice@example.com"})

# Find rows where a column is null
tp.why(df, col="email", where={"email": None})
```

### 📈 Production-Ready Performance

| Operation | Overhead | Notes |
|-----------|----------|-------|
| Filter (dropna, query) | 1.4-1.9x | Acceptable |
| Transform (fillna, replace) | 1.0-1.2x | Minimal |
| GroupBy | 1.0-1.2x | Minimal |
| Sort | 1.4x | Optimized |
| Scalar access (at/iat) | <1ms added | Fixed overhead |

Tested on DataFrames up to 1M rows with linear scaling.

### 🔒 Safety First

TracePipe never modifies your data or affects computation results:

```python
# Original pandas method ALWAYS runs first
# Lineage capture happens after, and failures are non-fatal
result = df.dropna()  # Guaranteed to work, even if tracking fails
```

---

## Two Modes

### CI Mode (Default)
Lightweight tracking for production pipelines:
- Step counts and retention rates
- Dropped row detection
- Merge mismatch warnings
- **No per-row provenance** (fast)

```python
tp.enable(mode="ci")
```

### Debug Mode
Full lineage for development and debugging:
- Complete row-level history
- Cell change tracking with before/after values
- GroupBy membership
- Merge parent tracking

```python
tp.enable(mode="debug", watch=["price", "amount"])
```

---

## API Reference

### Core Functions (5)

| Function | Purpose |
|----------|---------|
| `tp.enable(mode, watch)` | Start tracking |
| `tp.check(df)` | Health check with retention stats |
| `tp.trace(df, row, where)` | Trace a row's journey |
| `tp.why(df, col, row, where)` | Explain why a cell changed |
| `tp.report(df, path)` | Export HTML report |

### Control Functions

| Function | Purpose |
|----------|---------|
| `tp.disable()` | Stop tracking |
| `tp.reset()` | Clear all lineage data |
| `tp.stage(name)` | Label pipeline stages |

### Debug Namespace

For power users who need raw access:

```python
dbg = tp.debug.inspect()
dbg.steps              # All recorded operations
dbg.dropped_rows()     # Set of dropped row IDs
dbg.explain_row(42)    # Raw lineage for row 42
dbg.stats()            # Memory and tracking stats
dbg.export("json", "lineage.json")
```

---

## Data Quality Contracts

Validate your pipeline with fluent assertions:

```python
result = (tp.contract()
    .expect_unique("customer_id")
    .expect_no_nulls("email")
    .expect_retention(min_rate=0.9)
    .check(df))

result.raise_if_failed()  # Raises if any contract violated
```

---

## Snapshots & Diff

Compare DataFrame states:

```python
before = tp.snapshot(df)

# ... transformations ...

after = tp.snapshot(df)
diff = tp.diff(before, after)

print(f"Rows added: {diff.rows_added}")
print(f"Rows removed: {diff.rows_removed}")
print(f"Cells changed: {diff.cells_changed}")
```

---

## HTML Reports

Generate interactive lineage reports:

```python
tp.report(df, "pipeline_audit.html")
```

Opens a visual dashboard showing:
- Pipeline flow diagram
- Retention funnel
- Dropped rows by operation
- Cell change history

---

## What's Tracked

| Operation | Tracking | Completeness |
|-----------|----------|--------------|
| `dropna`, `drop_duplicates` | Dropped row IDs | FULL |
| `query`, `df[mask]` | Dropped row IDs | FULL |
| `head`, `tail`, `sample` | Dropped row IDs | FULL |
| `fillna`, `replace` | Cell diffs (watched cols) | FULL |
| `loc[]=`, `iloc[]=`, `at[]=` | Cell diffs | FULL |
| `merge`, `join` | Parent tracking | FULL |
| `groupby().agg()` | Group membership | FULL |
| `sort_values` | Reorder tracking | FULL |
| `apply`, `pipe` | Output tracked | PARTIAL |

---

## Limitations

TracePipe tracks pandas operations, not arbitrary Python code:

| Limitation | Workaround |
|------------|------------|
| Direct NumPy array modification | Use pandas methods |
| Mutable objects in cells (lists, dicts) | Use immutable types |
| Custom C extensions | Wrap with pandas operations |

---

## Example: ML Pipeline Audit

```python
import tracepipe as tp
import pandas as pd
import numpy as np

tp.enable(mode="debug", watch=["age", "income", "label"])

# Load and clean
df = pd.read_csv("training_data.csv")
df = df.dropna(subset=["label"])
df["income"] = df["income"].fillna(df["income"].median())
df = df[df["age"] >= 18]

# Feature engineering
df["age_bucket"] = pd.cut(df["age"], bins=[18, 30, 50, 100])
df["log_income"] = np.log1p(df["income"])

# Audit the pipeline
print(tp.check(df))
```
```
TracePipe Check: [OK] Pipeline healthy
  Mode: debug

Retention: 8234/10000 (82.3%)
Dropped: 1766 rows
  • DataFrame.dropna: 423
  • DataFrame.__getitem__[mask]: 1343

Value changes: 892 cells
  • DataFrame.fillna: 892 (income)
```

```python
# Why does this customer have log_income = 0?
print(tp.why(df, col="income", where={"customer_id": "C-789"}))
```
```
Cell History: row 156, column 'income'
  Current value: 45000.0
  [i] Was null at step 1 (later recovered)
      by: DataFrame.fillna

  History (1 change):
    None -> 45000.0
      by: DataFrame.fillna
```

```python
# Full journey of a specific row
print(tp.trace(df, where={"customer_id": "C-789"}))
```
```
Row 156 Journey:
  Status: [OK] Alive

  Events: 3
    [MODIFIED] DataFrame.fillna: income
    [MODIFIED] pd.cut: age_bucket
    [MODIFIED] DataFrame.__setitem__[log_income]: log_income
```

---

## Benchmarks

Run on MacBook Pro M1, pandas 2.0, Python 3.11:

### Overhead (10K rows, median of 10 runs)

| Operation | Baseline | With TracePipe | Overhead |
|-----------|----------|----------------|----------|
| dropna | 0.9ms | 1.7ms | 1.9x |
| query | 2.1ms | 3.0ms | 1.4x |
| fillna | 0.4ms | 0.4ms | 1.0x |
| groupby.sum | 1.2ms | 1.2ms | 1.0x |
| merge | 4.5ms | 12.6ms | 2.8x |
| sort_values | 1.1ms | 1.5ms | 1.4x |

### Scale (filter + dropna pipeline)

| Rows | Time | Throughput |
|------|------|------------|
| 10K | 5ms | 2M rows/sec |
| 100K | 35ms | 2.8M rows/sec |
| 1M | 320ms | 3.1M rows/sec |

### Memory

- Base overhead: ~40 bytes per tracked diff
- Typical pipeline: 2-3x memory vs baseline
- Spillover to disk available for large pipelines

---

## Documentation

📚 **[Full Documentation](https://gauthierpiarrette.github.io/tracepipe/)**

- [Getting Started](https://gauthierpiarrette.github.io/tracepipe/getting-started/quickstart/)
- [User Guide](https://gauthierpiarrette.github.io/tracepipe/guide/concepts/)
- [API Reference](https://gauthierpiarrette.github.io/tracepipe/api/)
- [Examples](https://gauthierpiarrette.github.io/tracepipe/examples/ml-pipeline/)

---

## Contributing

```bash
git clone https://github.com/gauthierpiarrette/tracepipe.git
cd tracepipe
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
ruff check tracepipe/ tests/

# Run benchmarks
python benchmarks/run_all.py
```

See [CONTRIBUTING](https://gauthierpiarrette.github.io/tracepipe/contributing/) for detailed guidelines.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Stop guessing where your rows went.</b><br>
  <code>pip install tracepipe</code>
</p>

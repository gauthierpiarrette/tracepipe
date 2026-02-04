<div align="center">

# TracePipe

### Row-level data lineage for pandas pipelines

**Know exactly where every row went, why values changed, and how your data transformed.**

[![PyPI version](https://img.shields.io/pypi/v/tracepipe.svg)](https://pypi.org/project/tracepipe/)
[![Python 3.9+](https://img.shields.io/pypi/pyversions/tracepipe.svg)](https://pypi.org/project/tracepipe/)
[![pandas 1.5-2.2](https://img.shields.io/badge/pandas-1.5--2.2-blue.svg)](https://github.com/gauthierpiarrette/tracepipe/blob/main/tests/test_version_matrix.py)
[![CI](https://github.com/gauthierpiarrette/tracepipe/actions/workflows/ci.yml/badge.svg)](https://github.com/gauthierpiarrette/tracepipe/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/gauthierpiarrette/tracepipe/branch/main/graph/badge.svg)](https://codecov.io/gh/gauthierpiarrette/tracepipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://gauthierpiarrette.github.io/tracepipe/)

[Getting Started](#getting-started) · [Documentation](https://gauthierpiarrette.github.io/tracepipe/) · [Examples](#real-world-example)

</div>

---

## Why TracePipe?

Data pipelines are black boxes. Rows vanish. Values change. You're left guessing.

```python
df = pd.read_csv("customers.csv")
df = df.dropna()                      # Some rows disappear
df = df.merge(regions, on="zip")      # New rows appear, some vanish
df["income"] = df["income"].fillna(0) # Values change silently
df = df[df["age"] >= 18]              # More rows gone
# What happened to customer C-789? 🤷
```

**TracePipe gives you the complete audit trail — zero code changes required.**

---

## Getting Started

```bash
pip install tracepipe
```

```python
import tracepipe as tp
import pandas as pd

tp.enable(mode="debug", watch=["income"])

df = pd.read_csv("customers.csv")
df = df.dropna()
df["income"] = df["income"].fillna(0)
df = df[df["age"] >= 18]

tp.check(df)  # See what happened
```

```
TracePipe Check: [OK] Pipeline healthy

Retention: 847/1000 (84.7%)
Dropped: 153 rows
  • DataFrame.dropna: 42
  • DataFrame.__getitem__[mask]: 111

Value changes: 23 cells modified
  • DataFrame.fillna: 23 (income)
```

That's it. **One import, full visibility.**

---

## Core API

| Function | What it does |
|----------|--------------|
| `tp.enable()` | Start tracking |
| `tp.check(df)` | Health check — retention, drops, changes |
| `tp.trace(df, where={"id": "C-789"})` | Follow a row's complete journey |
| `tp.why(df, col="income", row=5)` | Explain why a cell has its current value |
| `tp.report(df, "audit.html")` | Export interactive HTML report |

---

## Key Features

<table>
<tr>
<td width="50%">

### 🔍 Zero-Code Instrumentation
TracePipe patches pandas at runtime. Your existing code works unchanged.

### 📊 Complete Provenance
Track drops, transforms, merges, and cell-level changes with before/after values.

</td>
<td width="50%">

### 🎯 Business-Key Lookups
Find rows by their values: `tp.trace(df, where={"email": "alice@example.com"})`

### ⚡ Production-Ready
1.0-2.8x overhead (varies by operation). Tested on DataFrames up to 1M rows.

</td>
</tr>
</table>

---

## Real-World Example

```python
import tracepipe as tp
import pandas as pd

tp.enable(mode="debug", watch=["age", "income", "label"])

# Load and clean
df = pd.read_csv("training_data.csv")
df = df.dropna(subset=["label"])
df["income"] = df["income"].fillna(df["income"].median())
df = df[df["age"] >= 18]

# Audit
print(tp.check(df))
```

```
Retention: 8234/10000 (82.3%)
Dropped: 1766 rows
  • DataFrame.dropna: 423
  • DataFrame.__getitem__[mask]: 1343

Value changes: 892 cells
  • DataFrame.fillna: 892 (income)
```

```python
# Why does this customer have a filled income?
tp.why(df, col="income", where={"customer_id": "C-789"})
```

```
Cell History: row 156, column 'income'
  Current value: 45000.0
  [i] Was null at step 1 (later recovered)

  History (1 change):
    None -> 45000.0
      by: DataFrame.fillna
```

---

## Two Modes

| Mode | Use Case | What's Tracked |
|------|----------|----------------|
| **CI** (default) | Production pipelines | Step counts, retention rates, merge warnings |
| **Debug** | Development | Full row history, cell diffs, merge parents, group membership |

```python
tp.enable(mode="ci")     # Lightweight
tp.enable(mode="debug")  # Full lineage
```

---

## What's Tracked

| Operation | Coverage |
|-----------|----------|
| `dropna`, `drop_duplicates`, `query`, `df[mask]` | ✅ Full |
| `fillna`, `replace`, `loc[]=`, `iloc[]=` | ✅ Full (cell diffs) |
| `merge`, `join` | ✅ Full (parent tracking) |
| `groupby().agg()` | ✅ Full (group membership) |
| `sort_values`, `head`, `tail`, `sample` | ✅ Full |
| `apply`, `pipe` | ⚠️ Partial |

---

## Data Quality Contracts

```python
(tp.contract()
    .expect_unique("customer_id")
    .expect_no_nulls("email")
    .expect_retention(min_rate=0.9)
    .check(df)
    .raise_if_failed())
```

---

## Documentation

📚 **[Full Documentation](https://gauthierpiarrette.github.io/tracepipe/)**

- [Quickstart](https://gauthierpiarrette.github.io/tracepipe/getting-started/quickstart/)
- [User Guide](https://gauthierpiarrette.github.io/tracepipe/guide/concepts/)
- [API Reference](https://gauthierpiarrette.github.io/tracepipe/api/)
- [Examples](https://gauthierpiarrette.github.io/tracepipe/examples/ml-pipeline/)

---

## Known Limitations

TracePipe tracks **cell mutations**, **merge provenance**, **concat provenance**, and **duplicate drop decisions** reliably. A few patterns have limited tracking:

| Pattern | Status | Notes |
|---------|--------|-------|
| `df["col"] = df["col"].fillna(0)` | ✅ Tracked | Series + assignment |
| `df = df.fillna({"col": 0})` | ✅ Tracked | DataFrame-level fillna |
| `df.loc[mask, "col"] = val` | ✅ Tracked | Conditional assignment |
| `df.merge(other, on="key")` | ✅ Tracked | Full provenance in debug mode |
| `pd.concat([df1, df2])` | ✅ Tracked | Row IDs preserved with source DataFrame tracking (v0.4+) |
| `df.drop_duplicates()` | ✅ Tracked | Dropped rows map to kept representative (debug mode, v0.4+) |
| `pd.concat(axis=1)` | ⚠️ Partial | FULL only if all inputs have identical RIDs |
| Complex `apply`/`pipe` | ⚠️ Partial | Output tracked, internals opaque |

---

## Contributing

```bash
git clone https://github.com/gauthierpiarrette/tracepipe.git
cd tracepipe
pip install -e ".[dev]"
pytest tests/ -v
```

See [CONTRIBUTING](https://gauthierpiarrette.github.io/tracepipe/contributing/) for guidelines.

---

## License

MIT License. See [LICENSE](LICENSE).

---

<div align="center">

**Stop guessing where your rows went.**

```bash
pip install tracepipe
```

⭐ Star us on GitHub if TracePipe helps your data work!

</div>

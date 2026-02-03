# TracePipe

**Row-level data lineage tracking for pandas pipelines.**

TracePipe automatically tracks what happens to every row and cell in your DataFrame — drops, transformations, merges, and value changes. Zero code changes required.

<div class="grid cards" markdown>

- :material-rocket-launch: **[Quick Start](getting-started/quickstart.md)**

    Get up and running in 5 minutes

- :material-book-open: **[User Guide](guide/concepts.md)**

    Learn the core concepts and features

- :material-api: **[API Reference](api/index.md)**

    Complete function documentation

- :material-code-tags: **[Examples](examples/ml-pipeline.md)**

    Real-world usage patterns

</div>

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

Traditional debugging means `print()` statements, manual diffs, and guesswork.

---

## The Solution

```python
import tracepipe as tp
import pandas as pd

tp.enable(mode="debug", watch=["income"])

df = pd.read_csv("customers.csv")
df = df.dropna()
df["income"] = df["income"].fillna(0)
df = df.merge(regions, on="zip")
df = df[df["age"] >= 18]

# What actually happened to customer C-789?
print(tp.trace(df, where={"customer_id": "C-789"}))
```

```
Row 789 Journey:
  Status: [DROPPED]
  Dropped by: DataFrame.__getitem__[mask] (step 5)

  Events:
    [SURVIVED] DataFrame.dropna
    [MODIFIED] DataFrame.fillna: income (None → 0)
    [SURVIVED] DataFrame.merge
    [DROPPED]  DataFrame.__getitem__[mask]  ← age filter
```

Now you know: **C-789 had null income (filled to 0), survived the merge, but was dropped by the age filter.**

```python
# Pipeline health overview
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

**One import. Complete audit trail.**

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Zero-Code Instrumentation** | Works with existing pandas code unchanged |
| **Row-Level Tracking** | Know exactly where each row went |
| **Cell Provenance** | See before/after values for every change |
| **Merge Parent Tracking** | Understand which rows combined |
| **Data Contracts** | Validate retention rates and uniqueness |
| **HTML Reports** | Generate visual pipeline audits |

---

## Installation

```bash
pip install tracepipe
```

For optional features:

```bash
pip install tracepipe[arrow]   # Parquet/Arrow support
pip install tracepipe[all]     # All optional dependencies
```

---

## Quick Example

```python
import tracepipe as tp
import pandas as pd

# Enable tracking
tp.enable(mode="debug", watch=["price"])

# Your normal pandas code
df = pd.DataFrame({
    "product": ["A", "B", "C"],
    "price": [10.0, None, 30.0]
})
df = df.dropna()
df["price"] = df["price"] * 1.1

# Inspect what happened
print(tp.check(df))      # Health summary
print(tp.trace(df, 0))   # Row 0's journey
print(tp.why(df, "price", 0))  # Why price changed
```

---

## What's Tracked

| Operation | Tracking | Completeness |
|-----------|----------|--------------|
| `dropna`, `drop_duplicates` | Dropped row IDs | Full |
| `query`, `df[mask]` | Dropped row IDs | Full |
| `head`, `tail`, `sample` | Dropped row IDs | Full |
| `fillna`, `replace` | Cell diffs (watched cols) | Full |
| `loc[]=`, `iloc[]=`, `at[]=` | Cell diffs | Full |
| `merge`, `join` | Parent tracking | Full |
| `groupby().agg()` | Group membership | Full |
| `apply`, `pipe` | Output tracked | Partial |

---

## License

TracePipe is released under the [MIT License](https://opensource.org/licenses/MIT).

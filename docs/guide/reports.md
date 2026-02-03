# HTML Reports

Generate interactive visual reports of your pipeline.

## Basic Usage

```python
tp.report(df, "pipeline_audit.html")
```

This creates a standalone HTML file with:

- Pipeline flow diagram
- Retention funnel visualization
- Dropped rows breakdown
- Cell change history
- Interactive filtering

## Report Contents

### Pipeline Overview

Shows high-level statistics:

- Total rows processed
- Final row count
- Overall retention rate
- Number of steps

### Retention Funnel

Visual representation of how rows flow through each step:

```
Load:     ████████████████████ 1000 rows
dropna:   ████████████████░░░░  850 rows (-150)
filter:   ████████████░░░░░░░░  600 rows (-250)
merge:    ██████████████████░░  900 rows (+300)
final:    ████████████████░░░░  847 rows (-53)
```

### Drops by Operation

Breakdown of which operations dropped the most rows:

| Operation | Rows Dropped | % of Total |
|-----------|--------------|------------|
| DataFrame.dropna | 150 | 37% |
| DataFrame.__getitem__[mask] | 250 | 62% |
| DataFrame.drop_duplicates | 3 | 1% |

### Cell Changes

For watched columns, shows modification history:

| Column | Changes | Operations |
|--------|---------|------------|
| income | 423 | DataFrame.fillna |
| status | 89 | DataFrame.__setitem__ |

### Ghost Values

Last known values of dropped rows (debug mode only):

| Row ID | email | status | Dropped By |
|--------|-------|--------|------------|
| 42 | alice@... | active | dropna |
| 156 | bob@... | inactive | filter |

## Options

### Custom Title

```python
tp.report(df, "audit.html", title="Customer Pipeline - Q4 2024")
```

### Include Raw Data

```python
tp.report(df, "audit.html", include_data=True)
```

Adds a data preview table to the report. Use with caution for large DataFrames.

### Minimal Report

```python
tp.report(df, "audit.html", minimal=True)
```

Generates a simpler report without charts (faster, smaller file size).

## Programmatic Access

If you need the report data without HTML:

```python
# Get report data as dict
dbg = tp.debug.inspect()
report_data = dbg.export("dict")

# Contains:
# - steps: list of all operations
# - drops: dropped row details
# - changes: cell modifications
# - stats: summary statistics
```

## Viewing Reports

The generated HTML is self-contained (no external dependencies). Open it in any browser:

```bash
# macOS
open pipeline_audit.html

# Linux
xdg-open pipeline_audit.html

# Windows
start pipeline_audit.html
```

## Integration with Notebooks

In Jupyter notebooks, you can display reports inline:

```python
from IPython.display import HTML

tp.report(df, "audit.html")
HTML("audit.html")
```

Or use the display helper:

```python
# If available
tp.display_report(df)  # Renders inline in notebook
```

## Example Report

```python
import tracepipe as tp
import pandas as pd

tp.enable(mode="debug", watch=["income", "status"])

# Sample pipeline
df = pd.read_csv("customers.csv")
df = df.dropna(subset=["email"])
df["income"] = df["income"].fillna(df["income"].median())
df = df[df["age"] >= 18]
df = df.merge(segments, on="segment_id")

# Generate comprehensive report
tp.report(df, "customer_pipeline_audit.html", title="Customer ETL Pipeline")
```

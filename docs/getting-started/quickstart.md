# Quick Start

Get up and running with TracePipe in 5 minutes.

## 1. Enable Tracking

```python
import tracepipe as tp
import pandas as pd

# Start tracking with debug mode for full lineage
tp.enable(mode="debug", watch=["price", "quantity"])
```

The `watch` parameter specifies which columns to track for cell-level changes.

## 2. Run Your Pipeline

Write your pandas code as usual — TracePipe instruments it automatically:

```python
df = pd.DataFrame({
    "product": ["A", "B", "C", "D"],
    "price": [10.0, None, 30.0, 40.0],
    "quantity": [5, 10, 0, 8]
})

df = df.dropna()                    # Drops row B (null price)
df = df[df["quantity"] > 0]         # Drops row C (zero quantity)
df["total"] = df["price"] * df["quantity"]
```

## 3. Check Pipeline Health

```python
result = tp.check(df)
print(result)
```

Output:

```
TracePipe Check: [OK] Pipeline healthy
  Mode: debug

Retention: 50%
Dropped: 2 rows
  • DataFrame.dropna: 1
  • DataFrame.__getitem__[mask]: 1

Value changes: 2 cells
  • DataFrame.__setitem__[total]: 2
```

The `CheckResult` object provides convenient properties:

```python
result.passed       # True/False
result.retention    # 0.5 (row retention rate)
result.n_dropped    # 2 (total dropped rows)
result.drops_by_op  # {"DataFrame.dropna": 1, ...}
result.n_changes    # 2 (cell changes, debug mode only)
result.changes_by_op # {"DataFrame.__setitem__[total]": 2}
```

## 4. Trace a Row's Journey

```python
trace = tp.trace(df, where={"product": "A"})
print(trace)
```

Output:

```
Row 0 Journey:
  Status: [OK] Alive

  Events: 1
    [MODIFIED] DataFrame.__setitem__[total]: total
```

## 5. Explain a Cell's Value

```python
why = tp.why(df, col="total", row=0)
print(why)
```

Output:

```
Cell History: row 0, column 'total'
  Current value: 50.0

  History (1 change):
    None -> 50.0
      by: DataFrame.__setitem__[total]
```

## 6. Generate a Report

```python
tp.report(df, "pipeline_audit.html")
```

This creates an interactive HTML report with:

- Pipeline flow diagram
- Retention funnel visualization
- Dropped rows by operation
- Cell change history

## 7. Clean Up

```python
tp.disable()  # Stop tracking
tp.reset()    # Clear all lineage data
```

---

## Next Steps

- Learn about [CI vs Debug modes](modes.md)
- Explore [health checks](../guide/health-checks.md) in depth
- Set up [data contracts](../guide/contracts.md)
- See [real-world examples](../examples/ml-pipeline.md)

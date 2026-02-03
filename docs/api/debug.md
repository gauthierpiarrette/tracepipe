# Debug API

Advanced debugging and inspection tools.

## Accessing Debug Tools

### inspect

```python
tp.debug.inspect() -> DebugInspector
```

Get a debug inspector for raw lineage access.

**Returns:** `DebugInspector`

**Example:**

```python
dbg = tp.debug.inspect()
print(f"Steps recorded: {len(dbg.steps)}")
```

---

## DebugInspector Properties

### steps

```python
dbg.steps -> list[Step]
```

All recorded pipeline steps.

Each `Step` contains:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.operation` | `str` | Operation name |
| `.input_shape` | `tuple` | Input (rows, cols) |
| `.output_shape` | `tuple` | Output (rows, cols) |
| `.timestamp` | `datetime` | When it occurred |
| `.stage` | `str` | Pipeline stage (if set) |

**Example:**

```python
for step in dbg.steps:
    print(f"{step.operation}: {step.input_shape} → {step.output_shape}")
```

---

## DebugInspector Methods

### dropped_rows

```python
dbg.dropped_rows() -> set[int]
```

Get IDs of all dropped rows.

**Example:**

```python
dropped = dbg.dropped_rows()
print(f"Total dropped: {len(dropped)}")
```

---

### explain_row

```python
dbg.explain_row(row_id: int) -> RowExplanation
```

Get detailed explanation for a specific row.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `row_id` | `int` | Internal row ID |

**Returns:** `RowExplanation`

**Example:**

```python
for rid in list(dbg.dropped_rows())[:5]:
    explanation = dbg.explain_row(rid)
    print(f"Row {rid}: {explanation.status}")
```

---

### explain_group

```python
dbg.explain_group(group_key: Any) -> GroupExplanation
```

Explain which rows belonged to a group (after groupby).

**Example:**

```python
# After: df.groupby("category").sum()
explanation = dbg.explain_group("Electronics")
print(f"Group 'Electronics' had {len(explanation.member_ids)} rows")
```

---

### get_ghost_values

```python
dbg.get_ghost_values(row_id: int) -> dict[str, Any] | None
```

Get last known values of a dropped row.

**Example:**

```python
dropped_rid = list(dbg.dropped_rows())[0]
ghost = dbg.get_ghost_values(dropped_rid)
if ghost:
    print(f"Last values: {ghost}")
```

---

### stats

```python
dbg.stats() -> dict
```

Get tracking statistics.

**Returns:**

```python
{
    "steps_recorded": 15,
    "rows_tracked": 1000,
    "rows_dropped": 153,
    "cells_tracked": 5000,
    "memory_bytes": 102400,  # If psutil available
}
```

---

### export

```python
dbg.export(format: str, path: str | None = None) -> dict | None
```

Export lineage data.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `format` | `str` | `"json"`, `"dict"`, or `"csv"` |
| `path` | `str` | Output file (optional) |

**Returns:** Data dict if `path` is None, else None.

**Example:**

```python
# Export to file
dbg.export("json", "lineage.json")

# Get as dict
data = dbg.export("dict")
```

---

## Complete Example

```python
import tracepipe as tp

tp.enable(mode="debug", watch=["price", "status"])

# Run pipeline
df = pd.read_csv("data.csv")
df = df.dropna()
df["price"] = df["price"] * 1.1
df = df[df["price"] > 10]

# Deep inspection
dbg = tp.debug.inspect()

# Review all steps
print("Pipeline steps:")
for i, step in enumerate(dbg.steps):
    print(f"  {i+1}. {step.operation}")
    print(f"     {step.input_shape} → {step.output_shape}")

# Investigate dropped rows
dropped = dbg.dropped_rows()
print(f"\nDropped {len(dropped)} rows")

# Look at specific dropped rows
for rid in list(dropped)[:3]:
    ghost = dbg.get_ghost_values(rid)
    if ghost:
        print(f"  Row {rid}: price was {ghost.get('price')}")

# Export for external analysis
dbg.export("json", "pipeline_lineage.json")

# Stats
stats = dbg.stats()
print(f"\nMemory used: {stats.get('memory_bytes', 'N/A')} bytes")
```

# Row Tracing

Trace the complete journey of any row through your pipeline.

## Basic Usage

```python
# By row index
trace = tp.trace(df, row=0)
print(trace)

# By business key
trace = tp.trace(df, where={"customer_id": "C-12345"})
print(trace)
```

Output:

```
Row 42 Journey:
  Status: [OK] Alive

  Events: 3
    [SURVIVED] DataFrame.dropna
    [MODIFIED] DataFrame.fillna: income
    [SURVIVED] DataFrame.__getitem__[mask]
```

## The TraceResult Object

```python
trace = tp.trace(df, row=0)

# Access fields
trace.row_id           # int: internal row ID
trace.status           # str: "alive" or "dropped"
trace.events           # list[TraceEvent]: all events

# For dropped rows
trace.dropped_by       # str: operation that dropped the row
trace.dropped_at_step  # int: step number

# Export
trace.to_dict()        # dict representation
```

## Finding Rows

### By Index

```python
# Current DataFrame index
tp.trace(df, row=0)      # First row in current df
tp.trace(df, row=-1)     # Last row in current df
```

### By Business Key

```python
# Single key
tp.trace(df, where={"email": "alice@example.com"})

# Multiple keys (AND condition)
tp.trace(df, where={"region": "US", "status": "active"})

# Find row with null value
tp.trace(df, where={"email": None})
```

!!! tip "Use Business Keys"
    Business keys are more stable than row indices, which change as rows are filtered.

## Event Types

| Event Type | Description |
|------------|-------------|
| `SURVIVED` | Row passed through operation unchanged |
| `MODIFIED` | One or more cells changed |
| `DROPPED` | Row was removed |
| `CREATED` | Row first appeared (e.g., from merge) |

## Tracing Dropped Rows

You can trace rows that were dropped:

```python
dbg = tp.debug.inspect()

# Get IDs of dropped rows
dropped_ids = dbg.dropped_rows()

# Trace a specific dropped row
for rid in list(dropped_ids)[:5]:
    trace = dbg.explain_row(rid)
    print(f"Row {rid}: dropped by {trace.dropped_by}")
```

## Merge Parent Tracking

For rows created by merges, TracePipe tracks their parents:

```python
result = df1.merge(df2, on="id")
trace = tp.trace(result, row=0)

# In debug mode, you can see parent rows
if trace.merge_parents:
    print(f"Left parent: {trace.merge_parents.left}")
    print(f"Right parent: {trace.merge_parents.right}")
```

---

## Concat Origin Tracking (v0.4+)

When rows come from concatenated DataFrames, TracePipe tracks their source via `trace.origin`:

```python
df1 = pd.DataFrame({"a": [1, 2]})
df2 = pd.DataFrame({"a": [3, 4]})
result = pd.concat([df1, df2])

# Trace a row that came from df2
trace = tp.trace(result, row=2)
print(trace.origin)
# {"type": "concat", "source_df": 1, "step_id": 5}
```

The `.origin` property returns a unified dict with:

- `type`: `"concat"`, `"merge"`, or `None` (for original rows)
- `source_df`: Index in the concat list (0=first DataFrame, 1=second, etc.)
- `step_id`: Which pipeline step

Row IDs are preserved through `pd.concat(axis=0)`, so lineage chains correctly:

```python
# Transform df1 before concat
df1["a"] = df1["a"].fillna(0)

result = pd.concat([df1, df2])

# Rows from df1 still have their fillna history
trace = tp.trace(result, row=0)  # Shows fillna event from df1
```

---

## Duplicate Representative Tracking (v0.4+)

When `drop_duplicates` removes rows, TracePipe tracks which row "won" via `trace.representative`:

```python
df = pd.DataFrame({
    "key": ["A", "A", "B"],
    "value": [100, 200, 300]
})
df = df.drop_duplicates(subset=["key"], keep="first")

# Trace the dropped row (value=200)
trace = tp.trace(df, row=dropped_row_id)
print(trace.representative)
# {"kept_rid": 42, "subset": ["key"], "keep": "first"}
```

The `.representative` property is only set for rows dropped by `drop_duplicates`:

| `keep` Strategy | `.representative` |
|-----------------|-------------------|
| `keep='first'` | `{"kept_rid": 42, ...}` — first occurrence kept |
| `keep='last'` | `{"kept_rid": 45, ...}` — last occurrence kept |
| `keep=False` | `{"kept_rid": None, ...}` — all duplicates removed |

This answers "why did this row disappear?" — it wasn't deleted, it was deduplicated.

## Performance Considerations

- Row tracing in CI mode is limited (no individual row IDs)
- For large DataFrames, use `where=` with indexed columns for faster lookups
- Tracing many rows? Use `tp.debug.inspect()` for batch access

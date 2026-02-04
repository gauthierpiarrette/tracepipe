# Cell Provenance

Understand why a specific cell has its current value.

## Basic Usage

```python
# By row index and column
why = tp.why(df, col="income", row=0)
print(why)

# By business key
why = tp.why(df, col="income", where={"customer_id": "C-12345"})
print(why)
```

Output:

```
Cell History: row 42, column 'income'
  Current value: 45000.0
  [i] Was null at step 1 (later recovered)
      by: DataFrame.fillna

  History (1 change):
    None -> 45000.0
      by: DataFrame.fillna
```

## The WhyResult Object

```python
why = tp.why(df, col="income", row=0)

# Access fields
why.column           # str: column name
why.row_id           # int: internal row ID
why.current_value    # any: current cell value
why.history          # list[CellChange]: all changes

# Flags
why.was_null         # bool: was ever null
why.null_recovered   # bool: null was later filled

# Export
why.to_dict()        # dict representation
```

## Cell Change Records

Each change in history contains:

```python
for change in why.history:
    print(f"From: {change.old_value}")
    print(f"To: {change.new_value}")
    print(f"By: {change.operation}")
    print(f"At step: {change.step}")
```

## Requirements

!!! warning "Debug Mode Required"
    `tp.why()` requires debug mode with the column being watched:

    ```python
    tp.enable(mode="debug", watch=["income", "status"])
    ```

    Columns not in `watch` will not have cell history.

## Common Use Cases

### Finding Null Introduction

```python
# Where did this null come from?
why = tp.why(df, col="email", where={"email": None})

if why.was_null:
    print("This cell was null from the start")
else:
    # Find which operation set it to null
    for change in why.history:
        if change.new_value is None:
            print(f"Null introduced by: {change.operation}")
```

### Tracking Value Changes

```python
# How did this price get so high?
why = tp.why(df, col="price", row=0)

for change in why.history:
    pct_change = (change.new_value - change.old_value) / change.old_value * 100
    print(f"{change.operation}: {pct_change:+.1f}%")
```

### Auditing Sensitive Fields

```python
# Who touched the salary column?
for idx in range(len(df)):
    why = tp.why(df, col="salary", row=idx)
    if why.history:
        print(f"Row {idx}: {len(why.history)} changes")
```

## Finding Cells

### By Index

```python
tp.why(df, col="price", row=0)
tp.why(df, col="price", row=-1)  # Last row
```

### By Business Key

```python
tp.why(df, col="status", where={"order_id": "ORD-123"})
```

### Multiple Matches

If `where=` matches multiple rows, TracePipe returns the first match. For multiple rows, iterate:

```python
matching_rows = df[df["region"] == "US"]
for idx in range(len(matching_rows)):
    why = tp.why(matching_rows, col="price", row=idx)
    # Process each
```

## Ghost Values

For dropped rows, you can still query their last known values:

```python
dbg = tp.debug.inspect()

# Get ghost values for a specific dropped row
dropped_rid = list(dbg.dropped_rows())[0]
ghost = dbg.get_ghost_values(dropped_rid)
print(f"Last known values: {ghost}")
# {"age": 25, "salary": 50000}

# Or get all ghost rows as a DataFrame
ghost_df = dbg.ghost_rows()
print(ghost_df)
# DataFrame with __tp_row_id__, __tp_dropped_by__, and watched columns
```

The `get_ghost_values(row_id)` method returns a dict mapping column names to
their last known values, or `None` if the row wasn't found in ghost storage.

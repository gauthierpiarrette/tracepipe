# Snapshots & Diff

Compare DataFrame states at different points in your pipeline.

## Taking Snapshots

```python
# Capture current state
snapshot = tp.snapshot(df)

print(f"Rows: {snapshot.n_rows}")
print(f"Columns: {snapshot.columns}")
```

## Comparing Snapshots

```python
# Before transformation
before = tp.snapshot(df)

# Apply transformations
df = df.dropna()
df["price"] = df["price"] * 1.1

# After transformation
after = tp.snapshot(df)

# Compare
diff = tp.diff(before, after)
print(diff)
```

Output:

```
Snapshot Diff:
  - 153 rows removed
  ! 153 new drops

  Changes:
    - 847 cells modified
      price: 847
```

!!! tip "Enabling Cell-Level Diff"
    To see cell-level changes, create snapshots with `include_values=True`.

## The Snapshot Object

```python
snapshot = tp.snapshot(df)

# Access fields
snapshot.n_rows        # int: number of rows
snapshot.n_cols        # int: number of columns
snapshot.columns       # list[str]: column names
snapshot.dtypes        # dict: column dtypes
snapshot.row_ids       # set[int]: TracePipe row IDs (if available)
snapshot.timestamp     # datetime: when snapshot was taken

# Data access (optional, if include_data=True)
snapshot.data          # DataFrame copy (if captured)
```

## The DiffResult Object

```python
diff = tp.diff(before, after)

# Row-level changes (always available)
diff.rows_added        # set[int]: IDs of new rows
diff.rows_removed      # set[int]: IDs of removed rows
diff.new_drops         # set[int]: newly dropped row IDs
diff.recovered_rows    # set[int]: rows that were dropped but now exist

# Column changes
diff.columns_added     # list[str]: new columns
diff.columns_removed   # list[str]: removed columns

# Cell-level changes (requires include_values=True on both snapshots)
diff.cells_changed     # int: total modified cells
diff.changed_rows      # set[int]: IDs of rows with value changes
diff.changes_by_column # dict: {col: count}

# Stats changes
diff.stats_changes     # dict: {col: {metric: (old, new)}}
diff.drops_delta       # dict: {operation: delta_count}
```

!!! note "Cell-Level Diff Requirements"
    To get `cells_changed` and `changes_by_column`, both snapshots must be
    created with `include_values=True`:

    ```python
    before = tp.snapshot(df, include_values=True)
    # ... operations ...
    after = tp.snapshot(df, include_values=True)
    diff = tp.diff(before, after)
    print(f"{diff.cells_changed} cells modified")
    ```

## Options

### Include Data

By default, snapshots don't store the actual DataFrame data (for memory efficiency). To include it:

```python
snapshot = tp.snapshot(df, include_data=True)

# Now you can access the data
print(snapshot.data.head())
```

### Save to Disk

```python
# Save snapshot
tp.snapshot(df, path="checkpoint_1.npz")

# Load later
# (Requires include_data=True when saving)
```

## Use Cases

### Debugging Transformations

```python
def investigate_drop():
    before = tp.snapshot(df)
    result = df.dropna()
    after = tp.snapshot(result)

    diff = tp.diff(before, after)
    print(f"dropna removed {diff.rows_removed} rows")
    return result
```

### A/B Comparison

```python
# Original pipeline
tp.enable()
df_a = process_pipeline_v1(data)
snapshot_a = tp.snapshot(df_a)

# Modified pipeline
tp.reset()
df_b = process_pipeline_v2(data)
snapshot_b = tp.snapshot(df_b)

# Compare
diff = tp.diff(snapshot_a, snapshot_b)
print(f"V2 has {diff.rows_added - diff.rows_removed} net rows")
```

### Checkpoint Validation

```python
checkpoints = []

df = pd.read_csv("data.csv")
checkpoints.append(("load", tp.snapshot(df)))

df = df.dropna()
checkpoints.append(("clean", tp.snapshot(df)))

df = df.merge(lookup, on="id")
checkpoints.append(("enrich", tp.snapshot(df)))

# Review pipeline stages
for i in range(1, len(checkpoints)):
    name, snap = checkpoints[i]
    prev_name, prev_snap = checkpoints[i-1]
    diff = tp.diff(prev_snap, snap)
    print(f"{prev_name} → {name}: {diff.rows_removed} dropped, {diff.rows_added} added")
```

## Performance Notes

- Snapshots without data are very lightweight (just metadata)
- Snapshots with data create a full DataFrame copy
- For large DataFrames, consider snapshotting only row IDs (default behavior)

# Core Concepts

Understanding how TracePipe tracks your data.

## Row Identity

TracePipe assigns a unique ID to every row when it first appears in your pipeline. This ID persists through transformations:

```python
tp.enable(mode="debug")

df = pd.DataFrame({"a": [1, 2, 3]})  # Rows get IDs: 0, 1, 2
df = df[df["a"] > 1]                  # Row 0 dropped, IDs 1, 2 remain
df["b"] = df["a"] * 2                 # IDs unchanged
```

### How IDs Are Stored

TracePipe uses one of three storage strategies:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **WeakRef** (default) | IDs stored in a weak reference dictionary | Most pipelines |
| **Attrs** | IDs stored in `df.attrs` | Long-running sessions |
| **Column** | IDs stored as a hidden `__tracepipe_row_id__` column | Maximum reliability |

The hidden column is automatically stripped when exporting with `to_csv()` or `to_parquet()`.

---

## Lineage Events

TracePipe records several types of events:

### Drop Events

When rows are removed from a DataFrame:

```python
df = df.dropna()           # DROP event recorded
df = df[df["age"] >= 18]   # DROP event recorded
df = df.head(100)          # DROP event recorded
```

### Transform Events

When cell values change:

```python
df["price"] = df["price"] * 1.1    # TRANSFORM event (if "price" is watched)
df.loc[0, "status"] = "active"     # TRANSFORM event (if "status" is watched)
df["income"] = df["income"].fillna(0)  # TRANSFORM event
```

### Merge Events

When rows are combined from multiple DataFrames:

```python
result = df1.merge(df2, on="id")  # MERGE event with parent tracking
```

### Group Events

When rows are aggregated:

```python
grouped = df.groupby("category").sum()  # GROUP event with membership
```

---

## The Lineage Store

All events are stored in an in-memory lineage store. You can inspect it directly:

```python
dbg = tp.debug.inspect()

# View all steps
for step in dbg.steps:
    print(f"{step.operation}: {step.input_shape} -> {step.output_shape}")

# View dropped rows
print(f"Total dropped: {len(dbg.dropped_rows())}")

# Export raw lineage
dbg.export("json", "lineage.json")
```

---

## Ghost Values

In debug mode, TracePipe captures the last known values of dropped rows:

```python
tp.enable(mode="debug", watch=["email", "status"])

df = pd.DataFrame({
    "email": ["a@x.com", "b@x.com", None],
    "status": ["active", "inactive", "pending"]
})
df = df.dropna()  # Row 2 dropped, but its values are captured

# Ghost values are available in reports and debug.inspect()
```

This helps answer "what was the value of X when it was dropped?"

---

## Stages

Label pipeline stages for better organization:

```python
tp.stage("load")
df = pd.read_csv("data.csv")

tp.stage("clean")
df = df.dropna()
df = df.drop_duplicates()

tp.stage("transform")
df["total"] = df["price"] * df["quantity"]

# Stages appear in reports and check output
print(tp.check(df))
```

---

## Retention Rate

TracePipe tracks how many rows survive through your pipeline:

```
Retention = Final Rows / Maximum Rows Seen
```

!!! note "Multi-table Pipelines"
    For pipelines with merges, TracePipe uses the maximum row count seen across all steps as the baseline, accounting for row expansion from joins.

A retention rate below 50% triggers a warning by default. Configure this threshold:

```python
result = tp.check(df, retention_threshold=0.3)  # Warn below 30%
```

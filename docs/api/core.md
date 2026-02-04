# Core API

## Tracking Control

### enable

```python
tp.enable(
    mode: str = "ci",
    watch: list[str] | None = None,
    backend: str | None = None,
    identity: str | None = None,
) -> None
```

Start TracePipe tracking.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `str` | `"ci"` | Tracking mode: `"ci"` or `"debug"` |
| `watch` | `list[str]` | `None` | Columns to track for cell changes (debug mode) |
| `backend` | `str` | `None` | Lineage storage backend |
| `identity` | `str` | `None` | Row identity strategy |

**Example:**

```python
# CI mode - lightweight
tp.enable(mode="ci")

# Debug mode with watched columns
tp.enable(mode="debug", watch=["price", "quantity", "status"])
```

---

### disable

```python
tp.disable() -> None
```

Stop TracePipe tracking. Restores original pandas methods.

**Example:**

```python
tp.enable()
# ... tracked operations ...
tp.disable()  # Back to normal pandas
```

---

### reset

```python
tp.reset() -> None
```

Clear all lineage data. Does not disable tracking.

**Example:**

```python
tp.enable()
df = process_data_v1()
tp.check(df)

tp.reset()  # Clear lineage, keep tracking enabled
df = process_data_v2()
tp.check(df)
```

---

### register

```python
tp.register(*dfs: pd.DataFrame) -> None
```

Manually register DataFrames for tracking.

Use this when DataFrames are created before `tp.enable()` is called.

!!! note "Lineage Break"
    Calling `register()` assigns new row IDs, which breaks lineage from any prior transformations. Use it only for "entry point" DataFrames.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `*dfs` | `pd.DataFrame` | One or more DataFrames to register |

**Example:**

```python
# DataFrames created before enable()
customers = pd.read_csv("customers.csv")
orders = pd.read_csv("orders.csv")

tp.enable(mode="debug")
tp.register(customers, orders)  # Now they're tracked
```

---

### stage

```python
tp.stage(name: str) -> None
```

Label the current pipeline stage.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Stage name |

**Example:**

```python
tp.stage("load")
df = pd.read_csv("data.csv")

tp.stage("clean")
df = df.dropna()

tp.stage("transform")
df["total"] = df["price"] * df["qty"]
```

---

## Query Functions

### check

```python
tp.check(
    df: pd.DataFrame,
    *,
    retention_threshold: float | None = None,
    merge_expansion_threshold: float | None = None,
) -> CheckResult
```

Health check for a DataFrame's lineage.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | required | DataFrame to check |
| `retention_threshold` | `float` | `0.5` | Warn if retention below this |
| `merge_expansion_threshold` | `float` | `None` | Warn if merge expands beyond this |

**Returns:** `CheckResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `.passed` | `bool` | True if healthy |
| `.mode` | `str` | Current tracking mode |
| `.retention` | `float` | Row retention rate (0-1) |
| `.n_dropped` | `int` | Total dropped rows |
| `.n_changes` | `int` | Total cell changes |
| `.warnings` | `list[str]` | Any warnings |
| `.drops_by_op` | `dict` | Drops by operation |
| `.changes_by_op` | `dict` | Changes by operation |

**Example:**

```python
result = tp.check(df)
print(result)

if not result.passed:
    for warning in result.warnings:
        print(f"⚠ {warning}")
```

---

### trace

```python
tp.trace(
    df: pd.DataFrame,
    row: int | None = None,
    *,
    where: dict[str, Any] | None = None,
) -> TraceResult
```

Trace a row's journey through the pipeline.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | required | DataFrame containing the row |
| `row` | `int` | `None` | Row index (0-based) |
| `where` | `dict` | `None` | Business key lookup |

**Returns:** `TraceResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `.row_id` | `int` | Internal row ID |
| `.is_alive` | `bool` | True if row exists in current DataFrame |
| `.events` | `list` | All events for this row |
| `.dropped_at` | `dict` | Operation that dropped (if dropped) |
| `.origin` | `dict` | Where row came from: `{"type": "concat", "source_df": 1}` or `{"type": "merge", "left_parent": 10, "right_parent": 20}` |
| `.representative` | `dict` | If dropped by dedup: `{"kept_rid": 42, "subset": [...], "keep": "first"}` |

**Example:**

```python
# By index
trace = tp.trace(df, row=0)

# By business key
trace = tp.trace(df, where={"customer_id": "C-123"})

print(trace)
```

---

### why

```python
tp.why(
    df: pd.DataFrame,
    col: str,
    row: int | None = None,
    *,
    where: dict[str, Any] | None = None,
) -> WhyResult
```

Explain why a cell has its current value.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | required | DataFrame containing the cell |
| `col` | `str` | required | Column name |
| `row` | `int` | `None` | Row index (0-based) |
| `where` | `dict` | `None` | Business key lookup |

**Returns:** `WhyResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `.column` | `str` | Column name |
| `.row_id` | `int` | Internal row ID |
| `.current_value` | `Any` | Current cell value |
| `.history` | `list` | All changes to this cell |
| `.was_null` | `bool` | Was ever null |
| `.null_recovered` | `bool` | Null was later filled |

**Example:**

```python
why = tp.why(df, col="income", row=0)
print(why)

for change in why.history:
    print(f"{change.old_value} → {change.new_value}")
```

!!! warning "Requires Debug Mode"
    `tp.why()` requires debug mode with the column being watched.

---

## Output Functions

### report

```python
tp.report(
    df: pd.DataFrame,
    path: str,
    *,
    title: str | None = None,
    include_data: bool = False,
) -> None
```

Generate an HTML report.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | required | DataFrame to report on |
| `path` | `str` | required | Output file path |
| `title` | `str` | `None` | Report title |
| `include_data` | `bool` | `False` | Include data preview |

**Example:**

```python
tp.report(df, "audit.html", title="Pipeline Audit - 2024-01")
```

---

### snapshot

```python
tp.snapshot(
    df: pd.DataFrame,
    *,
    include_data: bool = False,
    path: str | None = None,
) -> Snapshot
```

Capture DataFrame state for later comparison.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | required | DataFrame to snapshot |
| `include_data` | `bool` | `False` | Store full data copy |
| `path` | `str` | `None` | Save to disk |

**Returns:** `Snapshot`

**Example:**

```python
before = tp.snapshot(df)
df = df.dropna()
after = tp.snapshot(df)
```

---

### diff

```python
tp.diff(
    before: Snapshot,
    after: Snapshot,
) -> DiffResult
```

Compare two snapshots.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `before` | `Snapshot` | Earlier snapshot |
| `after` | `Snapshot` | Later snapshot |

**Returns:** `DiffResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `.rows_added` | `int` | New rows |
| `.rows_removed` | `int` | Removed rows |
| `.cells_changed` | `int` | Modified cells |

**Example:**

```python
diff = tp.diff(before, after)
print(f"Removed: {diff.rows_removed} rows")
```

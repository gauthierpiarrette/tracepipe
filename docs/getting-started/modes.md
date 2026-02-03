# Tracking Modes

TracePipe offers two tracking modes optimized for different use cases.

## CI Mode (Default)

Lightweight tracking for production and CI/CD pipelines.

```python
tp.enable(mode="ci")
```

### What's Tracked

- ✅ Step counts and shapes
- ✅ Retention rates
- ✅ Dropped row counts (not individual IDs)
- ✅ Merge mismatch warnings
- ❌ Per-row provenance
- ❌ Cell change history
- ❌ Ghost values (last values before drop)

### When to Use

- Production data pipelines
- CI/CD validation
- Performance-critical code
- Large datasets (1M+ rows)

### Performance

Minimal overhead — typically < 10% slower than untracked pandas.

---

## Debug Mode

Full lineage tracking for development and debugging.

```python
tp.enable(mode="debug", watch=["price", "amount", "status"])
```

### What's Tracked

- ✅ Everything in CI mode, plus:
- ✅ Individual dropped row IDs
- ✅ Complete row-level history
- ✅ Cell change tracking (for watched columns)
- ✅ Before/after values
- ✅ Ghost values (captured at drop time)
- ✅ Merge parent tracking
- ✅ GroupBy membership

### When to Use

- Debugging data issues
- Understanding pipeline behavior
- Auditing transformations
- Investigating specific rows

### Performance

Higher overhead due to per-row tracking. Expect 1.5-3x slower depending on operations.

---

## The `watch` Parameter

In debug mode, specify which columns to track for cell-level changes:

```python
tp.enable(mode="debug", watch=["price", "quantity", "status"])
```

!!! tip "Watch Strategy"
    Only watch columns you care about. Watching all columns significantly increases memory usage and overhead.

### What Watching Enables

| Feature | Without Watch | With Watch |
|---------|---------------|------------|
| `tp.check()` retention | ✅ | ✅ |
| `tp.trace()` row journey | ✅ | ✅ |
| `tp.why()` cell history | ❌ | ✅ |
| Before/after values | ❌ | ✅ |
| Ghost values | ❌ | ✅ |

---

## Switching Modes

You can switch modes at any time:

```python
# Start in CI mode for initial load
tp.enable(mode="ci")
df = pd.read_csv("large_file.csv")

# Switch to debug for specific analysis
tp.reset()  # Clear previous lineage
tp.enable(mode="debug", watch=["target_column"])
df = df[df["flag"] == True]  # Now tracked in detail
```

---

## Mode Comparison

| Feature | CI Mode | Debug Mode |
|---------|---------|------------|
| Memory usage | Low | Medium-High |
| Performance overhead | < 10% | 50-200% |
| `tp.check()` | Full | Full |
| `tp.trace()` | Limited | Full |
| `tp.why()` | Not available | Full |
| Ghost values | No | Yes |
| Merge parents | Counts only | Full tracking |

---

## Configuration

Additional configuration options:

```python
tp.enable(
    mode="debug",
    watch=["col1", "col2"],
    # Advanced options via configure()
)

# Or use configure() separately
tp.configure(
    sample_rate=0.1,      # Sample 10% of rows (for huge datasets)
    max_tracked_rows=100000,  # Cap tracked rows
)
```

!!! note
    `sample_rate` and `max_tracked_rows` are planned features for handling very large datasets efficiently.

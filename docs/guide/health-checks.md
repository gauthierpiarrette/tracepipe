# Health Checks

The `tp.check()` function provides a comprehensive health audit of your pipeline.

## Basic Usage

```python
result = tp.check(df)
print(result)
```

Output:

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

## The CheckResult Object

`tp.check()` returns a `CheckResult` object with programmatic access:

```python
result = tp.check(df)

# Access fields
result.passed          # bool: True if pipeline is healthy
result.mode            # str: "ci" or "debug"
result.retention       # float: 0.0-1.0 retention rate
result.n_dropped       # int: total dropped rows
result.n_changes       # int: total cell changes
result.warnings        # list[str]: any warnings

# Breakdown by operation
result.drops_by_op     # dict: {operation: count}
result.changes_by_op   # dict: {operation: count}

# Export
result.to_dict()       # dict representation
```

## Warnings

TracePipe warns about common issues:

### Low Retention

```
⚠ Low retention (45.0%) — more than half of rows dropped
```

Triggered when retention falls below 50% (configurable).

### Merge Expansion

```
⚠ Merge expansion detected: 1000 → 1500 rows (50% increase)
```

Triggered when a merge produces significantly more rows than input.

### No Lineage

```
⚠ No lineage recorded — did you call tp.enable()?
```

Triggered when checking a DataFrame with no tracking data.

## Configuration

```python
# Custom thresholds
result = tp.check(
    df,
    retention_threshold=0.3,      # Warn below 30%
    merge_expansion_threshold=2.0  # Warn if merge doubles rows
)
```

## Using in CI/CD

```python
result = tp.check(df)

if not result.passed:
    print("Pipeline health check failed!")
    for warning in result.warnings:
        print(f"  • {warning}")
    sys.exit(1)
```

Or use contracts for stricter validation:

```python
tp.contract().expect_retention(min_rate=0.8).check(df).raise_if_failed()
```

## Check vs Contracts

| Feature | `tp.check()` | `tp.contract()` |
|---------|--------------|-----------------|
| Purpose | Quick health audit | Strict validation |
| Failure behavior | Returns result | Can raise exception |
| Customization | Thresholds | Full contract DSL |
| Use case | Development, monitoring | CI/CD gates |

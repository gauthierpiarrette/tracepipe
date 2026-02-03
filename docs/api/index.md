# API Reference

TracePipe provides a minimal API surface for maximum functionality.

## Quick Reference

### Core Functions (6)

| Function | Purpose |
|----------|---------|
| [`tp.enable()`](core.md#enable) | Start tracking |
| [`tp.disable()`](core.md#disable) | Stop tracking |
| [`tp.reset()`](core.md#reset) | Clear lineage data |
| [`tp.check()`](core.md#check) | Health audit |
| [`tp.trace()`](core.md#trace) | Row journey |
| [`tp.why()`](core.md#why) | Cell provenance |

### Additional Functions

| Function | Purpose |
|----------|---------|
| [`tp.stage()`](core.md#stage) | Label pipeline stages |
| [`tp.register()`](core.md#register) | Manually register DataFrames |
| [`tp.report()`](core.md#report) | Generate HTML report |
| [`tp.snapshot()`](core.md#snapshot) | Capture DataFrame state |
| [`tp.diff()`](core.md#diff) | Compare snapshots |

### Namespaces

| Namespace | Purpose |
|-----------|---------|
| [`tp.contracts`](contracts.md) | Data quality contracts |
| [`tp.debug`](debug.md) | Advanced debugging tools |

## Import

```python
import tracepipe as tp
```

All public functions are available directly from the `tp` namespace.

## Result Objects

All query functions return structured result objects:

| Function | Returns |
|----------|---------|
| `tp.check()` | `CheckResult` |
| `tp.trace()` | `TraceResult` |
| `tp.why()` | `WhyResult` |
| `tp.contract().check()` | `ContractResult` |
| `tp.snapshot()` | `Snapshot` |
| `tp.diff()` | `DiffResult` |

All result objects support:

- `print(result)` — Pretty-printed output
- `result.to_dict()` — Dictionary representation
- Boolean evaluation — `if result.passed:`

## Type Hints

TracePipe is fully typed. IDE autocompletion works out of the box:

```python
import tracepipe as tp

result = tp.check(df)  # Autocomplete shows CheckResult methods
result.retention       # Type: float
result.warnings        # Type: list[str]
```

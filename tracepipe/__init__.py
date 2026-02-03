# tracepipe/__init__.py
"""
TracePipe: Row-Level Data Lineage Tracking

Track every row, every change, every step in your pandas pipelines.

Quick Start:
    import tracepipe as tp
    import pandas as pd

    tp.enable(mode="debug", watch=["age", "salary"])

    df = pd.DataFrame({"age": [25, None, 35], "salary": [50000, 60000, None]})
    df = df.dropna()
    df["salary"] = df["salary"] * 1.1

    # Health audit
    result = tp.check(df)
    print(result)

    # Row journey
    trace = tp.trace(df, row=0)
    print(trace)

    # Cell provenance
    why = tp.why(df, col="salary", row=0)
    print(why)

    # Generate report
    tp.report(df, "audit.html")

Modes:
    - CI mode (default): Step stats, retention rates, merge mismatch detection.
      No per-row provenance. Fast for production.
    - DEBUG mode: Full per-row provenance, cell history, ghost values.

API Summary:
    Core (5 functions for 90% of use cases):
        tp.enable()   - Start tracking
        tp.check()    - Health audit → CheckResult
        tp.trace()    - Row journey → TraceResult
        tp.why()      - Cell provenance → WhyResult
        tp.report()   - HTML export

    Power features (via namespaces):
        tp.debug.inspect()           - Raw lineage access
        tp.contracts.contract()      - Data quality contracts
        tp.snapshot(), tp.diff()     - Pipeline state comparison

    All functions return structured result objects.
    Use print(result) for pretty output, result.to_dict() for data.
"""

# === CORE API (6 functions) ===
# === NAMESPACES ===
from . import contracts, debug
from .api import configure, disable, enable, register, reset, stage

# Re-export contract() at top level for convenience
from .contracts import contract

# === CONVENIENCE API (user-facing) ===
from .convenience import (
    CheckFailed,
    # Result types
    CheckResult,
    CheckWarning,
    TraceResult,
    WhyResult,
    check,
    find,
    report,
    trace,
    why,
)

# === CONFIGURATION ===
from .core import TracePipeConfig, TracePipeMode

# === SNAPSHOTS (top-level for convenience) ===
from .snapshot import DiffResult, Snapshot, diff, snapshot

# === VERSION ===
__version__ = "0.3.2"

# === MINIMAL __all__ ===
__all__ = [
    # Core control (6)
    "enable",
    "disable",
    "reset",
    "register",
    "stage",
    "configure",
    # Convenience API (5)
    "check",
    "find",
    "trace",
    "why",
    "report",
    # Result types (5)
    "CheckResult",
    "CheckWarning",
    "CheckFailed",
    "TraceResult",
    "WhyResult",
    # Snapshots (4)
    "snapshot",
    "diff",
    "Snapshot",
    "DiffResult",
    # Contracts (1)
    "contract",
    # Namespaces (2)
    "debug",
    "contracts",
    # Config (2)
    "TracePipeConfig",
    "TracePipeMode",
    # Version (1)
    "__version__",
]


def __dir__():
    """Control what shows up in IDE autocomplete - only essential functions."""
    return [
        # Primary API
        "enable",
        "disable",
        "reset",
        "register",
        "configure",
        "check",
        "find",
        "trace",
        "why",
        "report",
        # Snapshots
        "snapshot",
        "diff",
        # Contract
        "contract",
        # Namespaces
        "debug",
        "contracts",
        # Config
        "TracePipeConfig",
    ]

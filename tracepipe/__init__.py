# tracepipe/__init__.py
"""
TracePipe: Row-Level Data Lineage Tracking

Track every row, every change, every step in your pandas pipelines.

Quick Start:
    import tracepipe
    import pandas as pd

    tracepipe.enable()
    tracepipe.watch("age", "salary")  # Watch specific columns

    df = pd.DataFrame({"age": [25, None, 35], "salary": [50000, 60000, None]})
    df = df.dropna()
    df["salary"] = df["salary"] * 1.1

    # Query lineage
    row = tracepipe.explain(0)  # What happened to row 0?
    print(row.history())

    dropped = tracepipe.dropped_rows()  # Which rows were dropped?
    print(dropped)

Features:
    - Row-level tracking: Know exactly which rows were dropped and why
    - Cell-level diffs: See before/after values for watched columns
    - Aggregation lineage: Trace back from grouped results to source rows
    - Zero-copy design: Minimal overhead on your pipelines
    - Safe instrumentation: Never crashes your code

See IMPLEMENTATION_PLAN_v5.md for full documentation.
"""

from .api import (
    GroupLineageResult,
    # Result classes
    RowLineageResult,
    aggregation_groups,
    # Convenience functions
    alive_rows,
    clear_watch,
    configure,
    disable,
    dropped_rows,
    # Core control
    enable,
    # Query API
    explain,
    explain_group,
    explain_many,
    export_arrow,
    # Export
    export_json,
    mass_updates,
    register,
    reset,
    stage,
    stats,
    steps,
    unwatch,
    # Column watching
    watch,
    watch_all,
)
from .core import TracePipeConfig

# Export protocols for custom backend implementers
from .storage.base import LineageBackend, RowIdentityStrategy
from .visualization.html_export import save

__version__ = "0.2.0"

__all__ = [
    # Core API
    "enable",
    "disable",
    "reset",
    "configure",
    "watch",
    "watch_all",
    "unwatch",
    "clear_watch",
    "register",
    "stage",
    # Query API
    "explain",
    "explain_many",
    "explain_group",
    "dropped_rows",
    "alive_rows",
    "mass_updates",
    "steps",
    "aggregation_groups",
    # Export
    "export_json",
    "export_arrow",
    "stats",
    "save",
    # Configuration
    "TracePipeConfig",
    # Result classes
    "RowLineageResult",
    "GroupLineageResult",
    # Protocols (for custom backends)
    "LineageBackend",
    "RowIdentityStrategy",
    # Version
    "__version__",
]

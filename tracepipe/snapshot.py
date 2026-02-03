# tracepipe/snapshot.py
"""
Pipeline state snapshots and diff functionality.

Snapshots capture the current state of a pipeline for comparison,
debugging, and regression testing.

Features:
- Row ID tracking (which rows are alive/dropped)
- Column statistics (null rates, unique counts, min/max)
- Watched column values (columnar storage for efficiency)
- Cross-run comparison with summary-level diffing

Usage:
    # Capture state
    snap = tp.snapshot(df)

    # Save and load
    snap.save("baseline.json")
    baseline = Snapshot.load("baseline.json")

    # Compare
    result = tp.diff(baseline, current)
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .context import get_context


@dataclass
class ColumnStats:
    """Statistics for a single column."""

    name: str
    dtype: str
    null_count: int
    null_rate: float
    unique_count: int
    min_val: Any = None
    max_val: Any = None
    mean_val: Optional[float] = None


@dataclass
class WatchedColumnData:
    """
    Columnar storage for watched values (memory efficient).

    Instead of Dict[int, Dict[str, Any]] which creates Python objects per row,
    we store arrays that enable vectorized diffing and O(log n) lookup.
    """

    rids: np.ndarray  # Row IDs (sorted for binary search)
    columns: list[str]  # Column names
    values: dict[str, np.ndarray]  # col -> values array (aligned with rids)

    def get_value(self, rid: int, col: str) -> Optional[Any]:
        """Get value for a specific row/column (O(log n) lookup)."""
        if col not in self.values:
            return None
        i = np.searchsorted(self.rids, rid)
        if i < len(self.rids) and self.rids[i] == rid:
            return self.values[col][i]
        return None

    def to_dict_view(self, limit: int = 1000) -> dict[int, dict[str, Any]]:
        """Build dict view for small samples (for serialization)."""
        result = {}
        for i, rid in enumerate(self.rids[:limit]):
            result[int(rid)] = {col: self.values[col][i] for col in self.columns}
        return result


@dataclass
class Snapshot:
    """
    Pipeline state snapshot.

    Captures:
    - Row IDs present
    - Dropped row IDs with reasons
    - Watched column values (columnar storage)
    - Summary statistics
    """

    timestamp: float
    row_ids: set[int]
    dropped_ids: set[int]
    drops_by_op: dict[str, int]
    column_stats: dict[str, ColumnStats]
    watched_data: Optional[WatchedColumnData]
    total_steps: int
    mode: str

    @classmethod
    def capture(cls, df: pd.DataFrame, include_values: bool = False) -> "Snapshot":
        """
        Capture current pipeline state.

        Args:
            df: Current DataFrame
            include_values: If True, store watched column values (columnar)
        """
        ctx = get_context()

        row_ids = set()
        rids = ctx.row_manager.get_ids_array(df)
        if rids is not None:
            row_ids = set(rids.tolist())

        dropped_ids = set(ctx.store.get_dropped_rows())
        drops_by_op = ctx.store.get_dropped_by_step()

        # Column stats
        column_stats = {}
        for col in df.columns:
            if col.startswith("__tp"):
                continue

            null_count = df[col].isna().sum()
            stats = ColumnStats(
                name=col,
                dtype=str(df[col].dtype),
                null_count=int(null_count),
                null_rate=null_count / len(df) if len(df) > 0 else 0,
                unique_count=df[col].nunique(),
            )

            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    stats.min_val = float(df[col].min()) if not df[col].isna().all() else None
                    stats.max_val = float(df[col].max()) if not df[col].isna().all() else None
                    stats.mean_val = float(df[col].mean()) if not df[col].isna().all() else None
                except (TypeError, ValueError):
                    pass

            column_stats[col] = stats

        # Columnar watched values (memory efficient)
        watched_data = None
        if include_values and ctx.watched_columns and rids is not None:
            cols = list(ctx.watched_columns & set(df.columns))
            if cols:
                # Sort RIDs for binary search
                sort_idx = np.argsort(rids)
                sorted_rids = rids[sort_idx]

                # Extract values columnar (one array per column)
                values = {}
                for col in cols:
                    values[col] = df[col].values[sort_idx]

                watched_data = WatchedColumnData(
                    rids=sorted_rids,
                    columns=cols,
                    values=values,
                )

        return cls(
            timestamp=time.time(),
            row_ids=row_ids,
            dropped_ids=dropped_ids,
            drops_by_op=drops_by_op,
            column_stats=column_stats,
            watched_data=watched_data,
            total_steps=len(ctx.store.steps),
            mode=ctx.config.mode.value,
        )

    def save(self, path: str) -> None:
        """
        Save snapshot to file.

        Uses separate files for large watched values:
        - {path}: metadata, stats, row IDs (JSON)
        - {path}.npz: watched column arrays (if present)
        """
        base_path = Path(path)
        npz_path = base_path.with_suffix(".npz")

        # Metadata (always JSON)
        data = {
            "timestamp": self.timestamp,
            "row_ids": list(self.row_ids),
            "dropped_ids": list(self.dropped_ids),
            "drops_by_op": self.drops_by_op,
            "column_stats": {k: vars(v) for k, v in self.column_stats.items()},
            "watched_columns": self.watched_data.columns if self.watched_data else [],
            "has_watched_npz": self.watched_data is not None,
            "total_steps": self.total_steps,
            "mode": self.mode,
        }
        base_path.write_text(json.dumps(data, default=str))

        # Save watched values as npz (efficient for large data)
        if self.watched_data is not None:
            arrays = {"rids": self.watched_data.rids}
            for col, vals in self.watched_data.values.items():
                # Sanitize column name for npz key
                safe_col = col.replace(".", "_").replace(" ", "_")
                arrays[f"col_{safe_col}"] = vals
            # Save column name mapping
            arrays["_col_names"] = np.array(
                [col.replace(".", "_").replace(" ", "_") for col in self.watched_data.columns]
            )
            arrays["_col_names_original"] = np.array(self.watched_data.columns)
            np.savez_compressed(npz_path, **arrays)

    @classmethod
    def load(cls, path: str) -> "Snapshot":
        """
        Load snapshot from file.

        Loads watched values from npz if present.
        """
        base_path = Path(path)
        npz_path = base_path.with_suffix(".npz")

        data = json.loads(base_path.read_text())

        column_stats = {k: ColumnStats(**v) for k, v in data["column_stats"].items()}

        # Load watched data from npz if present
        watched_data = None
        if data.get("has_watched_npz") and npz_path.exists():
            with np.load(npz_path, allow_pickle=True) as npz:
                rids = npz["rids"]

                # Get original column names
                if "_col_names_original" in npz:
                    cols = list(npz["_col_names_original"])
                    safe_names = list(npz["_col_names"])
                    values = {}
                    for col, safe_col in zip(cols, safe_names):
                        key = f"col_{safe_col}"
                        if key in npz:
                            values[col] = npz[key]
                else:
                    # Legacy format
                    cols = data.get("watched_columns", [])
                    values = {}
                    for col in cols:
                        safe_col = col.replace(".", "_").replace(" ", "_")
                        key = f"col_{safe_col}"
                        if key in npz:
                            values[col] = npz[key]

                if len(values) > 0:
                    watched_data = WatchedColumnData(rids=rids, columns=cols, values=values)

        return cls(
            timestamp=data["timestamp"],
            row_ids=set(data["row_ids"]),
            dropped_ids=set(data["dropped_ids"]),
            drops_by_op=data["drops_by_op"],
            column_stats=column_stats,
            watched_data=watched_data,
            total_steps=data["total_steps"],
            mode=data["mode"],
        )

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Snapshot ({self.mode} mode)",
            f"  Rows: {len(self.row_ids)}",
            f"  Dropped: {len(self.dropped_ids)}",
            f"  Steps: {self.total_steps}",
            f"  Columns: {len(self.column_stats)}",
        ]
        if self.watched_data:
            lines.append(f"  Watched: {len(self.watched_data.columns)} columns")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<Snapshot rows={len(self.row_ids)} dropped={len(self.dropped_ids)} "
            f"steps={self.total_steps}>"
        )


@dataclass
class DiffResult:
    """Result of comparing two snapshots."""

    rows_added: set[int]
    rows_removed: set[int]
    new_drops: set[int]
    recovered_rows: set[int]
    drops_delta: dict[str, int]  # op -> change in count
    stats_changes: dict[str, dict[str, Any]]  # col -> {metric: (old, new)}

    def __repr__(self) -> str:
        lines = ["Snapshot Diff:"]

        if self.rows_added:
            lines.append(f"  + {len(self.rows_added)} rows added")
        if self.rows_removed:
            lines.append(f"  - {len(self.rows_removed)} rows removed")
        if self.new_drops:
            lines.append(f"  ! {len(self.new_drops)} new drops")
        if self.recovered_rows:
            lines.append(f"  * {len(self.recovered_rows)} recovered")

        if self.drops_delta:
            lines.append("  Drop changes by operation:")
            for op, delta in sorted(self.drops_delta.items(), key=lambda x: -abs(x[1])):
                sign = "+" if delta > 0 else ""
                lines.append(f"    {op}: {sign}{delta}")

        if self.stats_changes:
            lines.append("  Column stat changes:")
            for col, changes in list(self.stats_changes.items())[:5]:
                for metric, (old, new) in changes.items():
                    lines.append(f"    {col}.{metric}: {old} -> {new}")
            if len(self.stats_changes) > 5:
                lines.append(f"    ... and {len(self.stats_changes) - 5} more")

        if len(lines) == 1:
            lines.append("  No differences")

        return "\n".join(lines)

    @property
    def has_changes(self) -> bool:
        """True if there are any differences."""
        return bool(
            self.rows_added
            or self.rows_removed
            or self.new_drops
            or self.recovered_rows
            or self.drops_delta
            or self.stats_changes
        )

    def to_dict(self) -> dict:
        """Export to dictionary."""
        return {
            "rows_added": list(self.rows_added),
            "rows_removed": list(self.rows_removed),
            "new_drops": list(self.new_drops),
            "recovered_rows": list(self.recovered_rows),
            "drops_delta": self.drops_delta,
            "stats_changes": self.stats_changes,
        }


def diff(baseline: Snapshot, current: Snapshot) -> DiffResult:
    """
    Compare two snapshots.

    Note: Cross-run diff is SUMMARY-ONLY unless keys are stored.
    Row-level comparison only works within same session (same RID assignment).
    """
    rows_added = current.row_ids - baseline.row_ids
    rows_removed = baseline.row_ids - current.row_ids

    new_drops = current.dropped_ids - baseline.dropped_ids
    recovered_rows = baseline.dropped_ids - current.dropped_ids

    # Drops delta by operation
    all_ops = set(baseline.drops_by_op.keys()) | set(current.drops_by_op.keys())
    drops_delta = {}
    for op in all_ops:
        old = baseline.drops_by_op.get(op, 0)
        new = current.drops_by_op.get(op, 0)
        if old != new:
            drops_delta[op] = new - old

    # Stats changes
    stats_changes: dict[str, dict[str, Any]] = {}
    all_cols = set(baseline.column_stats.keys()) | set(current.column_stats.keys())
    for col in all_cols:
        old_stats = baseline.column_stats.get(col)
        new_stats = current.column_stats.get(col)

        if old_stats is None or new_stats is None:
            continue

        changes: dict[str, Any] = {}
        if old_stats.null_rate != new_stats.null_rate:
            changes["null_rate"] = (old_stats.null_rate, new_stats.null_rate)
        if old_stats.unique_count != new_stats.unique_count:
            changes["unique_count"] = (old_stats.unique_count, new_stats.unique_count)
        if old_stats.dtype != new_stats.dtype:
            changes["dtype"] = (old_stats.dtype, new_stats.dtype)

        if changes:
            stats_changes[col] = changes

    return DiffResult(
        rows_added=rows_added,
        rows_removed=rows_removed,
        new_drops=new_drops,
        recovered_rows=recovered_rows,
        drops_delta=drops_delta,
        stats_changes=stats_changes,
    )


def snapshot(df: pd.DataFrame, include_values: bool = False) -> Snapshot:
    """
    Capture current pipeline state.

    Args:
        df: Current DataFrame
        include_values: If True, store watched column values (columnar)

    Returns:
        Snapshot object
    """
    return Snapshot.capture(df, include_values)

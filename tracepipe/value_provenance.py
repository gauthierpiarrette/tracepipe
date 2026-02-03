# tracepipe/value_provenance.py
"""
Cell-level value provenance tracking.

Provides detailed history of how specific cell values changed
throughout the pipeline, including null introduction tracking.

Usage:
    # Get history of a specific cell
    history = tp.explain_value(row_id=123, column="price", df=result)

    # Analyze where nulls came from in a column
    analysis = tp.null_analysis("email", df)
"""

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from .context import get_context


@dataclass
class ValueEvent:
    """Single change event for a cell."""

    step_id: int
    operation: str
    old_value: Any
    new_value: Any
    change_type: str
    timestamp: float
    code_location: Optional[str]

    def to_dict(self) -> dict:
        """Export to dictionary."""
        return {
            "step_id": self.step_id,
            "operation": self.operation,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "change_type": self.change_type,
            "timestamp": self.timestamp,
            "code_location": self.code_location,
        }


@dataclass
class ValueHistory:
    """Complete history of a cell's value."""

    row_id: int
    column: str
    current_value: Any
    events: list[ValueEvent]
    became_null_at: Optional[int] = None  # step_id
    became_null_by: Optional[str] = None  # operation

    def __repr__(self) -> str:
        lines = [f"Value History: row {self.row_id}, column '{self.column}'"]
        lines.append(f"  Current: {self.current_value}")
        lines.append(f"  Changes: {len(self.events)}")

        if self.became_null_at:
            lines.append(f"  ! Became null at step {self.became_null_at} by {self.became_null_by}")

        for event in self.events[-5:]:
            lines.append(f"    {event.operation}: {event.old_value} -> {event.new_value}")

        if len(self.events) > 5:
            lines.append(f"    ... and {len(self.events) - 5} more events")

        return "\n".join(lines)

    @property
    def was_modified(self) -> bool:
        """True if value was ever modified."""
        return len(self.events) > 0

    @property
    def is_null(self) -> bool:
        """True if current value is null."""
        return pd.isna(self.current_value)

    def to_dict(self) -> dict:
        """Export to dictionary."""
        return {
            "row_id": self.row_id,
            "column": self.column,
            "current_value": self.current_value,
            "events": [e.to_dict() for e in self.events],
            "became_null_at": self.became_null_at,
            "became_null_by": self.became_null_by,
        }


def explain_value(
    row_id: int,
    column: str,
    df: Optional[pd.DataFrame] = None,
    follow_lineage: bool = True,
) -> ValueHistory:
    """
    Get complete history of a specific cell's value.

    Args:
        row_id: Row ID to trace
        column: Column name
        df: Optional DataFrame for current value lookup
        follow_lineage: If True, include pre-merge parent history (default: True)

    Returns:
        ValueHistory with all changes to this cell
    """
    ctx = get_context()
    store = ctx.store

    # Get current value if df provided
    current_value = None
    if df is not None:
        rids = ctx.row_manager.get_ids_array(df)
        if rids is not None:
            # Find position of this row_id
            matches = (rids == row_id).nonzero()[0]
            if len(matches) > 0 and column in df.columns:
                current_value = df.iloc[matches[0]][column]

    # Collect events - use lineage-aware method if requested
    if follow_lineage and hasattr(store, "get_cell_history_with_lineage"):
        # Get cell history including pre-merge parent history
        raw_events = store.get_cell_history_with_lineage(row_id, column)
    else:
        # Fallback to direct row_id lookup only
        raw_events = [e for e in store.get_row_history(row_id) if e["col"] == column]

    # Convert to ValueEvent objects
    events = []
    became_null_at = None
    became_null_by = None

    for diff in raw_events:
        events.append(
            ValueEvent(
                step_id=diff["step_id"],
                operation=diff.get("operation", "unknown"),
                old_value=diff["old_val"],
                new_value=diff["new_val"],
                change_type=diff.get("change_type", "UNKNOWN"),
                timestamp=diff.get("timestamp", 0) or 0,
                code_location=diff.get("code_location"),
            )
        )

        # Track when value became null
        if became_null_at is None and pd.isna(diff["new_val"]) and not pd.isna(diff["old_val"]):
            became_null_at = diff["step_id"]
            became_null_by = diff.get("operation", "unknown")

    # Events should already be sorted by step_id from lineage method
    events.sort(key=lambda e: e.step_id)

    return ValueHistory(
        row_id=row_id,
        column=column,
        current_value=current_value,
        events=events,
        became_null_at=became_null_at,
        became_null_by=became_null_by,
    )


@dataclass
class NullAnalysis:
    """Analysis of how nulls appeared in a column."""

    column: str
    total_nulls: int
    null_sources: dict[str, int]  # operation -> count
    sample_row_ids: list[int]

    def __repr__(self) -> str:
        lines = [f"Null Analysis: '{self.column}'"]
        lines.append(f"  Total nulls: {self.total_nulls}")

        if self.null_sources:
            lines.append("  Sources:")
            for op, count in sorted(self.null_sources.items(), key=lambda x: -x[1]):
                lines.append(f"    {op}: {count}")
        else:
            lines.append("  No tracked null introductions")

        if self.sample_row_ids:
            lines.append(f"  Sample row IDs: {self.sample_row_ids[:5]}")

        return "\n".join(lines)

    @property
    def has_untracked_nulls(self) -> bool:
        """True if some nulls were not tracked by TracePipe."""
        tracked = sum(self.null_sources.values())
        return tracked < self.total_nulls

    def to_dict(self) -> dict:
        """Export to dictionary."""
        return {
            "column": self.column,
            "total_nulls": self.total_nulls,
            "null_sources": self.null_sources,
            "sample_row_ids": self.sample_row_ids,
            "has_untracked_nulls": self.has_untracked_nulls,
        }


def null_analysis(column: str, df: pd.DataFrame) -> NullAnalysis:
    """
    Analyze how nulls appeared in a column.

    Returns breakdown of which operations introduced nulls.

    Args:
        column: Column name to analyze
        df: Current DataFrame

    Returns:
        NullAnalysis with breakdown of null sources
    """
    ctx = get_context()
    store = ctx.store

    if column not in df.columns:
        return NullAnalysis(column=column, total_nulls=0, null_sources={}, sample_row_ids=[])

    rids = ctx.row_manager.get_ids_array(df)
    if rids is None:
        return NullAnalysis(
            column=column,
            total_nulls=int(df[column].isna().sum()),
            null_sources={},
            sample_row_ids=[],
        )

    # Find null rows
    null_mask = df[column].isna()
    null_rids = set(rids[null_mask].tolist())

    # Track which operations introduced nulls
    null_sources: dict[str, int] = {}
    step_map = {s.step_id: s for s in store.steps}
    sample_ids: list[int] = []

    for diff in store._iter_all_diffs():
        if diff["col"] == column and diff["row_id"] in null_rids:
            if pd.isna(diff["new_val"]) and not pd.isna(diff["old_val"]):
                step = step_map.get(diff["step_id"])
                op = step.operation if step else "unknown"
                null_sources[op] = null_sources.get(op, 0) + 1
                if len(sample_ids) < 10:
                    sample_ids.append(diff["row_id"])

    return NullAnalysis(
        column=column,
        total_nulls=len(null_rids),
        null_sources=null_sources,
        sample_row_ids=sample_ids,
    )


def column_changes_summary(column: str, df: pd.DataFrame) -> dict[str, Any]:
    """
    Get summary of all changes to a column.

    Args:
        column: Column name
        df: Current DataFrame

    Returns:
        Dict with summary statistics
    """
    ctx = get_context()
    store = ctx.store

    rids = ctx.row_manager.get_ids_array(df)
    if rids is None:
        return {
            "column": column,
            "total_changes": 0,
            "changes_by_operation": {},
            "unique_rows_modified": 0,
        }

    rid_set = set(rids.tolist())
    changes_by_op: dict[str, int] = {}
    modified_rows: set = set()
    step_map = {s.step_id: s for s in store.steps}

    for diff in store._iter_all_diffs():
        if diff["col"] == column and diff["row_id"] in rid_set:
            step = step_map.get(diff["step_id"])
            op = step.operation if step else "unknown"
            changes_by_op[op] = changes_by_op.get(op, 0) + 1
            modified_rows.add(diff["row_id"])

    return {
        "column": column,
        "total_changes": sum(changes_by_op.values()),
        "changes_by_operation": changes_by_op,
        "unique_rows_modified": len(modified_rows),
    }

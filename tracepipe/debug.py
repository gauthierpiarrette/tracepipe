# tracepipe/debug.py
"""
Debug namespace for TracePipe power users.

This module provides low-level introspection and raw access to lineage data.
For most use cases, prefer the top-level convenience API (check, trace, why, report).

Usage:
    import tracepipe as tp

    # Access debug inspector
    dbg = tp.debug.inspect()
    dbg.steps              # All recorded steps
    dbg.dropped_rows()     # All dropped row IDs
    dbg.explain_row(42)    # Raw row lineage
    dbg.export("json")     # Export lineage data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from .context import get_context

if TYPE_CHECKING:
    from .api import GroupLineageResult, RowLineageResult
    from .core import StepEvent


@dataclass
class DebugInspector:
    """
    Debug inspector providing raw access to TracePipe internals.

    This is the primary entry point for power users who need
    low-level access to lineage data.
    """

    @property
    def enabled(self) -> bool:
        """True if TracePipe is currently enabled."""
        return get_context().enabled

    @property
    def mode(self) -> str:
        """Current mode: 'ci' or 'debug'."""
        return get_context().config.mode.value

    @property
    def steps(self) -> list[StepEvent]:
        """All recorded pipeline steps."""
        return get_context().store.steps

    @property
    def watched_columns(self) -> set:
        """Currently watched columns."""
        return get_context().watched_columns.copy()

    def watch(self, *columns: str) -> DebugInspector:
        """
        Add columns to watch for cell-level tracking.

        Args:
            *columns: Column names to watch.

        Returns:
            Self for chaining.
        """
        get_context().watched_columns.update(columns)
        return self

    @property
    def total_diffs(self) -> int:
        """Total number of diffs (including spilled)."""
        return get_context().store.total_diff_count

    @property
    def in_memory_diffs(self) -> int:
        """Number of diffs currently in memory."""
        return get_context().store.diff_count

    def dropped_rows(self, step_id: int | None = None) -> list[int]:
        """
        Get all dropped row IDs.

        Args:
            step_id: If provided, only return drops from this step.

        Returns:
            List of dropped row IDs.
        """
        return get_context().store.get_dropped_rows(step_id)

    def dropped_by_operation(self) -> dict:
        """Get count of dropped rows per operation."""
        return get_context().store.get_dropped_by_step()

    def alive_rows(self) -> list[int]:
        """Get all row IDs that are still alive (not dropped)."""
        ctx = get_context()
        all_registered = set(ctx.row_manager.all_registered_ids())
        dropped = set(ctx.store.get_dropped_rows())
        return sorted(all_registered - dropped)

    def explain_row(self, row_id: int) -> RowLineageResult:
        """
        Get lineage for a specific row.

        Returns a RowLineageResult object with:
            - row_id: int
            - is_alive: bool
            - dropped_at: Optional[str]
            - history(): List[dict]
            - cell_history(col): List[dict]
            - to_dict(): dict
        """
        from .api import RowLineageResult

        return RowLineageResult(row_id, get_context())

    def explain_group(self, group_key: str) -> GroupLineageResult:
        """Get aggregation group membership."""
        from .api import GroupLineageResult

        return GroupLineageResult(group_key, get_context())

    def aggregation_groups(self) -> list[str]:
        """List all tracked aggregation groups."""
        ctx = get_context()
        groups = []
        for mapping in ctx.store.aggregation_mappings:
            groups.extend(mapping.membership.keys())
        return groups

    def merge_stats(self, step_id: int | None = None) -> list[dict]:
        """Get merge operation statistics."""
        ctx = get_context()
        stats_list = ctx.store.get_merge_stats(step_id)
        return [
            {
                "step_id": sid,
                "left_rows": s.left_rows,
                "right_rows": s.right_rows,
                "result_rows": s.result_rows,
                "expansion_ratio": s.expansion_ratio,
                "left_match_rate": s.left_match_rate,
                "right_match_rate": s.right_match_rate,
                "how": s.how,
            }
            for sid, s in stats_list
        ]

    def mass_updates(self) -> list[dict]:
        """Get operations that exceeded cell diff threshold."""
        ctx = get_context()
        return [
            {
                "step_id": s.step_id,
                "operation": s.operation,
                "rows_affected": s.rows_affected,
                "stage": s.stage,
            }
            for s in ctx.store.steps
            if s.is_mass_update
        ]

    def ghost_rows(self, limit: int = 1000) -> pd.DataFrame:
        """
        Get dropped rows with their last-known values (DEBUG mode only).

        Returns DataFrame with columns:
            - __tp_row_id__: Original row ID
            - __tp_dropped_by__: Operation that dropped the row
            - [watched columns]: Last known values
        """
        ctx = get_context()
        return ctx.row_manager.get_ghost_rows(limit=limit)

    def stats(self) -> dict:
        """Get comprehensive tracking statistics."""
        ctx = get_context()
        return {
            "enabled": ctx.enabled,
            "mode": ctx.config.mode.value,
            "total_steps": len(ctx.store.steps),
            "total_diffs": ctx.store.total_diff_count,
            "in_memory_diffs": ctx.store.diff_count,
            "spilled_files": len(ctx.store.spilled_files),
            "watched_columns": list(ctx.watched_columns),
            "aggregation_groups": len(ctx.store.aggregation_mappings),
            "merge_mappings": len(ctx.store.merge_mappings),
            "features": {
                "merge_provenance": ctx.config.should_capture_merge_provenance,
                "ghost_row_values": ctx.config.should_capture_ghost_values,
                "cell_history": ctx.config.should_capture_cell_history,
            },
        }

    def export(self, format: str = "json", path: str | None = None) -> str | None:
        """
        Export lineage data.

        Args:
            format: "json" or "arrow"
            path: File path. If None, returns JSON string (json format only).

        Returns:
            JSON string if path is None and format is "json", else None.
        """
        ctx = get_context()

        if format == "json":
            json_str = ctx.store.to_json()
            if path:
                with open(path, "w") as f:
                    f.write(json_str)
                return None
            return json_str
        elif format == "arrow":
            if path is None:
                raise ValueError("path is required for arrow export")
            try:
                import pyarrow.parquet as pq
            except ImportError:
                raise ImportError(
                    "pyarrow is required for Arrow export. "
                    "Install with: pip install tracepipe[arrow]"
                ) from None
            table = ctx.store.to_arrow()
            pq.write_table(table, path)
            return None
        else:
            raise ValueError(f"Unknown format: {format}. Use 'json' or 'arrow'.")

    def register(self, df: pd.DataFrame) -> None:
        """Manually register a DataFrame for tracking."""
        ctx = get_context()
        if ctx.enabled:
            ctx.row_manager.register(df)

    def get_row_ids(self, df: pd.DataFrame) -> Any | None:
        """Get row IDs array for a DataFrame."""
        ctx = get_context()
        return ctx.row_manager.get_ids_array(df)

    def __repr__(self) -> str:
        ctx = get_context()
        if not ctx.enabled:
            return "<DebugInspector enabled=False>"
        return (
            f"<DebugInspector mode={ctx.config.mode.value} "
            f"steps={len(ctx.store.steps)} "
            f"diffs={ctx.store.total_diff_count}>"
        )


def inspect() -> DebugInspector:
    """
    Get a debug inspector for TracePipe internals.

    Returns:
        DebugInspector with access to steps, diffs, and raw lineage data.

    Example:
        dbg = tp.debug.inspect()
        print(dbg.steps)
        print(dbg.dropped_rows())
        dbg.export("json", "lineage.json")
    """
    return DebugInspector()


# Convenience aliases for common debug operations
def export_json(path: str) -> None:
    """Export lineage to JSON file."""
    inspect().export("json", path)


def export_arrow(path: str) -> None:
    """Export lineage to Parquet file."""
    inspect().export("arrow", path)


def find(
    df: pd.DataFrame,
    *,
    where: dict | None = None,
    predicate=None,
    limit: int = 50,
) -> list[int]:
    """
    Find row IDs matching a selector.

    This is a debug utility for discovering row IDs that can be used
    with trace() and why(). Row IDs are internal identifiers and should
    not be persisted across sessions.

    Args:
        df: DataFrame to search
        where: Exact match selector, e.g. {"status": "failed"}
        predicate: Vector predicate (df -> boolean Series)
        limit: Maximum number of IDs to return (default 50)

    Returns:
        List of internal row IDs (for use with trace/why row= parameter)

    Example:
        rids = tp.debug.find(df, where={"status": "failed"})
        for rid in rids[:3]:
            print(tp.trace(df, row=rid))
    """
    # Import here to avoid circular imports
    from .convenience import _resolve_predicate, _resolve_where

    ctx = get_context()

    if where:
        return _resolve_where(df, where, ctx, limit=limit)
    elif predicate:
        return _resolve_predicate(df, predicate, ctx, limit=limit)
    else:
        raise ValueError("Must provide 'where' or 'predicate'")

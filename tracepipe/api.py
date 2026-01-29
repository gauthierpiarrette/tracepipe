# tracepipe/api.py
"""
Public API for TracePipe.
"""

from __future__ import annotations

import sys
import types
from dataclasses import fields

import pandas as pd

from .context import TracePipeContext, get_context, reset_context, set_context
from .core import LineageGaps, TracePipeConfig
from .instrumentation.pandas_inst import instrument_pandas, uninstrument_pandas
from .storage.base import LineageBackend, RowIdentityStrategy


def _get_module() -> types.ModuleType:
    """Get the tracepipe module for fluent chaining."""
    return sys.modules["tracepipe"]


def enable(
    config: TracePipeConfig | None = None,
    auto_watch: bool = False,
    backend: LineageBackend | None = None,
    identity: RowIdentityStrategy | None = None,
) -> types.ModuleType:
    """
    Enable TracePipe lineage tracking.

    Args:
        config: Optional configuration
        auto_watch: If True, automatically watch columns with nulls
        backend: Optional custom storage backend (default: InMemoryLineageStore)
        identity: Optional custom row identity strategy (default: PandasRowIdentity)

    Returns:
        The tracepipe module for fluent chaining.

    Examples:
        # Basic usage (pandas + in-memory)
        tracepipe.enable()

        # Fluent chaining
        tracepipe.enable().watch("age", "salary")

        # With SQLite persistence (v2.1+)
        from tracepipe.storage.sqlite_backend import SQLiteLineageStore
        tracepipe.enable(backend=SQLiteLineageStore(config, "lineage.db"))

        # With Polars support (v2.1+)
        from tracepipe.storage.polars_identity import PolarsRowIdentity
        tracepipe.enable(identity=PolarsRowIdentity(config))
    """
    # Create context with custom backends if provided
    if backend is not None or identity is not None:
        ctx = TracePipeContext(config=config, backend=backend, identity=identity)
        set_context(ctx)
    else:
        ctx = get_context()
        if config:
            ctx.config = config

    if auto_watch:
        ctx.config.auto_watch = True

    if not ctx.enabled:
        instrument_pandas()
        ctx.enabled = True

    return _get_module()


def disable() -> types.ModuleType:
    """
    Disable TracePipe and restore original pandas methods.

    Note:
        This stops tracking but preserves lineage data collected so far.
        You can still query explain(), dropped_rows(), etc. after disabling.
        To clear all data, use reset() instead.

    Returns:
        The tracepipe module for fluent chaining.
    """
    ctx = get_context()

    if ctx.enabled:
        uninstrument_pandas()
        # Call cleanup if backend supports it
        if hasattr(ctx.store, "_cleanup_spillover"):
            ctx.store._cleanup_spillover()
        ctx.enabled = False

    return _get_module()


def reset() -> types.ModuleType:
    """
    Reset all tracking state for the current thread.

    This clears ALL lineage data, steps, watched columns, and row registrations.
    If tracking was enabled, it will be re-enabled with a fresh context.

    Use this when:
        - Starting fresh in a notebook cell
        - Running multiple independent analyses
        - Testing

    Returns:
        The tracepipe module for fluent chaining.
    """
    ctx = get_context()
    was_enabled = ctx.enabled

    if was_enabled:
        uninstrument_pandas()

    reset_context()

    if was_enabled:
        # Re-enable with fresh context
        enable()

    return _get_module()


def configure(**kwargs) -> types.ModuleType:
    """
    Update configuration.

    Args:
        **kwargs: Configuration options to update. Valid keys are:
            - max_diffs_in_memory: Maximum diffs before spilling to disk
            - max_diffs_per_step: Threshold for mass update detection
            - max_group_membership_size: Threshold for count-only groups
            - strict_mode: Raise exceptions on tracking errors
            - auto_watch: Auto-watch columns with null values
            - auto_watch_null_threshold: Null ratio threshold for auto-watch
            - spillover_dir: Directory for spilled data
            - use_hidden_column: Use hidden column for row tracking
            - warn_on_duplicate_index: Warn on duplicate DataFrame index
            - cleanup_spillover_on_disable: Clean up spilled files on disable

    Returns:
        The tracepipe module for fluent chaining.

    Raises:
        ValueError: If an invalid configuration key is provided.

    Examples:
        tracepipe.configure(max_diffs_per_step=1000)
        tracepipe.enable().configure(strict_mode=True).watch("amount")
    """
    ctx = get_context()

    # Validate keys against dataclass fields
    valid_keys = {f.name for f in fields(TracePipeConfig)}
    invalid_keys = set(kwargs.keys()) - valid_keys
    if invalid_keys:
        raise ValueError(
            f"Invalid configuration key(s): {invalid_keys}. "
            f"Valid keys are: {sorted(valid_keys)}"
        )

    for key, value in kwargs.items():
        setattr(ctx.config, key, value)

    return _get_module()


def watch(*columns: str) -> types.ModuleType:
    """
    Add columns to watch for cell-level changes.

    Args:
        *columns: Column names to watch.

    Returns:
        The tracepipe module for fluent chaining.

    Examples:
        tracepipe.watch("age", "salary")
        tracepipe.enable().watch("amount").watch("price")
    """
    ctx = get_context()
    ctx.watched_columns.update(columns)
    return _get_module()


def watch_all(df: pd.DataFrame) -> types.ModuleType:
    """
    Watch all columns in a DataFrame.

    Args:
        df: DataFrame whose columns to watch.

    Returns:
        The tracepipe module for fluent chaining.

    Examples:
        tracepipe.watch_all(df)
    """
    ctx = get_context()
    ctx.watched_columns.update(df.columns.tolist())
    return _get_module()


def unwatch(*columns: str) -> types.ModuleType:
    """
    Remove columns from watch list.

    Args:
        *columns: Column names to stop watching.

    Returns:
        The tracepipe module for fluent chaining.
    """
    ctx = get_context()
    ctx.watched_columns.difference_update(columns)
    return _get_module()


def clear_watch() -> types.ModuleType:
    """
    Clear all watched columns.

    Returns:
        The tracepipe module for fluent chaining.

    Examples:
        tracepipe.clear_watch().watch("new_column")
    """
    ctx = get_context()
    ctx.watched_columns.clear()
    return _get_module()


def register(df: pd.DataFrame) -> types.ModuleType:
    """
    Manually register a DataFrame for tracking.

    Use this for DataFrames created before enable() was called.

    Returns:
        The tracepipe module for fluent chaining.
    """
    ctx = get_context()
    if ctx.enabled:
        ctx.row_manager.register(df)
    return _get_module()


def stage(name: str):
    """Context manager for naming pipeline stages."""

    class StageContext:
        def __init__(self, stage_name: str):
            self.stage_name = stage_name
            self.previous_stage = None

        def __enter__(self):
            ctx = get_context()
            self.previous_stage = ctx.current_stage
            ctx.current_stage = self.stage_name
            return self

        def __exit__(self, *args):
            ctx = get_context()
            ctx.current_stage = self.previous_stage

    return StageContext(name)


# === QUERY API ===


class RowLineageResult:
    """Query result for a single row's journey."""

    def __init__(self, row_id: int, ctx: TracePipeContext):
        self.row_id = row_id
        self._ctx = ctx
        self._history = ctx.store.get_row_history(row_id)
        self._gaps = ctx.store.compute_gaps(row_id)

    @property
    def is_alive(self) -> bool:
        """Return True if row was not dropped."""
        return not any(h["change_type"] == "DROPPED" for h in self._history)

    @property
    def dropped_at(self) -> str | None:
        """Return operation name where row was dropped, or None."""
        for h in self._history:
            if h["change_type"] == "DROPPED":
                return h["operation"]
        return None

    def cell_history(self, column: str) -> list[dict]:
        """Get history for a specific column."""
        return [h for h in self._history if h["col"] == column]

    def history(self) -> list[dict]:
        """Get full history."""
        return self._history

    @property
    def gaps(self) -> LineageGaps:
        """Get lineage gaps."""
        return self._gaps

    @property
    def is_fully_tracked(self) -> bool:
        """Return True if no gaps in lineage."""
        return self._gaps.is_fully_tracked

    def to_dict(self) -> dict:
        """Export to dictionary."""
        return {
            "row_id": self.row_id,
            "is_alive": self.is_alive,
            "dropped_at": self.dropped_at,
            "is_fully_tracked": self.is_fully_tracked,
            "gaps_summary": self._gaps.summary(),
            "history": self._history,
        }

    def __repr__(self):
        status = "alive" if self.is_alive else f"dropped at {self.dropped_at}"
        return f"<RowLineage row_id={self.row_id} {status} events={len(self._history)}>"


class GroupLineageResult:
    """Query result for an aggregation group."""

    def __init__(self, group_key: str, ctx: TracePipeContext):
        self.group_key = group_key
        self._ctx = ctx
        self._info = ctx.store.get_group_members(group_key)

    @property
    def row_ids(self) -> list[int]:
        """Get list of row IDs in this group."""
        return self._info["row_ids"] if self._info else []

    @property
    def row_count(self) -> int:
        """Get number of rows in this group."""
        return self._info["row_count"] if self._info else 0

    @property
    def is_count_only(self) -> bool:
        """
        True if group exceeded max_group_membership_size threshold.

        When True, row_ids will be empty and only row_count is available.
        """
        return self._info.get("is_count_only", False) if self._info else False

    @property
    def group_column(self) -> str | None:
        """Get the column used for grouping."""
        return self._info["group_column"] if self._info else None

    @property
    def aggregation_functions(self) -> dict[str, str]:
        """Get the aggregation functions applied."""
        return self._info["agg_functions"] if self._info else {}

    def get_contributing_rows(self, limit: int = 100) -> list[RowLineageResult]:
        """
        Get lineage for contributing rows.

        Returns empty list if is_count_only is True.
        """
        if self.is_count_only:
            return []
        return [explain(row_id) for row_id in self.row_ids[:limit]]

    def to_dict(self) -> dict:
        """Export to dictionary."""
        return {
            "group_key": self.group_key,
            "group_column": self.group_column,
            "row_count": self.row_count,
            "row_ids": self.row_ids,
            "is_count_only": self.is_count_only,
            "aggregation_functions": self.aggregation_functions,
        }

    def __repr__(self):
        suffix = " (count only)" if self.is_count_only else ""
        return f"<GroupLineage key='{self.group_key}' rows={self.row_count}{suffix}>"


def explain(row_id: int) -> RowLineageResult:
    """Get lineage for a specific row."""
    ctx = get_context()
    return RowLineageResult(row_id, ctx)


def explain_many(row_ids: list[int]) -> list[RowLineageResult]:
    """
    Get lineage for multiple rows.

    Args:
        row_ids: List of row IDs to explain.

    Returns:
        List of RowLineageResult objects.

    Examples:
        results = tracepipe.explain_many([0, 1, 2])
        for row in results:
            print(row.is_alive, row.dropped_at)
    """
    ctx = get_context()
    return [RowLineageResult(row_id, ctx) for row_id in row_ids]


def explain_group(group_key: str) -> GroupLineageResult:
    """Get lineage for an aggregation group."""
    ctx = get_context()
    return GroupLineageResult(group_key, ctx)


def dropped_rows(by_step: bool = False) -> list[int] | dict[str, int]:
    """
    Get dropped row information.

    Args:
        by_step: If False (default), return list of dropped row IDs.
                 If True, return dict mapping operation names to drop counts.

    Returns:
        List of row IDs if by_step=False, or dict of {operation: count} if by_step=True.

    Examples:
        # Get all dropped row IDs
        dropped = tracepipe.dropped_rows()

        # Get counts by operation
        by_op = tracepipe.dropped_rows(by_step=True)
        # {'DataFrame.dropna': 5, 'DataFrame.query': 3}
    """
    ctx = get_context()
    if by_step:
        return ctx.store.get_dropped_by_step()
    return ctx.store.get_dropped_rows()


def alive_rows() -> list[int]:
    """
    Get all row IDs that are still alive (not dropped).

    Returns:
        List of row IDs that have not been dropped.

    Examples:
        alive = tracepipe.alive_rows()
        print(f"{len(alive)} rows survived the pipeline")
    """
    ctx = get_context()
    all_registered = set(ctx.row_manager.all_registered_ids())
    dropped = set(ctx.store.get_dropped_rows())
    return sorted(all_registered - dropped)


def mass_updates() -> list[dict]:
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


def steps() -> list[dict]:
    """Get all tracked steps."""
    ctx = get_context()
    return [
        {
            "step_id": s.step_id,
            "operation": s.operation,
            "stage": s.stage,
            "input_shape": s.input_shape,
            "output_shape": s.output_shape,
            "completeness": s.completeness.name,
            "is_mass_update": s.is_mass_update,
            "timestamp": s.timestamp,
            "code_file": s.code_file,
            "code_line": s.code_line,
        }
        for s in ctx.store.steps
    ]


def aggregation_groups() -> list[str]:
    """List all tracked aggregation groups."""
    ctx = get_context()
    groups = []
    for mapping in ctx.store.aggregation_mappings:
        groups.extend(mapping.membership.keys())
    return groups


# === EXPORT ===


def export_json(filepath: str) -> None:
    """Export lineage to JSON file."""
    ctx = get_context()
    with open(filepath, "w") as f:
        f.write(ctx.store.to_json())


def export_arrow(filepath: str) -> None:
    """
    Export lineage to Parquet file.

    Requires pyarrow to be installed.

    Args:
        filepath: Path to write the Parquet file.

    Raises:
        ImportError: If pyarrow is not installed.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for Arrow/Parquet export. "
            "Install it with: pip install tracepipe[arrow] or pip install pyarrow"
        ) from None

    ctx = get_context()
    table = ctx.store.to_arrow()
    pq.write_table(table, filepath)


def stats() -> dict:
    """Get tracking statistics."""
    ctx = get_context()
    return {
        "enabled": ctx.enabled,
        "total_steps": len(ctx.store.steps),
        "total_diffs": ctx.store.total_diff_count,
        "in_memory_diffs": ctx.store.diff_count,
        "spilled_files": len(ctx.store.spilled_files),
        "watched_columns": list(ctx.watched_columns),
        "aggregation_groups": len(ctx.store.aggregation_mappings),
    }

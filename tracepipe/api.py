# tracepipe/api.py
"""
Core API for TracePipe.

This module provides the foundational enable/disable/reset functions
and internal result classes. For user-facing functionality, see:
- convenience.py: check(), trace(), why(), report()
- debug.py: inspect(), export()
- contracts.py: contract()
- snapshot.py: snapshot(), diff()

Modes:
- CI: Fast stats and drop tracking. No merge provenance or ghost values.
- DEBUG: Full provenance with merge origin tracking and ghost row values.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Sequence
from dataclasses import fields

from .context import TracePipeContext, get_context, reset_context, set_context
from .core import LineageGaps, TracePipeConfig, TracePipeMode
from .instrumentation.pandas_inst import instrument_pandas, uninstrument_pandas
from .storage.base import LineageBackend, RowIdentityStrategy


def _get_module() -> types.ModuleType:
    """Get the tracepipe module for fluent chaining."""
    return sys.modules["tracepipe"]


def enable(
    config: TracePipeConfig | None = None,
    mode: TracePipeMode | str | None = None,
    *,
    watch: Sequence[str] | None = None,
    auto_watch: bool = False,
    backend: LineageBackend | None = None,
    identity: RowIdentityStrategy | None = None,
    merge_provenance: bool | None = None,
    ghost_row_values: bool | None = None,
    cell_history: bool | None = None,
    sample_rate: float | None = None,
    max_tracked_rows: int | None = None,
) -> types.ModuleType:
    """
    Enable TracePipe lineage tracking.

    Args:
        config: Optional configuration object
        mode: Operating mode - "ci" (fast) or "debug" (full provenance)
        watch: List of columns to watch for cell-level changes
        auto_watch: If True, automatically watch columns with nulls
        backend: Optional custom storage backend
        identity: Optional custom row identity strategy
        merge_provenance: Override: capture merge parent RIDs (DEBUG default: True)
        ghost_row_values: Override: capture last values of dropped rows
        cell_history: Override: capture cell-level changes
        sample_rate: Track only this fraction of rows (0.0-1.0)
        max_tracked_rows: Maximum rows to track (for large datasets)

    Returns:
        The tracepipe module for fluent chaining.

    Examples:
        # CI mode (fast, default)
        tp.enable()

        # Debug mode with watched columns
        tp.enable(mode="debug", watch=["age", "salary"])

        # Custom configuration
        tp.enable(mode="ci", merge_provenance=True)
    """
    # Get or create config
    # If config is provided explicitly, use it
    # Otherwise, start with existing context config (if any) or create new default
    if config is None:
        existing_ctx = get_context()
        config = existing_ctx.config  # Use existing config as base

    # Handle mode
    if mode is not None:
        if isinstance(mode, str):
            mode = TracePipeMode(mode.lower())
        config.mode = mode

    # Apply feature overrides
    if merge_provenance is not None:
        config.merge_provenance = merge_provenance
    if ghost_row_values is not None:
        config.ghost_row_values = ghost_row_values
    if cell_history is not None:
        config.cell_history = cell_history

    if auto_watch:
        config.auto_watch = True

    # Sampling config validation
    if sample_rate is not None or max_tracked_rows is not None:
        import warnings

        warnings.warn(
            "sample_rate and max_tracked_rows are not yet implemented. "
            "These parameters will be ignored.",
            UserWarning,
            stacklevel=2,
        )

    # Create context with custom backends if provided
    if backend is not None or identity is not None:
        ctx = TracePipeContext(config=config, backend=backend, identity=identity)
        set_context(ctx)
    else:
        ctx = get_context()
        ctx.config = config
        # Also update config in row_manager and store (they may have their own references)
        ctx.row_manager.config = config
        ctx.store.config = config

    # Add watched columns
    if watch:
        ctx.watched_columns.update(watch)

    if not ctx.enabled:
        instrument_pandas()
        ctx.enabled = True

    return _get_module()


def disable() -> types.ModuleType:
    """
    Disable TracePipe and restore original pandas methods.

    Note:
        This stops tracking but preserves lineage data collected so far.
        Use reset() to clear all data.

    Returns:
        The tracepipe module for fluent chaining.
    """
    ctx = get_context()

    if ctx.enabled:
        uninstrument_pandas()
        if hasattr(ctx.store, "_cleanup_spillover"):
            ctx.store._cleanup_spillover()
        ctx.enabled = False

    return _get_module()


def reset() -> types.ModuleType:
    """
    Reset all tracking state for the current thread.

    This clears ALL lineage data, steps, watched columns, and row registrations.
    If tracking was enabled, it will be re-enabled with a fresh context.

    Returns:
        The tracepipe module for fluent chaining.
    """
    ctx = get_context()
    was_enabled = ctx.enabled

    if was_enabled:
        uninstrument_pandas()

    reset_context()

    if was_enabled:
        enable()

    return _get_module()


def configure(**kwargs) -> types.ModuleType:
    """
    Update configuration.

    Args:
        **kwargs: Configuration options to update.

    Returns:
        The tracepipe module for fluent chaining.
    """
    ctx = get_context()

    valid_keys = {f.name for f in fields(TracePipeConfig)}
    invalid_keys = set(kwargs.keys()) - valid_keys
    if invalid_keys:
        raise ValueError(f"Invalid configuration key(s): {invalid_keys}")

    for key, value in kwargs.items():
        setattr(ctx.config, key, value)

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


def register(*dfs) -> types.ModuleType:
    """
    Register pre-existing DataFrames for tracking.

    Use this when DataFrames were created before tp.enable() was called.
    After registration, snapshots, ghost rows, and cell history will work.

    Args:
        *dfs: One or more DataFrames to register

    Returns:
        The tracepipe module for fluent chaining.

    Examples:
        # DataFrames created before enable
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})

        tp.enable()
        tp.register(df1, df2)  # Now they're tracked

        snap = tp.snapshot(df1)  # Works!
    """
    import pandas as pd

    ctx = get_context()

    if not ctx.enabled:
        import warnings

        warnings.warn(
            "TracePipe is not enabled. Call tp.enable() before tp.register().",
            UserWarning,
            stacklevel=2,
        )
        return _get_module()

    for df in dfs:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df).__name__}")

        # Only register if not already registered
        if ctx.row_manager.get_ids_array(df) is None:
            ctx.row_manager.register(df)

    return _get_module()


# === INTERNAL RESULT CLASSES (used by debug module) ===


class RowLineageResult:
    """Query result for a single row's journey."""

    def __init__(self, row_id: int, ctx: TracePipeContext):
        self.row_id = row_id
        self._ctx = ctx
        self._history: list[dict] | None = None
        self._gaps: LineageGaps | None = None
        self._drop_event: dict | None = None
        self._drop_event_checked: bool = False

    def _ensure_drop_event(self) -> None:
        if not self._drop_event_checked:
            self._drop_event = self._ctx.store.get_drop_event(self.row_id)
            self._drop_event_checked = True

    def _ensure_history(self) -> None:
        if self._history is None:
            self._history = self._ctx.store.get_row_history(self.row_id)

    def _ensure_gaps(self) -> None:
        if self._gaps is None:
            self._gaps = self._ctx.store.compute_gaps(self.row_id)

    @property
    def is_alive(self) -> bool:
        self._ensure_drop_event()
        return self._drop_event is None

    @property
    def dropped_at(self) -> str | None:
        self._ensure_drop_event()
        if self._drop_event is not None:
            return self._drop_event.get("operation")
        return None

    @property
    def dropped_step_id(self) -> int | None:
        self._ensure_drop_event()
        if self._drop_event is not None:
            return self._drop_event.get("step_id")
        return None

    def merge_origin(self) -> dict | None:
        return self._ctx.store.get_merge_origin(self.row_id)

    def cell_history(self, column: str) -> list[dict]:
        self._ensure_history()
        return [h for h in self._history if h["col"] == column]

    def history(self) -> list[dict]:
        self._ensure_history()
        return self._history

    @property
    def gaps(self) -> LineageGaps:
        self._ensure_gaps()
        return self._gaps

    @property
    def is_fully_tracked(self) -> bool:
        self._ensure_gaps()
        return self._gaps.is_fully_tracked

    def to_dict(self) -> dict:
        self._ensure_history()
        self._ensure_gaps()
        merge = self.merge_origin()
        return {
            "row_id": self.row_id,
            "is_alive": self.is_alive,
            "dropped_at": self.dropped_at,
            "dropped_step_id": self.dropped_step_id,
            "is_fully_tracked": self.is_fully_tracked,
            "gaps_summary": self._gaps.summary(),
            "merge_origin": merge,
            "history": self._history,
        }

    def __repr__(self):
        status = "alive" if self.is_alive else f"dropped at {self.dropped_at}"
        return f"<RowLineage row_id={self.row_id} {status} events={len(self.history())}>"


class GroupLineageResult:
    """Query result for an aggregation group."""

    def __init__(self, group_key: str, ctx: TracePipeContext):
        self.group_key = group_key
        self._ctx = ctx
        self._info = ctx.store.get_group_members(group_key)

    @property
    def row_ids(self) -> list[int]:
        return self._info["row_ids"] if self._info else []

    @property
    def row_count(self) -> int:
        return self._info["row_count"] if self._info else 0

    @property
    def is_count_only(self) -> bool:
        return self._info.get("is_count_only", False) if self._info else False

    @property
    def group_column(self) -> str | None:
        return self._info["group_column"] if self._info else None

    @property
    def aggregation_functions(self) -> dict[str, str]:
        return self._info["agg_functions"] if self._info else {}

    def to_dict(self) -> dict:
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

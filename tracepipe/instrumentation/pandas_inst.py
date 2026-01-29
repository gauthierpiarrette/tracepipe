# tracepipe/instrumentation/pandas_inst.py
"""
Pandas DataFrame instrumentation for row-level lineage tracking.
"""

import warnings
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd

from ..context import TracePipeContext, get_context
from ..core import ChangeType, CompletenessLevel
from ..safety import (
    TracePipeWarning,
    get_caller_info,
    wrap_pandas_filter_method,
    wrap_pandas_method,
    wrap_pandas_method_inplace,
)
from ..utils.value_capture import find_changed_indices_vectorized

# Store original methods for restore
_originals: dict[str, Any] = {}


# === FILTER CAPTURE ===


def _capture_filter(
    self: pd.DataFrame, args, kwargs, result, ctx: TracePipeContext, method_name: str
):
    """Capture lineage for filter operations (dropna, query, head, etc.)."""
    if not isinstance(result, pd.DataFrame):
        return

    source_ids = ctx.row_manager.get_ids(self)
    if source_ids is None:
        # Auto-register if not tracked
        ctx.row_manager.register(self)
        source_ids = ctx.row_manager.get_ids(self)
        if source_ids is None:
            return

    # Propagate IDs to result
    ctx.row_manager.propagate(self, result)

    # Find dropped rows (returns numpy array for performance)
    dropped_ids = ctx.row_manager.get_dropped_ids(self, result)

    if len(dropped_ids) > 0:
        code_file, code_line = get_caller_info(skip_frames=2)
        step_id = ctx.store.append_step(
            operation=f"DataFrame.{method_name}",
            stage=ctx.current_stage,
            code_file=code_file,
            code_line=code_line,
            params=_safe_params(kwargs),
            input_shape=self.shape,
            output_shape=result.shape,
        )

        # Bulk record all drops at once (10-50x faster than loop)
        ctx.store.append_bulk_drops(step_id, dropped_ids)


# === TRANSFORM CAPTURE ===


def _capture_transform(
    self: pd.DataFrame, args, kwargs, result, ctx: TracePipeContext, method_name: str
):
    """Capture lineage for transform operations (fillna, replace, astype)."""
    if not isinstance(result, pd.DataFrame):
        return

    source_ids = ctx.row_manager.get_ids(self)
    if source_ids is None:
        return

    # Propagate IDs
    ctx.row_manager.propagate(self, result)

    # Only track watched columns
    if not ctx.watched_columns:
        return

    # Determine affected columns
    affected = _get_affected_columns(method_name, args, kwargs, self.columns)
    cols_to_track = affected & ctx.watched_columns

    if not cols_to_track:
        return

    # Check if mass update
    if not ctx.store.should_track_cell_diffs(len(self)):
        code_file, code_line = get_caller_info(skip_frames=2)
        ctx.store.append_step(
            operation=f"DataFrame.{method_name}",
            stage=ctx.current_stage,
            code_file=code_file,
            code_line=code_line,
            params=_safe_params(kwargs),
            input_shape=self.shape,
            output_shape=result.shape,
            is_mass_update=True,
            rows_affected=len(self),
        )
        return

    # Find common index for vectorized comparison
    common_index = self.index.intersection(result.index).intersection(source_ids.index)
    if len(common_index) == 0:
        return

    ids_aligned = source_ids.reindex(common_index)
    ids_arr = ids_aligned.values

    # Check each column for changes using vectorized comparison
    all_changes = []  # Collect (row_id, col, old_val, new_val)

    for col in cols_to_track:
        if col not in self.columns or col not in result.columns:
            continue

        old_aligned = self[col].reindex(common_index)
        new_aligned = result[col].reindex(common_index)

        # Vectorized change detection (~50-100x faster)
        changed_mask = find_changed_indices_vectorized(old_aligned, new_aligned)

        if not changed_mask.any():
            continue

        changed_indices = np.where(changed_mask)[0]
        old_arr = old_aligned.values
        new_arr = new_aligned.values

        for i in changed_indices:
            all_changes.append((int(ids_arr[i]), col, old_arr[i], new_arr[i]))

    if not all_changes:
        return  # No changes detected, skip step creation

    # Create step only if there are actual changes
    code_file, code_line = get_caller_info(skip_frames=2)
    step_id = ctx.store.append_step(
        operation=f"DataFrame.{method_name}",
        stage=ctx.current_stage,
        code_file=code_file,
        code_line=code_line,
        params=_safe_params(kwargs),
        input_shape=self.shape,
        output_shape=result.shape,
    )

    # Record all changes
    for row_id, col, old_val, new_val in all_changes:
        ctx.store.append_diff(
            step_id=step_id,
            row_id=row_id,
            col=col,
            old_val=old_val,
            new_val=new_val,
            change_type=ChangeType.MODIFIED,
        )


# === APPLY/PIPE CAPTURE (PARTIAL) ===


def _capture_apply(
    self: pd.DataFrame, args, kwargs, result, ctx: TracePipeContext, method_name: str
):
    """Capture apply/pipe with PARTIAL completeness."""
    func = args[0] if args else kwargs.get("func")
    func_name = getattr(func, "__name__", "<lambda>")

    code_file, code_line = get_caller_info(skip_frames=2)
    step_id = ctx.store.append_step(
        operation=f"DataFrame.{method_name}({func_name})",
        stage=ctx.current_stage,
        code_file=code_file,
        code_line=code_line,
        params={"func": func_name},
        input_shape=self.shape,
        output_shape=result.shape if hasattr(result, "shape") else None,
        completeness=CompletenessLevel.PARTIAL,
    )

    # Still propagate and track output changes if result is DataFrame
    if isinstance(result, pd.DataFrame):
        ctx.row_manager.propagate(self, result)

        # Track changes to watched columns
        if ctx.watched_columns:
            _capture_cell_changes(ctx, step_id, self, result)


# === GROUPBY CAPTURE ===


def _capture_groupby(
    self: pd.DataFrame, args, kwargs, result, ctx: TracePipeContext, method_name: str
):
    """Capture groupby without re-calling groupby."""
    row_ids = ctx.row_manager.get_ids(self)
    if row_ids is None:
        return

    by = args[0] if args else kwargs.get("by")

    # Extract groups from RESULT (already computed)
    if hasattr(result, "groups"):
        # Clear any stale groupby state from same source (handles new groupby on same df)
        ctx.clear_groupby_for_source(id(self))

        ctx.push_groupby(
            {
                "source_df": self,
                "source_id": id(self),
                "row_ids": row_ids,
                "by": by,
                "groups": result.groups,
            }
        )


def _capture_agg(self, args, kwargs, result, ctx: TracePipeContext, method_name: str):
    """
    Capture aggregation and record group membership.

    Note: Uses peek_groupby() instead of pop_groupby() to support
    multiple aggregations on the same GroupBy object:

        grouped = df.groupby("category")
        means = grouped.mean()   # First agg - state preserved
        sums = grouped.sum()     # Second agg - still works!

    State is cleared when a new groupby() is called on the same source.
    """
    pending = ctx.peek_groupby()
    if pending is None:
        return

    row_ids = pending["row_ids"]
    source_df = pending["source_df"]
    max_membership = ctx.config.max_group_membership_size

    # Build membership mapping (with threshold for large groups)
    membership = {}
    for group_key, indices in pending["groups"].items():
        group_size = len(indices)

        if group_size > max_membership:
            # Large group - store count only to save memory
            # Use special marker: negative count indicates "count only"
            membership[str(group_key)] = [-group_size]
        else:
            # Normal group - store full membership
            member_ids = []
            for idx in indices:
                if idx in row_ids.index:
                    member_ids.append(int(row_ids.loc[idx]))
            membership[str(group_key)] = member_ids

    # Determine aggregation functions
    agg_funcs = {}
    if args:
        agg_arg = args[0]
        if isinstance(agg_arg, dict):
            agg_funcs = {k: str(v) for k, v in agg_arg.items()}
        elif isinstance(agg_arg, str):
            agg_funcs = {"_all_": agg_arg}
        elif isinstance(agg_arg, list):
            agg_funcs = {"_all_": str(agg_arg)}

    code_file, code_line = get_caller_info(skip_frames=2)
    step_id = ctx.store.append_step(
        operation=f"GroupBy.{method_name}",
        stage=ctx.current_stage,
        code_file=code_file,
        code_line=code_line,
        params={"by": str(pending["by"])},
        input_shape=source_df.shape,
        output_shape=result.shape if hasattr(result, "shape") else None,
    )

    ctx.store.append_aggregation(
        step_id=step_id,
        group_column=str(pending["by"]),
        membership=membership,
        agg_functions=agg_funcs,
    )

    # Register result with new IDs (aggregation creates new "rows")
    if isinstance(result, pd.DataFrame):
        ctx.row_manager.register(result)


# === MERGE/CONCAT (UNKNOWN - OUT OF SCOPE) ===


def _capture_merge(
    self: pd.DataFrame, args, kwargs, result, ctx: TracePipeContext, method_name: str
):
    """Mark merge as UNKNOWN completeness and reset lineage."""
    code_file, code_line = get_caller_info(skip_frames=2)
    ctx.store.append_step(
        operation=f"DataFrame.{method_name}",
        stage=ctx.current_stage,
        code_file=code_file,
        code_line=code_line,
        params={"how": kwargs.get("how", "inner")},
        input_shape=self.shape,
        output_shape=result.shape if hasattr(result, "shape") else None,
        completeness=CompletenessLevel.UNKNOWN,
    )

    warnings.warn(
        f"TracePipe: {method_name}() resets row lineage. "
        f"Rows in result cannot be traced back to source rows.",
        TracePipeWarning,
    )

    # Register result with NEW IDs
    if isinstance(result, pd.DataFrame):
        ctx.row_manager.register(result)


def _capture_concat(args, kwargs, result, ctx: TracePipeContext):
    """Capture pd.concat (module-level function)."""
    code_file, code_line = get_caller_info(skip_frames=2)
    ctx.store.append_step(
        operation="pd.concat",
        stage=ctx.current_stage,
        code_file=code_file,
        code_line=code_line,
        params={"axis": kwargs.get("axis", 0)},
        input_shape=None,
        output_shape=result.shape if hasattr(result, "shape") else None,
        completeness=CompletenessLevel.UNKNOWN,
    )

    warnings.warn("TracePipe: pd.concat() resets row lineage.", TracePipeWarning)

    if isinstance(result, pd.DataFrame):
        ctx.row_manager.register(result)


# === INDEX OPERATIONS ===


def _capture_reset_index(
    self: pd.DataFrame, args, kwargs, result, ctx: TracePipeContext, method_name: str
):
    """Handle reset_index which changes index alignment."""
    if not isinstance(result, pd.DataFrame):
        return

    drop = kwargs.get("drop", False)
    if args and len(args) > 0:
        # reset_index(drop=True) might pass drop as positional
        pass

    if drop:
        ctx.row_manager.realign_for_reset_index(self, result)
    else:
        ctx.row_manager.propagate(self, result)


def _capture_set_index(
    self: pd.DataFrame, args, kwargs, result, ctx: TracePipeContext, method_name: str
):
    """Handle set_index."""
    if isinstance(result, pd.DataFrame):
        ctx.row_manager.propagate(self, result)


def _capture_sort_values(
    self: pd.DataFrame, args, kwargs, result, ctx: TracePipeContext, method_name: str
):
    """Handle sort_values with order tracking."""
    if not isinstance(result, pd.DataFrame):
        return

    source_ids = ctx.row_manager.get_ids(self)
    if source_ids is None:
        return

    ctx.row_manager.propagate(self, result)

    by = args[0] if args else kwargs.get("by")
    ascending = kwargs.get("ascending", True)

    code_file, code_line = get_caller_info(skip_frames=2)
    step_id = ctx.store.append_step(
        operation="DataFrame.sort_values",
        stage=ctx.current_stage,
        code_file=code_file,
        code_line=code_line,
        params={"by": str(by), "ascending": ascending},
        input_shape=self.shape,
        output_shape=result.shape,
    )

    # Record reorder for each row
    result_ids = ctx.row_manager.get_ids(result)
    if result_ids is not None:
        for new_pos, (idx, row_id) in enumerate(result_ids.items()):
            # Find old position
            try:
                old_pos = list(source_ids.index).index(idx)
                if old_pos != new_pos:
                    ctx.store.append_diff(
                        step_id=step_id,
                        row_id=int(row_id),
                        col="__position__",
                        old_val=old_pos,
                        new_val=new_pos,
                        change_type=ChangeType.REORDERED,
                    )
            except (ValueError, KeyError):
                pass


# === COPY CAPTURE ===


def _capture_copy(
    self: pd.DataFrame, args, kwargs, result, ctx: TracePipeContext, method_name: str
):
    """
    Capture df.copy() - propagate row IDs to the copy.

    Without this, copies would lose their row identity and become untracked.
    """
    if not isinstance(result, pd.DataFrame):
        return

    source_ids = ctx.row_manager.get_ids(self)
    if source_ids is None:
        return

    # Propagate IDs to the copy (same rows, new DataFrame object)
    ctx.row_manager.propagate(self, result)


# === DROP CAPTURE ===


def _capture_drop(
    self: pd.DataFrame, args, kwargs, result, ctx: TracePipeContext, method_name: str
):
    """
    Capture df.drop() - handles both row and column drops.

    - Row drops (axis=0): Track as filter operation
    - Column drops (axis=1): Track as schema change (step metadata only)
    """
    if not isinstance(result, pd.DataFrame):
        return

    axis = kwargs.get("axis", 0)
    if args and isinstance(args[0], int):
        axis = args[0]

    source_ids = ctx.row_manager.get_ids(self)

    if axis == 0 or axis == "index":
        # Row drop - similar to filter
        if source_ids is None:
            return

        ctx.row_manager.propagate(self, result)
        dropped_ids = ctx.row_manager.get_dropped_ids(self, result)

        if len(dropped_ids) > 0:
            labels = kwargs.get("labels") or kwargs.get("index") or (args[0] if args else None)
            code_file, code_line = get_caller_info(skip_frames=2)
            step_id = ctx.store.append_step(
                operation="DataFrame.drop",
                stage=ctx.current_stage,
                code_file=code_file,
                code_line=code_line,
                params={"axis": "index", "labels": str(labels)[:100]},
                input_shape=self.shape,
                output_shape=result.shape,
            )

            # Bulk record all drops at once
            ctx.store.append_bulk_drops(step_id, dropped_ids)
    else:
        # Column drop - schema change, just propagate IDs
        if source_ids is not None:
            ctx.row_manager.propagate(self, result)

        columns = kwargs.get("columns") or kwargs.get("labels") or (args[0] if args else None)
        code_file, code_line = get_caller_info(skip_frames=2)
        ctx.store.append_step(
            operation="DataFrame.drop",
            stage=ctx.current_stage,
            code_file=code_file,
            code_line=code_line,
            params={"axis": "columns", "columns": str(columns)[:100]},
            input_shape=self.shape,
            output_shape=result.shape,
        )


# === __getitem__ DISPATCH ===


def _capture_getitem(
    self: pd.DataFrame, args, kwargs, result, ctx: TracePipeContext, method_name: str
):
    """
    Dispatch __getitem__ based on key type.

    - df['col'] -> Series (ignore)
    - df[['a','b']] -> DataFrame column select (propagate)
    - df[mask] -> DataFrame row filter (track drops)
    - df[slice] -> DataFrame row slice (track drops)
    """
    if len(args) != 1:
        return

    key = args[0]

    # Series result - column access, not row filter
    if isinstance(result, pd.Series):
        return

    if not isinstance(result, pd.DataFrame):
        return

    # Boolean mask - row filter
    if isinstance(key, (pd.Series, np.ndarray)) and getattr(key, "dtype", None) is np.dtype("bool"):
        # Skip if we're inside a named filter op (e.g., drop_duplicates)
        # to avoid double-counting drops
        if ctx._filter_op_depth > 0:
            ctx.row_manager.propagate(self, result)
            return
        _capture_filter(self, args, kwargs, result, ctx, "__getitem__[mask]")
        return

    # List of columns - column selection
    if isinstance(key, list):
        ctx.row_manager.propagate(self, result)
        return

    # Slice - row selection
    if isinstance(key, slice):
        # Skip if we're inside a named filter op
        if ctx._filter_op_depth > 0:
            ctx.row_manager.propagate(self, result)
            return
        _capture_filter(self, args, kwargs, result, ctx, "__getitem__[slice]")
        return

    # Default: propagate
    ctx.row_manager.propagate(self, result)


# === __setitem__ CAPTURE ===


def _capture_setitem_with_before(
    self: pd.DataFrame, key: str, before_values: pd.Series, ctx: TracePipeContext
):
    """
    Capture column assignment with before/after values.

    Called after assignment completes, with before_values captured earlier.
    Uses vectorized comparison for ~50-100x speedup over row-by-row .loc access.
    """
    source_ids = ctx.row_manager.get_ids(self)
    if source_ids is None:
        return

    after_values = self[key]

    # Align series to same index for vectorized comparison
    common_index = before_values.index.intersection(after_values.index).intersection(
        source_ids.index
    )
    if len(common_index) == 0:
        return

    before_aligned = before_values.reindex(common_index)
    after_aligned = after_values.reindex(common_index)
    ids_aligned = source_ids.reindex(common_index)

    # Vectorized: find which rows changed (~50-100x faster than loop)
    changed_mask = find_changed_indices_vectorized(before_aligned, after_aligned)

    if not changed_mask.any():
        return  # No changes, skip step creation

    code_file, code_line = get_caller_info(skip_frames=2)
    step_id = ctx.store.append_step(
        operation=f"DataFrame.__setitem__[{key}]",
        stage=ctx.current_stage,
        code_file=code_file,
        code_line=code_line,
        params={"column": str(key)},
        input_shape=self.shape,
        output_shape=self.shape,
    )

    # Extract only changed values (numpy arrays for fast access)
    changed_indices = np.where(changed_mask)[0]
    old_arr = before_aligned.values
    new_arr = after_aligned.values
    ids_arr = ids_aligned.values

    # Only loop over changed rows (typically small fraction of total)
    for i in changed_indices:
        ctx.store.append_diff(
            step_id=step_id,
            row_id=int(ids_arr[i]),
            col=key,
            old_val=old_arr[i],
            new_val=new_arr[i],
            change_type=ChangeType.MODIFIED,
        )


def _capture_setitem_new_column(self: pd.DataFrame, key: str, ctx: TracePipeContext):
    """
    Capture assignment to a new column (no before values).

    Uses vectorized array access for performance.
    """
    source_ids = ctx.row_manager.get_ids(self)
    if source_ids is None:
        return

    new_values = self[key]

    # Align to common index
    common_index = new_values.index.intersection(source_ids.index)
    if len(common_index) == 0:
        return

    code_file, code_line = get_caller_info(skip_frames=2)
    step_id = ctx.store.append_step(
        operation=f"DataFrame.__setitem__[{key}]",
        stage=ctx.current_stage,
        code_file=code_file,
        code_line=code_line,
        params={"column": str(key), "is_new_column": True},
        input_shape=self.shape,
        output_shape=self.shape,
    )

    # Use numpy arrays for fast access (avoid .loc per row)
    new_aligned = new_values.reindex(common_index)
    ids_aligned = source_ids.reindex(common_index)

    new_arr = new_aligned.values
    ids_arr = ids_aligned.values

    for i in range(len(ids_arr)):
        ctx.store.append_diff(
            step_id=step_id,
            row_id=int(ids_arr[i]),
            col=key,
            old_val=None,
            new_val=new_arr[i],
            change_type=ChangeType.ADDED,
        )


def _wrap_setitem(original):
    """
    Wrap __setitem__ to capture column assignments.

    Captures BEFORE state for existing columns, then executes assignment,
    then records the diff with actual old/new values.
    """

    @wraps(original)
    def wrapper(self, key, value):
        ctx = get_context()

        # === CAPTURE BEFORE STATE ===
        before_values = None
        is_new_column = False
        should_track = False

        if ctx.enabled and isinstance(key, str):
            if key in ctx.watched_columns:
                should_track = True
                if key in self.columns:
                    # Existing column - capture before values
                    try:
                        before_values = self[key].copy()
                    except Exception:
                        pass
                else:
                    # New column
                    is_new_column = True

        # === EXECUTE ORIGINAL ===
        original(self, key, value)

        # === CAPTURE AFTER STATE ===
        if should_track:
            try:
                if is_new_column:
                    _capture_setitem_new_column(self, key, ctx)
                elif before_values is not None:
                    _capture_setitem_with_before(self, key, before_values, ctx)
            except Exception as e:
                if ctx.config.strict_mode:
                    from ..safety import TracePipeError

                    raise TracePipeError(f"__setitem__ instrumentation failed: {e}") from e
                else:
                    warnings.warn(f"TracePipe: __setitem__ failed: {e}", TracePipeWarning)

    return wrapper


# === AUTO-REGISTRATION ===


def _wrap_dataframe_reader(original, reader_name: str):
    """Wrap pd.read_csv etc. to auto-register."""

    @wraps(original)
    def wrapper(*args, **kwargs):
        result = original(*args, **kwargs)

        ctx = get_context()
        if ctx.enabled and isinstance(result, pd.DataFrame):
            ctx.row_manager.register(result)

        return result

    return wrapper


def _wrap_dataframe_init(original):
    """Wrap DataFrame.__init__ for auto-registration."""

    @wraps(original)
    def wrapper(self, *args, **kwargs):
        original(self, *args, **kwargs)

        ctx = get_context()
        if ctx.enabled:
            if ctx.row_manager.get_ids(self) is None:
                ctx.row_manager.register(self)

    return wrapper


# === EXPORT HOOKS (Auto-strip hidden column) ===


def _make_export_wrapper(original):
    """Create a wrapper that strips hidden column before export."""

    @wraps(original)
    def wrapper(self, *args, **kwargs):
        ctx = get_context()
        if ctx.enabled:
            clean_df = ctx.row_manager.strip_hidden_column(self)
            return original(clean_df, *args, **kwargs)
        return original(self, *args, **kwargs)

    return wrapper


# === HELPER FUNCTIONS ===


def _safe_params(kwargs: dict) -> dict:
    """Extract safe (serializable) params from kwargs."""
    safe = {}
    for k, v in kwargs.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            safe[k] = v
        elif isinstance(v, (list, tuple)) and all(
            isinstance(x, (str, int, float, bool)) for x in v
        ):
            safe[k] = list(v)
        else:
            safe[k] = str(type(v).__name__)
    return safe


def _get_affected_columns(method_name: str, args, kwargs, all_columns: pd.Index) -> set[str]:
    """Determine which columns are affected by an operation."""
    if method_name == "fillna":
        value = args[0] if args else kwargs.get("value")
        if isinstance(value, dict):
            return set(value.keys())
        return set(all_columns)

    elif method_name == "replace":
        return set(all_columns)

    elif method_name == "astype":
        dtype = args[0] if args else kwargs.get("dtype")
        if isinstance(dtype, dict):
            return set(dtype.keys())
        return set(all_columns)

    elif method_name == "__setitem__":
        key = args[0] if args else None
        if isinstance(key, str):
            return {key}
        elif isinstance(key, list):
            return set(key)

    return set(all_columns)


def _capture_cell_changes(
    ctx: TracePipeContext, step_id: int, before: pd.DataFrame, after: pd.DataFrame
):
    """
    Capture cell-level changes between two DataFrames.

    Uses vectorized comparison for ~50-100x speedup.
    """
    source_ids = ctx.row_manager.get_ids(before)
    if source_ids is None:
        return

    cols_to_track = ctx.watched_columns & set(before.columns) & set(after.columns)
    if not cols_to_track:
        return

    # Find common index for vectorized comparison
    common_index = before.index.intersection(after.index).intersection(source_ids.index)
    if len(common_index) == 0:
        return

    ids_aligned = source_ids.reindex(common_index)
    ids_arr = ids_aligned.values

    for col in cols_to_track:
        old_aligned = before[col].reindex(common_index)
        new_aligned = after[col].reindex(common_index)

        # Vectorized change detection
        changed_mask = find_changed_indices_vectorized(old_aligned, new_aligned)

        if not changed_mask.any():
            continue

        changed_indices = np.where(changed_mask)[0]
        old_arr = old_aligned.values
        new_arr = new_aligned.values

        for i in changed_indices:
            ctx.store.append_diff(
                step_id=step_id,
                row_id=int(ids_arr[i]),
                col=col,
                old_val=old_arr[i],
                new_val=new_arr[i],
                change_type=ChangeType.MODIFIED,
            )


# === INSTRUMENTATION SETUP ===


def instrument_pandas():
    """Install all pandas instrumentation."""
    global _originals

    if _originals:
        # Already instrumented
        return

    # === DataFrame filter methods ===
    # Use wrap_pandas_filter_method to prevent double-counting when
    # methods like drop_duplicates internally call __getitem__
    filter_methods = ["dropna", "drop_duplicates", "query", "head", "tail", "sample"]
    for method_name in filter_methods:
        if hasattr(pd.DataFrame, method_name):
            original = getattr(pd.DataFrame, method_name)
            _originals[f"DataFrame.{method_name}"] = original
            wrapped = wrap_pandas_filter_method(method_name, original, _capture_filter)
            setattr(pd.DataFrame, method_name, wrapped)

    # === DataFrame transform methods (with inplace support) ===
    transform_methods = ["fillna", "replace"]
    for method_name in transform_methods:
        if hasattr(pd.DataFrame, method_name):
            original = getattr(pd.DataFrame, method_name)
            _originals[f"DataFrame.{method_name}"] = original
            wrapped = wrap_pandas_method_inplace(method_name, original, _capture_transform)
            setattr(pd.DataFrame, method_name, wrapped)

    # === astype (no inplace) ===
    _originals["DataFrame.astype"] = pd.DataFrame.astype
    pd.DataFrame.astype = wrap_pandas_method("astype", pd.DataFrame.astype, _capture_transform)

    # === copy (preserves row identity) ===
    _originals["DataFrame.copy"] = pd.DataFrame.copy
    pd.DataFrame.copy = wrap_pandas_method("copy", pd.DataFrame.copy, _capture_copy)

    # === drop (row/column removal) ===
    _originals["DataFrame.drop"] = pd.DataFrame.drop
    pd.DataFrame.drop = wrap_pandas_method("drop", pd.DataFrame.drop, _capture_drop)

    # === apply/pipe ===
    _originals["DataFrame.apply"] = pd.DataFrame.apply
    pd.DataFrame.apply = wrap_pandas_method("apply", pd.DataFrame.apply, _capture_apply)

    _originals["DataFrame.pipe"] = pd.DataFrame.pipe
    pd.DataFrame.pipe = wrap_pandas_method("pipe", pd.DataFrame.pipe, _capture_apply)

    # === groupby ===
    _originals["DataFrame.groupby"] = pd.DataFrame.groupby
    pd.DataFrame.groupby = wrap_pandas_method("groupby", pd.DataFrame.groupby, _capture_groupby)

    # === GroupBy aggregation methods ===
    from pandas.core.groupby import DataFrameGroupBy

    for agg_method in ["agg", "aggregate", "sum", "mean", "count", "min", "max", "std", "var"]:
        if hasattr(DataFrameGroupBy, agg_method):
            original = getattr(DataFrameGroupBy, agg_method)
            _originals[f"DataFrameGroupBy.{agg_method}"] = original
            wrapped = wrap_pandas_method(agg_method, original, _capture_agg)
            setattr(DataFrameGroupBy, agg_method, wrapped)

    # === merge ===
    _originals["DataFrame.merge"] = pd.DataFrame.merge
    pd.DataFrame.merge = wrap_pandas_method("merge", pd.DataFrame.merge, _capture_merge)

    _originals["DataFrame.join"] = pd.DataFrame.join
    pd.DataFrame.join = wrap_pandas_method("join", pd.DataFrame.join, _capture_merge)

    # === Index operations ===
    _originals["DataFrame.reset_index"] = pd.DataFrame.reset_index
    pd.DataFrame.reset_index = wrap_pandas_method(
        "reset_index", pd.DataFrame.reset_index, _capture_reset_index
    )

    _originals["DataFrame.set_index"] = pd.DataFrame.set_index
    pd.DataFrame.set_index = wrap_pandas_method(
        "set_index", pd.DataFrame.set_index, _capture_set_index
    )

    _originals["DataFrame.sort_values"] = pd.DataFrame.sort_values
    pd.DataFrame.sort_values = wrap_pandas_method(
        "sort_values", pd.DataFrame.sort_values, _capture_sort_values
    )

    # === __getitem__ ===
    _originals["DataFrame.__getitem__"] = pd.DataFrame.__getitem__
    pd.DataFrame.__getitem__ = wrap_pandas_method(
        "__getitem__", pd.DataFrame.__getitem__, _capture_getitem
    )

    # === __setitem__ (column assignment) ===
    _originals["DataFrame.__setitem__"] = pd.DataFrame.__setitem__
    pd.DataFrame.__setitem__ = _wrap_setitem(pd.DataFrame.__setitem__)

    # === Readers (auto-registration) ===
    readers = [
        "read_csv",
        "read_excel",
        "read_parquet",
        "read_json",
        "read_sql",
        "read_feather",
        "read_pickle",
    ]
    for reader_name in readers:
        if hasattr(pd, reader_name):
            original = getattr(pd, reader_name)
            _originals[f"pd.{reader_name}"] = original
            setattr(pd, reader_name, _wrap_dataframe_reader(original, reader_name))

    # === DataFrame.__init__ ===
    _originals["DataFrame.__init__"] = pd.DataFrame.__init__
    pd.DataFrame.__init__ = _wrap_dataframe_init(pd.DataFrame.__init__)

    # === Export methods (auto-strip hidden column) ===
    _originals["DataFrame.to_csv"] = pd.DataFrame.to_csv
    pd.DataFrame.to_csv = _make_export_wrapper(pd.DataFrame.to_csv)

    _originals["DataFrame.to_parquet"] = pd.DataFrame.to_parquet
    pd.DataFrame.to_parquet = _make_export_wrapper(pd.DataFrame.to_parquet)

    # === pd.concat ===
    _originals["pd.concat"] = pd.concat

    def wrapped_concat(*args, **kwargs):
        result = _originals["pd.concat"](*args, **kwargs)
        ctx = get_context()
        if ctx.enabled:
            _capture_concat(args, kwargs, result, ctx)
        return result

    pd.concat = wrapped_concat


def uninstrument_pandas():
    """Restore original pandas methods."""
    global _originals

    for key, original in _originals.items():
        parts = key.split(".")
        if parts[0] == "pd":
            setattr(pd, parts[1], original)
        elif parts[0] == "DataFrame":
            setattr(pd.DataFrame, parts[1], original)
        elif parts[0] == "DataFrameGroupBy":
            from pandas.core.groupby import DataFrameGroupBy

            setattr(DataFrameGroupBy, parts[1], original)

    _originals.clear()

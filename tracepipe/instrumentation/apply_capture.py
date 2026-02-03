# tracepipe/instrumentation/apply_capture.py
"""
apply() and pipe() instrumentation for TracePipe.

Challenge: User functions are opaque - we can't see what they do internally.
Solution: Capture before/after state and mark as PARTIAL completeness.

Operations tracked:
| Pattern                      | Tracking              | Completeness |
|------------------------------|-----------------------|--------------|
| df.apply(func, axis=0)       | Column-wise diffs     | PARTIAL      |
| df.apply(func, axis=1)       | Row-wise diffs        | PARTIAL      |
| df.pipe(func)                | Before/after snapshot | PARTIAL      |
| df.transform(func)           | Before/after diffs    | PARTIAL      |
| df.assign(**kwargs)          | Per-column diffs      | FULL         |

Key insight: We don't try to understand the function - we just diff the result.
"""

import warnings
from functools import wraps
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..context import get_context
from ..core import ChangeType, CompletenessLevel
from ..safety import TracePipeWarning, get_caller_info


def wrap_apply():
    """
    Wrap DataFrame.apply to capture before/after diffs.

    apply() is inherently PARTIAL because we can't see inside the function.
    However, we can still track:
    - Which rows/columns changed
    - What the before/after values were
    """
    original_apply = pd.DataFrame.apply

    @wraps(original_apply)
    def tracked_apply(self, func, axis=0, raw=False, result_type=None, args=(), **kwargs):
        ctx = get_context()

        if not ctx.enabled:
            return original_apply(self, func, axis, raw, result_type, args, **kwargs)

        # Capture before state for watched columns
        before_values = _capture_watched_state(self, ctx)
        source_rids = ctx.row_manager.get_ids_array(self)
        if source_rids is None:
            ctx.row_manager.register(self)
            source_rids = ctx.row_manager.get_ids_array(self)

        # Run apply
        result = original_apply(self, func, axis, raw, result_type, args, **kwargs)

        try:
            _capture_apply_result(
                source_df=self,
                result=result,
                before_values=before_values,
                source_rids=source_rids,
                axis=axis,
                func=func,
                ctx=ctx,
            )
        except Exception as e:
            if ctx.config.strict_mode:
                raise
            warnings.warn(f"TracePipe: apply() capture failed: {e}", TracePipeWarning)

        return result

    pd.DataFrame.apply = tracked_apply
    pd.DataFrame._tp_original_apply = original_apply


def wrap_pipe():
    """
    Wrap DataFrame.pipe to capture before/after state.

    pipe() passes the entire DataFrame to a function - we track the transformation.
    """
    original_pipe = pd.DataFrame.pipe

    @wraps(original_pipe)
    def tracked_pipe(self, func, *args, **kwargs):
        ctx = get_context()

        if not ctx.enabled:
            return original_pipe(self, func, *args, **kwargs)

        # Capture before state
        before_values = _capture_watched_state(self, ctx)
        source_rids = ctx.row_manager.get_ids_array(self)
        if source_rids is None:
            ctx.row_manager.register(self)
            source_rids = ctx.row_manager.get_ids_array(self)

        # Run pipe
        result = original_pipe(self, func, *args, **kwargs)

        try:
            _capture_pipe_result(
                source_df=self,
                result=result,
                before_values=before_values,
                source_rids=source_rids,
                func=func,
                ctx=ctx,
            )
        except Exception as e:
            if ctx.config.strict_mode:
                raise
            warnings.warn(f"TracePipe: pipe() capture failed: {e}", TracePipeWarning)

        return result

    pd.DataFrame.pipe = tracked_pipe
    pd.DataFrame._tp_original_pipe = original_pipe


def wrap_transform():
    """Wrap DataFrame.transform for element-wise transformations."""
    original_transform = pd.DataFrame.transform

    @wraps(original_transform)
    def tracked_transform(self, func, axis=0, *args, **kwargs):
        ctx = get_context()

        if not ctx.enabled:
            return original_transform(self, func, axis, *args, **kwargs)

        before_values = _capture_watched_state(self, ctx)
        source_rids = ctx.row_manager.get_ids_array(self)
        if source_rids is None:
            ctx.row_manager.register(self)
            source_rids = ctx.row_manager.get_ids_array(self)

        result = original_transform(self, func, axis, *args, **kwargs)

        try:
            _capture_transform_result(
                source_df=self,
                result=result,
                before_values=before_values,
                source_rids=source_rids,
                func=func,
                ctx=ctx,
            )
        except Exception as e:
            if ctx.config.strict_mode:
                raise
            warnings.warn(f"TracePipe: transform() capture failed: {e}", TracePipeWarning)

        return result

    pd.DataFrame.transform = tracked_transform
    pd.DataFrame._tp_original_transform = original_transform


def wrap_assign():
    """
    Wrap DataFrame.assign for FULL completeness tracking.

    assign() is explicit about what columns are being created/modified,
    so we can track with FULL completeness.
    """
    original_assign = pd.DataFrame.assign

    @wraps(original_assign)
    def tracked_assign(self, **kwargs):
        ctx = get_context()

        if not ctx.enabled:
            return original_assign(self, **kwargs)

        # Capture before state for columns being modified
        before_values = {}
        source_rids = ctx.row_manager.get_ids_array(self)
        if source_rids is None:
            ctx.row_manager.register(self)
            source_rids = ctx.row_manager.get_ids_array(self)

        for col in kwargs.keys():
            if col in ctx.watched_columns and col in self.columns:
                before_values[col] = self[col].values.copy()

        # Run assign
        result = original_assign(self, **kwargs)

        try:
            _capture_assign_result(
                source_df=self,
                result=result,
                before_values=before_values,
                source_rids=source_rids,
                assigned_cols=list(kwargs.keys()),
                ctx=ctx,
            )
        except Exception as e:
            if ctx.config.strict_mode:
                raise
            warnings.warn(f"TracePipe: assign() capture failed: {e}", TracePipeWarning)

        return result

    pd.DataFrame.assign = tracked_assign
    pd.DataFrame._tp_original_assign = original_assign


# ============ CAPTURE HELPERS ============


def _capture_watched_state(df: pd.DataFrame, ctx) -> dict:
    """Capture current values of watched columns."""
    state = {}
    for col in ctx.watched_columns:
        if col in df.columns:
            state[col] = df[col].values.copy()
    return state


def _capture_apply_result(
    source_df: pd.DataFrame,
    result: Any,
    before_values: dict,
    source_rids: np.ndarray,
    axis: int,
    func: Callable,
    ctx,
) -> None:
    """Capture diffs from apply() result."""
    store = ctx.store
    row_mgr = ctx.row_manager

    code_file, code_line = get_caller_info(skip_frames=4)

    func_name = getattr(func, "__name__", "anonymous")

    if isinstance(result, pd.DataFrame):
        # Result is DataFrame - may have same or different shape
        step_id = store.append_step(
            operation=f"DataFrame.apply({func_name})",
            stage=ctx.current_stage,
            code_file=code_file,
            code_line=code_line,
            params={"axis": axis, "func": func_name},
            input_shape=source_df.shape,
            output_shape=result.shape,
            completeness=CompletenessLevel.PARTIAL,
        )

        # Propagate RIDs if same length
        if len(result) == len(source_df):
            row_mgr.set_result_rids(result, source_rids)

            # Capture diffs for watched columns
            _capture_column_diffs(result, before_values, source_rids, step_id, store)
        else:
            # Different length - register new IDs, track as filter
            new_rids = row_mgr.register(result)
            if new_rids is not None:
                dropped = row_mgr.compute_dropped_ids(source_rids, new_rids)
                if len(dropped) > 0:
                    store.append_bulk_drops(step_id, dropped)

    elif isinstance(result, pd.Series):
        # Result is Series - aggregation or single column
        store.append_step(
            operation=f"DataFrame.apply({func_name})",
            stage=ctx.current_stage,
            code_file=code_file,
            code_line=code_line,
            params={"axis": axis, "func": func_name, "result_type": "Series"},
            input_shape=source_df.shape,
            output_shape=result.shape,
            completeness=CompletenessLevel.PARTIAL,
        )


def _capture_pipe_result(
    source_df: pd.DataFrame,
    result: Any,
    before_values: dict,
    source_rids: np.ndarray,
    func: Callable,
    ctx,
) -> None:
    """Capture diffs from pipe() result."""
    store = ctx.store
    row_mgr = ctx.row_manager

    code_file, code_line = get_caller_info(skip_frames=4)
    func_name = getattr(func, "__name__", "anonymous")

    if isinstance(result, pd.DataFrame):
        step_id = store.append_step(
            operation=f"DataFrame.pipe({func_name})",
            stage=ctx.current_stage,
            code_file=code_file,
            code_line=code_line,
            params={"func": func_name},
            input_shape=source_df.shape,
            output_shape=result.shape,
            completeness=CompletenessLevel.PARTIAL,
        )

        if len(result) == len(source_df):
            # Same length - preserve RIDs and track diffs
            row_mgr.set_result_rids(result, source_rids)
            _capture_column_diffs(result, before_values, source_rids, step_id, store)
        else:
            # Different length - treat as filter
            new_rids = row_mgr.register(result)
            if new_rids is not None:
                dropped = row_mgr.compute_dropped_ids(source_rids, new_rids)
                if len(dropped) > 0:
                    store.append_bulk_drops(step_id, dropped)
    else:
        # Non-DataFrame result (e.g., scalar, dict)
        store.append_step(
            operation=f"DataFrame.pipe({func_name})",
            stage=ctx.current_stage,
            code_file=code_file,
            code_line=code_line,
            params={"func": func_name, "result_type": type(result).__name__},
            input_shape=source_df.shape,
            output_shape=None,
            completeness=CompletenessLevel.PARTIAL,
        )


def _capture_transform_result(
    source_df: pd.DataFrame,
    result: pd.DataFrame,
    before_values: dict,
    source_rids: np.ndarray,
    func: Callable,
    ctx,
) -> None:
    """Capture diffs from transform() result."""
    store = ctx.store
    row_mgr = ctx.row_manager

    code_file, code_line = get_caller_info(skip_frames=4)
    func_name = getattr(func, "__name__", "anonymous")

    step_id = store.append_step(
        operation=f"DataFrame.transform({func_name})",
        stage=ctx.current_stage,
        code_file=code_file,
        code_line=code_line,
        params={"func": func_name},
        input_shape=source_df.shape,
        output_shape=result.shape,
        completeness=CompletenessLevel.PARTIAL,
    )

    # transform() preserves shape
    row_mgr.set_result_rids(result, source_rids)
    _capture_column_diffs(result, before_values, source_rids, step_id, store)


def _capture_assign_result(
    source_df: pd.DataFrame,
    result: pd.DataFrame,
    before_values: dict,
    source_rids: np.ndarray,
    assigned_cols: list,
    ctx,
) -> None:
    """Capture diffs from assign() result."""
    store = ctx.store
    row_mgr = ctx.row_manager

    code_file, code_line = get_caller_info(skip_frames=4)

    step_id = store.append_step(
        operation="DataFrame.assign",
        stage=ctx.current_stage,
        code_file=code_file,
        code_line=code_line,
        params={"columns": assigned_cols[:5]},
        input_shape=source_df.shape,
        output_shape=result.shape,
        completeness=CompletenessLevel.FULL,  # assign() is explicit
    )

    # assign() returns new DataFrame with same rows
    row_mgr.set_result_rids(result, source_rids)
    _capture_column_diffs(result, before_values, source_rids, step_id, store)


def _capture_column_diffs(
    result_df: pd.DataFrame,
    before_values: dict,
    rids: np.ndarray,
    step_id: int,
    store,
) -> None:
    """
    Capture diffs for all watched columns using vectorized comparison.

    Uses find_changed_indices_vectorized for 50-100x speedup over row-by-row.
    """
    from ..utils.value_capture import find_changed_indices_vectorized

    for col, old_vals in before_values.items():
        if col not in result_df.columns:
            continue

        new_vals = result_df[col].values

        # Vectorized change detection
        old_series = pd.Series(old_vals)
        new_series = pd.Series(new_vals)
        changed_mask = find_changed_indices_vectorized(old_series, new_series)

        if not changed_mask.any():
            continue

        # Only loop over changed indices (typically small fraction)
        changed_indices = np.where(changed_mask)[0]
        for i in changed_indices:
            store.append_diff(
                step_id=step_id,
                row_id=int(rids[i]),
                col=col,
                old_val=old_vals[i],
                new_val=new_vals[i],
                change_type=ChangeType.MODIFIED,
            )


def instrument_apply_pipe():
    """Install all apply/pipe instrumentation."""
    wrap_apply()
    wrap_pipe()
    wrap_transform()
    wrap_assign()


def uninstrument_apply_pipe():
    """Restore original apply/pipe behavior."""
    for method in ["apply", "pipe", "transform", "assign"]:
        orig_attr = f"_tp_original_{method}"
        if hasattr(pd.DataFrame, orig_attr):
            setattr(pd.DataFrame, method, getattr(pd.DataFrame, orig_attr))
            delattr(pd.DataFrame, orig_attr)

# tracepipe/instrumentation/merge_capture.py
"""
Merge provenance using position column injection.

CI Mode: Stats only (fast)
DEBUG Mode: Full parent RID mapping (position column injection)
"""

import uuid
import warnings
from functools import wraps

import numpy as np
import pandas as pd

from ..context import get_context
from ..core import CompletenessLevel, ConcatMapping, MergeMapping, MergeStats
from ..safety import TracePipeWarning, get_caller_info


def wrap_merge_with_lineage(original_merge):
    """
    Wrap DataFrame.merge with lineage capture.
    """

    @wraps(original_merge)
    def wrapper(self, right, *args, **kwargs):
        ctx = get_context()

        if not ctx.enabled:
            return original_merge(self, right, *args, **kwargs)

        if ctx.config.should_capture_merge_provenance:
            return _merge_with_provenance(original_merge, self, right, args, kwargs, ctx)
        else:
            return _merge_with_stats_only(original_merge, self, right, args, kwargs, ctx)

    return wrapper


def _merge_with_provenance(original_merge, left, right, args, kwargs, ctx):
    """
    Merge with full provenance (debug mode).
    Uses position column injection.

    Guards against validate= and other merge errors.
    """
    row_mgr = ctx.row_manager
    store = ctx.store

    # Get/register source RIDs
    left_rids = row_mgr.get_ids_array(left)
    right_rids = row_mgr.get_ids_array(right)

    if left_rids is None:
        left_rids = row_mgr.register(left)
    if right_rids is None:
        right_rids = row_mgr.register(right)

    # Generate unique position column names
    token = uuid.uuid4().hex[:12]
    left_pos_col = f"__tp_lp_{token}__"
    right_pos_col = f"__tp_rp_{token}__"

    # Inject position columns (int32 for memory efficiency)
    left_tracked = left.assign(**{left_pos_col: np.arange(len(left), dtype=np.int32)})
    right_tracked = right.assign(**{right_pos_col: np.arange(len(right), dtype=np.int32)})

    # Run merge in try/except to handle validate= errors
    try:
        result_tracked = original_merge(left_tracked, right_tracked, *args, **kwargs)
    except Exception as e:
        # Merge failed (e.g., validate="1:1" violation)
        # Record error step for debuggability, then re-raise
        return _record_merge_error_and_reraise(e, left, right, kwargs, ctx)

    # Check for collision, don't rerun merge
    if left_pos_col not in result_tracked.columns or right_pos_col not in result_tracked.columns:
        warnings.warn(
            "TracePipe: Position column collision in merge. Provenance marked PARTIAL.",
            UserWarning,
        )
        # Don't rerun - just drop tracking columns and continue
        result = result_tracked.drop(
            columns=[c for c in [left_pos_col, right_pos_col] if c in result_tracked.columns],
            errors="ignore",
        )
        return _finalize_merge_partial(result, left, right, kwargs, ctx)

    # Extract indexers
    left_indexer = result_tracked[left_pos_col].values
    right_indexer = result_tracked[right_pos_col].values

    # Drop tracking columns
    result = result_tracked.drop(columns=[left_pos_col, right_pos_col])

    # Build mapping with vectorized parent lookup
    # Handle NaN for outer joins
    left_valid = ~pd.isna(left_indexer)
    right_valid = ~pd.isna(right_indexer)

    left_parent_rids = np.full(len(result), -1, dtype=np.int64)
    right_parent_rids = np.full(len(result), -1, dtype=np.int64)

    left_parent_rids[left_valid] = left_rids[left_indexer[left_valid].astype(np.int64)]
    right_parent_rids[right_valid] = right_rids[right_indexer[right_valid].astype(np.int64)]

    # Assign new RIDs to result
    result_rids = row_mgr.register(result)

    # Record step
    code_file, code_line = get_caller_info(skip_frames=3)

    step_id = store.append_step(
        operation="DataFrame.merge",
        stage=ctx.current_stage,
        code_file=code_file,
        code_line=code_line,
        params=_merge_params(kwargs),
        input_shape=(left.shape, right.shape),
        output_shape=result.shape,
        completeness=CompletenessLevel.FULL,
    )

    # Sort mapping arrays by out_rids for O(log n) lookup
    sort_idx = np.argsort(result_rids)
    sorted_out_rids = result_rids[sort_idx]
    sorted_left_parents = left_parent_rids[sort_idx]
    sorted_right_parents = right_parent_rids[sort_idx]

    mapping = MergeMapping(
        step_id=step_id,
        out_rids=sorted_out_rids,
        left_parent_rids=sorted_left_parents,
        right_parent_rids=sorted_right_parents,
    )
    store.merge_mappings.append(mapping)

    # Compute stats from indexers
    stats = _compute_stats_from_indexers(
        left_indexer[left_valid].astype(np.int64),
        right_indexer[right_valid].astype(np.int64),
        len(left),
        len(right),
        len(result),
        kwargs,
    )
    store.merge_stats.append((step_id, stats))

    return result


def _finalize_merge_partial(result, left, right, kwargs, ctx):
    """Finalize merge when provenance capture failed."""
    row_mgr = ctx.row_manager
    store = ctx.store

    row_mgr.register(result)

    code_file, code_line = get_caller_info(skip_frames=4)

    step_id = store.append_step(
        operation="DataFrame.merge",
        stage=ctx.current_stage,
        code_file=code_file,
        code_line=code_line,
        params=_merge_params(kwargs),
        input_shape=(left.shape, right.shape),
        output_shape=result.shape,
        completeness=CompletenessLevel.PARTIAL,
    )

    stats = _compute_stats_approximate(len(left), len(right), len(result), kwargs)
    store.merge_stats.append((step_id, stats))

    return result


def _record_merge_error_and_reraise(error: Exception, left, right, kwargs, ctx):
    """
    Record merge error step for debuggability, then re-raise.

    This handles cases like validate="1:1" violations where pandas raises
    but we still want to record what was attempted.
    """
    store = ctx.store

    code_file, code_line = get_caller_info(skip_frames=4)

    # Record error step with exception info
    params = _merge_params(kwargs)
    params["_error"] = str(error)[:200]  # Truncate long error messages
    params["_error_type"] = type(error).__name__

    store.append_step(
        operation="DataFrame.merge (error)",
        stage=ctx.current_stage,
        code_file=code_file,
        code_line=code_line,
        params=params,
        input_shape=(left.shape, right.shape),
        output_shape=None,  # No result
        completeness=CompletenessLevel.PARTIAL,
    )

    # Re-raise the original exception so user code behaves normally
    raise error


def _merge_with_stats_only(original_merge, left, right, args, kwargs, ctx):
    """
    Merge with stats only (CI mode).
    Fast: no position injection.

    Also handles merge errors for debuggability.
    """
    try:
        result = original_merge(left, right, *args, **kwargs)
    except Exception as e:
        return _record_merge_error_and_reraise(e, left, right, kwargs, ctx)

    row_mgr = ctx.row_manager
    store = ctx.store

    row_mgr.register(result)

    code_file, code_line = get_caller_info(skip_frames=3)

    # CI mode merge = PARTIAL (we know it's a merge, but no parent mapping)
    step_id = store.append_step(
        operation="DataFrame.merge",
        stage=ctx.current_stage,
        code_file=code_file,
        code_line=code_line,
        params=_merge_params(kwargs),
        input_shape=(left.shape, right.shape),
        output_shape=result.shape,
        completeness=CompletenessLevel.PARTIAL,
    )

    stats = _compute_stats_approximate(len(left), len(right), len(result), kwargs)
    store.merge_stats.append((step_id, stats))

    return result


def _compute_stats_from_indexers(
    left_indexer: np.ndarray,
    right_indexer: np.ndarray,
    n_left: int,
    n_right: int,
    n_result: int,
    kwargs: dict,
) -> MergeStats:
    """Compute accurate merge stats from indexers."""

    # Match rates
    left_match_rate = len(np.unique(left_indexer)) / n_left if n_left > 0 else 0
    right_match_rate = len(np.unique(right_indexer)) / n_right if n_right > 0 else 0

    # Dup rates (rows appearing more than once)
    if len(left_indexer) > 0:
        left_counts = np.bincount(left_indexer, minlength=n_left)
        left_dup_rate = (left_counts > 1).sum() / n_left if n_left > 0 else 0
    else:
        left_dup_rate = 0

    if len(right_indexer) > 0:
        right_counts = np.bincount(right_indexer, minlength=n_right)
        right_dup_rate = (right_counts > 1).sum() / n_right if n_right > 0 else 0
    else:
        right_dup_rate = 0

    return MergeStats(
        left_rows=n_left,
        right_rows=n_right,
        result_rows=n_result,
        expansion_ratio=n_result / max(n_left, n_right, 1),
        left_match_rate=left_match_rate,
        right_match_rate=right_match_rate,
        left_dup_rate=left_dup_rate,
        right_dup_rate=right_dup_rate,
        how=kwargs.get("how", "inner"),
    )


def _compute_stats_approximate(
    n_left: int, n_right: int, n_result: int, kwargs: dict
) -> MergeStats:
    """Approximate stats (CI mode - fast, skip expensive computations)."""
    return MergeStats(
        left_rows=n_left,
        right_rows=n_right,
        result_rows=n_result,
        expansion_ratio=n_result / max(n_left, n_right, 1),
        left_match_rate=-1.0,  # Unknown in CI mode
        right_match_rate=-1.0,
        left_dup_rate=-1.0,
        right_dup_rate=-1.0,
        how=kwargs.get("how", "inner"),
    )


def _merge_params(kwargs: dict) -> dict:
    return {
        "how": kwargs.get("how", "inner"),
        "on": str(kwargs.get("on", kwargs.get("left_on", "")))[:50],
    }


# ============ JOIN WRAPPER ============


def wrap_join_with_lineage(original_join):
    """
    Wrap DataFrame.join with lineage capture.
    Similar to merge but uses index-based joining.
    """

    @wraps(original_join)
    def wrapper(self, other, *args, **kwargs):
        ctx = get_context()

        if not ctx.enabled:
            return original_join(self, other, *args, **kwargs)

        # Run join
        try:
            result = original_join(self, other, *args, **kwargs)
        except Exception as e:
            if ctx.config.strict_mode:
                raise
            warnings.warn(f"TracePipe: Join failed: {e}", TracePipeWarning)
            raise

        row_mgr = ctx.row_manager
        store = ctx.store

        # Register result
        row_mgr.register(result)

        code_file, code_line = get_caller_info(skip_frames=2)

        # Record step
        step_id = store.append_step(
            operation="DataFrame.join",
            stage=ctx.current_stage,
            code_file=code_file,
            code_line=code_line,
            params={"how": kwargs.get("how", "left")},
            input_shape=(self.shape, other.shape if hasattr(other, "shape") else None),
            output_shape=result.shape,
            completeness=CompletenessLevel.PARTIAL,  # Join is complex, mark PARTIAL
        )

        # Compute basic stats
        n_left = len(self)
        n_right = len(other) if hasattr(other, "__len__") else 0
        n_result = len(result)

        stats = MergeStats(
            left_rows=n_left,
            right_rows=n_right,
            result_rows=n_result,
            expansion_ratio=n_result / max(n_left, 1),
            left_match_rate=-1.0,
            right_match_rate=-1.0,
            left_dup_rate=-1.0,
            right_dup_rate=-1.0,
            how=kwargs.get("how", "left"),
        )
        store.merge_stats.append((step_id, stats))

        return result

    return wrapper


# ============ CONCAT WRAPPER ============


def wrap_concat_with_lineage(original_concat):
    """
    Wrap pd.concat with lineage capture.

    For axis=0 (vertical concat):
    - Preserves row IDs from source DataFrames (FULL provenance)
    - Tracks which source DataFrame each row came from

    For axis=1 (horizontal concat):
    - Propagates RIDs if all inputs have identical RID arrays
    - Otherwise marks as PARTIAL
    """

    @wraps(original_concat)
    def wrapper(objs, *args, **kwargs):
        ctx = get_context()

        if not ctx.enabled:
            return original_concat(objs, *args, **kwargs)

        axis = kwargs.get("axis", 0)

        # === BEFORE: Capture source RIDs from all tracked DataFrames ===
        source_data = []  # [(rids_copy, shape, original_index), ...]
        try:
            objs_list = list(objs) if hasattr(objs, "__iter__") else [objs]
        except TypeError:
            objs_list = [objs]

        for i, obj in enumerate(objs_list):
            if isinstance(obj, pd.DataFrame) and len(obj) > 0:
                rids = ctx.row_manager.get_ids_array(obj)
                if rids is None:
                    rids = ctx.row_manager.register(obj)
                # IMPORTANT: Make a copy to avoid mutation issues
                source_data.append((rids.copy(), obj.shape, i))

        # === RUN ORIGINAL ===
        try:
            result = original_concat(objs_list, *args, **kwargs)
        except Exception:
            raise  # Don't store mapping on failure

        if not isinstance(result, pd.DataFrame):
            return result

        row_mgr = ctx.row_manager
        store = ctx.store
        code_file, code_line = get_caller_info(skip_frames=2)

        # Compute input shapes for step metadata
        input_shapes = [sd[1] for sd in source_data]

        # === AXIS=0: Vertical concat with FULL provenance ===
        if axis == 0 and source_data:
            return _concat_axis0_with_provenance(
                result, source_data, input_shapes, code_file, code_line, ctx
            )

        # === AXIS=1: Horizontal concat ===
        elif axis == 1 and source_data:
            return _concat_axis1_with_provenance(
                result, source_data, input_shapes, code_file, code_line, ctx
            )

        # === FALLBACK: Unknown axis or no source data ===
        else:
            row_mgr.register(result)
            store.append_step(
                operation="pd.concat",
                stage=ctx.current_stage,
                code_file=code_file,
                code_line=code_line,
                params={
                    "axis": axis,
                    "n_inputs": len(source_data),
                },
                input_shape=tuple(input_shapes) if input_shapes else None,
                output_shape=result.shape,
                completeness=CompletenessLevel.PARTIAL,
            )
            return result

    return wrapper


def _concat_axis0_with_provenance(result, source_data, input_shapes, code_file, code_line, ctx):
    """
    Handle axis=0 concat with FULL row provenance.

    Preserves source RIDs and tracks which source DF each row came from.
    """
    row_mgr = ctx.row_manager
    store = ctx.store

    # Build concatenated RID array and source index array
    all_rids = np.concatenate([sd[0] for sd in source_data])
    all_source_idx = np.concatenate(
        [np.full(len(sd[0]), sd[2], dtype=np.int32) for sd in source_data]
    )

    # Validate: length must match result
    if len(all_rids) != len(result):
        # Mismatch - some objects contributed differently (empty DFs, Series, etc.)
        # Degrade gracefully to PARTIAL
        row_mgr.register(result)
        store.append_step(
            operation="pd.concat",
            stage=ctx.current_stage,
            code_file=code_file,
            code_line=code_line,
            params={
                "axis": 0,
                "n_inputs": len(source_data),
                "_length_mismatch": True,
            },
            input_shape=tuple(input_shapes) if input_shapes else None,
            output_shape=result.shape,
            completeness=CompletenessLevel.PARTIAL,
        )
        return result

    # Propagate RIDs to result (preserving lineage!)
    row_mgr.set_result_rids(result, all_rids.copy())

    # Build sorted arrays for O(log n) lookup
    sort_order = np.argsort(all_rids)
    out_rids_sorted = all_rids[sort_order].copy()
    out_pos_sorted = sort_order.copy()

    # Record step with FULL completeness
    step_id = store.append_step(
        operation="pd.concat",
        stage=ctx.current_stage,
        code_file=code_file,
        code_line=code_line,
        params={
            "axis": 0,
            "n_inputs": len(source_data),
        },
        input_shape=tuple(input_shapes) if input_shapes else None,
        output_shape=result.shape,
        completeness=CompletenessLevel.FULL,
    )

    # Store mapping
    mapping = ConcatMapping(
        step_id=step_id,
        out_rids=all_rids.copy(),
        source_indices=all_source_idx.copy(),
        out_rids_sorted=out_rids_sorted,
        out_pos_sorted=out_pos_sorted,
        source_shapes=list(input_shapes),
    )
    store.concat_mappings.append(mapping)

    return result


def _concat_axis1_with_provenance(result, source_data, input_shapes, code_file, code_line, ctx):
    """
    Handle axis=1 concat with best-effort provenance.

    If all inputs have identical RID arrays, propagate them (FULL).
    Otherwise, mark as PARTIAL and register new RIDs.
    """
    row_mgr = ctx.row_manager
    store = ctx.store

    # Check if all inputs have the same RIDs in same order
    first_rids = source_data[0][0]
    all_same = all(
        len(sd[0]) == len(first_rids) and np.array_equal(sd[0], first_rids) for sd in source_data
    )

    if all_same and len(first_rids) == len(result):
        # All inputs have identical RIDs - propagate them
        row_mgr.set_result_rids(result, first_rids.copy())
        completeness = CompletenessLevel.FULL
    else:
        # Misaligned or different RIDs - register new RIDs
        row_mgr.register(result)
        completeness = CompletenessLevel.PARTIAL

    store.append_step(
        operation="pd.concat",
        stage=ctx.current_stage,
        code_file=code_file,
        code_line=code_line,
        params={
            "axis": 1,
            "n_inputs": len(source_data),
        },
        input_shape=tuple(input_shapes) if input_shapes else None,
        output_shape=result.shape,
        completeness=completeness,
    )

    return result

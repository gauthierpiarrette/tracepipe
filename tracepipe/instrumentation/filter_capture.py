# tracepipe/instrumentation/filter_capture.py
"""
Mask-first filter capture with PARTIAL fallback.

Operation Matrix:
| Operation        | Mask Derivation                          | Completeness |
|------------------|------------------------------------------|--------------|
| dropna           | ~df[subset].isna().any(axis=1)          | FULL         |
| drop_duplicates  | ~df.duplicated(subset, keep)            | FULL         |
| query (simple)   | df.eval(expr)                           | FULL         |
| query (complex)  | fallback                                | PARTIAL      |
| head(n)          | positions [0:n]                         | FULL         |
| tail(n)          | positions [-n:]                         | FULL         |
| sample           | result.index positions                  | FULL         |
| __getitem__[mask]| mask directly available                 | FULL         |
| other            | index-label fallback                    | PARTIAL      |
"""

import warnings
from functools import wraps
from typing import Optional

import numpy as np
import pandas as pd

from ..context import TracePipeContext, get_context
from ..core import CompletenessLevel
from ..safety import TracePipeWarning, get_caller_info

# ============ MASK DERIVATION FUNCTIONS ============


def derive_dropna_mask(
    df: pd.DataFrame, args: tuple, kwargs: dict
) -> tuple[np.ndarray, CompletenessLevel]:
    """
    Derive kept mask for dropna.

    Guards against missing columns in subset.
    """
    axis = kwargs.get("axis", 0)
    if axis != 0 and axis != "index":
        return np.ones(len(df), dtype=bool), CompletenessLevel.FULL

    how = kwargs.get("how", "any")
    subset = kwargs.get("subset", None)
    thresh = kwargs.get("thresh", None)

    # Guard missing columns
    if subset is not None:
        valid_cols = [c for c in subset if c in df.columns]
        if not valid_cols:
            # No valid columns to check - keep all rows
            return np.ones(len(df), dtype=bool), CompletenessLevel.PARTIAL
        if len(valid_cols) != len(subset):
            # Some columns missing - still compute but mark partial
            check_df = df[valid_cols]
            completeness = CompletenessLevel.PARTIAL
        else:
            check_df = df[valid_cols]
            completeness = CompletenessLevel.FULL
    else:
        check_df = df
        completeness = CompletenessLevel.FULL

    if thresh is not None:
        kept_mask = check_df.notna().sum(axis=1) >= thresh
    elif how == "any":
        kept_mask = ~check_df.isna().any(axis=1)
    else:
        kept_mask = ~check_df.isna().all(axis=1)

    return kept_mask.values, completeness


def derive_drop_duplicates_mask(
    df: pd.DataFrame, args: tuple, kwargs: dict
) -> tuple[np.ndarray, CompletenessLevel]:
    """Derive kept mask for drop_duplicates."""
    subset = kwargs.get("subset", None)
    keep = kwargs.get("keep", "first")

    # Guard missing columns
    if subset is not None:
        valid_cols = [c for c in subset if c in df.columns]
        if not valid_cols:
            return np.ones(len(df), dtype=bool), CompletenessLevel.PARTIAL
        if len(valid_cols) != len(subset):
            subset = valid_cols
            completeness = CompletenessLevel.PARTIAL
        else:
            completeness = CompletenessLevel.FULL
    else:
        completeness = CompletenessLevel.FULL

    kept_mask = ~df.duplicated(subset=subset, keep=keep)
    return kept_mask.values, completeness


def derive_query_mask(
    df: pd.DataFrame, args: tuple, kwargs: dict
) -> tuple[Optional[np.ndarray], CompletenessLevel]:
    """
    Derive kept mask for query.

    More complete unsafe pattern checking.
    """
    expr = args[0] if args else kwargs.get("expr", "")

    # Check for engine/parser that make eval unreliable
    engine = kwargs.get("engine", None)
    parser = kwargs.get("parser", None)
    local_dict = kwargs.get("local_dict", None)
    global_dict = kwargs.get("global_dict", None)

    # Mark PARTIAL if non-default engine/parser
    if engine == "python" or parser == "python":
        return None, CompletenessLevel.PARTIAL

    if local_dict is not None or global_dict is not None:
        return None, CompletenessLevel.PARTIAL

    # Check for unsafe patterns
    unsafe_patterns = [
        "@",  # Local variable reference
        "`",  # Backtick column names
        "index",  # Index reference (case variations)
        "Index",
        "level_",  # MultiIndex level
    ]

    if any(p in expr for p in unsafe_patterns):
        return None, CompletenessLevel.PARTIAL

    try:
        mask = df.eval(expr)
        if isinstance(mask, pd.Series) and mask.dtype == bool:
            return mask.values, CompletenessLevel.FULL
        else:
            return None, CompletenessLevel.PARTIAL
    except Exception:
        return None, CompletenessLevel.PARTIAL


def derive_head_positions(df: pd.DataFrame, args: tuple, kwargs: dict) -> np.ndarray:
    """Derive positions for head(n)."""
    n = args[0] if args else kwargs.get("n", 5)
    n = min(max(0, n), len(df))
    return np.arange(n, dtype=np.int64)


def derive_tail_positions(df: pd.DataFrame, args: tuple, kwargs: dict) -> np.ndarray:
    """Derive positions for tail(n)."""
    n = args[0] if args else kwargs.get("n", 5)
    n = min(max(0, n), len(df))
    return np.arange(len(df) - n, len(df), dtype=np.int64)


def derive_sample_positions(
    source_df: pd.DataFrame, result_df: pd.DataFrame
) -> tuple[Optional[np.ndarray], CompletenessLevel]:
    """
    Derive positions for sample by matching result index to source.

    If source index has duplicates, get_indexer returns first match
    and we may map wrong rows. Mark PARTIAL in this case.
    """
    # Duplicate index makes position mapping unreliable
    if source_df.index.has_duplicates:
        return None, CompletenessLevel.PARTIAL

    try:
        positions = source_df.index.get_indexer(result_df.index)
        if -1 in positions:
            return None, CompletenessLevel.PARTIAL
        return positions.astype(np.int64), CompletenessLevel.FULL
    except Exception:
        return None, CompletenessLevel.PARTIAL


# ============ UNIFIED FILTER WRAPPER ============


def wrap_filter_method(method_name: str, original_method):
    """
    Create a wrapper for filter methods with mask-first capture.
    """

    @wraps(original_method)
    def wrapper(self, *args, **kwargs):
        ctx = get_context()

        # Increment filter depth to prevent internal loc/iloc from creating steps
        if ctx.enabled:
            ctx._filter_op_depth += 1

        try:
            # === ALWAYS RUN ORIGINAL FIRST ===
            result = original_method(self, *args, **kwargs)
        finally:
            if ctx.enabled:
                ctx._filter_op_depth -= 1

        if not ctx.enabled:
            return result

        if not isinstance(result, pd.DataFrame):
            return result

        try:
            _capture_filter_with_mask(
                source_df=self,
                result_df=result,
                method_name=method_name,
                args=args,
                kwargs=kwargs,
                ctx=ctx,
            )
        except Exception as e:
            if ctx.config.strict_mode:
                raise
            warnings.warn(
                f"TracePipe: Filter capture failed for {method_name}: {e}",
                TracePipeWarning,
            )

        return result

    return wrapper


def _capture_filter_with_mask(
    source_df: pd.DataFrame,
    result_df: pd.DataFrame,
    method_name: str,
    args: tuple,
    kwargs: dict,
    ctx: TracePipeContext,
) -> None:
    """
    Core filter capture logic with mask-first design.
    """
    row_mgr = ctx.row_manager
    store = ctx.store

    # Ensure source is registered
    source_rids = row_mgr.get_ids_array(source_df)
    if source_rids is None:
        row_mgr.register(source_df)
        source_rids = row_mgr.get_ids_array(source_df)
        if source_rids is None:
            return

    n_before = len(source_df)

    # === DERIVE MASK/POSITIONS ===
    kept_mask: Optional[np.ndarray] = None
    positions: Optional[np.ndarray] = None
    completeness = CompletenessLevel.FULL

    if method_name == "dropna":
        kept_mask, completeness = derive_dropna_mask(source_df, args, kwargs)

    elif method_name == "drop_duplicates":
        kept_mask, completeness = derive_drop_duplicates_mask(source_df, args, kwargs)

    elif method_name == "query":
        kept_mask, completeness = derive_query_mask(source_df, args, kwargs)

    elif method_name == "head":
        positions = derive_head_positions(source_df, args, kwargs)

    elif method_name == "tail":
        positions = derive_tail_positions(source_df, args, kwargs)

    elif method_name == "sample":
        positions, completeness = derive_sample_positions(source_df, result_df)

    elif method_name == "__getitem__[mask]":
        if args and hasattr(args[0], "dtype") and args[0].dtype == bool:
            key = args[0]
            kept_mask = key.values if isinstance(key, pd.Series) else np.asarray(key)

    # === PROPAGATE RIDs ===
    # Maintain kept_mask explicitly in each branch to avoid confusion
    kept_mask_final: Optional[np.ndarray] = None
    result_rids: Optional[np.ndarray] = None

    if kept_mask is not None:
        # Mask-derived branch: kept_mask is directly available
        result_rids = row_mgr.propagate_by_mask(source_df, result_df, kept_mask)
        kept_mask_final = kept_mask

    elif positions is not None:
        # Position-derived branch: build kept_mask from positions
        result_rids = row_mgr.propagate_by_positions(source_df, result_df, positions)
        kept_mask_final = np.zeros(n_before, dtype=bool)
        kept_mask_final[positions] = True

    else:
        # FALLBACK: Index-label matching (mark as PARTIAL)
        # Don't use compute_dropped_with_positions here - positions aren't reliable
        completeness = CompletenessLevel.PARTIAL
        result_rids = _propagate_by_index_fallback(row_mgr, source_df, result_df)
        kept_mask_final = None  # Cannot reliably determine mask

    # === COMPUTE DROPPED ===
    # Use kept_mask_final consistently
    if kept_mask_final is not None:
        # We have a reliable mask - use it for accurate drop computation
        dropped_rids, dropped_positions = row_mgr.compute_dropped_with_positions(
            source_rids, kept_mask_final
        )
    elif result_rids is not None:
        # Fallback: use setdiff (no position info, but correct IDs)
        dropped_rids = row_mgr.compute_dropped_ids(source_rids, result_rids)
    else:
        dropped_rids = np.array([], dtype=np.int64)

    n_dropped = len(dropped_rids)

    # === RECORD STEP (ALWAYS - even if no rows dropped) ===
    code_file, code_line = get_caller_info(skip_frames=4)
    step_id = store.append_step(
        operation=f"DataFrame.{method_name}",
        stage=ctx.current_stage,
        code_file=code_file,
        code_line=code_line,
        params=_safe_filter_params(method_name, args, kwargs),
        input_shape=source_df.shape,
        output_shape=result_df.shape,
        completeness=completeness,
    )

    # === RECORD DROPS ===
    if n_dropped > 0:
        store.append_bulk_drops(step_id, dropped_rids)

        # Capture ghost values (debug mode)
        if kept_mask_final is not None:
            dropped_mask = ~kept_mask_final
            row_mgr.capture_ghost_values(
                source_df=source_df,
                dropped_mask=dropped_mask,
                dropped_by=f"DataFrame.{method_name}",
                step_id=step_id,
                watched_columns=ctx.watched_columns,
            )
        elif result_rids is not None and len(dropped_rids) > 0:
            # Fallback: derive dropped_mask from dropped_rids
            # Build mask by checking which source RIDs were dropped
            dropped_set = set(dropped_rids)
            dropped_mask = np.array([rid in dropped_set for rid in source_rids], dtype=bool)
            row_mgr.capture_ghost_values(
                source_df=source_df,
                dropped_mask=dropped_mask,
                dropped_by=f"DataFrame.{method_name}",
                step_id=step_id,
                watched_columns=ctx.watched_columns,
            )


def _propagate_by_index_fallback(
    row_mgr, source_df: pd.DataFrame, result_df: pd.DataFrame
) -> Optional[np.ndarray]:
    """
    Fallback propagation using index labels.
    Used when we can't derive mask/positions.
    This is BEST EFFORT and marked PARTIAL.
    """
    source_ids = row_mgr.get_ids(source_df)
    if source_ids is None:
        return row_mgr.register(result_df)

    try:
        result_ids = source_ids.reindex(result_df.index)

        new_mask = result_ids.isna()
        if new_mask.any():
            n_new = new_mask.sum()
            new_rids = np.arange(row_mgr._next_row_id, row_mgr._next_row_id + n_new, dtype=np.int64)
            row_mgr._next_row_id += n_new
            result_ids.loc[new_mask] = new_rids

        result_rids = result_ids.values.astype(np.int64)
        row_mgr.set_result_rids(result_df, result_rids)
        return result_rids
    except Exception:
        return row_mgr.register(result_df)


def _safe_filter_params(method_name: str, args: tuple, kwargs: dict) -> dict:
    """Extract safe params for step metadata."""
    params = {}

    if method_name == "dropna":
        params["how"] = kwargs.get("how", "any")
        params["subset"] = str(kwargs.get("subset", "all"))[:50]
        params["thresh"] = kwargs.get("thresh")
    elif method_name == "drop_duplicates":
        params["keep"] = kwargs.get("keep", "first")
        params["subset"] = str(kwargs.get("subset", "all"))[:50]
    elif method_name == "query":
        params["expr"] = str(args[0] if args else kwargs.get("expr", ""))[:100]
    elif method_name in ("head", "tail"):
        params["n"] = args[0] if args else kwargs.get("n", 5)
    elif method_name == "sample":
        params["n"] = kwargs.get("n")
        params["frac"] = kwargs.get("frac")
        params["random_state"] = kwargs.get("random_state")

    return params


# ============ BOOLEAN INDEXING WRAPPER ============


def wrap_getitem_filter(original_getitem):
    """
    Wrap DataFrame.__getitem__ to capture boolean indexing filters.
    """

    @wraps(original_getitem)
    def wrapper(self, key):
        ctx = get_context()

        result = original_getitem(self, key)

        if not ctx.enabled:
            return result

        # Skip if we're inside a filter operation (prevents double-counting)
        if ctx._filter_op_depth > 0:
            return result

        # Only capture boolean masks that produce DataFrames
        if not isinstance(result, pd.DataFrame):
            return result

        # Check if key is a boolean mask
        is_boolean_mask = False
        if isinstance(key, pd.Series) and key.dtype == bool:
            is_boolean_mask = True
        elif isinstance(key, np.ndarray) and key.dtype == bool:
            is_boolean_mask = True
        elif isinstance(key, list) and key and isinstance(key[0], bool):
            is_boolean_mask = True

        if not is_boolean_mask:
            return result

        try:
            _capture_filter_with_mask(
                source_df=self,
                result_df=result,
                method_name="__getitem__[mask]",
                args=(key,),
                kwargs={},
                ctx=ctx,
            )
        except Exception as e:
            if ctx.config.strict_mode:
                raise
            warnings.warn(f"TracePipe: Boolean indexing capture failed: {e}", TracePipeWarning)

        return result

    return wrapper

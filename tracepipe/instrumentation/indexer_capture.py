# tracepipe/instrumentation/indexer_capture.py
"""
loc/iloc instrumentation for TracePipe.

Operations tracked:
| Pattern                      | Type       | Completeness |
|------------------------------|------------|--------------|
| df.loc[mask]                 | Filter     | FULL         |
| df.loc[mask, 'col']          | Filter     | FULL         |
| df.iloc[0:5]                 | Filter     | FULL         |
| df.iloc[[1,3,5]]             | Filter     | FULL         |
| df.loc[mask, 'col'] = val    | Transform  | FULL         |
| df.iloc[0:5, 0] = val        | Transform  | FULL         |
| df.loc[mask] = other_df      | Transform  | PARTIAL      |

Key insight: We wrap the indexer's __getitem__ and __setitem__, not DataFrame.loc itself.
"""

import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from ..context import get_context
from ..core import ChangeType, CompletenessLevel
from ..safety import TracePipeWarning, get_caller_info


class _CallableLocIndexer:
    """
    Wrapper for the indexer returned by df.loc(axis=...).

    This is used internally by pandas methods like dropna.
    We skip tracking when _filter_op_depth > 0 to avoid double-counting
    since the outer operation (e.g., dropna) will track the drops.
    """

    def __init__(self, indexer, parent_df: pd.DataFrame):
        self._indexer = indexer
        self._parent_df = parent_df

    def __getitem__(self, key):
        """Pass through to underlying indexer and propagate RIDs."""
        ctx = get_context()
        result = self._indexer[key]

        if not ctx.enabled:
            return result

        # Skip step tracking if we're inside a filter operation (parent will track)
        # But still propagate RIDs for the result
        if isinstance(result, pd.DataFrame):
            try:
                row_mgr = ctx.row_manager
                source_df = self._parent_df

                source_rids = row_mgr.get_ids_array(source_df)
                if source_rids is None:
                    row_mgr.register(source_df)
                    source_rids = row_mgr.get_ids_array(source_df)

                # Propagate RIDs to result
                if hasattr(key, "dtype") and key.dtype == bool:
                    mask = key.values if isinstance(key, pd.Series) else np.asarray(key)
                    row_mgr.propagate_by_mask(source_df, result, mask)
                else:
                    row_mgr.register(result)

                # Only create step if NOT inside a filter operation
                if ctx._filter_op_depth == 0:
                    store = ctx.store
                    if hasattr(key, "dtype") and key.dtype == bool:
                        mask = key.values if isinstance(key, pd.Series) else np.asarray(key)
                        dropped_rids, _ = row_mgr.compute_dropped_with_positions(source_rids, mask)
                        completeness = CompletenessLevel.FULL
                    else:
                        result_rids = row_mgr.get_ids_array(result)
                        dropped_rids = row_mgr.compute_dropped_ids(source_rids, result_rids)
                        completeness = CompletenessLevel.PARTIAL

                    code_file, code_line = get_caller_info(skip_frames=4)
                    step_id = store.append_step(
                        operation="DataFrame.loc(axis)[]",
                        stage=ctx.current_stage,
                        code_file=code_file,
                        code_line=code_line,
                        params={},
                        input_shape=source_df.shape,
                        output_shape=result.shape,
                        completeness=completeness,
                    )

                    if len(dropped_rids) > 0:
                        store.append_bulk_drops(step_id, dropped_rids)
            except Exception as e:
                if ctx.config.strict_mode:
                    raise
                warnings.warn(f"TracePipe: loc(axis)[] capture failed: {e}", TracePipeWarning)

        return result

    def __setitem__(self, key, value):
        """Pass through setitem."""
        self._indexer[key] = value


class TrackedLocIndexer:
    """
    Wrapper around pandas _LocIndexer that captures lineage.

    Usage (internal - user never sees this):
        df.loc  # Returns TrackedLocIndexer wrapping the real _LocIndexer
    """

    def __init__(self, indexer, parent_df: pd.DataFrame):
        self._indexer = indexer
        self._parent_df = parent_df

    def __getattr__(self, name):
        """Proxy any other attribute access to the underlying indexer."""
        return getattr(self._indexer, name)

    def __call__(self, axis=None):
        """
        Support for df.loc(axis=...) callable form used internally by pandas.

        This is used by dropna and other methods that call self.loc(axis=axis)[mask].
        """
        # Return a callable-aware indexer that wraps the result of calling the original
        return _CallableLocIndexer(self._indexer(axis), self._parent_df)

    def __getitem__(self, key) -> Union[pd.DataFrame, pd.Series, Any]:
        """
        Capture filter operations via loc[].

        Handles:
        - df.loc[mask] -> DataFrame (filter)
        - df.loc[mask, 'col'] -> Series (filter + column select)
        - df.loc[mask, ['a', 'b']] -> DataFrame (filter + column select)
        - df.loc['label'] -> Row (single row access)
        """
        ctx = get_context()

        # Always run original first
        result = self._indexer[key]

        if not ctx.enabled:
            return result

        try:
            self._capture_loc_getitem(key, result, ctx)
        except Exception as e:
            if ctx.config.strict_mode:
                raise
            warnings.warn(f"TracePipe: loc[] capture failed: {e}", TracePipeWarning)

        return result

    def __setitem__(self, key, value) -> None:
        """
        Capture transform operations via loc[] = value.

        Handles:
        - df.loc[mask, 'col'] = scalar
        - df.loc[mask, 'col'] = array
        - df.loc[mask, ['a', 'b']] = values
        - df.loc[mask] = other_df (PARTIAL - complex assignment)
        """
        ctx = get_context()

        # Capture before state for watched columns
        before_values = None
        affected_cols = None
        if ctx.enabled and ctx.watched_columns:
            before_values, affected_cols = self._capture_before_state(key, ctx)

        # Always run original
        self._indexer[key] = value

        if not ctx.enabled:
            return

        try:
            self._capture_loc_setitem(key, value, before_values, affected_cols, ctx)
        except Exception as e:
            if ctx.config.strict_mode:
                raise
            warnings.warn(f"TracePipe: loc[]= capture failed: {e}", TracePipeWarning)

    def _capture_loc_getitem(self, key, result, ctx) -> None:
        """Capture filter via loc[]."""
        if not isinstance(result, pd.DataFrame):
            return  # Series or scalar - not a filter operation

        row_mgr = ctx.row_manager
        source_df = self._parent_df

        source_rids = row_mgr.get_ids_array(source_df)
        if source_rids is None:
            row_mgr.register(source_df)
            source_rids = row_mgr.get_ids_array(source_df)

        # Derive kept mask from key
        kept_mask, completeness = self._derive_loc_mask(key, source_df)

        # Always propagate RIDs
        if kept_mask is not None:
            row_mgr.propagate_by_mask(source_df, result, kept_mask)
        else:
            row_mgr.register(result)

        # Skip step tracking if we're inside a filter operation (parent will track)
        if ctx._filter_op_depth > 0:
            return

        store = ctx.store

        if kept_mask is not None:
            dropped_rids, _ = row_mgr.compute_dropped_with_positions(source_rids, kept_mask)
        else:
            completeness = CompletenessLevel.PARTIAL
            result_rids = row_mgr.get_ids_array(result)
            dropped_rids = row_mgr.compute_dropped_ids(source_rids, result_rids)

        code_file, code_line = get_caller_info(skip_frames=4)
        step_id = store.append_step(
            operation="DataFrame.loc[]",
            stage=ctx.current_stage,
            code_file=code_file,
            code_line=code_line,
            params={"key_type": type(key).__name__},
            input_shape=source_df.shape,
            output_shape=result.shape,
            completeness=completeness,
        )

        if len(dropped_rids) > 0:
            store.append_bulk_drops(step_id, dropped_rids)

    def _derive_loc_mask(
        self, key, df: pd.DataFrame
    ) -> tuple[Optional[np.ndarray], CompletenessLevel]:
        """
        Derive boolean mask from loc key.

        Key types:
        - Boolean array/Series -> mask directly
        - Slice -> convert to positional mask
        - List of labels -> index.isin()
        - Single label -> single row mask
        - Tuple (row_key, col_key) -> handle row_key
        """
        row_key = key[0] if isinstance(key, tuple) else key

        # Boolean mask
        if hasattr(row_key, "dtype") and row_key.dtype == bool:
            mask = row_key.values if isinstance(row_key, pd.Series) else np.asarray(row_key)
            if len(mask) == len(df):
                return mask, CompletenessLevel.FULL

        # List of labels
        if isinstance(row_key, (list, np.ndarray)) and not (
            hasattr(row_key, "dtype") and row_key.dtype == bool
        ):
            mask = df.index.isin(row_key)
            return (
                mask.to_numpy() if hasattr(mask, "to_numpy") else np.asarray(mask),
                CompletenessLevel.FULL,
            )

        # Slice
        if isinstance(row_key, slice):
            try:
                # Get positional indices from label slice
                start_idx = df.index.get_loc(row_key.start) if row_key.start is not None else 0
                stop_idx = df.index.get_loc(row_key.stop) if row_key.stop is not None else len(df)
                # loc slice is inclusive on both ends
                if isinstance(start_idx, int) and isinstance(stop_idx, int):
                    mask = np.zeros(len(df), dtype=bool)
                    mask[start_idx : stop_idx + 1] = True
                    return mask, CompletenessLevel.FULL
            except (KeyError, TypeError):
                pass

        # Single label
        if not isinstance(row_key, (list, np.ndarray, slice, pd.Series)):
            try:
                mask = df.index == row_key
                return (
                    mask.to_numpy() if hasattr(mask, "to_numpy") else np.asarray(mask),
                    CompletenessLevel.FULL,
                )
            except Exception:
                pass

        return None, CompletenessLevel.PARTIAL

    def _capture_before_state(self, key, ctx) -> tuple[Optional[dict], Optional[list]]:
        """Capture values before assignment for watched columns."""
        col_key = key[1] if isinstance(key, tuple) and len(key) > 1 else None

        # Determine affected columns
        if col_key is None:
            affected_cols = list(ctx.watched_columns & set(self._parent_df.columns))
        elif isinstance(col_key, str):
            affected_cols = [col_key] if col_key in ctx.watched_columns else []
        elif isinstance(col_key, list):
            affected_cols = [c for c in col_key if c in ctx.watched_columns]
        else:
            affected_cols = []

        if not affected_cols:
            return None, None

        # Derive affected rows
        mask, _ = self._derive_loc_mask(key, self._parent_df)
        if mask is None:
            return None, affected_cols

        # Capture values (vectorized per column)
        rids = ctx.row_manager.get_ids_array(self._parent_df)
        if rids is None:
            return None, affected_cols

        before = {}
        affected_positions = np.where(mask)[0]
        for col in affected_cols:
            before[col] = {
                "rids": rids[affected_positions].copy(),
                "values": self._parent_df[col].values[affected_positions].copy(),
            }

        return before, affected_cols

    def _capture_loc_setitem(self, key, value, before_values, affected_cols, ctx) -> None:
        """Capture transform via loc[] = value."""
        if before_values is None or not affected_cols:
            return

        store = ctx.store

        code_file, code_line = get_caller_info(skip_frames=4)
        step_id = store.append_step(
            operation="DataFrame.loc[]=",
            stage=ctx.current_stage,
            code_file=code_file,
            code_line=code_line,
            params={"columns": affected_cols[:3]},
            input_shape=self._parent_df.shape,
            output_shape=self._parent_df.shape,
            completeness=CompletenessLevel.FULL,
        )

        from ..utils.value_capture import values_equal

        for col in affected_cols:
            if col not in before_values:
                continue

            rids = before_values[col]["rids"]
            old_vals = before_values[col]["values"]

            # Get current positions for these rids
            mask, _ = self._derive_loc_mask(key, self._parent_df)
            if mask is None:
                continue

            new_vals = self._parent_df[col].values[np.where(mask)[0]]

            # Vectorized diff detection
            for rid, old_val, new_val in zip(rids, old_vals, new_vals):
                if not values_equal(old_val, new_val):
                    store.append_diff(
                        step_id=step_id,
                        row_id=int(rid),
                        col=col,
                        old_val=old_val,
                        new_val=new_val,
                        change_type=ChangeType.MODIFIED,
                    )


class TrackedILocIndexer:
    """
    Wrapper around pandas _iLocIndexer that captures lineage.

    Similar to TrackedLocIndexer but uses positional indexing.
    """

    def __init__(self, indexer, parent_df: pd.DataFrame):
        self._indexer = indexer
        self._parent_df = parent_df

    def __getattr__(self, name):
        """Proxy any other attribute access to the underlying indexer."""
        return getattr(self._indexer, name)

    def __getitem__(self, key) -> Union[pd.DataFrame, pd.Series, Any]:
        """Capture filter via iloc[]."""
        ctx = get_context()
        result = self._indexer[key]

        if not ctx.enabled:
            return result

        try:
            self._capture_iloc_getitem(key, result, ctx)
        except Exception as e:
            if ctx.config.strict_mode:
                raise
            warnings.warn(f"TracePipe: iloc[] capture failed: {e}", TracePipeWarning)

        return result

    def __setitem__(self, key, value) -> None:
        """Capture transform via iloc[] = value."""
        ctx = get_context()

        before_values = None
        affected_cols = None
        if ctx.enabled and ctx.watched_columns:
            before_values, affected_cols = self._capture_before_state(key, ctx)

        self._indexer[key] = value

        if not ctx.enabled:
            return

        try:
            self._capture_iloc_setitem(key, value, before_values, affected_cols, ctx)
        except Exception as e:
            if ctx.config.strict_mode:
                raise
            warnings.warn(f"TracePipe: iloc[]= capture failed: {e}", TracePipeWarning)

    def _capture_iloc_getitem(self, key, result, ctx) -> None:
        """Capture filter via iloc[]."""
        if not isinstance(result, pd.DataFrame):
            return

        row_mgr = ctx.row_manager
        source_df = self._parent_df

        source_rids = row_mgr.get_ids_array(source_df)
        if source_rids is None:
            row_mgr.register(source_df)
            source_rids = row_mgr.get_ids_array(source_df)

        # Derive positions from key
        positions = self._derive_iloc_positions(key, source_df)

        # Always propagate RIDs
        if positions is not None:
            row_mgr.propagate_by_positions(source_df, result, positions)
        else:
            row_mgr.register(result)

        # Skip step tracking if we're inside a filter operation (parent will track)
        if ctx._filter_op_depth > 0:
            return

        store = ctx.store

        if positions is not None:
            kept_mask = np.zeros(len(source_df), dtype=bool)
            kept_mask[positions] = True
            dropped_rids, _ = row_mgr.compute_dropped_with_positions(source_rids, kept_mask)
            completeness = CompletenessLevel.FULL
        else:
            completeness = CompletenessLevel.PARTIAL
            result_rids = row_mgr.get_ids_array(result)
            dropped_rids = row_mgr.compute_dropped_ids(source_rids, result_rids)

        code_file, code_line = get_caller_info(skip_frames=4)
        step_id = store.append_step(
            operation="DataFrame.iloc[]",
            stage=ctx.current_stage,
            code_file=code_file,
            code_line=code_line,
            params={"key_type": type(key).__name__},
            input_shape=source_df.shape,
            output_shape=result.shape,
            completeness=completeness,
        )

        if len(dropped_rids) > 0:
            store.append_bulk_drops(step_id, dropped_rids)

    def _derive_iloc_positions(self, key, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Derive position array from iloc key."""
        row_key = key[0] if isinstance(key, tuple) else key
        n = len(df)

        # Integer
        if isinstance(row_key, int):
            pos = row_key if row_key >= 0 else n + row_key
            return np.array([pos], dtype=np.int64)

        # Slice
        if isinstance(row_key, slice):
            indices = range(*row_key.indices(n))
            return np.array(list(indices), dtype=np.int64)

        # List/array of integers
        if isinstance(row_key, (list, np.ndarray)):
            arr = np.asarray(row_key, dtype=np.int64)
            # Handle negative indices
            arr = np.where(arr < 0, n + arr, arr)
            return arr

        # Boolean array
        if hasattr(row_key, "dtype") and row_key.dtype == bool:
            return np.where(row_key)[0].astype(np.int64)

        return None

    def _capture_before_state(self, key, ctx):
        """Capture values before assignment for watched columns."""
        col_key = key[1] if isinstance(key, tuple) and len(key) > 1 else None

        # Determine affected columns by position
        if col_key is None:
            affected_cols = list(ctx.watched_columns & set(self._parent_df.columns))
        elif isinstance(col_key, int):
            col_name = self._parent_df.columns[col_key]
            affected_cols = [col_name] if col_name in ctx.watched_columns else []
        elif isinstance(col_key, (list, np.ndarray)):
            col_names = [self._parent_df.columns[i] for i in col_key]
            affected_cols = [c for c in col_names if c in ctx.watched_columns]
        elif isinstance(col_key, slice):
            col_names = self._parent_df.columns[col_key].tolist()
            affected_cols = [c for c in col_names if c in ctx.watched_columns]
        else:
            affected_cols = []

        if not affected_cols:
            return None, None

        positions = self._derive_iloc_positions(key, self._parent_df)
        if positions is None:
            return None, affected_cols

        rids = ctx.row_manager.get_ids_array(self._parent_df)
        if rids is None:
            return None, affected_cols

        before = {}
        for col in affected_cols:
            before[col] = {
                "rids": rids[positions].copy(),
                "values": self._parent_df[col].values[positions].copy(),
                "positions": positions.copy(),
            }

        return before, affected_cols

    def _capture_iloc_setitem(self, key, value, before_values, affected_cols, ctx) -> None:
        """Capture transform via iloc[] = value."""
        if before_values is None or not affected_cols:
            return

        store = ctx.store

        code_file, code_line = get_caller_info(skip_frames=4)
        step_id = store.append_step(
            operation="DataFrame.iloc[]=",
            stage=ctx.current_stage,
            code_file=code_file,
            code_line=code_line,
            params={"columns": affected_cols[:3]},
            input_shape=self._parent_df.shape,
            output_shape=self._parent_df.shape,
            completeness=CompletenessLevel.FULL,
        )

        from ..utils.value_capture import values_equal

        for col in affected_cols:
            if col not in before_values:
                continue

            rids = before_values[col]["rids"]
            old_vals = before_values[col]["values"]
            positions = before_values[col]["positions"]
            new_vals = self._parent_df[col].values[positions]

            for rid, old_val, new_val in zip(rids, old_vals, new_vals):
                if not values_equal(old_val, new_val):
                    store.append_diff(
                        step_id=step_id,
                        row_id=int(rid),
                        col=col,
                        old_val=old_val,
                        new_val=new_val,
                        change_type=ChangeType.MODIFIED,
                    )


# Store original properties for restore
_original_loc = None
_original_iloc = None
_original_at = None
_original_iat = None


class TrackedAtIndexer:
    """
    Wrapper around pandas _AtIndexer that captures scalar assignments.

    .at is optimized for scalar access by label.
    """

    def __init__(self, indexer, df):
        self._indexer = indexer
        self._df = df

    def __getitem__(self, key):
        return self._indexer[key]

    def __setitem__(self, key, value) -> None:
        """Capture scalar assignment via at[row, col] = value."""
        ctx = get_context()
        if not ctx or not ctx.enabled:
            self._indexer[key] = value
            return

        row_label, col = key
        col_str = str(col)

        # Check if column is watched
        should_track = col_str in ctx.watched_columns if ctx.watched_columns else False

        # Capture before state
        old_val = None
        if should_track:
            try:
                old_val = self._df.at[row_label, col]
            except (KeyError, IndexError):
                pass

        # Execute original
        self._indexer[key] = value

        # Capture after state
        if should_track and ctx.store:
            from .pandas_inst import get_caller_info

            code_file, code_line = get_caller_info(skip_frames=2)
            step_id = ctx.store.append_step(
                operation="DataFrame.at[]=",
                stage=ctx.current_stage,
                code_file=code_file,
                code_line=code_line,
                params={"row": str(row_label), "col": col_str},
                input_shape=self._df.shape,
                output_shape=self._df.shape,
            )

            # Get row_id for this position
            try:
                row_pos = self._df.index.get_loc(row_label)
                if isinstance(row_pos, int):
                    rids = ctx.row_manager.get_ids_array(self._df)
                    if rids is None:
                        ctx.row_manager.register(self._df)
                        rids = ctx.row_manager.get_ids_array(self._df)
                    if rids is not None and row_pos < len(rids):
                        from ..core import ChangeType

                        row_id = int(rids[row_pos])
                        ctx.store.append_diff(
                            step_id=step_id,
                            row_id=row_id,
                            col=col_str,
                            old_val=old_val,
                            new_val=value,
                            change_type=ChangeType.MODIFIED,
                        )
            except (KeyError, TypeError):
                pass


class TrackedIAtIndexer:
    """
    Wrapper around pandas _iAtIndexer that captures scalar assignments.

    .iat is optimized for scalar access by integer position.
    """

    def __init__(self, indexer, df):
        self._indexer = indexer
        self._df = df

    def __getitem__(self, key):
        return self._indexer[key]

    def __setitem__(self, key, value) -> None:
        """Capture scalar assignment via iat[row, col] = value."""
        ctx = get_context()
        if not ctx or not ctx.enabled:
            self._indexer[key] = value
            return

        row_pos, col_pos = key
        col_str = self._df.columns[col_pos] if col_pos < len(self._df.columns) else str(col_pos)

        # Check if column is watched
        should_track = col_str in ctx.watched_columns if ctx.watched_columns else False

        # Capture before state
        old_val = None
        if should_track:
            try:
                old_val = self._df.iat[row_pos, col_pos]
            except (KeyError, IndexError):
                pass

        # Execute original
        self._indexer[key] = value

        # Capture after state
        if should_track and ctx.store:
            from .pandas_inst import get_caller_info

            code_file, code_line = get_caller_info(skip_frames=2)
            step_id = ctx.store.append_step(
                operation="DataFrame.iat[]=",
                stage=ctx.current_stage,
                code_file=code_file,
                code_line=code_line,
                params={"row": row_pos, "col": col_str},
                input_shape=self._df.shape,
                output_shape=self._df.shape,
            )

            # Get row_id for this position
            try:
                rids = ctx.row_manager.get_ids_array(self._df)
                if rids is None:
                    ctx.row_manager.register(self._df)
                    rids = ctx.row_manager.get_ids_array(self._df)
                if rids is not None and row_pos < len(rids):
                    from ..core import ChangeType

                    row_id = int(rids[row_pos])
                    ctx.store.append_diff(
                        step_id=step_id,
                        row_id=row_id,
                        col=col_str,
                        old_val=old_val,
                        new_val=value,
                        change_type=ChangeType.MODIFIED,
                    )
            except (KeyError, TypeError):
                pass


def instrument_indexers():
    """
    Install tracked indexers for loc, iloc, at, iat.

    Monkey-patches DataFrame.loc, DataFrame.iloc, DataFrame.at, DataFrame.iat properties.
    """
    global _original_loc, _original_iloc, _original_at, _original_iat

    if _original_loc is not None:
        # Already instrumented
        return

    _original_loc = pd.DataFrame.loc.fget
    _original_iloc = pd.DataFrame.iloc.fget
    _original_at = pd.DataFrame.at.fget
    _original_iat = pd.DataFrame.iat.fget

    @property
    def tracked_loc(self):
        return TrackedLocIndexer(_original_loc(self), self)

    @property
    def tracked_iloc(self):
        return TrackedILocIndexer(_original_iloc(self), self)

    @property
    def tracked_at(self):
        return TrackedAtIndexer(_original_at(self), self)

    @property
    def tracked_iat(self):
        return TrackedIAtIndexer(_original_iat(self), self)

    pd.DataFrame.loc = tracked_loc
    pd.DataFrame.iloc = tracked_iloc
    pd.DataFrame.at = tracked_at
    pd.DataFrame.iat = tracked_iat


def uninstrument_indexers():
    """Restore original loc/iloc/at/iat."""
    global _original_loc, _original_iloc, _original_at, _original_iat

    if _original_loc is not None:
        pd.DataFrame.loc = property(_original_loc)
        _original_loc = None
    if _original_iloc is not None:
        pd.DataFrame.iloc = property(_original_iloc)
        _original_iloc = None
    if _original_at is not None:
        pd.DataFrame.at = property(_original_at)
        _original_at = None
    if _original_iat is not None:
        pd.DataFrame.iat = property(_original_iat)
        _original_iat = None

# tracepipe/storage/row_identity.py
"""
Row identity tracking with positional propagation.

Key invariants:
- Every registered DataFrame has RIDs aligned to its index
- Propagation is POSITIONAL, not index-label based
- Ghost values are captured in debug mode only
- NO DataFrame mutation by default

Identity Storage Options:
- REGISTRY (default): WeakKeyDictionary, no mutation
  - If weakref fails, auto-degrades to ATTRS with one-time warning
- ATTRS: df.attrs token for long sessions
- COLUMN: hidden column (opt-in only)
"""

import logging
import uuid
import weakref
from collections import OrderedDict
from typing import Optional

import numpy as np
import pandas as pd

from ..core import GhostRowInfo, IdentityStorage, TracePipeConfig

logger = logging.getLogger(__name__)

_TRACEPIPE_ROW_ID_COL = "__tracepipe_row_id__"
_TRACEPIPE_TOKEN_ATTR = "_tracepipe_token"


class PandasRowIdentity:
    """
    Row identity tracking with positional propagation.

    Implements: RowIdentityStrategy protocol
    """

    # Cap token registry size to prevent unbounded growth
    MAX_TOKEN_REGISTRY_SIZE: int = 50_000

    def __init__(self, config: TracePipeConfig):
        self.config = config
        self._next_row_id: int = 0

        # Use WeakKeyDictionary for proper GC
        # Maps DataFrame object -> rids_array
        self._registry: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

        # Fallback for non-weakref-able DataFrames
        # Maps token -> rids_array (token stored in df.attrs)
        # Uses OrderedDict for FIFO eviction
        self._token_registry: OrderedDict[str, np.ndarray] = OrderedDict()

        # Ghost row storage (debug mode only)
        self._ghost_rows: dict[int, GhostRowInfo] = {}

        # Track if we've warned about weakref fallback
        self._weakref_fallback_warned: bool = False

    def _generate_token(self) -> str:
        """Generate unique token for DataFrame identification."""
        return uuid.uuid4().hex

    def register(
        self,
        df: pd.DataFrame,
        row_ids: Optional[np.ndarray] = None,
        warn_duplicate_index: bool = True,
    ) -> np.ndarray:
        """
        Register a DataFrame and assign row IDs.

        Returns:
            numpy array of row IDs (int64)
        """
        if warn_duplicate_index and self.config.warn_on_duplicate_index:
            if df.index.has_duplicates:
                logger.debug("DataFrame has duplicate index values. Row identity may be ambiguous.")

        n = len(df)
        if row_ids is None:
            row_ids = np.arange(self._next_row_id, self._next_row_id + n, dtype=np.int64)
            self._next_row_id += n
        else:
            row_ids = np.asarray(row_ids, dtype=np.int64)

        # Explicit storage selection based on config
        if self.config.identity_storage == IdentityStorage.COLUMN:
            # COLUMN mode: hidden column only
            df[_TRACEPIPE_ROW_ID_COL] = row_ids

        elif self.config.identity_storage == IdentityStorage.ATTRS:
            # ATTRS mode: df.attrs token only
            token = self._generate_token()
            df.attrs[_TRACEPIPE_TOKEN_ATTR] = token
            self._add_to_token_registry(token, row_ids)

        else:  # REGISTRY mode (default)
            # Try WeakKeyDictionary first
            try:
                self._registry[df] = row_ids
            except TypeError:
                # Auto-degrade to attrs silently (log at debug level)
                if not self._weakref_fallback_warned:
                    logger.debug(
                        "DataFrame not weakref-able; using df.attrs fallback. "
                        "This is safe but uses slightly more memory."
                    )
                    self._weakref_fallback_warned = True

                token = self._generate_token()
                df.attrs[_TRACEPIPE_TOKEN_ATTR] = token
                self._add_to_token_registry(token, row_ids)

        return row_ids

    def _add_to_token_registry(self, token: str, row_ids: np.ndarray) -> None:
        """
        Add token to registry with FIFO eviction if over cap.

        Prevents unbounded growth of _token_registry in attrs fallback mode.
        """
        # Evict oldest tokens if over cap
        while len(self._token_registry) >= self.MAX_TOKEN_REGISTRY_SIZE:
            self._token_registry.popitem(last=False)  # Remove oldest (FIFO)

        self._token_registry[token] = row_ids

    def get_ids(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Get row IDs as Series (for index-based operations)."""
        rids = self.get_ids_array(df)
        if rids is not None:
            return pd.Series(rids, index=df.index, dtype=np.int64)
        return None

    def get_ids_array(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Get row IDs as numpy array (for vectorized operations)."""
        # 1. Try WeakKeyDictionary (fastest, for REGISTRY mode)
        try:
            if df in self._registry:
                rids = self._registry[df]
                if len(rids) == len(df):
                    return rids
        except TypeError:
            pass

        # 2. Try attrs token (for ATTRS mode or REGISTRY fallback)
        token = df.attrs.get(_TRACEPIPE_TOKEN_ATTR)
        if token and token in self._token_registry:
            rids = self._token_registry[token]
            if len(rids) == len(df):
                return rids

        # 3. Try hidden column (for COLUMN mode)
        # Access underlying data directly to avoid triggering instrumented __getitem__
        if _TRACEPIPE_ROW_ID_COL in df.columns:
            col_idx = df.columns.get_loc(_TRACEPIPE_ROW_ID_COL)
            # Use _iget_item_cache for direct column access (bypasses instrumentation)
            return df._get_column_array(col_idx).astype(np.int64)

        return None

    def set_result_rids(self, result_df: pd.DataFrame, rids: np.ndarray) -> None:
        """
        Set RIDs for a result DataFrame (internal controlled assignment).

        Uses same storage logic as register().
        """
        rids = np.asarray(rids, dtype=np.int64)

        if self.config.identity_storage == IdentityStorage.COLUMN:
            result_df[_TRACEPIPE_ROW_ID_COL] = rids

        elif self.config.identity_storage == IdentityStorage.ATTRS:
            token = self._generate_token()
            result_df.attrs[_TRACEPIPE_TOKEN_ATTR] = token
            self._add_to_token_registry(token, rids)

        else:  # REGISTRY mode
            try:
                self._registry[result_df] = rids
            except TypeError:
                # Fallback to attrs (FIFO eviction)
                token = self._generate_token()
                result_df.attrs[_TRACEPIPE_TOKEN_ATTR] = token
                self._add_to_token_registry(token, rids)

    # ========== POSITIONAL PROPAGATION METHODS ==========

    def propagate_by_mask(
        self, source_df: pd.DataFrame, result_df: pd.DataFrame, kept_mask: np.ndarray
    ) -> np.ndarray:
        """
        Propagate RIDs using boolean mask (for filter operations).

        Args:
            source_df: Original DataFrame
            result_df: Filtered DataFrame
            kept_mask: Boolean array where True = row kept

        Returns:
            Array of RIDs for result_df
        """
        source_rids = self.get_ids_array(source_df)
        if source_rids is None:
            return self.register(result_df)

        result_rids = source_rids[kept_mask]
        self.set_result_rids(result_df, result_rids)
        return result_rids

    def propagate_by_positions(
        self, source_df: pd.DataFrame, result_df: pd.DataFrame, positions: np.ndarray
    ) -> np.ndarray:
        """
        Propagate RIDs using position indices (for head/tail/sample).
        """
        source_rids = self.get_ids_array(source_df)
        if source_rids is None:
            return self.register(result_df)

        result_rids = source_rids[positions]
        self.set_result_rids(result_df, result_rids)
        return result_rids

    def propagate_by_permutation(
        self, source_df: pd.DataFrame, result_df: pd.DataFrame, perm: np.ndarray
    ) -> np.ndarray:
        """
        Propagate RIDs using permutation array (for sort_values).
        """
        source_rids = self.get_ids_array(source_df)
        if source_rids is None:
            return self.register(result_df)

        result_rids = source_rids[perm]
        self.set_result_rids(result_df, result_rids)
        return result_rids

    def propagate(self, source_df: pd.DataFrame, result_df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Propagate row IDs from source to result DataFrame.

        Backwards compatible method - uses index-based fallback.
        For better accuracy, use propagate_by_mask/positions/permutation.
        """
        source_ids = self.get_ids(source_df)
        if source_ids is None:
            return None

        if result_df is source_df:
            return source_ids

        try:
            if result_df.index.equals(source_df.index):
                # Same index - direct copy
                result_ids = source_ids.copy()
            elif result_df.index.isin(source_df.index).all():
                # Result index is subset/reorder of source
                result_ids = source_ids.loc[result_df.index].copy()
            else:
                # Partial overlap or new indices
                result_ids = source_ids.reindex(result_df.index)
                # Rows not in source get new IDs
                new_mask = result_ids.isna()
                if new_mask.any():
                    new_count = new_mask.sum()
                    new_row_ids = np.arange(
                        self._next_row_id, self._next_row_id + new_count, dtype=np.int64
                    )
                    self._next_row_id += new_count
                    result_ids.loc[new_mask] = new_row_ids
                result_ids = result_ids.astype("int64")
        except Exception:
            # Fallback: positional alignment
            if len(result_df) <= len(source_df):
                result_ids = pd.Series(
                    source_ids.values[: len(result_df)],
                    index=result_df.index,
                    dtype="int64",
                )
            else:
                # Result is larger - assign new IDs to extras
                base_ids = list(source_ids.values)
                extra_count = len(result_df) - len(source_df)
                extra_ids = list(range(self._next_row_id, self._next_row_id + extra_count))
                self._next_row_id += extra_count
                result_ids = pd.Series(base_ids + extra_ids, index=result_df.index, dtype="int64")

        # Register the result
        rids_array = result_ids.values.astype(np.int64)
        self.set_result_rids(result_df, rids_array)
        return result_ids

    # ========== DROP COMPUTATION ==========

    def compute_dropped_ids(self, source_rids: np.ndarray, result_rids: np.ndarray) -> np.ndarray:
        """
        Compute which RIDs were dropped.

        Returns:
            Array of dropped RIDs (sorted)
        """
        # assume_unique=False because arrays are not guaranteed sorted
        return np.setdiff1d(source_rids, result_rids, assume_unique=False)

    def compute_dropped_with_positions(
        self, source_rids: np.ndarray, kept_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute dropped IDs with original positions preserved.

        Returns:
            (dropped_rids, dropped_positions) - both in original order
        """
        dropped_mask = ~kept_mask
        dropped_positions = np.where(dropped_mask)[0]
        dropped_rids = source_rids[dropped_mask]
        return dropped_rids, dropped_positions

    def get_dropped_ids(self, source_df: pd.DataFrame, result_df: pd.DataFrame) -> np.ndarray:
        """
        Get row IDs that were dropped between source and result.

        Backwards compatible method.
        """
        source_ids = self.get_ids_array(source_df)
        result_ids = self.get_ids_array(result_df)

        if source_ids is None:
            return np.array([], dtype="int64")
        if result_ids is None:
            return np.asarray(source_ids, dtype="int64")

        return np.setdiff1d(source_ids, result_ids)

    # ========== GHOST ROW TRACKING ==========

    def capture_ghost_values(
        self,
        source_df: pd.DataFrame,
        dropped_mask: np.ndarray,
        dropped_by: str,
        step_id: int,
        watched_columns: set[str],
    ) -> None:
        """
        Capture last-known values for dropped rows (debug mode only).

        Uses vectorized extraction, not iloc per cell.
        """
        if not self.config.should_capture_ghost_values:
            return

        source_rids = self.get_ids_array(source_df)
        if source_rids is None:
            return

        dropped_positions = np.where(dropped_mask)[0]
        dropped_rids = source_rids[dropped_mask]

        if len(dropped_rids) == 0:
            return

        # Limit to prevent memory explosion
        max_ghosts = self.config.max_ghost_rows
        if len(dropped_rids) > max_ghosts:
            sample_idx = np.random.choice(len(dropped_rids), max_ghosts, replace=False)
            sample_idx.sort()  # Keep relative order
            dropped_rids = dropped_rids[sample_idx]
            dropped_positions = dropped_positions[sample_idx]

        # Determine columns to capture
        cols_to_capture = list(watched_columns & set(source_df.columns))
        if not cols_to_capture:
            cols_to_capture = list(source_df.columns)[:5]

        # Vectorized extraction (one slice per column)
        values_matrix: dict[str, np.ndarray] = {}
        for col in cols_to_capture:
            try:
                values_matrix[col] = source_df[col].values[dropped_positions]
            except Exception:
                pass

        # Build ghost row info from pre-extracted values
        for i, (rid, pos) in enumerate(zip(dropped_rids, dropped_positions)):
            values = {col: vals[i] for col, vals in values_matrix.items()}

            self._ghost_rows[int(rid)] = GhostRowInfo(
                row_id=int(rid),
                last_values=values,
                dropped_by=dropped_by,
                dropped_step=step_id,
                original_position=int(pos),
            )

    def get_ghost_rows(self, limit: int = 1000) -> pd.DataFrame:
        """
        Get dropped rows with last-known values.

        Returns DataFrame with:
        - __tp_row_id__: Original row ID
        - __tp_dropped_by__: Operation that dropped the row
        - __tp_dropped_step__: Step ID
        - __tp_original_position__: Position in original DataFrame
        - [original columns]: Last known values
        """
        if not self._ghost_rows:
            return pd.DataFrame()

        # Sort by original position for natural order
        sorted_ghosts = sorted(self._ghost_rows.values(), key=lambda g: g.original_position)[:limit]

        rows = []
        for info in sorted_ghosts:
            row = {
                "__tp_row_id__": info.row_id,
                "__tp_dropped_by__": info.dropped_by,
                "__tp_dropped_step__": info.dropped_step,
                "__tp_original_position__": info.original_position,
                **info.last_values,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    # ========== CLEANUP ==========

    def clear(self) -> None:
        """Reset all state."""
        self._registry.clear()
        self._token_registry.clear()
        self._ghost_rows.clear()
        self._next_row_id = 0

    def cleanup(self) -> None:
        """Remove stale entries (backwards compatible)."""
        # WeakKeyDictionary handles this automatically for registry.
        # Token registry uses FIFO eviction in _add_to_token_registry.
        pass

    def strip_hidden_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove hidden column for export."""
        if _TRACEPIPE_ROW_ID_COL in df.columns:
            return df.drop(columns=[_TRACEPIPE_ROW_ID_COL])
        return df

    def all_registered_ids(self) -> list[int]:
        """
        Get all row IDs that have ever been registered.
        """
        all_ids: set[int] = set()

        # From WeakKeyDictionary
        for rids in self._registry.values():
            all_ids.update(rids.tolist())

        # From token registry
        for rids in self._token_registry.values():
            all_ids.update(rids.tolist())

        return sorted(all_ids)

    def realign_for_reset_index(
        self, original_df: pd.DataFrame, new_df: pd.DataFrame
    ) -> Optional[pd.Series]:
        """Handle reset_index(drop=True) which changes index."""
        old_ids = self.get_ids_array(original_df)
        if old_ids is None:
            return None

        # Same values, new index
        self.set_result_rids(new_df, old_ids)
        return pd.Series(old_ids, index=new_df.index, dtype="int64")

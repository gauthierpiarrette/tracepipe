# tracepipe/storage/row_identity.py
"""
Row identity tracking for pandas DataFrames.

Uses: Registry + Hidden Column fallback.
"""

import warnings
import weakref
from typing import Optional

import numpy as np
import pandas as pd

from ..core import TracePipeConfig

_TRACEPIPE_ROW_ID_COL = "__tracepipe_row_id__"


class PandasRowIdentity:
    """
    Hybrid row identity tracking for pandas DataFrames.

    Implements: RowIdentityStrategy protocol

    Handles:
    - Standard operations (filter, sort, copy)
    - reset_index(drop=True)
    - Duplicate indices (with warning)
    - Chained operations

    Future alternatives:
    - PolarsRowIdentity: Uses Polars row numbers and lazy evaluation
    - SparkRowIdentity: Uses monotonically_increasing_id() or RDD zipWithIndex
    """

    def __init__(self, config: TracePipeConfig):
        self.config = config
        self._registry: dict[int, pd.Series] = {}
        self._df_refs: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._next_row_id: int = 0

    def register(
        self,
        df: pd.DataFrame,
        row_ids: Optional[pd.Series] = None,
        warn_duplicate_index: bool = True,
    ) -> pd.Series:
        """
        Register a DataFrame and assign row IDs.

        Args:
            df: DataFrame to register
            row_ids: Optional pre-assigned IDs (for propagation)
            warn_duplicate_index: Warn if index has duplicates

        Returns:
            Series of row IDs aligned to df.index
        """
        # Check for duplicate index
        if warn_duplicate_index and self.config.warn_on_duplicate_index:
            if df.index.has_duplicates:
                warnings.warn(
                    "TracePipe: DataFrame has duplicate index values. "
                    "Row identity may be ambiguous for duplicates.",
                    UserWarning,
                )

        if row_ids is None:
            # Generate new sequential IDs
            new_ids = list(range(self._next_row_id, self._next_row_id + len(df)))
            self._next_row_id += len(df)
            row_ids = pd.Series(new_ids, index=df.index, dtype="int64")
        else:
            # Ensure alignment
            if not row_ids.index.equals(df.index):
                row_ids = row_ids.copy()
                row_ids.index = df.index

        obj_id = id(df)
        self._registry[obj_id] = row_ids
        self._df_refs[obj_id] = df

        # Optionally embed in DataFrame
        if self.config.use_hidden_column:
            df[_TRACEPIPE_ROW_ID_COL] = row_ids.values

        return row_ids

    def get_ids(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Get row IDs for a DataFrame."""
        # 1. Try registry (fast path)
        obj_id = id(df)
        if obj_id in self._registry:
            stored = self._registry[obj_id]
            # Verify alignment still valid
            if len(stored) == len(df) and stored.index.equals(df.index):
                return stored

        # 2. Try hidden column (fallback)
        if _TRACEPIPE_ROW_ID_COL in df.columns:
            row_ids = df[_TRACEPIPE_ROW_ID_COL].copy()
            row_ids.index = df.index
            # Re-register for future lookups
            self._registry[obj_id] = row_ids
            self._df_refs[obj_id] = df
            return row_ids

        # 3. Not tracked
        return None

    def propagate(self, source_df: pd.DataFrame, result_df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Propagate row IDs from source to result DataFrame.

        Handles:
        - Filtering (fewer rows)
        - Reordering (same rows, different order)
        - Mixed operations
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
                    new_row_ids = list(range(self._next_row_id, self._next_row_id + new_count))
                    self._next_row_id += new_count
                    result_ids.loc[new_mask] = new_row_ids
                result_ids = result_ids.astype("int64")
        except Exception:
            # Fallback: positional alignment
            if len(result_df) <= len(source_df):
                result_ids = pd.Series(
                    source_ids.values[: len(result_df)], index=result_df.index, dtype="int64"
                )
            else:
                # Result is larger - assign new IDs to extras
                base_ids = list(source_ids.values)
                extra_count = len(result_df) - len(source_df)
                extra_ids = list(range(self._next_row_id, self._next_row_id + extra_count))
                self._next_row_id += extra_count
                result_ids = pd.Series(base_ids + extra_ids, index=result_df.index, dtype="int64")

        return self.register(result_df, result_ids, warn_duplicate_index=False)

    def realign_for_reset_index(
        self, original_df: pd.DataFrame, new_df: pd.DataFrame
    ) -> Optional[pd.Series]:
        """Handle reset_index(drop=True) which changes index."""
        old_ids = self.get_ids(original_df)
        if old_ids is None:
            return None

        # Same values, new index
        new_ids = pd.Series(old_ids.values, index=new_df.index, dtype="int64")
        return self.register(new_df, new_ids, warn_duplicate_index=False)

    def get_dropped_ids(self, source_df: pd.DataFrame, result_df: pd.DataFrame) -> np.ndarray:
        """
        Get row IDs that were dropped between source and result.

        Uses numpy's setdiff1d for vectorized performance (~50x faster
        than Python set operations for large DataFrames).

        Returns:
            numpy array of dropped row IDs (empty array if none dropped)
        """
        source_ids = self.get_ids(source_df)
        result_ids = self.get_ids(result_df)

        if source_ids is None:
            return np.array([], dtype="int64")
        if result_ids is None:
            return np.asarray(source_ids.values, dtype="int64")

        # Vectorized set difference - O(n log n) in C instead of O(n) in Python
        return np.setdiff1d(source_ids.values, result_ids.values)

    def strip_hidden_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove hidden column for export."""
        if _TRACEPIPE_ROW_ID_COL in df.columns:
            return df.drop(columns=[_TRACEPIPE_ROW_ID_COL])
        return df

    def cleanup(self) -> None:
        """Remove stale entries."""
        stale = [k for k in list(self._registry.keys()) if k not in self._df_refs]
        for k in stale:
            del self._registry[k]

    def all_registered_ids(self) -> list[int]:
        """
        Get all row IDs that have ever been registered.

        Returns:
            List of all registered row IDs.
        """
        all_ids = set()
        for row_ids in self._registry.values():
            all_ids.update(row_ids.values.tolist())
        return sorted(all_ids)

# tracepipe/utils/value_capture.py
"""
Value capture and comparison utilities with complete NA handling.
"""

from typing import Any

import numpy as np
import pandas as pd

# Interned type strings (avoid allocating same strings repeatedly)
_TYPE_NULL = "null"
_TYPE_BOOL = "bool"
_TYPE_INT = "int"
_TYPE_FLOAT = "float"
_TYPE_STR = "str"
_TYPE_DATETIME = "datetime"
_TYPE_OTHER = "other"


def capture_typed_value(value: Any) -> tuple[Any, str]:
    """
    Convert value to (python_native, type_string) for storage.

    Handles:
    - None, np.nan, pd.NA, pd.NaT
    - numpy scalars (not JSON-serializable)
    - Standard Python types
    - Datetime types

    Returns:
        Tuple of (native_value, type_string)
    """
    # Handle all NA types first (pd.isna handles None, np.nan, pd.NA, pd.NaT)
    try:
        if pd.isna(value):
            return None, _TYPE_NULL
    except (ValueError, TypeError):
        # pd.isna can fail on some types (e.g., lists)
        pass

    # numpy scalar -> Python native (CRITICAL for JSON serialization)
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass

    # Type mapping (order matters: bool before int)
    if isinstance(value, bool):
        return value, _TYPE_BOOL
    elif isinstance(value, (int, np.integer)):
        return int(value), _TYPE_INT
    elif isinstance(value, (float, np.floating)):
        return float(value), _TYPE_FLOAT
    elif isinstance(value, str):
        return value, _TYPE_STR
    elif isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value), _TYPE_DATETIME
    else:
        # Fallback: stringify for storage
        return str(value), _TYPE_OTHER


def values_equal(a: Any, b: Any) -> bool:
    """
    Compare two values, handling NA correctly.

    pd.isna(x) == pd.isna(y) handles the case where both are NA.
    """
    try:
        a_na = pd.isna(a)
        b_na = pd.isna(b)

        if a_na and b_na:
            return True
        if a_na or b_na:
            return False
        return a == b
    except (ValueError, TypeError):
        # Fallback for unhashable types
        return str(a) == str(b)


def find_changed_indices_vectorized(old_series: pd.Series, new_series: pd.Series) -> np.ndarray:
    """
    Find indices where values changed, using vectorized operations.

    ~50-100x faster than row-by-row .loc[] access for large DataFrames.

    Args:
        old_series: Series of old values (must be aligned with new_series)
        new_series: Series of new values

    Returns:
        Boolean mask array where True indicates the value changed
    """
    old_arr = old_series.values
    new_arr = new_series.values
    n = len(old_arr)

    if n == 0:
        return np.array([], dtype=bool)

    # Vectorized NA detection
    old_na = pd.isna(old_arr)
    new_na = pd.isna(new_arr)

    # NA status changed (one is NA, other isn't)
    na_status_changed = old_na != new_na

    # For non-NA values, check if they differ
    # Handle mixed types safely by comparing element-by-element for non-NA
    both_not_na = ~old_na & ~new_na

    # Initialize values_differ as False
    values_differ = np.zeros(n, dtype=bool)

    # Only compare where both are non-NA
    non_na_indices = np.where(both_not_na)[0]
    if len(non_na_indices) > 0:
        # Try vectorized comparison first (fast path for homogeneous arrays)
        try:
            with np.errstate(invalid="ignore"):
                values_differ[non_na_indices] = old_arr[non_na_indices] != new_arr[non_na_indices]
        except (TypeError, ValueError):
            # Fallback for mixed types: element-by-element comparison
            for i in non_na_indices:
                try:
                    values_differ[i] = old_arr[i] != new_arr[i]
                except (TypeError, ValueError):
                    # Different types that can't be compared - treat as different
                    values_differ[i] = True

    changed_mask = (both_not_na & values_differ) | na_status_changed

    return changed_mask

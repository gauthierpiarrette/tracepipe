# tracepipe/instrumentation/series_capture.py
"""
Series method instrumentation for TracePipe.

Challenge: Series operations are often chained and may not be assigned back.
Solution: Track when Series is extracted, wrap common methods, capture on assignment.

Operations tracked:
| Pattern                           | Tracking                    | Completeness |
|-----------------------------------|-----------------------------|--------------|
| df['col'].fillna(val)             | Method call + assignment    | FULL         |
| df['col'].replace(...)            | Method call + assignment    | FULL         |
| df['col'].str.upper()             | Method call + assignment    | FULL         |
| df['col'].dt.year                 | Method call + assignment    | FULL         |
| df['col'].apply(func)             | Before/after diff           | PARTIAL      |
| df['col'] = series                | Assignment diff             | FULL         |

Key insight: We track at ASSIGNMENT time, not method call time.
This handles arbitrary chains: df['col'] = df['other'].str.strip().str.upper()
"""

import warnings
import weakref
from functools import wraps

import pandas as pd

from ..context import get_context
from ..core import ChangeType, CompletenessLevel
from ..safety import TracePipeWarning, get_caller_info


class TrackedSeries(pd.Series):
    """
    Series subclass that tracks its origin DataFrame and column.

    When assigned back to a DataFrame, we can compute the diff.

    Note: This is created only when extracting from a tracked DataFrame.
    Regular Series operations remain unchanged.

    Memory Safety:
    - _tp_source_df_ref is a weakref to prevent memory leaks
    - Source DataFrame can be garbage collected independently
    """

    _metadata = ["_tp_source_df_ref", "_tp_source_col", "_tp_source_rids", "_tp_last_op"]

    @property
    def _constructor(self):
        return TrackedSeries

    @property
    def _constructor_expanddim(self):
        return pd.DataFrame

    @property
    def _tp_source_df(self):
        """Get source DataFrame from weakref (may return None if GC'd)."""
        ref = getattr(self, "_tp_source_df_ref", None)
        if ref is not None:
            return ref()
        return None

    @_tp_source_df.setter
    def _tp_source_df(self, df):
        """Store source DataFrame as weakref."""
        if df is not None:
            self._tp_source_df_ref = weakref.ref(df)
        else:
            self._tp_source_df_ref = None


def wrap_series_extraction():
    """
    Wrap DataFrame.__getitem__ to return TrackedSeries for single column access.

    This allows us to track the origin of Series that may be modified and assigned back.
    """
    original_getitem = pd.DataFrame.__getitem__

    @wraps(original_getitem)
    def tracked_getitem(self, key):
        result = original_getitem(self, key)

        ctx = get_context()
        if not ctx.enabled:
            return result

        # Skip internal tracking operations to avoid recursion
        if ctx._filter_op_depth > 0:
            return result

        # Skip internal tracepipe columns
        if isinstance(key, str) and key.startswith("__tracepipe"):
            return result

        # Only wrap single-column Series access
        if isinstance(key, str) and isinstance(result, pd.Series):
            rids = ctx.row_manager.get_ids_array(self)
            if rids is not None:
                # Convert to TrackedSeries
                tracked = TrackedSeries(result)
                tracked._tp_source_df = self
                tracked._tp_source_col = key
                tracked._tp_source_rids = rids.copy()
                return tracked

        return result

    pd.DataFrame.__getitem__ = tracked_getitem
    pd.DataFrame._tp_original_getitem_series = original_getitem


def wrap_series_assignment():
    """
    Wrap DataFrame.__setitem__ to capture diffs when assigning Series.

    Note: For watched columns, _wrap_setitem (pandas_inst.py) already captures
    the assignment. This wrapper only captures for NON-watched columns when
    a TrackedSeries is assigned, to avoid double-logging.

    Handles:
    - df['col'] = series  (where series may have been modified)
    - df['col'] = scalar  (broadcast assignment)
    - df['col'] = array   (direct assignment)
    """
    original_setitem = pd.DataFrame.__setitem__

    @wraps(original_setitem)
    def tracked_setitem(self, key, value):
        ctx = get_context()

        # For watched columns, _wrap_setitem already captures - skip to avoid double-logging
        # We only capture here for NON-watched columns when a TrackedSeries is involved
        should_capture_here = False
        before_values = None

        if (
            ctx.enabled
            and isinstance(key, str)
            and key in self.columns
            and key not in ctx.watched_columns  # Only capture NON-watched columns here
            and isinstance(value, TrackedSeries)  # Only for TrackedSeries assignments
        ):
            rids = ctx.row_manager.get_ids_array(self)
            if rids is not None:
                should_capture_here = True
                before_values = {
                    "rids": rids.copy(),
                    "values": self[key].values.copy(),
                }

        # Always run original (which may be _wrap_setitem's wrapper)
        original_setitem(self, key, value)

        if not ctx.enabled:
            return

        if not should_capture_here or before_values is None:
            return

        try:
            _capture_series_assignment(self, key, value, before_values, ctx)
        except Exception as e:
            if ctx.config.strict_mode:
                raise
            warnings.warn(f"TracePipe: Series assignment capture failed: {e}", TracePipeWarning)

    pd.DataFrame.__setitem__ = tracked_setitem
    pd.DataFrame._tp_original_setitem_series = original_setitem


def _capture_series_assignment(df, key, value, before_values, ctx):
    """Capture diffs from Series assignment."""
    from ..utils.value_capture import values_equal

    store = ctx.store
    rids = before_values["rids"]
    old_vals = before_values["values"]
    new_vals = df[key].values

    # Determine completeness based on value type
    if isinstance(value, TrackedSeries):
        # Can trace back to source
        completeness = CompletenessLevel.FULL
        operation = f"Series.{_infer_series_operation(value)}"
    elif hasattr(value, "apply") or callable(value):
        completeness = CompletenessLevel.PARTIAL
        operation = "Series.transform"
    else:
        completeness = CompletenessLevel.FULL
        operation = "DataFrame[]="

    code_file, code_line = get_caller_info(skip_frames=4)
    step_id = store.append_step(
        operation=operation,
        stage=ctx.current_stage,
        code_file=code_file,
        code_line=code_line,
        params={"column": key},
        input_shape=df.shape,
        output_shape=df.shape,
        completeness=completeness,
    )

    # Track diffs for changed values
    for rid, old_val, new_val in zip(rids, old_vals, new_vals):
        if not values_equal(old_val, new_val):
            store.append_diff(
                step_id=step_id,
                row_id=int(rid),
                col=key,
                old_val=old_val,
                new_val=new_val,
                change_type=ChangeType.MODIFIED,
            )


def _infer_series_operation(series: TrackedSeries) -> str:
    """
    Infer the operation that produced this Series.

    Best effort - returns generic name if unknown.
    """
    if hasattr(series, "_tp_last_op") and series._tp_last_op is not None:
        return series._tp_last_op
    return "transform"


# ============ STRING ACCESSOR WRAPPERS ============

# Store original accessors module-level for restore
_original_str_methods = {}

# Use a WeakKeyDictionary to track series references without modifying accessor
_str_accessor_series_map = weakref.WeakKeyDictionary()


def wrap_str_accessor():
    """
    Wrap StringMethods to track operations.

    We wrap the individual methods rather than __init__ since pandas
    doesn't allow adding new attributes to accessor instances.
    """
    global _original_str_methods
    from pandas.core.strings.accessor import StringMethods

    # Wrap common string methods to preserve TrackedSeries
    for method_name in [
        "lower",
        "upper",
        "strip",
        "lstrip",
        "rstrip",
        "replace",
        "slice",
        "split",
        "contains",
        "startswith",
        "endswith",
        "len",
        "extract",
        "findall",
        "cat",
        "get",
        "pad",
        "center",
        "ljust",
        "rjust",
        "zfill",
        "wrap",
        "title",
        "capitalize",
        "swapcase",
        "normalize",
    ]:
        if hasattr(StringMethods, method_name):
            _wrap_str_method(StringMethods, method_name)


def _wrap_str_method(cls, method_name):
    """Wrap a single string method to preserve TrackedSeries."""
    global _original_str_methods

    if method_name in _original_str_methods:
        return  # Already wrapped

    original = getattr(cls, method_name)
    _original_str_methods[method_name] = original

    @wraps(original)
    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        # Try to get the originating series from the accessor's internal _orig
        series = getattr(self, "_orig", None)
        if isinstance(result, pd.Series) and isinstance(series, TrackedSeries):
            tracked = TrackedSeries(result)
            # Copy weakref directly to avoid creating strong reference
            tracked._tp_source_df_ref = getattr(series, "_tp_source_df_ref", None)
            tracked._tp_source_col = getattr(series, "_tp_source_col", None)
            tracked._tp_source_rids = getattr(series, "_tp_source_rids", None)
            tracked._tp_last_op = f"str.{method_name}"
            return tracked
        return result

    setattr(cls, method_name, wrapped)


def instrument_series():
    """Install all Series instrumentation."""
    wrap_series_extraction()
    wrap_series_assignment()
    wrap_str_accessor()
    # Note: DateTime accessor (.dt) wrapping is not implemented.
    # Most datetime operations don't require cell-level tracking.


def uninstrument_series():
    """Restore original Series behavior."""
    global _original_str_methods

    if hasattr(pd.DataFrame, "_tp_original_getitem_series"):
        pd.DataFrame.__getitem__ = pd.DataFrame._tp_original_getitem_series
        delattr(pd.DataFrame, "_tp_original_getitem_series")
    if hasattr(pd.DataFrame, "_tp_original_setitem_series"):
        pd.DataFrame.__setitem__ = pd.DataFrame._tp_original_setitem_series
        delattr(pd.DataFrame, "_tp_original_setitem_series")

    # Restore str methods
    if _original_str_methods:
        try:
            from pandas.core.strings.accessor import StringMethods

            for method_name, original in _original_str_methods.items():
                setattr(StringMethods, method_name, original)
            _original_str_methods.clear()
        except ImportError:
            pass

# tracepipe/safety.py
"""
Safe instrumentation wrappers.

CRITICAL: Execute original FIRST, capture lineage SECOND, return ALWAYS.
"""

import inspect
import traceback
import warnings
from functools import wraps
from typing import Callable

from .context import get_context


class TracePipeWarning(UserWarning):
    """Warning for non-fatal instrumentation issues."""

    pass


class TracePipeError(Exception):
    """Error raised in strict mode."""

    pass


def get_caller_info(skip_frames: int = 2) -> tuple:
    """
    Get caller's file and line number, skipping library frames.

    Walks up the stack to find the first frame that's in user code,
    not in pandas, numpy, or tracepipe internals.

    Args:
        skip_frames: Minimum frames to skip (default 2 for wrapper + this func)

    Returns:
        (filename, line_number) or (None, None)
    """
    import os

    # Get tracepipe package directory to skip it specifically
    tracepipe_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tracepipe_pkg = os.path.join(tracepipe_dir, "tracepipe")

    # Paths/patterns to skip (library code)
    SKIP_PATTERNS = (
        "/pandas/",
        "/numpy/",
        "site-packages",
        "<frozen",
        "<string>",
    )

    try:
        frame = inspect.currentframe()
        # Skip minimum frames first
        for _ in range(skip_frames + 1):
            if frame is None:
                return None, None
            frame = frame.f_back

        # Now walk up until we find user code
        max_depth = 20  # Safety limit
        for _ in range(max_depth):
            if frame is None:
                return None, None

            filename = frame.f_code.co_filename
            abs_filename = os.path.abspath(filename)

            # Check if this is tracepipe package code
            if abs_filename.startswith(tracepipe_pkg):
                frame = frame.f_back
                continue

            # Check if this is other library code
            is_library = any(pattern in filename for pattern in SKIP_PATTERNS)

            if not is_library:
                return filename, frame.f_lineno

            frame = frame.f_back

        return None, None
    except Exception:
        return None, None
    finally:
        del frame


def _make_wrapper(
    method_name: str, original_method: Callable, capture_func: Callable, mode: str = "standard"
) -> Callable:
    """
    Factory for pandas method wrappers with lineage capture.

    CRITICAL: Execute original FIRST, capture lineage SECOND, return ALWAYS.

    Args:
        method_name: Name for error messages
        original_method: The original pandas method
        capture_func: func(self, args, kwargs, result, ctx, method_name)
        mode: "standard", "filter", "inplace", or "transform"
    """

    @wraps(original_method)
    def wrapper(self, *args, **kwargs):
        ctx = get_context()

        # === PRE-EXECUTION SETUP ===
        before_snapshot = None
        is_inplace = kwargs.get("inplace", False)

        if mode == "filter" and ctx.enabled:
            ctx._filter_op_depth += 1
        elif mode == "transform" and ctx.enabled:
            # Suppress __setitem__ recording during transform ops (fillna, replace)
            # to avoid double-counting the same cell change
            ctx._in_transform_op += 1
            # Also handle inplace for transform operations
            if is_inplace:
                try:
                    before_snapshot = self.copy()
                except Exception:
                    pass
        elif mode == "inplace" and ctx.enabled and is_inplace:
            try:
                before_snapshot = self.copy()
            except Exception:
                pass

        # === EXECUTE ORIGINAL (SACRED) ===
        try:
            result = original_method(self, *args, **kwargs)
        finally:
            if mode == "filter" and ctx.enabled:
                ctx._filter_op_depth -= 1
            elif mode == "transform" and ctx.enabled:
                ctx._in_transform_op -= 1

        # === CAPTURE LINEAGE (SIDE EFFECT) ===
        # Skip capture if we're inside a filter operation (prevents recursion during export)
        if ctx.enabled and ctx._filter_op_depth == 0:
            try:
                # Handle inplace for both "inplace" and "transform" modes
                if (mode == "inplace" or mode == "transform") and is_inplace:
                    if before_snapshot is not None:
                        capture_func(before_snapshot, args, kwargs, self, ctx, method_name)
                elif (mode == "inplace" or mode == "transform") and result is not None:
                    capture_func(self, args, kwargs, result, ctx, method_name)
                else:
                    capture_func(self, args, kwargs, result, ctx, method_name)
            except Exception as e:
                if ctx.config.strict_mode:
                    raise TracePipeError(
                        f"Instrumentation failed for {method_name}: {e}\n{traceback.format_exc()}"
                    ) from e
                else:
                    warnings.warn(
                        f"TracePipe: {method_name} instrumentation failed: {e}. "
                        f"Lineage may be incomplete.",
                        TracePipeWarning,
                    )

        # === RETURN RESULT (ALWAYS) ===
        return result

    return wrapper


def wrap_pandas_method(
    method_name: str, original_method: Callable, capture_func: Callable
) -> Callable:
    """Wrap a pandas method with lineage capture."""
    return _make_wrapper(method_name, original_method, capture_func, mode="standard")


def wrap_pandas_filter_method(
    method_name: str, original_method: Callable, capture_func: Callable
) -> Callable:
    """Wrap a pandas filter method (dropna, drop_duplicates, etc.)."""
    return _make_wrapper(method_name, original_method, capture_func, mode="filter")


def wrap_pandas_method_inplace(
    method_name: str, original_method: Callable, capture_func: Callable
) -> Callable:
    """Wrap a pandas method that supports inplace=True."""
    return _make_wrapper(method_name, original_method, capture_func, mode="inplace")


def wrap_pandas_transform_method(
    method_name: str, original_method: Callable, capture_func: Callable
) -> Callable:
    """Wrap a pandas transform method (fillna, replace) that may trigger internal setitem.

    These methods modify column values and pandas internally uses setitem.
    We suppress setitem recording during these ops to avoid double-counting.
    """
    return _make_wrapper(method_name, original_method, capture_func, mode="transform")

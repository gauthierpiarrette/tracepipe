"""
Pandas DataFrame instrumentation for automatic lineage capture.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from tracepipe.core import (
    DataSnapshot,
    OperationType,
    get_code_location,
    get_context,
)

_ORIGINAL_METHODS: Dict[str, Callable] = {}

_HIGH_VALUE_METHODS = [
    "fillna",
    "dropna",
    "drop_duplicates",
    "merge",
    "join",
    "concat",
    "groupby",
    "pivot",
    "pivot_table",
    "melt",
    "apply",
    "transform",
    "agg",
    "aggregate",
    "query",
    "filter",
    "rename",
    "astype",
    "sort_values",
    "sort_index",
    "reset_index",
    "set_index",
    "head",
    "tail",
    "sample",
    "nlargest",
    "nsmallest",
]

_NOISE_METHODS = [
    "__getitem__",
    "mean",
    "sum",
    "min",
    "max",
    "count",
    "std",
    "var",
]

_TRANSFORM_METHODS = [
    "apply",
    "transform",
    "pipe",
    "map",
    "replace",
    "fillna",
    "dropna",
    "drop_duplicates",
    "rename",
    "astype",
    "sort_values",
    "sort_index",
    "reset_index",
    "set_index",
    "melt",
    "pivot",
    "pivot_table",
    "stack",
    "unstack",
    "explode",
    "clip",
    "diff",
    "pct_change",
    "shift",
    "interpolate",
    "bfill",
    "ffill",
    "head",
    "tail",
    "sample",
    "nlargest",
    "nsmallest",
    "assign",
    "copy",
]

_FILTER_METHODS = [
    "query",
    "filter",
    "where",
    "mask",
]

_JOIN_METHODS = [
    "merge",
    "join",
    "concat",
]

_AGGREGATE_METHODS = [
    "groupby",
    "resample",
    "agg",
    "aggregate",
]

def _classify_operation(method_name: str) -> OperationType:
    if method_name in _FILTER_METHODS or method_name.endswith("__getitem__"):
        return OperationType.FILTER
    if method_name in _JOIN_METHODS:
        return OperationType.JOIN
    if method_name in _AGGREGATE_METHODS:
        return OperationType.AGGREGATE
    if method_name in _TRANSFORM_METHODS:
        return OperationType.TRANSFORM
    if method_name == "assign":
        return OperationType.ASSIGN
    if method_name == "copy":
        return OperationType.COPY
    return OperationType.UNKNOWN


def _wrap_method(cls: type, method_name: str, original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        ctx = get_context()
        
        if not ctx.enabled:
            return original(self, *args, **kwargs)
        
        if method_name in _NOISE_METHODS:
            return original(self, *args, **kwargs)
        
        result = original(self, *args, **kwargs)
        
        if isinstance(result, (pd.DataFrame, pd.Series)) and method_name in _HIGH_VALUE_METHODS:
            operation = _classify_operation(method_name)
            params = _extract_params(method_name, args, kwargs)
            
            ctx.graph.add_node(
                operation=operation,
                operation_name=f"DataFrame.{method_name}",
                input_data=self,
                output_data=result,
                parameters=params,
                code_location=get_code_location(depth=3),
            )
        
        return result
    
    return wrapper


def _wrap_setitem(original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapper(self, key, value):
        ctx = get_context()
        input_snapshot = None
        
        if ctx.enabled and isinstance(key, str):
            input_snapshot = DataSnapshot.from_dataframe(self)
        
        original(self, key, value)
        
        if ctx.enabled and isinstance(key, str):
            col_name = key
            is_new = col_name not in (input_snapshot.columns if input_snapshot else [])
            
            op_name = f"Added column '{col_name}'" if is_new else f"Updated column '{col_name}'"
            
            ctx.graph.add_node(
                operation=OperationType.ASSIGN,
                operation_name=op_name,
                input_data=None,
                output_data=self,
                parameters={"column": col_name, "is_new": is_new},
                code_location=get_code_location(depth=3),
            )
    
    return wrapper


def _wrap_init(original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        original(self, *args, **kwargs)
        ctx = get_context()
        
        if ctx.enabled:
            ctx.graph.add_node(
                operation=OperationType.CREATE,
                operation_name="DataFrame.__init__",
                input_data=None,
                output_data=self,
                parameters={"args_count": len(args)},
                code_location=get_code_location(depth=3),
            )
    
    return wrapper


def _extract_params(method_name: str, args: Tuple, kwargs: Dict) -> Dict[str, Any]:
    params = {}
    
    try:
        if method_name == "merge":
            if "on" in kwargs:
                params["on"] = kwargs["on"]
            if "how" in kwargs:
                params["how"] = kwargs["how"]
            if "left_on" in kwargs:
                params["left_on"] = kwargs["left_on"]
            if "right_on" in kwargs:
                params["right_on"] = kwargs["right_on"]
        elif method_name == "groupby":
            if args:
                by = args[0]
                params["by"] = by if isinstance(by, (str, list)) else str(type(by))
            if "by" in kwargs:
                params["by"] = kwargs["by"]
        elif method_name == "sort_values":
            if args:
                params["by"] = args[0]
            if "by" in kwargs:
                params["by"] = kwargs["by"]
            if "ascending" in kwargs:
                params["ascending"] = kwargs["ascending"]
        elif method_name == "fillna":
            if args:
                val = args[0]
                params["value"] = val if not isinstance(val, (pd.DataFrame, pd.Series)) else "<DataFrame>"
            if "value" in kwargs:
                val = kwargs["value"]
                params["value"] = val if not isinstance(val, (pd.DataFrame, pd.Series)) else "<DataFrame>"
            if "method" in kwargs:
                params["method"] = kwargs["method"]
        elif method_name == "dropna":
            if "axis" in kwargs:
                params["axis"] = kwargs["axis"]
            if "how" in kwargs:
                params["how"] = kwargs["how"]
            if "subset" in kwargs:
                params["subset"] = kwargs["subset"]
        elif method_name == "rename":
            if "columns" in kwargs:
                params["columns"] = list(kwargs["columns"].keys()) if isinstance(kwargs["columns"], dict) else str(kwargs["columns"])
        elif method_name == "astype":
            if args:
                params["dtype"] = str(args[0])
            if "dtype" in kwargs:
                params["dtype"] = str(kwargs["dtype"])
        elif method_name == "query":
            if args:
                params["expr"] = args[0]
        elif method_name == "filter":
            if "items" in kwargs:
                params["items"] = kwargs["items"]
            if "like" in kwargs:
                params["like"] = kwargs["like"]
            if "regex" in kwargs:
                params["regex"] = kwargs["regex"]
        elif method_name in ("head", "tail", "sample"):
            if args:
                params["n"] = args[0]
            if "n" in kwargs:
                params["n"] = kwargs["n"]
    except Exception:
        pass
    
    return params


def instrument_pandas():
    ctx = get_context()
    if ctx.is_instrumented("pandas"):
        return
    
    for method_name in _HIGH_VALUE_METHODS:
        if hasattr(pd.DataFrame, method_name):
            original = getattr(pd.DataFrame, method_name)
            if callable(original):
                key = f"DataFrame.{method_name}"
                if key not in _ORIGINAL_METHODS:
                    _ORIGINAL_METHODS[key] = original
                    setattr(pd.DataFrame, method_name, _wrap_method(pd.DataFrame, method_name, original))
    
    series_methods = ["fillna", "dropna", "astype", "copy"]
    for method_name in series_methods:
        if hasattr(pd.Series, method_name):
            original = getattr(pd.Series, method_name)
            if callable(original):
                key = f"Series.{method_name}"
                if key not in _ORIGINAL_METHODS:
                    _ORIGINAL_METHODS[key] = original
                    setattr(pd.Series, method_name, _wrap_method(pd.Series, method_name, original))
    
    if hasattr(pd.DataFrame, "__setitem__"):
        original_setitem = getattr(pd.DataFrame, "__setitem__")
        _ORIGINAL_METHODS["DataFrame.__setitem__"] = original_setitem
        setattr(pd.DataFrame, "__setitem__", _wrap_setitem(original_setitem))
    
    ctx.mark_instrumented("pandas")


def uninstrument_pandas():
    for key, original in _ORIGINAL_METHODS.items():
        parts = key.split(".")
        if len(parts) == 2:
            cls_name, method_name = parts
            cls = pd.DataFrame if cls_name == "DataFrame" else pd.Series
            setattr(cls, method_name, original)
    
    _ORIGINAL_METHODS.clear()
    get_context()._instrumented_modules.discard("pandas")

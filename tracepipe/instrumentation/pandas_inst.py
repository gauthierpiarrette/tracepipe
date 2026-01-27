"""
Pandas DataFrame instrumentation for automatic lineage capture.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from tracepipe.core import (
    LineageGraph,
    OperationType,
    get_code_location,
    get_context,
    get_graph,
)

_ORIGINAL_METHODS: Dict[str, Callable] = {}
_TRANSFORM_METHODS = [
    "apply",
    "transform",
    "pipe",
    "map",
    "applymap",
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
    "round",
    "abs",
    "diff",
    "pct_change",
    "shift",
    "rolling",
    "expanding",
    "ewm",
    "interpolate",
    "bfill",
    "ffill",
    "head",
    "tail",
    "sample",
    "nlargest",
    "nsmallest",
    "assign",
]

_FILTER_METHODS = [
    "query",
    "filter",
    "where",
    "mask",
    "loc.__getitem__",
    "iloc.__getitem__",
    "__getitem__",
]

_JOIN_METHODS = [
    "merge",
    "join",
    "concat",
    "append",
    "combine",
    "combine_first",
    "update",
]

_AGGREGATE_METHODS = [
    "groupby",
    "resample",
    "agg",
    "aggregate",
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "count",
    "std",
    "var",
    "sem",
    "describe",
    "value_counts",
    "nunique",
    "cumsum",
    "cummax",
    "cummin",
    "cumprod",
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
        
        result = original(self, *args, **kwargs)
        
        if isinstance(result, (pd.DataFrame, pd.Series)):
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
        original(self, key, value)
        
        if ctx.enabled:
            ctx.graph.add_node(
                operation=OperationType.ASSIGN,
                operation_name=f"DataFrame.__setitem__[{key}]",
                input_data=None,
                output_data=self,
                parameters={"column": str(key)},
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
    
    all_methods = set(_TRANSFORM_METHODS + _JOIN_METHODS + _AGGREGATE_METHODS)
    all_methods.add("copy")
    all_methods.add("__getitem__")
    
    for method_name in all_methods:
        if hasattr(pd.DataFrame, method_name):
            original = getattr(pd.DataFrame, method_name)
            if callable(original):
                key = f"DataFrame.{method_name}"
                if key not in _ORIGINAL_METHODS:
                    _ORIGINAL_METHODS[key] = original
                    setattr(pd.DataFrame, method_name, _wrap_method(pd.DataFrame, method_name, original))
    
    if "__setitem__" not in _ORIGINAL_METHODS:
        _ORIGINAL_METHODS["DataFrame.__setitem__"] = pd.DataFrame.__setitem__
        pd.DataFrame.__setitem__ = _wrap_setitem(pd.DataFrame.__setitem__)
    
    series_methods = ["apply", "map", "transform", "fillna", "dropna", "replace", "astype", "copy"]
    for method_name in series_methods:
        if hasattr(pd.Series, method_name):
            original = getattr(pd.Series, method_name)
            if callable(original):
                key = f"Series.{method_name}"
                if key not in _ORIGINAL_METHODS:
                    _ORIGINAL_METHODS[key] = original
                    setattr(pd.Series, method_name, _wrap_method(pd.Series, method_name, original))
    
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

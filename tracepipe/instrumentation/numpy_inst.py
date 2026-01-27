"""
NumPy array instrumentation for automatic lineage capture.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from tracepipe.core import (
    OperationType,
    get_code_location,
    get_context,
)

_ORIGINAL_FUNCTIONS: Dict[str, Callable] = {}

_TRACKED_FUNCTIONS = [
    "array",
    "zeros",
    "ones",
    "empty",
    "full",
    "arange",
    "linspace",
    "logspace",
    "eye",
    "identity",
    "diag",
    "concatenate",
    "stack",
    "vstack",
    "hstack",
    "dstack",
    "column_stack",
    "row_stack",
    "split",
    "hsplit",
    "vsplit",
    "dsplit",
    "reshape",
    "ravel",
    "flatten",
    "transpose",
    "swapaxes",
    "expand_dims",
    "squeeze",
    "dot",
    "matmul",
    "inner",
    "outer",
    "tensordot",
    "einsum",
    "where",
    "select",
    "clip",
    "abs",
    "sqrt",
    "exp",
    "log",
    "log10",
    "log2",
    "sin",
    "cos",
    "tan",
    "sum",
    "mean",
    "std",
    "var",
    "min",
    "max",
    "argmin",
    "argmax",
    "sort",
    "argsort",
    "unique",
    "pad",
    "roll",
    "flip",
    "rot90",
    "tile",
    "repeat",
    "copy",
    "astype",
    "asarray",
    "ascontiguousarray",
    "nan_to_num",
    "isnan",
    "isinf",
    "isfinite",
]


def _wrap_numpy_function(func_name: str, original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        ctx = get_context()
        
        if not ctx.enabled:
            return original(*args, **kwargs)
        
        input_arrays = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                input_arrays.append(arg)
        for val in kwargs.values():
            if isinstance(val, np.ndarray):
                input_arrays.append(val)
        
        result = original(*args, **kwargs)
        
        if isinstance(result, np.ndarray):
            input_data = input_arrays[0] if len(input_arrays) == 1 else None
            
            parent_ids = []
            for arr in input_arrays:
                node_id = ctx.graph.find_node_for_data(arr)
                if node_id:
                    parent_ids.append(node_id)
            
            params = {"function": func_name}
            if func_name in ("reshape", "expand_dims", "squeeze"):
                if "shape" in kwargs:
                    params["shape"] = kwargs["shape"]
                elif len(args) > 1:
                    params["shape"] = args[1] if not isinstance(args[1], np.ndarray) else None
            elif func_name in ("concatenate", "stack", "vstack", "hstack"):
                params["num_arrays"] = len(args[0]) if args else 0
            
            ctx.graph.add_node(
                operation=OperationType.NUMPY_OP,
                operation_name=f"numpy.{func_name}",
                input_data=input_data,
                output_data=result,
                parameters=params,
                parent_ids=parent_ids if parent_ids else None,
                code_location=get_code_location(depth=3),
            )
        
        return result
    
    return wrapper


def _wrap_ndarray_method(method_name: str, original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        ctx = get_context()
        
        if not ctx.enabled:
            return original(self, *args, **kwargs)
        
        result = original(self, *args, **kwargs)
        
        if isinstance(result, np.ndarray):
            params = {"method": method_name}
            
            ctx.graph.add_node(
                operation=OperationType.NUMPY_OP,
                operation_name=f"ndarray.{method_name}",
                input_data=self,
                output_data=result,
                parameters=params,
                code_location=get_code_location(depth=3),
            )
        
        return result
    
    return wrapper


_NDARRAY_METHODS = [
    "reshape",
    "transpose",
    "T",
    "flatten",
    "ravel",
    "squeeze",
    "copy",
    "astype",
    "clip",
    "round",
    "sum",
    "mean",
    "std",
    "var",
    "min",
    "max",
    "argmin",
    "argmax",
    "sort",
    "argsort",
]


def instrument_numpy():
    ctx = get_context()
    if ctx.is_instrumented("numpy"):
        return
    
    for func_name in _TRACKED_FUNCTIONS:
        if hasattr(np, func_name):
            original = getattr(np, func_name)
            if callable(original):
                key = f"numpy.{func_name}"
                if key not in _ORIGINAL_FUNCTIONS:
                    _ORIGINAL_FUNCTIONS[key] = original
                    setattr(np, func_name, _wrap_numpy_function(func_name, original))
    
    for method_name in _NDARRAY_METHODS:
        if hasattr(np.ndarray, method_name):
            original = getattr(np.ndarray, method_name)
            if callable(original) and not isinstance(original, property):
                key = f"ndarray.{method_name}"
                if key not in _ORIGINAL_FUNCTIONS:
                    _ORIGINAL_FUNCTIONS[key] = original
                    try:
                        setattr(np.ndarray, method_name, _wrap_ndarray_method(method_name, original))
                    except (TypeError, AttributeError):
                        pass
    
    ctx.mark_instrumented("numpy")


def uninstrument_numpy():
    for key, original in _ORIGINAL_FUNCTIONS.items():
        if key.startswith("numpy."):
            func_name = key.split(".")[1]
            setattr(np, func_name, original)
        elif key.startswith("ndarray."):
            method_name = key.split(".")[1]
            try:
                setattr(np.ndarray, method_name, original)
            except (TypeError, AttributeError):
                pass
    
    _ORIGINAL_FUNCTIONS.clear()
    get_context()._instrumented_modules.discard("numpy")

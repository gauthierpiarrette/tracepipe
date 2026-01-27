"""
Instrumentation modules for automatic lineage capture.
"""
from tracepipe.instrumentation.pandas_inst import instrument_pandas
from tracepipe.instrumentation.numpy_inst import instrument_numpy
from tracepipe.instrumentation.sklearn_inst import instrument_sklearn

__all__ = ["instrument_pandas", "instrument_numpy", "instrument_sklearn"]

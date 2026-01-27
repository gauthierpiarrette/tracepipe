"""
TracePipe: Automatic data lineage & debugging for ML pipelines.

Usage:
    import tracepipe
    tracepipe.enable()  # Instruments pandas, numpy, sklearn automatically
    
    # Your ML pipeline code here...
    df = pd.read_csv("data.csv")
    df = df.fillna(0)
    
    # Explain what happened
    lineage = tracepipe.explain(df)
    lineage.show()  # Opens interactive HTML visualization
"""
from tracepipe.core import (
    LineageGraph,
    LineageNode,
    DataSnapshot,
    OperationType,
    get_context,
    get_graph,
    stage,
)
from tracepipe.api import (
    explain,
    get_lineage,
    summary,
    print_summary,
    LineageResult,
)
from tracepipe.export import (
    export_to_json,
    export_to_openlineage,
    LineageExporter,
)
from tracepipe.visualization import (
    LineageVisualizer,
    render_lineage_html,
)

__version__ = "1.0.0"
__all__ = [
    "enable",
    "disable",
    "reset",
    "stage",
    "explain",
    "get_lineage",
    "summary",
    "print_summary",
    "export_to_json",
    "export_to_openlineage",
    "LineageResult",
    "LineageGraph",
    "LineageNode",
    "DataSnapshot",
    "LineageVisualizer",
    "LineageExporter",
    "OperationType",
    "get_graph",
]


def enable(
    pandas: bool = True,
    numpy: bool = True,
    sklearn: bool = True,
) -> None:
    """
    Enable tracepipe lineage tracking.
    
    This instruments pandas, numpy, and sklearn (if available) to automatically
    capture data transformations. Call this at the start of your script.
    
    Args:
        pandas: Instrument pandas DataFrame operations (default: True)
        numpy: Instrument numpy array operations (default: True)
        sklearn: Instrument sklearn transformers and models (default: True)
    
    Example:
        >>> import tracepipe
        >>> tracepipe.enable()
        >>> # Now all DataFrame operations are tracked
        >>> df = pd.DataFrame({"a": [1, 2, 3]})
        >>> df = df.fillna(0)
    """
    ctx = get_context()
    ctx.enable()
    
    if pandas:
        from tracepipe.instrumentation import instrument_pandas
        instrument_pandas()
    
    if numpy:
        from tracepipe.instrumentation import instrument_numpy
        instrument_numpy()
    
    if sklearn:
        try:
            from tracepipe.instrumentation import instrument_sklearn
            instrument_sklearn()
        except ImportError:
            pass


def disable() -> None:
    """
    Disable tracepipe lineage tracking.
    
    This stops capturing new operations but preserves the existing lineage graph.
    Use reset() to also clear the graph.
    """
    get_context().disable()


def reset() -> None:
    """
    Reset tracepipe state.
    
    This clears all captured lineage data and resets the stage stack.
    Does not uninstrument pandas/numpy/sklearn.
    """
    get_context().reset()


def is_enabled() -> bool:
    """Check if tracepipe lineage tracking is currently enabled."""
    return get_context().enabled

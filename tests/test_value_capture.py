# tests/test_value_capture.py
"""
Tests for tracepipe/utils/value_capture.py - Value capture utilities.
"""

import numpy as np
import pandas as pd

import tracepipe


class TestValueCapture:
    """Tests for value capture utilities."""

    def test_captures_various_types(self):
        """Value capture handles various Python types."""
        tracepipe.enable()
        tracepipe.watch("a")

        df = pd.DataFrame({"a": [1, "string", 3.14, True]})
        df["a"] = [10, "new_string", 31.4, False]

        row = tracepipe.explain(0)
        history = row.cell_history("a")
        assert len(history) >= 1

    def test_captures_numpy_types(self):
        """Value capture handles numpy types."""
        tracepipe.enable()
        tracepipe.watch("a")

        df = pd.DataFrame({"a": np.array([1, 2, 3], dtype=np.int64)})
        df["a"] = df["a"] + 10

        row = tracepipe.explain(0)
        history = row.cell_history("a")
        assert len(history) >= 1

    def test_captures_none(self):
        """Value capture handles None correctly."""
        tracepipe.enable()
        tracepipe.watch("a")

        df = pd.DataFrame({"a": [1, None, 3]})
        df["a"] = df["a"].fillna(0)

        row = tracepipe.explain(1)
        history = row.cell_history("a")
        assert len(history) >= 1

    def test_captures_nan(self):
        """Value capture handles NaN correctly."""
        tracepipe.enable()
        tracepipe.watch("a")

        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        df["a"] = df["a"].fillna(0)

        row = tracepipe.explain(1)
        history = row.cell_history("a")
        assert len(history) >= 1

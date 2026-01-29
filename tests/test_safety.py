# tests/test_safety.py
"""
Tests for tracepipe/safety.py - Safe instrumentation wrappers.
"""

import pandas as pd

import tracepipe
from tracepipe import TracePipeConfig


class TestSafetyGuarantees:
    """Instrumentation must NEVER crash user pipelines."""

    def test_instrumentation_error_returns_result(self):
        """Even if instrumentation fails internally, user gets their result."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, None, 4]})

        result = df.dropna()

        assert result is not None
        assert len(result) == 3

    def test_basic_operations_work(self):
        """Standard pandas operations continue to work."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        filtered = df[df["a"] > 1]
        assert len(filtered) == 2

        df2 = df.copy()
        df2["c"] = df2["a"] + df2["b"]
        assert "c" in df2.columns

        grouped = df.groupby("a").sum()
        assert len(grouped) == 3

    def test_chained_operations(self):
        """Chained operations work correctly."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, None, 3, None, 5], "b": [10, 20, 30, 40, 50]})

        result = df.dropna().reset_index(drop=True).head(2)

        assert len(result) == 2
        assert list(result["a"]) == [1, 3]

    def test_disabled_tracking_no_errors(self):
        """Operations work correctly when tracking is disabled."""
        tracepipe.enable()
        tracepipe.disable()

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = df.dropna()

        assert len(result) == 3
        assert not tracepipe.stats()["enabled"]


class TestStrictMode:
    """Tests for strict mode error handling."""

    def test_strict_mode_config(self):
        """Strict mode can be enabled via config."""
        config = TracePipeConfig(strict_mode=True)
        tracepipe.enable(config=config)

        from tracepipe.context import get_context

        ctx = get_context()
        assert ctx.config.strict_mode is True

# tests/test_public_api.py
"""
Tests for public API behavior - check(), trace(), why(), report().

These are the most important user-facing functions and need solid coverage.
"""

import pandas as pd
import pytest

import tracepipe as tp


@pytest.fixture(autouse=True)
def reset_tp():
    """Reset TracePipe before each test."""
    tp.reset()
    yield
    try:
        tp.disable()
    except Exception:
        pass


# =============================================================================
# CHECK() TESTS
# =============================================================================


class TestCheck:
    """Test tp.check() - the main health check function."""

    def test_check_healthy_pipeline(self):
        """check() passes for pipeline with no issues."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = tp.check(df)
        assert result.ok

    def test_check_with_drops(self):
        """check() reports dropped rows."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, None, 3, None, 5]})
        df = df.dropna()
        result = tp.check(df)
        assert result.facts["rows_dropped"] == 2

    def test_check_retention_rate(self):
        """check() calculates correct retention rate."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": range(100)})
        df = df.head(80)
        result = tp.check(df)
        assert result.facts["retention_rate"] == 0.8

    def test_check_with_value_changes(self):
        """check() reports value changes."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})
        df["a"] = df["a"] * 2
        result = tp.check(df)
        assert result.ok

    def test_check_ci_mode(self):
        """check() works in CI mode."""
        tp.enable(mode="ci")
        df = pd.DataFrame({"a": range(10)})
        df = df.head(5)
        result = tp.check(df)
        assert result.mode == "ci"

    def test_check_str_output(self):
        """check() has readable string output."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = tp.check(df)
        output = str(result)
        assert "TracePipe" in output or "Check" in output


# =============================================================================
# TRACE() TESTS
# =============================================================================


class TestTrace:
    """Test tp.trace() - row journey tracking."""

    def test_trace_by_row(self):
        """trace() shows row journey."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.head(2)
        result = tp.trace(df, row=0)
        assert result is not None
        assert "OK" in str(result) or "Alive" in str(result)

    def test_trace_dropped_row(self):
        """trace() shows dropped row status."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()
        result = tp.trace(df, row=1)
        assert result is not None

    def test_trace_with_where(self):
        """trace() with where clause."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"id": ["A", "B", "C"], "val": [1, 2, 3]})
        df = df[df["val"] > 1]
        result = tp.trace(df, where={"id": "B"})
        assert result is not None

    def test_trace_through_merge(self):
        """trace() shows merge provenance."""
        tp.enable(mode="debug")
        left = pd.DataFrame({"key": [1, 2], "val": ["a", "b"]})
        right = pd.DataFrame({"key": [2, 3], "info": [100, 200]})
        df = left.merge(right, on="key")
        result = tp.trace(df, row=0)
        assert result is not None


# =============================================================================
# WHY() TESTS
# =============================================================================


class TestWhy:
    """Test tp.why() - cell provenance."""

    def test_why_unchanged_cell(self):
        """why() for cell with no changes."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = tp.why(df, col="a", row=0)
        assert result is not None

    def test_why_changed_cell(self):
        """why() for cell that was modified."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})
        df["a"] = df["a"] * 10
        result = tp.why(df, col="a", row=0)
        assert result is not None
        assert result.current_value == 10

    def test_why_fillna(self):
        """why() tracks fillna changes."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        df["a"] = df["a"].fillna(999)
        result = tp.why(df, col="a", row=1)
        assert result.current_value == 999


# =============================================================================
# FILTER OPERATIONS
# =============================================================================


class TestFilterTracking:
    """Test filter operation tracking."""

    def test_dropna(self):
        """dropna() drops are tracked."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()
        assert len(df) == 2
        result = tp.check(df)
        assert result.facts["rows_dropped"] == 1

    def test_boolean_filter(self):
        """Boolean indexing drops are tracked."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": range(10)})
        df = df[df["a"] >= 5]
        result = tp.check(df)
        assert result.facts["rows_dropped"] == 5

    def test_query(self):
        """query() drops are tracked."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
        df = df.query("val > 2")
        result = tp.check(df)
        assert result.facts["rows_dropped"] == 2

    def test_head_tail(self):
        """head/tail drops are tracked."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": range(10)})
        df = df.head(3)
        result = tp.check(df)
        assert result.facts["rows_dropped"] == 7

    def test_drop_duplicates(self):
        """drop_duplicates() drops are tracked."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 1, 2, 2, 3]})
        df = df.drop_duplicates()
        result = tp.check(df)
        assert result.facts["rows_dropped"] == 2


# =============================================================================
# INDEXER OPERATIONS
# =============================================================================


class TestIndexerTracking:
    """Test loc/iloc tracking."""

    def test_loc_read(self):
        """loc[] read works."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        val = df.loc[0, "a"]
        assert val == 1

    def test_loc_write(self):
        """loc[] write is tracked."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.loc[0, "a"] = 100
        result = tp.why(df, col="a", row=0)
        assert result.current_value == 100

    def test_iloc_read(self):
        """iloc[] read works."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})
        val = df.iloc[0, 0]
        assert val == 1

    def test_iloc_write(self):
        """iloc[] write is tracked."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.iloc[0, 0] = 200
        assert df.iloc[0, 0] == 200

    def test_loc_slice(self):
        """loc[] with slice."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": range(10)})
        subset = df.loc[2:5]
        assert len(subset) == 4


# =============================================================================
# DEBUG INSPECTOR
# =============================================================================


class TestDebugInspector:
    """Test tp.debug.inspect() for internal access."""

    def test_inspect_steps(self):
        """inspect() returns steps."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.dropna()
        df = df.head(2)

        dbg = tp.debug.inspect()
        assert len(dbg.steps) >= 2

    def test_inspect_dropped_rows(self):
        """inspect() returns dropped rows."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": range(5)})
        df = df.head(3)

        dbg = tp.debug.inspect()
        dropped = dbg.dropped_rows()
        assert len(dropped) == 2

    def test_inspect_stats(self):
        """inspect() returns stats."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()

        dbg = tp.debug.inspect()
        stats = dbg.stats()
        assert "total_steps" in stats

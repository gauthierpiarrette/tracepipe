# tests/test_integration.py
"""
Integration tests for TracePipe - End-to-end scenarios and edge cases.
"""

import pandas as pd

import tracepipe


def dbg():
    """Helper to access debug inspector."""
    return tracepipe.debug.inspect()


class TestComplexPipelines:
    """Tests for complex multi-step pipelines."""

    def test_etl_pipeline(self):
        """Typical ETL pipeline works correctly."""
        tracepipe.enable(watch=["amount"])

        # Extract
        with tracepipe.stage("extract"):
            df = pd.DataFrame(
                {
                    "id": [1, 2, 3, 4, 5],
                    "amount": [100.0, None, 300.0, None, 500.0],
                    "category": ["A", "B", "A", "B", "A"],
                }
            )

        # Transform
        with tracepipe.stage("transform"):
            df = df.dropna()
            df["amount"] = df["amount"] * 1.1

        # Load (simulated)
        with tracepipe.stage("load"):
            _result = df.copy()

        # Verify tracking
        stats = dbg().stats()
        assert stats["total_steps"] >= 2  # dropna + setitem at minimum

        dropped = dbg().dropped_rows()
        assert len(dropped) == 2

        steps = dbg().steps
        transform_steps = [s for s in steps if s.stage == "transform"]
        assert len(transform_steps) >= 1  # dropna and setitem happen in transform

    def test_aggregation_pipeline(self):
        """Aggregation pipeline with group tracking."""
        tracepipe.enable()

        df = pd.DataFrame(
            {
                "region": ["East", "West", "East", "West", "East"],
                "sales": [100, 200, 150, 250, 300],
            }
        )

        _summary = df.groupby("region").agg({"sales": ["sum", "mean"]})

        # Verify group membership
        east_group = dbg().explain_group("East")
        assert east_group.row_count == 3

        west_group = dbg().explain_group("West")
        assert west_group.row_count == 2


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_dataframe(self):
        """Empty DataFrames don't cause errors."""
        tracepipe.enable()
        df = pd.DataFrame({"a": []})

        result = df.dropna()

        assert len(result) == 0
        assert dbg().stats()["enabled"]

    def test_single_row_dataframe(self):
        """Single row DataFrames work correctly."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1]})

        _result = df.head(1)

        row = dbg().explain_row(0)
        assert row is not None
        assert row.is_alive

    def test_special_column_names(self):
        """Columns with special characters work."""
        tracepipe.enable(watch=["col with spaces", "col/slash"])

        df = pd.DataFrame({"col with spaces": [1, 2, 3], "col/slash": [4, 5, 6]})

        df["col with spaces"] = df["col with spaces"] * 2

        steps = dbg().steps
        assert len(steps) >= 1

    def test_large_dataframe(self):
        """Large DataFrames don't cause excessive memory usage."""
        tracepipe.enable()
        tracepipe.configure(max_diffs_per_step=1000)

        df = pd.DataFrame({"a": range(100_000)})
        df = df.head(50_000)

        # Should complete without error
        stats = dbg().stats()
        assert stats["enabled"]

    def test_deeply_chained_operations(self):
        """Deeply chained operations maintain tracking."""
        tracepipe.enable()

        df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        _result = (
            df.query("a > 2")
            .reset_index(drop=True)
            .head(5)
            .copy()
            .sort_values("a")
            .reset_index(drop=True)
        )

        # Tracking should still work
        stats = dbg().stats()
        assert stats["total_steps"] >= 2  # At least some steps tracked


class TestConcurrencySimulation:
    """Tests simulating concurrent-like patterns."""

    def test_multiple_dataframes(self):
        """Multiple DataFrames can be tracked simultaneously."""
        tracepipe.enable()

        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [4, 5, 6]})
        df3 = pd.DataFrame({"c": [7, 8, 9]})

        df1 = df1.head(2)
        df2 = df2.head(2)
        df3 = df3.head(2)

        dropped = dbg().dropped_rows()

        # Each df dropped 1 row
        assert len(dropped) == 3

    def test_reuse_dataframe_name(self):
        """Reusing variable name works correctly."""
        tracepipe.enable()

        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.head(2)

        df = pd.DataFrame({"a": [10, 20, 30, 40]})
        df = df.head(2)

        # Both operations should be tracked
        stats = dbg().stats()
        assert stats["total_steps"] >= 2


class TestErrorRecovery:
    """Tests for error handling and recovery."""

    def test_invalid_explain_row(self):
        """explain_row() with invalid row ID returns result."""
        tracepipe.enable()
        _df = pd.DataFrame({"a": [1, 2, 3]})

        # Row ID that was never created
        row = dbg().explain_row(999)

        # Should return a result (possibly with empty history)
        assert row is not None

    def test_invalid_explain_group(self):
        """explain_group() with invalid key returns result."""
        tracepipe.enable()
        df = pd.DataFrame({"cat": ["A"], "val": [1]})
        df.groupby("cat").sum()

        # Group that doesn't exist
        group = dbg().explain_group("Z")

        # Should return a result (possibly with no members)
        assert group is not None

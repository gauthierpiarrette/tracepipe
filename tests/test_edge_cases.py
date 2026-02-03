# tests/test_edge_cases.py
"""
Edge case tests and stress tests.

Covers:
- Unusual DataFrame structures (MultiIndex, duplicate index, etc.)
- Boundary conditions (empty, single row, very large)
- Error recovery scenarios
- Memory and performance edge cases
- Concurrent/sequential operations
"""

import gc

import numpy as np
import pandas as pd
import pytest

import tracepipe as tp

# =============================================================================
# UNUSUAL DATAFRAME STRUCTURES
# =============================================================================


class TestMultiIndexDataFrames:
    """Test DataFrames with MultiIndex."""

    def test_multiindex_rows(self):
        """DataFrame with MultiIndex rows is tracked."""
        tp.enable(mode="debug")

        arrays = [
            ["A", "A", "B", "B"],
            [1, 2, 1, 2],
        ]
        index = pd.MultiIndex.from_arrays(arrays, names=["letter", "number"])
        df = pd.DataFrame({"value": [10, 20, 30, 40]}, index=index)

        df = df[df["value"] > 15]
        result = tp.check(df)

        assert result.facts["rows_dropped"] == 1

    def test_multiindex_columns(self):
        """DataFrame with MultiIndex columns is tracked."""
        tp.enable(mode="debug")

        arrays = [
            ["A", "A", "B", "B"],
            ["one", "two", "one", "two"],
        ]
        columns = pd.MultiIndex.from_arrays(arrays)
        df = pd.DataFrame(
            np.random.randn(3, 4),
            columns=columns,
        )

        df = df.iloc[:2]
        result = tp.check(df)

        assert result.facts["rows_dropped"] == 1

    def test_multiindex_both(self):
        """DataFrame with MultiIndex on both axes."""
        tp.enable(mode="debug")

        row_idx = pd.MultiIndex.from_product([["A", "B"], [1, 2]])
        col_idx = pd.MultiIndex.from_product([["X", "Y"], ["a", "b"]])
        df = pd.DataFrame(
            np.random.randn(4, 4),
            index=row_idx,
            columns=col_idx,
        )

        df = df.iloc[:2]
        result = tp.check(df)

        assert result.facts["rows_dropped"] == 2


class TestDuplicateIndex:
    """Test DataFrames with duplicate index values."""

    def test_duplicate_index_filter(self):
        """Filtering with duplicate index works."""
        tp.enable(mode="debug")

        df = pd.DataFrame(
            {"value": [1, 2, 3, 4, 5]},
            index=[0, 0, 1, 1, 2],  # Duplicate indices
        )

        df = df[df["value"] > 2]
        result = tp.check(df)

        assert result.facts["rows_dropped"] == 2

    def test_duplicate_index_transform(self):
        """Transform with duplicate index works."""
        tp.enable(mode="debug", watch=["value"])

        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=["a", "a", "b"],
        )

        df["value"] = df["value"] * 2
        result = tp.check(df)

        # Transform doesn't drop rows
        assert result.facts["rows_dropped"] == 0


class TestSpecialValues:
    """Test handling of special values."""

    def test_inf_values(self):
        """DataFrame with inf values is tracked."""
        tp.enable(mode="debug")

        df = pd.DataFrame({"a": [1.0, np.inf, -np.inf, 4.0]})
        df = df[np.isfinite(df["a"])]

        result = tp.check(df)

        assert result.facts["rows_dropped"] == 2

    def test_nan_variations(self):
        """Different NaN representations are handled."""
        tp.enable(mode="debug")

        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, None, pd.NA if hasattr(pd, "NA") else np.nan],
            }
        )

        original_len = len(df)
        df = df.dropna()
        result = tp.check(df)

        assert result.facts["rows_dropped"] == original_len - len(df)

    def test_empty_strings(self):
        """Empty strings vs None distinction."""
        tp.enable(mode="debug")

        df = pd.DataFrame({"a": ["hello", "", None, "world"]})
        df = df.dropna()

        result = tp.check(df)

        # Only None is dropped, not empty string
        assert result.facts["rows_dropped"] == 1


# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================


class TestBoundaryConditions:
    """Test boundary conditions."""

    def test_empty_dataframe_operations(self):
        """Operations on empty DataFrame don't crash."""
        tp.enable(mode="debug")

        df = pd.DataFrame({"a": []})

        # These should all work without error
        df = df.dropna()
        df = df[df["a"] > 0] if len(df) > 0 else df

        result = tp.check(df)
        assert result.facts["rows_dropped"] == 0

    def test_single_row_all_operations(self):
        """All operations work on single-row DataFrame."""
        tp.enable(mode="debug", watch=["value"])

        df = pd.DataFrame({"id": [1], "value": [100.0]})

        # Filter (keeps row)
        df = df[df["value"] > 50]
        # Transform
        df["value"] = df["value"] * 2
        # Sort (no-op but should work)
        df = df.sort_values("value")

        result = tp.check(df)
        assert len(df) == 1

    def test_single_column_dataframe(self):
        """Single column DataFrame works."""
        tp.enable(mode="debug")

        df = pd.DataFrame({"only_col": [1, 2, 3, 4, 5]})
        df = df[df["only_col"] > 2]

        result = tp.check(df)
        assert result.facts["rows_dropped"] == 2

    def test_wide_dataframe(self):
        """Wide DataFrame (many columns) works."""
        tp.enable(mode="debug")

        # 100 columns
        data = {f"col_{i}": range(10) for i in range(100)}
        df = pd.DataFrame(data)

        df = df.iloc[:5]
        result = tp.check(df)

        assert result.facts["rows_dropped"] == 5

    def test_filter_to_empty(self):
        """Filter that results in empty DataFrame."""
        tp.enable(mode="debug")

        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df[df["a"] > 100]  # All rows filtered

        result = tp.check(df)
        assert result.facts["rows_dropped"] == 3
        assert len(df) == 0


# =============================================================================
# STRESS TESTS
# =============================================================================


class TestStress:
    """Stress tests for performance and memory."""

    @pytest.mark.slow
    def test_large_dataframe_100k(self):
        """100K row DataFrame works correctly."""
        tp.enable(mode="ci")

        df = pd.DataFrame(
            {
                "id": range(100_000),
                "value": np.random.randn(100_000),
            }
        )

        df = df[df["value"] > 0]
        result = tp.check(df)

        # Roughly half should remain
        assert 0.4 < result.facts["retention_rate"] < 0.6

    @pytest.mark.slow
    def test_many_operations(self):
        """Many sequential operations are tracked."""
        tp.enable(mode="debug")

        df = pd.DataFrame({"a": range(1000)})

        # 50 filter operations
        for i in range(50):
            if len(df) > 10:
                df = df.iloc[:-1]

        result = tp.check(df)
        assert result.facts["rows_dropped"] == 50

    @pytest.mark.slow
    def test_chained_merges(self):
        """Multiple chained merges work."""
        tp.enable(mode="debug")

        df1 = pd.DataFrame({"key": range(100), "v1": range(100)})
        df2 = pd.DataFrame({"key": range(50, 150), "v2": range(100)})
        df3 = pd.DataFrame({"key": range(75, 125), "v3": range(50)})

        result = df1.merge(df2, on="key").merge(df3, on="key")

        check_result = tp.check(result)
        assert check_result is not None

    def test_repeated_enable_disable(self):
        """Repeated enable/disable cycles work."""
        for _ in range(10):
            tp.enable(mode="debug")
            df = pd.DataFrame({"a": [1, 2, 3]})
            df = df.dropna()
            tp.check(df)
            tp.disable()
            tp.reset()

        # Should complete without error

    def test_memory_cleanup_on_reset(self):
        """Reset clears memory appropriately."""
        tp.enable(mode="debug", watch=["value"])

        # Create some data
        for _ in range(10):
            df = pd.DataFrame({"value": range(1000)})
            df["value"] = df["value"] * 2

        tp.reset()
        gc.collect()

        # After reset, new operations should work
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = tp.check(df)
        assert result.facts["rows_dropped"] == 0


# =============================================================================
# ERROR RECOVERY
# =============================================================================


class TestErrorRecovery:
    """Test error recovery scenarios."""

    def test_tracking_continues_after_error_in_user_code(self):
        """Tracking continues after user code error."""
        tp.enable(mode="debug")

        df = pd.DataFrame({"a": [1, 2, 3]})

        try:
            # This will raise an error
            _ = df["nonexistent_column"]
        except KeyError:
            pass

        # Tracking should still work
        df = df[df["a"] > 1]
        result = tp.check(df)

        assert result.facts["rows_dropped"] == 1

    def test_disable_after_error(self):
        """Disable works even after errors."""
        tp.enable(mode="debug")

        try:
            df = pd.DataFrame({"a": [1, 2, 3]})
            df = df[df["nonexistent"] > 0]
        except KeyError:
            pass

        # Should not raise
        tp.disable()

    def test_reset_after_error(self):
        """Reset works even after errors."""
        tp.enable(mode="debug")

        try:
            df = pd.DataFrame({"a": [1, 2, 3]})
            df.loc[999, "a"] = 1  # May cause issues depending on version
        except Exception:
            pass

        # Should not raise
        tp.reset()


# =============================================================================
# SEQUENTIAL OPERATIONS
# =============================================================================


class TestSequentialOperations:
    """Test sequences of operations."""

    def test_filter_transform_filter(self):
        """Filter-transform-filter sequence."""
        tp.enable(mode="debug", watch=["value"])

        df = pd.DataFrame({"value": range(100)})

        df = df[df["value"] > 20]  # Keep 79 rows
        df["value"] = df["value"] * 2  # Transform
        df = df[df["value"] < 100]  # Filter more

        result = tp.check(df)
        assert result.facts["rows_dropped"] > 20

    def test_transform_merge_filter(self):
        """Transform-merge-filter sequence."""
        tp.enable(mode="debug")

        df1 = pd.DataFrame({"key": [1, 2, 3], "val": [10, 20, 30]})
        df2 = pd.DataFrame({"key": [2, 3, 4], "extra": ["a", "b", "c"]})

        df1["val"] = df1["val"] * 2
        merged = df1.merge(df2, on="key")
        result_df = merged[merged["val"] > 30]

        result = tp.check(result_df)
        assert result is not None

    def test_groupby_filter_transform(self):
        """GroupBy-filter-transform sequence."""
        tp.enable(mode="debug")

        df = pd.DataFrame(
            {
                "cat": ["A", "A", "B", "B", "C"],
                "val": [1, 2, 3, 4, 5],
            }
        )

        # GroupBy
        grouped = df.groupby("cat")["val"].sum().reset_index()
        # Filter
        grouped = grouped[grouped["val"] > 2]
        # Transform
        grouped["val"] = grouped["val"] * 10

        result = tp.check(grouped)
        assert len(grouped) >= 2


# =============================================================================
# COPY BEHAVIOR
# =============================================================================


class TestCopyBehavior:
    """Test copy-related behavior."""

    def test_explicit_copy(self):
        """Explicit copy is tracked."""
        tp.enable(mode="debug")

        df = pd.DataFrame({"a": [1, 2, 3]})
        df_copy = df.copy()

        df_copy = df_copy[df_copy["a"] > 1]

        result = tp.check(df_copy)
        assert result.facts["rows_dropped"] == 1

    def test_slice_behavior(self):
        """Slice behavior is tracked correctly."""
        tp.enable(mode="debug")

        df = pd.DataFrame({"a": range(10)})
        df_slice = df[df["a"] > 5]

        result = tp.check(df_slice)
        assert result.facts["rows_dropped"] == 6

    def test_inplace_operations(self):
        """Inplace operations are tracked."""
        tp.enable(mode="debug", watch=["value"])

        df = pd.DataFrame({"value": [1.0, None, 3.0]})
        df.fillna(0, inplace=True)

        result = tp.check(df)
        # Inplace fillna should still be tracked
        assert result is not None


# =============================================================================
# SPECIAL COLUMN NAMES
# =============================================================================


class TestSpecialColumnNames:
    """Test handling of special column names."""

    def test_column_with_spaces(self):
        """Columns with spaces work."""
        tp.enable(mode="debug", watch=["my column"])

        df = pd.DataFrame({"my column": [1, 2, 3]})
        df["my column"] = df["my column"] * 2

        result = tp.check(df)
        assert result is not None

    def test_column_with_special_chars(self):
        """Columns with special characters work."""
        tp.enable(mode="debug")

        df = pd.DataFrame({"col.with" + ".dots": [1, 2, 3]})
        df = df[df["col.with.dots"] > 1]

        result = tp.check(df)
        assert result.facts["rows_dropped"] == 1

    def test_numeric_column_names(self):
        """Numeric column names work."""
        tp.enable(mode="debug")

        df = pd.DataFrame({0: [1, 2, 3], 1: [4, 5, 6]})
        df = df[df[0] > 1]

        result = tp.check(df)
        assert result.facts["rows_dropped"] == 1

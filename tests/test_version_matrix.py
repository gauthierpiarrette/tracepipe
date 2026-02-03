# tests/test_version_matrix.py
"""
Tests for pandas version-specific behavior and compatibility.

These tests ensure TracePipe works correctly across all supported pandas versions:
- pandas 1.5.x
- pandas 2.0.x
- pandas 2.1.x
- pandas 2.2.x

Version-gated tests use skipif markers; version-agnostic tests should pass on all versions.
"""

import pandas as pd
import pytest

import tracepipe as tp
from tests.conftest import PANDAS_VERSION, PANDAS_VERSION_STR

# =============================================================================
# VERSION-AGNOSTIC TESTS (must pass on ALL pandas versions)
# =============================================================================


class TestAllVersionsBasicOps:
    """Basic operations that must work identically on all pandas versions."""

    @pytest.mark.parametrize("dtype", ["int64", "float64", "object"])
    def test_dropna_all_dtypes(self, dtype):
        """dropna() tracking works for all dtypes."""
        tp.enable(mode="debug")

        if dtype == "int64":
            # Use nullable int for null support
            df = pd.DataFrame({"a": pd.array([1, None, 3], dtype="Int64")})
        elif dtype == "float64":
            df = pd.DataFrame({"a": [1.0, None, 3.0]})
        else:
            df = pd.DataFrame({"a": ["x", None, "z"]})

        df = df.dropna()
        result = tp.check(df)

        assert result.facts["rows_dropped"] == 1

    @pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
    def test_merge_all_join_types(self, how):
        """merge() tracking works for all join types."""
        tp.enable(mode="debug")

        left = pd.DataFrame({"key": [1, 2], "val": ["a", "b"]})
        right = pd.DataFrame({"key": [2, 3], "val2": ["x", "y"]})

        result_df = left.merge(right, on="key", how=how)
        result = tp.check(result_df)

        # Just verify it works without error
        assert result is not None

    @pytest.mark.parametrize("agg", ["sum", "mean", "count", "min", "max"])
    def test_groupby_all_aggregations(self, agg):
        """groupby() tracking works for all aggregation types."""
        tp.enable(mode="debug")

        df = pd.DataFrame(
            {
                "cat": ["A", "A", "B", "B"],
                "val": [1.0, 2.0, 3.0, 4.0],
            }
        )

        result_df = getattr(df.groupby("cat"), agg)()

        # Just verify it works without error
        assert len(result_df) == 2


class TestAllVersionsFilters:
    """Filter operations that must work on all pandas versions."""

    def test_boolean_indexing(self):
        """Boolean indexing works on all versions."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": range(10)})

        df = df[df["a"] > 5]
        result = tp.check(df)

        assert result.facts["rows_dropped"] == 6

    def test_query_string(self):
        """query() with string works on all versions."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": range(10)})

        df = df.query("a > 5")
        result = tp.check(df)

        assert result.facts["rows_dropped"] == 6

    def test_isin_filter(self):
        """isin() filter works on all versions."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        df = df[df["a"].isin([2, 4])]
        result = tp.check(df)

        assert result.facts["rows_dropped"] == 3

    def test_head_tail_sample(self):
        """head/tail/sample work on all versions."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": range(100)})

        df_head = df.head(10)
        df_tail = df.tail(10)
        df_sample = df.sample(n=10, random_state=42)

        assert len(df_head) == 10
        assert len(df_tail) == 10
        assert len(df_sample) == 10


class TestAllVersionsTransforms:
    """Transform operations that must work on all pandas versions."""

    def test_fillna_scalar(self):
        """fillna() with scalar works on all versions."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1.0, None, 3.0]})

        df["a"] = df["a"].fillna(0)

        assert df["a"].isna().sum() == 0

    def test_replace_value(self):
        """replace() with value works on all versions."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})

        df["a"] = df["a"].replace(2, 999)

        assert 999 in df["a"].values

    def test_astype_conversion(self):
        """astype() works on all versions."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})

        df["a"] = df["a"].astype(float)

        assert df["a"].dtype == float


# =============================================================================
# PANDAS 1.5 SPECIFIC TESTS
# =============================================================================


@pytest.mark.skipif(PANDAS_VERSION >= (2, 0), reason="pandas 1.5 specific")
class TestPandas15Specific:
    """Tests for pandas 1.5.x specific behavior."""

    def test_basic_operations_15(self):
        """Basic operations work on pandas 1.5."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df[df["a"] > 1]

        result = tp.check(df)
        assert result.facts["rows_dropped"] == 1


# =============================================================================
# PANDAS 2.0+ SPECIFIC TESTS
# =============================================================================


@pytest.mark.skipif(PANDAS_VERSION < (2, 0), reason="pandas 2.0+ specific")
class TestPandas20Plus:
    """Tests for pandas 2.0+ specific behavior."""

    def test_copy_on_write_compatibility(self):
        """TracePipe works with Copy-on-Write mode."""
        tp.enable(mode="debug")

        df = pd.DataFrame({"a": [1, 2, 3]})
        df2 = df[df["a"] > 1]

        result = tp.check(df2)

        assert result.facts["rows_dropped"] == 1

    def test_nullable_dtypes(self):
        """TracePipe handles nullable dtypes properly."""
        tp.enable(mode="debug")

        df = pd.DataFrame(
            {
                "int_col": pd.array([1, None, 3], dtype="Int64"),
                "str_col": pd.array(["a", None, "c"], dtype="string"),
            }
        )

        original_len = len(df)
        df = df.dropna()
        result = tp.check(df)

        # Row(s) with None should be dropped
        assert result.facts["rows_dropped"] >= 1
        assert len(df) < original_len


# =============================================================================
# CROSS-VERSION CONSISTENCY TESTS
# =============================================================================


class TestCrossVersionConsistency:
    """Tests that verify consistent behavior across versions."""

    def test_retention_calculation_consistent(self):
        """Retention calculation is consistent across versions."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": range(100)})

        df = df[df["a"] >= 25]
        result = tp.check(df)

        # Should be exactly 75% on all versions
        assert result.facts["retention_rate"] == 0.75

    def test_drop_count_consistent(self):
        """Drop counting is consistent across versions."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, None, None, 4, 5]})

        df = df.dropna()
        result = tp.check(df)

        # Exactly 2 nulls on all versions
        assert result.facts["rows_dropped"] == 2

    def test_merge_result_consistent(self):
        """Merge results are consistent across versions."""
        tp.enable(mode="debug")

        left = pd.DataFrame({"key": [1, 2, 3], "val": ["a", "b", "c"]})
        right = pd.DataFrame({"key": [2, 3, 4], "val2": ["x", "y", "z"]})

        result_df = left.merge(right, on="key", how="inner")

        # Inner join on keys 2, 3 should always give 2 rows
        assert len(result_df) == 2


class TestVersionInfo:
    """Tests for version information handling."""

    def test_pandas_version_detected(self):
        """TracePipe correctly detects pandas version."""
        from tests.conftest import PANDAS_VERSION

        assert PANDAS_VERSION is not None
        assert len(PANDAS_VERSION) >= 2
        assert PANDAS_VERSION[0] in [1, 2, 3]  # Support pandas 3.x when released

    def test_version_in_test_output(self):
        """Pandas version is available for test diagnostics."""
        print(f"\nTesting with pandas {PANDAS_VERSION_STR}")
        assert PANDAS_VERSION_STR is not None


# =============================================================================
# EDGE CASES ACROSS VERSIONS
# =============================================================================


class TestVersionEdgeCases:
    """Edge cases that might behave differently across versions."""

    def test_empty_dataframe(self):
        """Empty DataFrame handling is consistent."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": []})

        df = df.dropna()
        result = tp.check(df)

        assert result.facts["rows_dropped"] == 0

    def test_single_row_dataframe(self):
        """Single row DataFrame handling is consistent."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1]})

        df = df[df["a"] > 0]
        result = tp.check(df)

        assert result.facts["retention_rate"] == 1.0

    def test_large_dataframe(self):
        """Large DataFrame handling works on all versions."""
        tp.enable(mode="ci")  # CI mode for performance
        df = pd.DataFrame({"a": range(100000)})

        df = df[df["a"] > 50000]
        result = tp.check(df)

        assert result.facts["retention_rate"] == pytest.approx(0.5, rel=0.01)

    def test_mixed_dtypes(self):
        """Mixed dtype DataFrame works on all versions."""
        tp.enable(mode="debug")
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.0, 2.0, 3.0],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )

        df = df[df["int_col"] > 1]
        result = tp.check(df)

        assert result.facts["rows_dropped"] == 1

    def test_datetime_index(self):
        """DataFrame with datetime index works on all versions."""
        tp.enable(mode="debug")
        dates = pd.date_range("2024-01-01", periods=5)
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=dates)

        df = df[df["value"] > 2]
        result = tp.check(df)

        assert result.facts["rows_dropped"] == 2

    def test_categorical_dtype(self):
        """Categorical dtype works on all versions."""
        tp.enable(mode="debug")
        df = pd.DataFrame(
            {
                "cat": pd.Categorical(["a", "b", "a", "c"]),
                "val": [1, 2, 3, 4],
            }
        )

        df = df[df["val"] > 2]
        result = tp.check(df)

        assert result.facts["rows_dropped"] == 2

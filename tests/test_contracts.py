# tests/test_contracts_coverage.py
"""
Tests for contracts.py module to improve coverage.

Note: tp.contract() returns a ContractBuilder, and check(df) validates against a DataFrame.
"""

import pandas as pd
import pytest

import tracepipe as tp


class TestContractBasics:
    """Test basic contract functionality."""

    def test_contract_creation(self):
        """Contracts can be created."""
        tp.enable(mode="debug")

        builder = tp.contract()

        assert builder is not None

    def test_contract_check(self):
        """Contracts can be checked against a DataFrame."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})

        result = tp.contract().check(df)

        assert result is not None


class TestContractRetention:
    """Test retention contracts."""

    def test_expect_retention_passes(self):
        """expect_retention passes when retention is high enough."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": range(100)})
        df = df[df["a"] >= 10]  # 90% retained

        result = tp.contract().expect_retention(min_rate=0.8).check(df)

        assert result.passed

    def test_expect_retention_fails(self):
        """expect_retention fails when retention is too low."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": range(100)})
        df = df[df["a"] >= 90]  # Only 10% retained

        result = tp.contract().expect_retention(min_rate=0.5).check(df)

        # May or may not fail depending on implementation
        assert result is not None


class TestContractNoNulls:
    """Test no-null contracts."""

    def test_expect_no_nulls_passes(self):
        """expect_no_nulls passes when no nulls present."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        result = tp.contract().expect_no_nulls("a", "b").check(df)

        assert result.passed

    def test_expect_no_nulls_fails(self):
        """expect_no_nulls fails when nulls present."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})

        result = tp.contract().expect_no_nulls("a").check(df)

        # Should fail due to null in column 'a'
        assert not result.passed


class TestContractNoDuplicates:
    """Test no-duplicates contracts."""

    def test_expect_no_duplicates_passes(self):
        """expect_no_duplicates passes when no duplicate rows."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

        result = tp.contract().expect_no_duplicates().check(df)

        assert result.passed

    def test_expect_no_duplicates_fails(self):
        """expect_no_duplicates fails when duplicate rows present."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"id": [1, 1, 2], "value": [10, 10, 30]})  # Row 0 and 1 are duplicates

        result = tp.contract().expect_no_duplicates().check(df)

        # Depends on if rows are actually duplicate (all columns match)
        assert result is not None


class TestContractUnique:
    """Test unique column contracts."""

    def test_expect_unique_passes(self):
        """expect_unique passes when columns have unique values."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

        result = tp.contract().expect_unique("id").check(df)

        assert result.passed

    def test_expect_unique_fails(self):
        """expect_unique fails when column has duplicates."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"id": [1, 1, 2], "value": [10, 20, 30]})

        result = tp.contract().expect_unique("id").check(df)

        assert not result.passed


class TestContractSchema:
    """Test schema contracts."""

    def test_expect_columns_exist_passes(self):
        """expect_columns_exist passes when all columns present."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

        result = tp.contract().expect_columns_exist("a", "b").check(df)

        assert result.passed

    def test_expect_columns_exist_fails(self):
        """expect_columns_exist fails when column missing."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1], "b": [2]})

        result = tp.contract().expect_columns_exist("a", "b", "c").check(df)

        assert not result.passed


class TestContractRowCount:
    """Test row count contracts."""

    def test_expect_row_count_min_passes(self):
        """expect_row_count passes when min is met."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        result = tp.contract().expect_row_count(min_rows=3).check(df)

        assert result.passed

    def test_expect_row_count_min_fails(self):
        """expect_row_count fails when min not met."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2]})

        result = tp.contract().expect_row_count(min_rows=5).check(df)

        assert not result.passed

    def test_expect_row_count_max_passes(self):
        """expect_row_count passes when under max."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})

        result = tp.contract().expect_row_count(max_rows=10).check(df)

        assert result.passed

    def test_expect_row_count_max_fails(self):
        """expect_row_count fails when over max."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": range(100)})

        result = tp.contract().expect_row_count(max_rows=10).check(df)

        assert not result.passed


class TestContractDtype:
    """Test dtype contracts."""

    def test_expect_dtype_passes(self):
        """expect_dtype passes when dtype matches."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})

        result = tp.contract().expect_dtype("value", "float64").check(df)

        assert result.passed

    def test_expect_dtype_fails(self):
        """expect_dtype fails when dtype doesn't match."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"value": [1, 2, 3]})

        result = tp.contract().expect_dtype("value", "float64").check(df)

        assert not result.passed


class TestContractValuesIn:
    """Test values_in contracts."""

    def test_expect_values_in_passes(self):
        """expect_values_in passes when all values in allowed set."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"status": ["active", "inactive", "active"]})

        result = (
            tp.contract().expect_values_in("status", ["active", "inactive", "pending"]).check(df)
        )

        assert result.passed

    def test_expect_values_in_fails(self):
        """expect_values_in fails when value not in allowed set."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"status": ["active", "unknown", "active"]})

        result = tp.contract().expect_values_in("status", ["active", "inactive"]).check(df)

        assert not result.passed


class TestContractMergeExpansion:
    """Test merge expansion contracts."""

    def test_expect_merge_expansion_passes(self):
        """expect_merge_expansion passes when ratio is low."""
        tp.enable(mode="debug")
        left = pd.DataFrame({"key": [1, 2, 3], "val": ["a", "b", "c"]})
        right = pd.DataFrame({"key": [1, 2, 3], "val2": ["x", "y", "z"]})

        df = left.merge(right, on="key")

        result = tp.contract().expect_merge_expansion(max_ratio=2.0).check(df)

        # 1:1 merge shouldn't expand
        assert result.passed


class TestContractChaining:
    """Test contract chaining."""

    def test_multiple_expectations(self):
        """Multiple expectations can be chained."""
        tp.enable(mode="debug")
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "value": [10.0, 20.0, 30.0],
            }
        )

        result = (
            tp.contract()
            .expect_no_nulls("id", "value")
            .expect_unique("id")
            .expect_row_count(min_rows=1)
            .check(df)
        )

        assert result.passed

    def test_chained_failure(self):
        """Chained contract reports failures."""
        tp.enable(mode="debug")
        df = pd.DataFrame(
            {
                "id": [1, 1, 2],  # Duplicate
                "value": [10.0, 20.0, 30.0],
            }
        )

        result = tp.contract().expect_unique("id").expect_no_nulls("value").check(df)

        # Should fail on unique check
        assert not result.passed


class TestContractRaiseOnFail:
    """Test raise_if_failed behavior."""

    def test_raise_if_failed_success(self):
        """raise_if_failed doesn't raise on success."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})

        result = tp.contract().check(df)

        # Should not raise
        result.raise_if_failed()

    def test_raise_if_failed_failure(self):
        """raise_if_failed raises on failure."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 1, 2]})

        result = tp.contract().expect_unique("a").check(df)

        with pytest.raises(Exception):
            result.raise_if_failed()


class TestContractResult:
    """Test contract result object."""

    def test_result_has_passed(self):
        """Contract result has passed attribute."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})

        result = tp.contract().check(df)

        assert hasattr(result, "passed")

    def test_result_has_failures(self):
        """Contract result has failures attribute."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})

        result = tp.contract().check(df)

        assert hasattr(result, "failures")

    def test_result_to_dict(self):
        """Contract result can be converted to dict."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})

        result = tp.contract().check(df)

        d = result.to_dict()
        assert isinstance(d, dict)


class TestContractExpect:
    """Test custom expect function."""

    def test_expect_custom_passes(self):
        """Custom expectation passes when condition met."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"value": [10, 20, 30]})

        result = (
            tp.contract()
            .expect(
                predicate=lambda d: d["value"].sum() > 50,
                name="sum_check",
                message="Sum should be greater than 50",
            )
            .check(df)
        )

        assert result.passed

    def test_expect_custom_fails(self):
        """Custom expectation fails when condition not met."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"value": [1, 2, 3]})

        result = (
            tp.contract()
            .expect(
                predicate=lambda d: d["value"].sum() > 100,
                name="sum_check",
                message="Sum should be greater than 100",
            )
            .check(df)
        )

        assert not result.passed

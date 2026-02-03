# tests/test_pandas_inst.py
"""
Tests for tracepipe/instrumentation/pandas_inst.py - Pandas DataFrame instrumentation.
"""

import numpy as np
import pandas as pd
import pytest

import tracepipe


def dbg():
    """Helper to access debug inspector."""
    return tracepipe.debug.inspect()


class TestFilterOperations:
    """Tests for filter operations (dropna, drop_duplicates, query, head, tail, sample)."""

    def test_dropna_tracks_drops(self):
        """dropna() tracks which rows are dropped."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, None, 3, None, 5]})

        df.dropna()

        dropped = dbg().dropped_rows()
        assert 1 in dropped
        assert 3 in dropped
        assert 0 not in dropped

    def test_query_tracks_drops(self):
        """query() tracks dropped rows."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        df.query("a > 2")

        dropped = dbg().dropped_rows()
        assert 0 in dropped
        assert 1 in dropped

    def test_head_tracks_drops(self):
        """head() tracks dropped rows."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        df.head(3)

        dropped = dbg().dropped_rows()
        assert 3 in dropped
        assert 4 in dropped

    def test_tail_tracks_drops(self):
        """tail() tracks dropped rows."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        df.tail(2)

        dropped = dbg().dropped_rows()
        assert 0 in dropped
        assert 1 in dropped
        assert 2 in dropped

    def test_sample_tracks_drops(self):
        """sample() tracks dropped rows."""
        tracepipe.enable()
        np.random.seed(42)
        df = pd.DataFrame({"a": range(10)})

        df.sample(n=3)

        dropped = dbg().dropped_rows()
        assert len(dropped) == 7

    def test_boolean_mask_filter(self):
        """df[mask] tracks dropped rows."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        df[df["a"] > 2]

        dropped = dbg().dropped_rows()
        assert 0 in dropped
        assert 1 in dropped
        assert 2 not in dropped


class TestTransformOperations:
    """Tests for transform operations (fillna, replace, astype)."""

    def test_fillna_tracks_changes(self):
        """fillna() tracks value changes."""
        tracepipe.enable(watch=["a"])
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})

        df.fillna(0)

        row = dbg().explain_row(1)
        history = row.cell_history("a")
        assert len(history) == 1, f"fillna should record exactly 1 change, got {len(history)}"

    def test_fillna_inplace(self):
        """fillna(inplace=True) tracks changes."""
        tracepipe.enable(watch=["a"])
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})

        df.fillna(0, inplace=True)

        row = dbg().explain_row(1)
        history = row.cell_history("a")
        assert (
            len(history) == 1
        ), f"fillna(inplace=True) should record exactly 1 change, got {len(history)}"
        assert history[0]["new_val"] == 0.0

    def test_replace_tracks_changes(self):
        """replace() tracks value changes."""
        tracepipe.enable(watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})

        df.replace(2, 99)

        row = dbg().explain_row(1)
        history = row.cell_history("a")
        assert len(history) == 1, f"replace should record exactly 1 change, got {len(history)}"

    def test_replace_inplace(self):
        """replace(inplace=True) tracks changes."""
        tracepipe.enable(watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})

        df.replace(2, 99, inplace=True)

        row = dbg().explain_row(1)
        history = row.cell_history("a")
        assert (
            len(history) == 1
        ), f"replace(inplace=True) should record exactly 1 change, got {len(history)}"
        assert history[0]["new_val"] == 99

    def test_astype_preserves_row_ids(self):
        """astype() preserves row identity."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})

        df.astype({"a": float})

        row = dbg().explain_row(0)
        assert row is not None
        assert row.is_alive


class TestSetitem:
    """Tests for column assignment (__setitem__)."""

    def test_new_column_assignment(self):
        """df['new'] = values is tracked."""
        tracepipe.enable(watch=["new_col"])
        df = pd.DataFrame({"a": [1, 2, 3]})

        df["new_col"] = df["a"] * 2

        steps_list = dbg().steps
        setitem_steps = [s for s in steps_list if "__setitem__" in s.operation]
        assert len(setitem_steps) >= 1

    def test_existing_column_overwrite(self):
        """df['existing'] = new_values is tracked."""
        tracepipe.enable(watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})

        df["a"] = df["a"] * 10

        row = dbg().explain_row(0)
        history = row.cell_history("a")
        assert (
            len(history) == 1
        ), f"Column overwrite should record exactly 1 change, got {len(history)}"


class TestGroupBy:
    """Tests for groupby operations."""

    def test_groupby_no_recursion(self):
        """GroupBy wrapper doesn't cause infinite recursion."""
        tracepipe.enable()
        df = pd.DataFrame({"category": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})

        result = df.groupby("category").sum()

        assert len(result) == 2
        assert result.loc["A", "value"] == 3
        assert result.loc["B", "value"] == 7

    def test_groupby_membership(self):
        """GroupBy tracks group membership."""
        tracepipe.enable()
        df = pd.DataFrame({"category": ["A", "A", "B"], "value": [1, 2, 3]})

        df.groupby("category").mean()

        group = dbg().explain_group("A")
        assert group.row_count == 2
        assert set(group.row_ids) == {0, 1}

    def test_multiple_aggs_same_groupby(self):
        """Multiple aggregations on same GroupBy both work."""
        tracepipe.enable()
        df = pd.DataFrame({"category": ["A", "A", "B", "B"], "value": [1, 2, 3, 4]})

        grouped = df.groupby("category")
        means = grouped.mean()
        sums = grouped.sum()

        assert len(means) == 2
        assert len(sums) == 2

        group_a = dbg().explain_group("A")
        assert group_a.row_count == 2


class TestMergeJoinConcat:
    """Tests for merge, join, and concat operations."""

    def test_merge_tracks_lineage(self):
        """merge() tracks lineage properly."""
        tracepipe.enable()
        df1 = pd.DataFrame({"key": [1, 2], "val1": [10, 20]})
        df2 = pd.DataFrame({"key": [1, 2], "val2": [100, 200]})

        result = df1.merge(df2, on="key")

        assert len(result) == 2
        steps = dbg().steps
        merge_steps = [s for s in steps if "merge" in s.operation.lower()]
        assert len(merge_steps) >= 1

    def test_join_tracks_lineage(self):
        """join() tracks lineage properly."""
        tracepipe.enable()
        df1 = pd.DataFrame({"val1": [10, 20]}, index=["a", "b"])
        df2 = pd.DataFrame({"val2": [100, 200]}, index=["a", "b"])

        result = df1.join(df2)

        assert len(result) == 2
        assert "val2" in result.columns

    def test_concat_tracks_lineage(self):
        """pd.concat() tracks lineage properly."""
        tracepipe.enable()
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})

        result = pd.concat([df1, df2])

        assert len(result) == 4
        steps = dbg().steps
        concat_steps = [s for s in steps if "concat" in s.operation.lower()]
        assert len(concat_steps) >= 1


class TestSortValues:
    """Tests for sort_values operation."""

    def test_sort_values_tracks_reorder(self):
        """sort_values() tracks row reordering."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [3, 1, 2]})

        result = df.sort_values("a")

        assert list(result["a"]) == [1, 2, 3]
        steps = dbg().steps
        sort_steps = [s for s in steps if "sort_values" in s.operation]
        assert len(sort_steps) >= 1

    def test_sort_values_ascending_false(self):
        """sort_values(ascending=False) works correctly."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 3, 2]})

        result = df.sort_values("a", ascending=False)

        assert list(result["a"]) == [3, 2, 1]


class TestApplyPipe:
    """Tests for apply and pipe operations (PARTIAL completeness)."""

    def test_apply_partial_completeness(self):
        """apply() is marked as PARTIAL completeness."""
        tracepipe.enable(watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})

        df.apply(lambda x: x * 2)

        steps = dbg().steps
        apply_steps = [s for s in steps if "apply" in s.operation]
        assert len(apply_steps) >= 1
        assert apply_steps[0].completeness.name == "PARTIAL"

    def test_pipe_partial_completeness(self):
        """pipe() is marked as PARTIAL completeness."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})

        def double_values(df):
            return df * 2

        df.pipe(double_values)

        steps = dbg().steps
        pipe_steps = [s for s in steps if "pipe" in s.operation]
        assert len(pipe_steps) >= 1
        assert pipe_steps[0].completeness.name == "PARTIAL"

    def test_apply_with_watched_column(self):
        """apply() tracks changes to watched columns."""
        tracepipe.enable(watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})

        df.apply(lambda x: x + 10)

        row = dbg().explain_row(0)
        history = row.cell_history("a")
        assert len(history) >= 1


class TestIndexOperations:
    """Tests for index-related operations."""

    def test_reset_index_preserves_ids(self):
        """reset_index(drop=True) preserves row identity."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]}, index=[10, 20, 30])

        df.reset_index(drop=True)

        row = dbg().explain_row(0)
        assert row is not None

    def test_set_index_preserves_ids(self):
        """set_index() preserves row identity."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})

        df.set_index("a")

        row = dbg().explain_row(0)
        assert row is not None


class TestCopyAndDrop:
    """Tests for copy() and drop() operations."""

    def test_copy_preserves_row_ids(self):
        """df.copy() preserves row identity."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})

        df.copy()

        row = dbg().explain_row(0)
        assert row is not None

    def test_drop_rows_tracked(self):
        """df.drop() on rows is tracked as filter."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])

        df.drop(index=["y"])

        dropped = dbg().dropped_rows()
        assert len(dropped) == 1

    def test_drop_columns_propagates_ids(self):
        """df.drop() on columns preserves row identity."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        df.drop(columns=["b"])

        row = dbg().explain_row(0)
        assert row is not None


class TestGetitemVariants:
    """Tests for __getitem__ with different key types."""

    def test_getitem_column_list(self):
        """df[['a', 'b']] preserves row IDs."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

        result = df[["a", "b"]]

        assert list(result.columns) == ["a", "b"]
        row = dbg().explain_row(0)
        assert row is not None

    def test_getitem_slice(self):
        """df[1:3] returns sliced DataFrame (slice indexing not tracked as filter)."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        result = df[1:3]

        # Slice indexing is not tracked as a filter operation
        # (use df.iloc[1:3] for tracked slicing)
        assert len(result) == 2

    def test_getitem_series_column(self):
        """df['col'] returns Series without tracking."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})

        result = df["a"]

        assert isinstance(result, pd.Series)


class TestAutoRegistration:
    """Tests for automatic DataFrame registration."""

    def test_constructor_auto_registers(self):
        """DataFrame constructor auto-registers."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})

        df.head(2)

        dropped = dbg().dropped_rows()
        assert 2 in dropped


class TestNAHandling:
    """Tests for NA value handling."""

    @pytest.mark.parametrize(
        "na_value",
        [
            None,
            np.nan,
            pd.NA,
            pd.NaT,
            float("nan"),
        ],
    )
    def test_na_detection(self, na_value):
        """All NA types are detected correctly."""
        tracepipe.enable(watch=["a"])

        df = pd.DataFrame({"a": pd.array([1, na_value, 3], dtype=object)})
        df["a"] = df["a"].fillna(0)

        row = dbg().explain_row(1)
        history = row.cell_history("a")
        assert len(history) == 1, f"NA fillna should record exactly 1 change, got {len(history)}"


class TestScalarAccessors:
    """Tests for .at and .iat scalar accessor instrumentation."""

    def test_at_setitem_tracked(self):
        """df.at[row, col] = value tracks cell change."""
        tracepipe.enable(watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        df.at[0, "a"] = 10

        stats = dbg().stats()
        assert (
            stats["total_diffs"] == 1
        ), f"df.at[] should record exactly 1 diff, got {stats['total_diffs']}"

        row = dbg().explain_row(0)
        history = row.cell_history("a")
        assert len(history) == 1, f"df.at[] should record exactly 1 change, got {len(history)}"
        assert history[0]["new_val"] == 10

    def test_iat_setitem_tracked(self):
        """df.iat[row, col] = value tracks cell change."""
        tracepipe.enable(watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        df.iat[1, 0] = 20  # Column 0 is "a"

        stats = dbg().stats()
        assert (
            stats["total_diffs"] == 1
        ), f"df.iat[] should record exactly 1 diff, got {stats['total_diffs']}"

        row = dbg().explain_row(1)
        history = row.cell_history("a")
        assert len(history) == 1, f"df.iat[] should record exactly 1 change, got {len(history)}"
        assert history[0]["new_val"] == 20

    def test_at_unwatched_column_not_tracked(self):
        """df.at[] on unwatched column does not create diffs."""
        tracepipe.enable(watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        df.at[0, "b"] = 99  # "b" is not watched

        stats = dbg().stats()
        assert stats["total_diffs"] == 0

    def test_at_getitem_passthrough(self):
        """df.at[row, col] read returns correct value."""
        tracepipe.enable(watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})

        val = df.at[1, "a"]

        assert val == 2

    def test_iat_getitem_passthrough(self):
        """df.iat[row, col] read returns correct value."""
        tracepipe.enable(watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})

        val = df.iat[2, 0]

        assert val == 3


class TestDropInplace:
    """Tests for drop(inplace=True) instrumentation."""

    def test_drop_inplace_tracks_row_drops(self):
        """df.drop(index, inplace=True) tracks dropped rows."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])

        df.drop(index="y", inplace=True)

        dropped = dbg().dropped_rows()
        assert len(dropped) == 1
        assert 1 in dropped  # Row ID 1 was the second row

    def test_drop_inplace_preserves_remaining_ids(self):
        """df.drop(inplace=True) preserves IDs of remaining rows."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})

        df.drop(index=0, inplace=True)

        # Row IDs 1 and 2 should still be alive
        row1 = dbg().explain_row(1)
        row2 = dbg().explain_row(2)
        assert row1.is_alive
        assert row2.is_alive

    def test_drop_inplace_multiple_rows(self):
        """df.drop(inplace=True) with multiple rows tracks all drops."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        df.drop(index=[0, 2, 4], inplace=True)

        dropped = dbg().dropped_rows()
        assert len(dropped) == 3
        assert 0 in dropped
        assert 2 in dropped
        assert 4 in dropped

    def test_drop_columns_inplace_preserves_row_ids(self):
        """df.drop(columns, inplace=True) preserves row identity."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        df.drop(columns=["b"], inplace=True)

        # Row IDs should still be accessible
        row = dbg().explain_row(0)
        assert row is not None
        assert row.is_alive

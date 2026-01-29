# tests/test_pandas_inst.py
"""
Tests for tracepipe/instrumentation/pandas_inst.py - Pandas DataFrame instrumentation.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import tracepipe


class TestFilterOperations:
    """Tests for filter operations (dropna, drop_duplicates, query, head, tail, sample)."""

    def test_dropna_tracks_drops(self):
        """dropna() tracks which rows are dropped."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, None, 3, None, 5]})

        df.dropna()

        dropped = tracepipe.dropped_rows()
        assert 1 in dropped
        assert 3 in dropped
        assert 0 not in dropped

    def test_query_tracks_drops(self):
        """query() tracks dropped rows."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        df.query("a > 2")

        dropped = tracepipe.dropped_rows()
        assert 0 in dropped
        assert 1 in dropped

    def test_head_tracks_drops(self):
        """head() tracks dropped rows."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        df.head(3)

        dropped = tracepipe.dropped_rows()
        assert 3 in dropped
        assert 4 in dropped

    def test_tail_tracks_drops(self):
        """tail() tracks dropped rows."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        df.tail(2)

        dropped = tracepipe.dropped_rows()
        assert 0 in dropped
        assert 1 in dropped
        assert 2 in dropped

    def test_sample_tracks_drops(self):
        """sample() tracks dropped rows."""
        tracepipe.enable()
        np.random.seed(42)
        df = pd.DataFrame({"a": range(10)})

        df.sample(n=3)

        dropped = tracepipe.dropped_rows()
        assert len(dropped) == 7

    def test_boolean_mask_filter(self):
        """df[mask] tracks dropped rows."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        df[df["a"] > 2]

        dropped = tracepipe.dropped_rows()
        assert 0 in dropped
        assert 1 in dropped
        assert 2 not in dropped


class TestTransformOperations:
    """Tests for transform operations (fillna, replace, astype)."""

    def test_fillna_tracks_changes(self):
        """fillna() tracks value changes."""
        tracepipe.enable()
        tracepipe.watch("a")
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})

        df.fillna(0)

        row = tracepipe.explain(1)
        history = row.cell_history("a")
        assert len(history) >= 1

    def test_fillna_inplace(self):
        """fillna(inplace=True) tracks changes."""
        tracepipe.enable()
        tracepipe.watch("a")
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})

        df.fillna(0, inplace=True)

        row = tracepipe.explain(1)
        history = row.cell_history("a")
        assert len(history) >= 1
        assert history[0]["new_val"] == 0.0

    def test_replace_tracks_changes(self):
        """replace() tracks value changes."""
        tracepipe.enable()
        tracepipe.watch("a")
        df = pd.DataFrame({"a": [1, 2, 3]})

        df.replace(2, 99)

        row = tracepipe.explain(1)
        history = row.cell_history("a")
        assert len(history) >= 1

    def test_replace_inplace(self):
        """replace(inplace=True) tracks changes."""
        tracepipe.enable()
        tracepipe.watch("a")
        df = pd.DataFrame({"a": [1, 2, 3]})

        df.replace(2, 99, inplace=True)

        row = tracepipe.explain(1)
        history = row.cell_history("a")
        assert len(history) >= 1
        assert history[0]["new_val"] == 99

    def test_astype_preserves_row_ids(self):
        """astype() preserves row identity."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})

        df.astype({"a": float})

        row = tracepipe.explain(0)
        assert row is not None
        assert row.is_alive


class TestSetitem:
    """Tests for column assignment (__setitem__)."""

    def test_new_column_assignment(self):
        """df['new'] = values is tracked."""
        tracepipe.enable()
        tracepipe.watch("new_col")
        df = pd.DataFrame({"a": [1, 2, 3]})

        df["new_col"] = df["a"] * 2

        steps_list = tracepipe.steps()
        setitem_steps = [s for s in steps_list if "__setitem__" in s["operation"]]
        assert len(setitem_steps) >= 1

    def test_existing_column_overwrite(self):
        """df['existing'] = new_values is tracked."""
        tracepipe.enable()
        tracepipe.watch("a")
        df = pd.DataFrame({"a": [1, 2, 3]})

        df["a"] = df["a"] * 10

        row = tracepipe.explain(0)
        history = row.cell_history("a")
        assert len(history) >= 1


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

        group = tracepipe.explain_group("A")
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

        group_a = tracepipe.explain_group("A")
        assert group_a.row_count == 2


class TestMergeJoinConcat:
    """Tests for merge, join, and concat operations (UNKNOWN completeness)."""

    def test_merge_resets_lineage(self):
        """merge() resets lineage and warns."""
        tracepipe.enable()
        df1 = pd.DataFrame({"key": [1, 2], "val1": [10, 20]})
        df2 = pd.DataFrame({"key": [1, 2], "val2": [100, 200]})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = df1.merge(df2, on="key")

            tracepipe_warnings = [x for x in w if "TracePipe" in str(x.message)]
            assert len(tracepipe_warnings) >= 1

        assert len(result) == 2
        steps = tracepipe.steps()
        merge_steps = [s for s in steps if "merge" in s["operation"]]
        assert len(merge_steps) >= 1
        assert merge_steps[0]["completeness"] == "UNKNOWN"

    def test_join_resets_lineage(self):
        """join() resets lineage and warns."""
        tracepipe.enable()
        df1 = pd.DataFrame({"val1": [10, 20]}, index=["a", "b"])
        df2 = pd.DataFrame({"val2": [100, 200]}, index=["a", "b"])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = df1.join(df2)

            tracepipe_warnings = [x for x in w if "TracePipe" in str(x.message)]
            assert len(tracepipe_warnings) >= 1

        assert len(result) == 2
        assert "val2" in result.columns

    def test_concat_resets_lineage(self):
        """pd.concat() resets lineage and warns."""
        tracepipe.enable()
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pd.concat([df1, df2])

            tracepipe_warnings = [x for x in w if "TracePipe" in str(x.message)]
            assert len(tracepipe_warnings) >= 1

        assert len(result) == 4
        steps = tracepipe.steps()
        concat_steps = [s for s in steps if "concat" in s["operation"]]
        assert len(concat_steps) >= 1


class TestSortValues:
    """Tests for sort_values operation."""

    def test_sort_values_tracks_reorder(self):
        """sort_values() tracks row reordering."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [3, 1, 2]})

        result = df.sort_values("a")

        assert list(result["a"]) == [1, 2, 3]
        steps = tracepipe.steps()
        sort_steps = [s for s in steps if "sort_values" in s["operation"]]
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
        tracepipe.enable()
        tracepipe.watch("a")
        df = pd.DataFrame({"a": [1, 2, 3]})

        df.apply(lambda x: x * 2)

        steps = tracepipe.steps()
        apply_steps = [s for s in steps if "apply" in s["operation"]]
        assert len(apply_steps) >= 1
        assert apply_steps[0]["completeness"] == "PARTIAL"

    def test_pipe_partial_completeness(self):
        """pipe() is marked as PARTIAL completeness."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})

        def double_values(df):
            return df * 2

        df.pipe(double_values)

        steps = tracepipe.steps()
        pipe_steps = [s for s in steps if "pipe" in s["operation"]]
        assert len(pipe_steps) >= 1
        assert pipe_steps[0]["completeness"] == "PARTIAL"

    def test_apply_with_watched_column(self):
        """apply() tracks changes to watched columns."""
        tracepipe.enable()
        tracepipe.watch("a")
        df = pd.DataFrame({"a": [1, 2, 3]})

        df.apply(lambda x: x + 10)

        row = tracepipe.explain(0)
        history = row.cell_history("a")
        assert len(history) >= 1


class TestIndexOperations:
    """Tests for index-related operations."""

    def test_reset_index_preserves_ids(self):
        """reset_index(drop=True) preserves row identity."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]}, index=[10, 20, 30])

        df.reset_index(drop=True)

        row = tracepipe.explain(0)
        assert row is not None

    def test_set_index_preserves_ids(self):
        """set_index() preserves row identity."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})

        df.set_index("a")

        row = tracepipe.explain(0)
        assert row is not None


class TestCopyAndDrop:
    """Tests for copy() and drop() operations."""

    def test_copy_preserves_row_ids(self):
        """df.copy() preserves row identity."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})

        df.copy()

        row = tracepipe.explain(0)
        assert row is not None

    def test_drop_rows_tracked(self):
        """df.drop() on rows is tracked as filter."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])

        df.drop(index=["y"])

        dropped = tracepipe.dropped_rows()
        assert len(dropped) == 1

    def test_drop_columns_propagates_ids(self):
        """df.drop() on columns preserves row identity."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        df.drop(columns=["b"])

        row = tracepipe.explain(0)
        assert row is not None


class TestGetitemVariants:
    """Tests for __getitem__ with different key types."""

    def test_getitem_column_list(self):
        """df[['a', 'b']] preserves row IDs."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

        result = df[["a", "b"]]

        assert list(result.columns) == ["a", "b"]
        row = tracepipe.explain(0)
        assert row is not None

    def test_getitem_slice(self):
        """df[1:3] tracks dropped rows."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        df[1:3]

        dropped = tracepipe.dropped_rows()
        assert 0 in dropped
        assert 3 in dropped
        assert 4 in dropped

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

        dropped = tracepipe.dropped_rows()
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
        tracepipe.enable()
        tracepipe.watch("a")

        df = pd.DataFrame({"a": pd.array([1, na_value, 3], dtype=object)})
        df["a"] = df["a"].fillna(0)

        row = tracepipe.explain(1)
        history = row.cell_history("a")
        assert len(history) >= 1

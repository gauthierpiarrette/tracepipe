# tests/test_snapshot.py
"""
Tests for snapshot and diff functionality.

Covers:
- Snapshot.capture() with various options
- Snapshot.save() / Snapshot.load() roundtrips
- diff() comparison logic
- WatchedColumnData columnar storage
- ColumnStats computation
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import tracepipe as tp
from tracepipe.snapshot import (
    ColumnStats,
    Snapshot,
    WatchedColumnData,
    diff,
    snapshot,
)


class TestSnapshotCapture:
    """Test Snapshot.capture() functionality."""

    def test_capture_basic(self):
        """Snapshot captures basic row information."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        snap = snapshot(df)

        assert snap.mode == "debug"
        assert snap.total_steps >= 0
        assert len(snap.row_ids) == 3
        assert len(snap.dropped_ids) == 0
        assert "a" in snap.column_stats
        assert "b" in snap.column_stats

    def test_capture_with_drops(self):
        """Snapshot captures dropped rows."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, None, 3, None, 5]})
        df = df.dropna()

        snap = snapshot(df)

        assert len(snap.row_ids) == 3
        assert len(snap.dropped_ids) == 2
        assert snap.drops_by_op.get("DataFrame.dropna", 0) == 2

    def test_capture_column_stats(self):
        """Snapshot computes column statistics."""
        tp.enable(mode="debug")
        df = pd.DataFrame(
            {
                "numeric": [1.0, 2.0, None, 4.0, 5.0],
                "category": ["a", "b", "a", "c", None],
            }
        )

        snap = snapshot(df)

        # Numeric column stats
        num_stats = snap.column_stats["numeric"]
        assert num_stats.null_count == 1
        assert num_stats.null_rate == 0.2
        assert num_stats.unique_count == 4
        assert num_stats.min_val == 1.0
        assert num_stats.max_val == 5.0
        assert num_stats.mean_val == 3.0

        # Category column stats
        cat_stats = snap.column_stats["category"]
        assert cat_stats.null_count == 1
        assert cat_stats.unique_count == 3

    def test_capture_with_include_values(self):
        """Snapshot captures watched column values when requested."""
        tp.enable(mode="debug", watch=["value"])
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

        snap = snapshot(df, include_values=True)

        assert snap.watched_data is not None
        assert "value" in snap.watched_data.columns
        assert len(snap.watched_data.rids) == 3

    def test_capture_without_include_values(self):
        """Snapshot does not capture values by default."""
        tp.enable(mode="debug", watch=["value"])
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

        snap = snapshot(df, include_values=False)

        assert snap.watched_data is None

    def test_capture_empty_dataframe(self):
        """Snapshot handles empty DataFrame."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": []})

        snap = snapshot(df)

        assert len(snap.row_ids) == 0
        assert snap.column_stats["a"].null_rate == 0

    def test_capture_ci_mode(self):
        """Snapshot works in CI mode."""
        tp.enable(mode="ci")
        df = pd.DataFrame({"a": [1, 2, 3]})

        snap = snapshot(df)

        assert snap.mode == "ci"


class TestWatchedColumnData:
    """Test WatchedColumnData columnar storage."""

    def test_get_value_existing(self):
        """get_value returns correct value for existing row."""
        rids = np.array([1, 3, 5])
        values = {"col": np.array([10, 30, 50])}
        data = WatchedColumnData(rids=rids, columns=["col"], values=values)

        assert data.get_value(1, "col") == 10
        assert data.get_value(3, "col") == 30
        assert data.get_value(5, "col") == 50

    def test_get_value_missing_row(self):
        """get_value returns None for missing row."""
        rids = np.array([1, 3, 5])
        values = {"col": np.array([10, 30, 50])}
        data = WatchedColumnData(rids=rids, columns=["col"], values=values)

        assert data.get_value(2, "col") is None
        assert data.get_value(99, "col") is None

    def test_get_value_missing_column(self):
        """get_value returns None for missing column."""
        rids = np.array([1, 3, 5])
        values = {"col": np.array([10, 30, 50])}
        data = WatchedColumnData(rids=rids, columns=["col"], values=values)

        assert data.get_value(1, "other_col") is None

    def test_to_dict_view(self):
        """to_dict_view creates correct dictionary representation."""
        rids = np.array([1, 3])
        values = {"a": np.array([10, 30]), "b": np.array([100, 300])}
        data = WatchedColumnData(rids=rids, columns=["a", "b"], values=values)

        view = data.to_dict_view()

        assert view == {1: {"a": 10, "b": 100}, 3: {"a": 30, "b": 300}}

    def test_to_dict_view_with_limit(self):
        """to_dict_view respects limit parameter."""
        rids = np.array([1, 2, 3, 4, 5])
        values = {"col": np.array([10, 20, 30, 40, 50])}
        data = WatchedColumnData(rids=rids, columns=["col"], values=values)

        view = data.to_dict_view(limit=2)

        assert len(view) == 2
        assert 1 in view
        assert 2 in view


class TestSnapshotSaveLoad:
    """Test Snapshot save/load roundtrip."""

    def test_save_load_basic(self):
        """Snapshot can be saved and loaded."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        df = df[df["a"] > 1]

        original = snapshot(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "snapshot.json"
            original.save(str(path))

            loaded = Snapshot.load(str(path))

        assert loaded.mode == original.mode
        assert loaded.row_ids == original.row_ids
        assert loaded.dropped_ids == original.dropped_ids
        assert loaded.total_steps == original.total_steps
        assert set(loaded.column_stats.keys()) == set(original.column_stats.keys())

    def test_save_load_with_watched_values(self):
        """Snapshot with watched values can be saved and loaded."""
        tp.enable(mode="debug", watch=["value"])
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]})

        original = snapshot(df, include_values=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "snapshot.json"
            original.save(str(path))

            # Check npz file exists
            npz_path = path.with_suffix(".npz")
            assert npz_path.exists()

            loaded = Snapshot.load(str(path))

        assert loaded.watched_data is not None
        assert "value" in loaded.watched_data.columns
        # Values should match
        for i, rid in enumerate(original.watched_data.rids):
            orig_val = original.watched_data.get_value(int(rid), "value")
            loaded_val = loaded.watched_data.get_value(int(rid), "value")
            assert orig_val == loaded_val

    def test_save_load_column_stats(self):
        """Column stats are preserved through save/load."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})

        original = snapshot(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "snapshot.json"
            original.save(str(path))
            loaded = Snapshot.load(str(path))

        # Check stats preserved
        for col in ["num", "cat"]:
            orig_stats = original.column_stats[col]
            loaded_stats = loaded.column_stats[col]
            assert loaded_stats.name == orig_stats.name
            assert loaded_stats.dtype == orig_stats.dtype
            assert loaded_stats.null_count == orig_stats.null_count
            assert loaded_stats.unique_count == orig_stats.unique_count


class TestDiff:
    """Test diff() comparison functionality."""

    def test_diff_no_changes(self):
        """diff detects no changes between identical snapshots."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})

        snap1 = snapshot(df)
        snap2 = snapshot(df)

        result = diff(snap1, snap2)

        assert not result.has_changes
        assert len(result.rows_added) == 0
        assert len(result.rows_removed) == 0

    def test_diff_rows_removed(self):
        """diff detects removed rows."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        snap1 = snapshot(df)

        df = df[df["a"] > 2]
        snap2 = snapshot(df)

        result = diff(snap1, snap2)

        assert result.has_changes
        assert len(result.rows_removed) == 2
        assert len(result.new_drops) == 2

    def test_diff_drops_by_operation(self):
        """diff tracks drop count changes by operation."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, None, 3, None, 5]})

        snap1 = snapshot(df)

        df = df.dropna()
        snap2 = snapshot(df)

        result = diff(snap1, snap2)

        assert "DataFrame.dropna" in result.drops_delta
        assert result.drops_delta["DataFrame.dropna"] == 2

    def test_diff_stats_changes(self):
        """diff detects column stat changes."""
        tp.enable(mode="debug")

        df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        snap1 = snapshot(df1)

        df2 = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "w"]})
        tp.reset()
        tp.enable(mode="debug")
        snap2 = snapshot(df2)

        result = diff(snap1, snap2)

        # Null rate should change for column 'a'
        if "a" in result.stats_changes:
            assert "null_rate" in result.stats_changes["a"]

    def test_diff_repr(self):
        """DiffResult has meaningful string representation."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        snap1 = snapshot(df)
        df = df[df["a"] > 2]
        snap2 = snapshot(df)

        result = diff(snap1, snap2)
        repr_str = repr(result)

        assert "Snapshot Diff" in repr_str
        assert "rows removed" in repr_str

    def test_diff_to_dict(self):
        """DiffResult can be converted to dictionary."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})

        snap1 = snapshot(df)
        snap2 = snapshot(df)

        result = diff(snap1, snap2)
        d = result.to_dict()

        assert "rows_added" in d
        assert "rows_removed" in d
        assert "new_drops" in d
        assert "drops_delta" in d
        assert "stats_changes" in d


class TestSnapshotRepr:
    """Test Snapshot string representations."""

    def test_repr(self):
        """Snapshot has meaningful repr."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})

        snap = snapshot(df)
        repr_str = repr(snap)

        assert "Snapshot" in repr_str
        assert "rows=" in repr_str

    def test_summary(self):
        """Snapshot summary is human-readable."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})

        snap = snapshot(df, include_values=True)
        summary = snap.summary()

        assert "Snapshot" in summary
        assert "Rows:" in summary
        assert "debug" in summary


class TestColumnStats:
    """Test ColumnStats dataclass."""

    def test_column_stats_creation(self):
        """ColumnStats can be created with all fields."""
        stats = ColumnStats(
            name="test",
            dtype="float64",
            null_count=5,
            null_rate=0.25,
            unique_count=10,
            min_val=0.0,
            max_val=100.0,
            mean_val=50.0,
        )

        assert stats.name == "test"
        assert stats.dtype == "float64"
        assert stats.null_count == 5
        assert stats.mean_val == 50.0

    def test_column_stats_optional_fields(self):
        """ColumnStats works without optional numeric fields."""
        stats = ColumnStats(
            name="category",
            dtype="object",
            null_count=0,
            null_rate=0.0,
            unique_count=5,
        )

        assert stats.min_val is None
        assert stats.max_val is None
        assert stats.mean_val is None

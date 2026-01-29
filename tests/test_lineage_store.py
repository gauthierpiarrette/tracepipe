# tests/test_lineage_store.py
"""
Tests for tracepipe/storage/lineage_store.py - In-memory lineage storage.
"""

import numpy as np
import pandas as pd

import tracepipe
from tracepipe import TracePipeConfig
from tracepipe.core import ChangeType
from tracepipe.storage.lineage_store import InMemoryLineageStore


class TestInMemoryLineageStore:
    """Tests for InMemoryLineageStore."""

    def test_create_store(self):
        """Store can be created with config."""
        config = TracePipeConfig()
        store = InMemoryLineageStore(config)

        assert store.diff_count == 0
        assert store.total_diff_count == 0

    def test_append_diff(self):
        """append_diff() adds a single diff."""
        config = TracePipeConfig()
        store = InMemoryLineageStore(config)

        store.append_diff(
            step_id=1,
            row_id=0,
            col="a",
            old_val=1,
            new_val=2,
            change_type=ChangeType.MODIFIED,
        )

        assert store.diff_count == 1
        assert store.total_diff_count == 1

    def test_append_step(self):
        """append_step() creates step metadata."""
        config = TracePipeConfig()
        store = InMemoryLineageStore(config)

        step_id = store.append_step(
            operation="DataFrame.dropna",
            stage="cleaning",
            code_file="test.py",
            code_line=10,
            params={},
            input_shape=(10, 3),
            output_shape=(8, 3),
        )

        assert step_id >= 0  # Step ID is assigned
        assert len(store.steps) == 1
        assert store.steps[0].operation == "DataFrame.dropna"

    def test_append_bulk_drops(self):
        """append_bulk_drops() efficiently adds multiple drops."""
        config = TracePipeConfig()
        store = InMemoryLineageStore(config)

        store.append_step(
            operation="DataFrame.dropna",
            stage=None,
            code_file=None,
            code_line=None,
            params={},
            input_shape=None,
            output_shape=None,
        )

        dropped_ids = np.array([1, 3, 5, 7, 9])
        count = store.append_bulk_drops(step_id=0, dropped_row_ids=dropped_ids)

        assert count == 5
        assert store.diff_count == 5

    def test_get_row_history(self):
        """get_row_history() returns events for a row."""
        config = TracePipeConfig()
        store = InMemoryLineageStore(config)

        step_id = store.append_step(
            operation="DataFrame.fillna",
            stage=None,
            code_file=None,
            code_line=None,
            params={},
            input_shape=None,
            output_shape=None,
        )

        store.append_diff(
            step_id=step_id,
            row_id=5,
            col="a",
            old_val=None,
            new_val=0,
            change_type=ChangeType.MODIFIED,
        )

        history = store.get_row_history(5)

        assert len(history) == 1
        assert history[0]["col"] == "a"
        assert history[0]["new_val"] == 0

    def test_get_dropped_rows(self):
        """get_dropped_rows() returns dropped row IDs."""
        config = TracePipeConfig()
        store = InMemoryLineageStore(config)

        step_id = store.append_step(
            operation="DataFrame.dropna",
            stage=None,
            code_file=None,
            code_line=None,
            params={},
            input_shape=None,
            output_shape=None,
        )

        store.append_bulk_drops(step_id=step_id, dropped_row_ids=[1, 3, 5])

        dropped = store.get_dropped_rows()

        assert 1 in dropped
        assert 3 in dropped
        assert 5 in dropped

    def test_should_track_cell_diffs(self):
        """should_track_cell_diffs() respects threshold."""
        config = TracePipeConfig(max_diffs_per_step=100)
        store = InMemoryLineageStore(config)

        assert store.should_track_cell_diffs(50) is True
        assert store.should_track_cell_diffs(150) is False

    def test_to_json(self):
        """to_json() exports valid JSON."""
        config = TracePipeConfig()
        store = InMemoryLineageStore(config)

        _step_id = store.append_step(
            operation="test",
            stage=None,
            code_file=None,
            code_line=None,
            params={},
            input_shape=None,
            output_shape=None,
        )

        json_str = store.to_json()

        assert "tracepipe_version" in json_str
        assert "steps" in json_str


class TestMemoryManagement:
    """Tests for memory management and mass update detection."""

    def test_tracking_works_without_oom(self):
        """Basic tracking doesn't cause memory issues."""
        tracepipe.enable()
        tracepipe.watch("value")

        df = pd.DataFrame({"value": range(10_000)})
        df["value"] = df["value"] + 1

        stats = tracepipe.stats()
        assert stats["enabled"]

    def test_mass_update_detection(self):
        """Mass updates are detected when threshold exceeded."""
        tracepipe.enable()
        tracepipe.configure(max_diffs_per_step=100)
        tracepipe.watch("value")

        df = pd.DataFrame({"value": [np.nan] * 1000})
        df = df.fillna(0)

        mass = tracepipe.mass_updates()
        assert len(mass) >= 1


class TestLargeGroupMembership:
    """Tests for large group handling."""

    def test_large_group_count_only(self):
        """Groups exceeding threshold store count only."""
        tracepipe.enable()
        tracepipe.configure(max_group_membership_size=5)

        df = pd.DataFrame({"category": ["A"] * 10, "value": range(10)})
        df.groupby("category").sum()

        group = tracepipe.explain_group("A")
        assert group.row_count == 10
        assert group.is_count_only is True
        assert group.row_ids == []


class TestLineageGaps:
    """Tests for lineage gap computation."""

    def test_fully_tracked_row(self):
        """Row with full tracking reports no gaps."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.dropna()

        row = tracepipe.explain(0)
        assert row.is_fully_tracked

    def test_partial_completeness_gap(self):
        """apply() creates a gap in lineage."""
        tracepipe.enable()
        tracepipe.watch("a")
        df = pd.DataFrame({"a": [1, 2, 3]})

        df.apply(lambda x: x * 2)

        row = tracepipe.explain(0)
        gaps = row.gaps
        assert gaps is not None

    def test_gaps_summary(self):
        """LineageGaps.summary() returns readable string."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.dropna()

        row = tracepipe.explain(0)
        summary = row.gaps.summary()
        assert isinstance(summary, str)

# tests/test_api.py
"""
Tests for tracepipe/api.py - Public API functions and query result classes.
"""

import pandas as pd

import tracepipe
from tracepipe import TracePipeConfig


class TestEnableDisable:
    """Tests for enable/disable/reset functions."""

    def test_enable_creates_context(self):
        """enable() creates tracking context."""
        tracepipe.enable()

        assert tracepipe.stats()["enabled"] is True

    def test_disable_stops_tracking(self):
        """disable() stops tracking."""
        tracepipe.enable()
        tracepipe.disable()

        assert tracepipe.stats()["enabled"] is False

    def test_reset_clears_state(self):
        """reset() clears all tracking state."""
        tracepipe.enable()
        tracepipe.watch("a")
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()

        assert len(tracepipe.steps()) > 0

        tracepipe.reset()

        assert len(tracepipe.steps()) == 0
        assert tracepipe.stats()["total_diffs"] == 0

    def test_multiple_enable_disable_cycles(self):
        """Multiple enable/disable cycles work correctly."""
        for _ in range(3):
            tracepipe.enable()
            df = pd.DataFrame({"a": [1, None, 3]})
            df = df.dropna()
            assert tracepipe.stats()["enabled"] is True
            tracepipe.disable()
            assert tracepipe.stats()["enabled"] is False


class TestConfigure:
    """Tests for configure() function."""

    def test_configure_updates_config(self):
        """configure() updates configuration."""
        tracepipe.enable()
        tracepipe.configure(max_diffs_per_step=500)

        from tracepipe.context import get_context

        ctx = get_context()
        assert ctx.config.max_diffs_per_step == 500

    def test_configure_invalid_key_raises(self):
        """configure() raises ValueError for invalid keys."""
        tracepipe.enable()

        import pytest

        with pytest.raises(ValueError) as exc_info:
            tracepipe.configure(invalid_key=123)

        assert "Invalid configuration key" in str(exc_info.value)
        assert "invalid_key" in str(exc_info.value)

    def test_configure_typo_raises(self):
        """configure() catches common typos."""
        tracepipe.enable()

        import pytest

        # Common typo: max_diff_in_memory instead of max_diffs_in_memory
        with pytest.raises(ValueError):
            tracepipe.configure(max_diff_in_memory=1000)


class TestWatchUnwatch:
    """Tests for watch/unwatch functions."""

    def test_watch_adds_columns(self):
        """watch() adds columns to watched set."""
        tracepipe.enable()

        tracepipe.watch("col1", "col2")

        stats = tracepipe.stats()
        assert "col1" in stats["watched_columns"]
        assert "col2" in stats["watched_columns"]

    def test_unwatch_removes_columns(self):
        """unwatch() removes columns from watched set."""
        tracepipe.enable()
        tracepipe.watch("col1", "col2")

        tracepipe.unwatch("col1")

        stats = tracepipe.stats()
        assert "col1" not in stats["watched_columns"]
        assert "col2" in stats["watched_columns"]

    def test_watch_all(self):
        """watch_all() adds all columns from DataFrame."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

        tracepipe.watch_all(df)

        stats = tracepipe.stats()
        assert "a" in stats["watched_columns"]
        assert "b" in stats["watched_columns"]
        assert "c" in stats["watched_columns"]

    def test_clear_watch(self):
        """clear_watch() removes all watched columns."""
        tracepipe.enable()
        tracepipe.watch("col1", "col2", "col3")

        tracepipe.clear_watch()

        stats = tracepipe.stats()
        assert len(stats["watched_columns"]) == 0


class TestFluentChaining:
    """Tests for fluent API chaining."""

    def test_enable_returns_module(self):
        """enable() returns tracepipe module for chaining."""
        result = tracepipe.enable()

        assert result is tracepipe

    def test_fluent_chaining(self):
        """Can chain enable().watch()."""
        tracepipe.enable().watch("col1", "col2")

        stats = tracepipe.stats()
        assert "col1" in stats["watched_columns"]
        assert "col2" in stats["watched_columns"]

    def test_full_fluent_chain(self):
        """Full fluent chain works."""
        tracepipe.enable().configure(max_diffs_per_step=1000).watch("a")

        from tracepipe.context import get_context

        ctx = get_context()
        assert ctx.config.max_diffs_per_step == 1000
        assert "a" in ctx.watched_columns

    def test_disable_returns_module(self):
        """disable() returns tracepipe module for chaining."""
        tracepipe.enable()
        result = tracepipe.disable()

        assert result is tracepipe


class TestRegister:
    """Tests for register() function."""

    def test_register_manual(self):
        """register() manually registers a DataFrame."""
        tracepipe.enable()

        tracepipe.disable()
        df = pd.DataFrame({"a": [1, 2, 3]})
        tracepipe.enable()

        tracepipe.register(df)

        _df2 = df.head(2)
        dropped = tracepipe.dropped_rows()
        assert 2 in dropped


class TestStage:
    """Tests for stage() context manager."""

    def test_stage_context_manager(self):
        """Stage context manager sets stage name."""
        tracepipe.enable()

        with tracepipe.stage("cleaning"):
            df = pd.DataFrame({"a": [1, None, 3]})
            df = df.dropna()

        steps_list = tracepipe.steps()
        cleaning_steps = [s for s in steps_list if s["stage"] == "cleaning"]
        assert len(cleaning_steps) >= 1


class TestQueryFunctions:
    """Tests for query API functions."""

    def test_dropped_rows(self):
        """dropped_rows() returns dropped row IDs."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()

        dropped = tracepipe.dropped_rows()

        assert 1 in dropped

    def test_dropped_rows_by_step(self):
        """dropped_rows(by_step=True) returns correct counts."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, None, 3, None, 5]})
        df = df.dropna()

        by_step = tracepipe.dropped_rows(by_step=True)

        assert isinstance(by_step, dict)
        assert sum(by_step.values()) == 2

    def test_alive_rows(self):
        """alive_rows() returns non-dropped row IDs."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        df = df.head(3)

        alive = tracepipe.alive_rows()

        assert 0 in alive
        assert 1 in alive
        assert 2 in alive
        assert 3 not in alive
        assert 4 not in alive

    def test_explain_many(self):
        """explain_many() returns multiple RowLineageResults."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        df = df.head(3)

        results = tracepipe.explain_many([0, 1, 4])

        assert len(results) == 3
        assert results[0].is_alive is True
        assert results[1].is_alive is True
        assert results[2].is_alive is False

    def test_aggregation_groups(self):
        """aggregation_groups() lists tracked groups."""
        tracepipe.enable()
        df = pd.DataFrame({"cat": ["A", "B", "A"], "val": [1, 2, 3]})
        df.groupby("cat").sum()

        groups = tracepipe.aggregation_groups()

        assert "A" in groups
        assert "B" in groups

    def test_steps(self):
        """steps() returns step metadata."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()

        steps = tracepipe.steps()

        assert len(steps) >= 1
        assert "operation" in steps[0]

    def test_stats(self):
        """stats() returns tracking statistics."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()

        s = tracepipe.stats()

        assert s["enabled"] is True
        assert s["total_steps"] >= 1
        assert s["total_diffs"] >= 1


class TestRowLineageResult:
    """Tests for RowLineageResult query object."""

    def test_explain_returns_result(self):
        """explain() returns RowLineageResult."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.dropna()

        row = tracepipe.explain(0)

        assert row is not None
        assert row.row_id == 0

    def test_is_alive_true(self):
        """is_alive property returns True for surviving rows."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.head(2)

        row = tracepipe.explain(0)

        assert row.is_alive is True

    def test_is_alive_false(self):
        """is_alive property returns False for dropped rows."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.head(2)

        row = tracepipe.explain(2)

        assert row.is_alive is False

    def test_dropped_at(self):
        """dropped_at property returns operation name."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.head(2)

        row = tracepipe.explain(2)

        assert row.dropped_at is not None
        assert "head" in row.dropped_at.lower()

    def test_cell_history(self):
        """cell_history() returns column-specific history."""
        tracepipe.enable()
        tracepipe.watch("a")
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        df["a"] = df["a"] * 10

        row = tracepipe.explain(0)
        history = row.cell_history("a")

        assert len(history) >= 1

    def test_history(self):
        """history() returns full history."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()

        row = tracepipe.explain(1)
        history = row.history()

        assert len(history) >= 1

    def test_gaps(self):
        """gaps property returns LineageGaps."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.dropna()

        row = tracepipe.explain(0)

        assert row.gaps is not None
        assert hasattr(row.gaps, "is_fully_tracked")

    def test_to_dict(self):
        """to_dict() returns serializable dict."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()

        row = tracepipe.explain(1)
        data = row.to_dict()

        assert "row_id" in data
        assert "is_alive" in data
        assert "dropped_at" in data
        assert "history" in data

    def test_repr(self):
        """__repr__ returns readable string."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.dropna()

        row = tracepipe.explain(0)

        assert "RowLineage" in repr(row)
        assert "row_id=0" in repr(row)


class TestGroupLineageResult:
    """Tests for GroupLineageResult query object."""

    def test_explain_group_returns_result(self):
        """explain_group() returns GroupLineageResult."""
        tracepipe.enable()
        df = pd.DataFrame({"cat": ["A", "A"], "val": [1, 2]})
        df.groupby("cat").sum()

        group = tracepipe.explain_group("A")

        assert group is not None
        assert group.group_key == "A"

    def test_row_ids(self):
        """row_ids property returns member IDs."""
        tracepipe.enable()
        df = pd.DataFrame({"cat": ["A", "A", "B"], "val": [1, 2, 3]})
        df.groupby("cat").mean()

        group = tracepipe.explain_group("A")

        assert group.row_count == 2
        assert set(group.row_ids) == {0, 1}

    def test_group_column(self):
        """group_column property returns correct column."""
        tracepipe.enable()
        df = pd.DataFrame({"cat": ["A", "A"], "val": [1, 2]})
        df.groupby("cat").sum()

        group = tracepipe.explain_group("A")

        assert group.group_column == "cat"

    def test_aggregation_functions(self):
        """aggregation_functions property returns agg info."""
        tracepipe.enable()
        df = pd.DataFrame({"cat": ["A", "A"], "val": [1, 2]})
        df.groupby("cat").agg({"val": "sum"})

        group = tracepipe.explain_group("A")

        assert group.aggregation_functions is not None

    def test_get_contributing_rows(self):
        """get_contributing_rows() returns row lineage results."""
        tracepipe.enable()
        df = pd.DataFrame({"cat": ["A", "A", "B"], "val": [1, 2, 3]})
        df.groupby("cat").sum()

        group = tracepipe.explain_group("A")
        contributing = group.get_contributing_rows(limit=10)

        assert len(contributing) == 2

    def test_to_dict(self):
        """to_dict() returns serializable dict."""
        tracepipe.enable()
        df = pd.DataFrame({"cat": ["A", "A"], "val": [1, 2]})
        df.groupby("cat").sum()

        group = tracepipe.explain_group("A")
        data = group.to_dict()

        assert "group_key" in data
        assert "row_count" in data
        assert "row_ids" in data

    def test_repr(self):
        """__repr__ returns readable string."""
        tracepipe.enable()
        df = pd.DataFrame({"cat": ["A", "A"], "val": [1, 2]})
        df.groupby("cat").sum()

        group = tracepipe.explain_group("A")

        assert "GroupLineage" in repr(group)
        assert "rows=2" in repr(group)


class TestExport:
    """Tests for export functions."""

    def test_export_json(self, tmp_path):
        """export_json() creates valid JSON file."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()

        filepath = tmp_path / "lineage.json"
        tracepipe.export_json(str(filepath))

        assert filepath.exists()
        content = filepath.read_text()
        assert "tracepipe_version" in content

    def test_save_html(self, tmp_path):
        """save() creates HTML report."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()

        filepath = tmp_path / "report.html"
        tracepipe.save(str(filepath))

        assert filepath.exists()
        content = filepath.read_text()
        assert "TracePipe" in content

    def test_export_arrow_import_error(self, tmp_path, monkeypatch):
        """export_arrow() gives helpful error when pyarrow not installed."""
        import builtins
        import sys

        tracepipe.enable()
        _df = pd.DataFrame({"a": [1, 2, 3]})

        # Remove pyarrow from sys.modules if present
        _orig_modules = sys.modules.copy()
        if "pyarrow" in sys.modules:
            del sys.modules["pyarrow"]
        if "pyarrow.parquet" in sys.modules:
            del sys.modules["pyarrow.parquet"]

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyarrow.parquet" or name == "pyarrow":
                raise ImportError("No module named 'pyarrow'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        import pytest

        filepath = tmp_path / "lineage.parquet"
        with pytest.raises(ImportError) as exc_info:
            tracepipe.export_arrow(str(filepath))

        assert "pyarrow" in str(exc_info.value)
        assert "pip install" in str(exc_info.value)


class TestProtocolExtensibility:
    """Tests for protocol-based extensibility."""

    def test_custom_backend_accepted(self):
        """enable() accepts custom LineageBackend."""
        from tracepipe.storage.lineage_store import InMemoryLineageStore

        config = TracePipeConfig()
        custom_backend = InMemoryLineageStore(config)

        tracepipe.enable(backend=custom_backend)

        assert tracepipe.stats()["enabled"] is True

    def test_custom_identity_accepted(self):
        """enable() accepts custom RowIdentityStrategy."""
        from tracepipe.storage.row_identity import PandasRowIdentity

        config = TracePipeConfig()
        custom_identity = PandasRowIdentity(config)

        tracepipe.enable(identity=custom_identity)
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.head(2)

        dropped = tracepipe.dropped_rows()
        assert 2 in dropped

    def test_protocol_isinstance_check(self):
        """Default implementations satisfy Protocol."""
        from tracepipe.storage.base import LineageBackend, RowIdentityStrategy
        from tracepipe.storage.lineage_store import InMemoryLineageStore
        from tracepipe.storage.row_identity import PandasRowIdentity

        config = TracePipeConfig()
        backend = InMemoryLineageStore(config)
        identity = PandasRowIdentity(config)

        assert isinstance(backend, LineageBackend)
        assert isinstance(identity, RowIdentityStrategy)

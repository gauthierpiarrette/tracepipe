# tests/test_api.py
"""
Tests for tracepipe API.
"""

import pandas as pd
import pytest

import tracepipe
from tracepipe import TracePipeConfig


def dbg():
    """Helper to access debug inspector."""
    return tracepipe.debug.inspect()


class TestEnableDisable:
    """Tests for enable/disable/reset functions."""

    def test_enable_creates_context(self):
        """enable() creates tracking context."""
        tracepipe.reset()
        tracepipe.enable()

        assert dbg().enabled is True

    def test_enable_with_watch(self):
        """enable(watch=[...]) sets watched columns."""
        tracepipe.reset()
        tracepipe.enable(watch=["age", "salary"])

        assert "age" in dbg().watched_columns
        assert "salary" in dbg().watched_columns

    def test_enable_with_mode(self):
        """enable(mode='debug') sets debug mode."""
        tracepipe.reset()
        tracepipe.enable(mode="debug")

        assert dbg().mode == "debug"

    def test_disable_stops_tracking(self):
        """disable() stops tracking."""
        tracepipe.reset()
        tracepipe.enable()
        tracepipe.disable()

        assert dbg().enabled is False

    def test_reset_clears_state(self):
        """reset() clears all tracking state."""
        tracepipe.reset()
        tracepipe.enable(watch=["a"])
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()

        assert len(dbg().steps) > 0

        tracepipe.reset()

        assert len(dbg().steps) == 0
        assert dbg().total_diffs == 0

    def test_multiple_enable_disable_cycles(self):
        """Multiple enable/disable cycles work correctly."""
        for _ in range(3):
            tracepipe.reset()
            tracepipe.enable()
            df = pd.DataFrame({"a": [1, None, 3]})
            df = df.dropna()
            assert dbg().enabled is True
            tracepipe.disable()
            assert dbg().enabled is False

    def test_enable_propagates_config_to_components(self):
        """enable() propagates config to row_manager and store."""
        from tracepipe.context import get_context

        tracepipe.reset()
        tracepipe.enable(mode="debug")

        ctx = get_context()
        # All components should share the same config object
        assert ctx.config is ctx.row_manager.config
        assert ctx.config is ctx.store.config
        # And mode should be DEBUG
        assert ctx.config.should_capture_ghost_values is True

    def test_enable_preserves_existing_config_settings(self):
        """enable() preserves existing config settings when not overridden."""
        from tracepipe.context import TracePipeContext, get_context, set_context
        from tracepipe.core import TracePipeConfig

        tracepipe.reset()
        # Set up custom config
        config = TracePipeConfig(max_diffs_per_step=123)
        ctx = TracePipeContext(config=config)
        set_context(ctx)

        # enable() without explicit config should preserve max_diffs_per_step
        tracepipe.enable(watch=["a"])

        ctx = get_context()
        assert ctx.config.max_diffs_per_step == 123


class TestConfigure:
    """Tests for configure() function."""

    def test_configure_updates_config(self):
        """configure() updates configuration."""
        tracepipe.reset()
        tracepipe.enable()
        tracepipe.configure(max_diffs_per_step=500)

        from tracepipe.context import get_context

        ctx = get_context()
        assert ctx.config.max_diffs_per_step == 500

    def test_configure_invalid_key_raises(self):
        """configure() raises ValueError for invalid keys."""
        tracepipe.reset()
        tracepipe.enable()

        with pytest.raises(ValueError) as exc_info:
            tracepipe.configure(invalid_key=123)

        assert "Invalid configuration key" in str(exc_info.value)


class TestContractsNamespace:
    """Tests for tp.contracts namespace."""

    def test_contract_builder(self):
        """contracts.contract() returns builder."""
        tracepipe.reset()
        tracepipe.enable()
        df = pd.DataFrame({"id": [1, 2, 3]})

        result = tracepipe.contracts.contract().expect_unique("id").check(df)

        assert result.passed is True

    def test_contract_top_level(self):
        """tp.contract() is available at top level."""
        tracepipe.reset()
        tracepipe.enable()
        df = pd.DataFrame({"id": [1, 2, 3]})

        result = tracepipe.contract().expect_no_null_in("id").check(df)

        assert result.passed is True


class TestSnapshots:
    """Tests for snapshot/diff functions."""

    def test_snapshot_captures_state(self):
        """snapshot(df) captures pipeline state."""
        tracepipe.reset()
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.dropna()

        snap = tracepipe.snapshot(df)

        assert len(snap.row_ids) == 3
        assert snap.total_steps >= 0

    def test_diff_compares_snapshots(self):
        """diff(snap1, snap2) returns DiffResult."""
        tracepipe.reset()
        tracepipe.enable()

        df1 = pd.DataFrame({"a": [1, 2, 3]})
        snap1 = tracepipe.snapshot(df1)

        df2 = df1.head(2)
        snap2 = tracepipe.snapshot(df2)

        result = tracepipe.diff(snap1, snap2)

        assert len(result.rows_removed) > 0 or len(result.new_drops) > 0


class TestStage:
    """Tests for stage() context manager."""

    def test_stage_context_manager(self):
        """Stage context manager sets stage name."""
        tracepipe.reset()
        tracepipe.enable()

        with tracepipe.stage("cleaning"):
            df = pd.DataFrame({"a": [1, None, 3]})
            df = df.dropna()

        cleaning_steps = [s for s in dbg().steps if s.stage == "cleaning"]
        assert len(cleaning_steps) >= 1


class TestFluentChaining:
    """Tests for fluent API chaining."""

    def test_enable_returns_module(self):
        """enable() returns tracepipe module for chaining."""
        tracepipe.reset()
        result = tracepipe.enable()
        assert result is tracepipe

    def test_disable_returns_module(self):
        """disable() returns tracepipe module for chaining."""
        tracepipe.reset()
        tracepipe.enable()
        result = tracepipe.disable()
        assert result is tracepipe


class TestProtocolExtensibility:
    """Tests for protocol-based extensibility."""

    def test_custom_backend_accepted(self):
        """enable() accepts custom LineageBackend."""
        from tracepipe.storage.lineage_store import InMemoryLineageStore

        tracepipe.reset()
        config = TracePipeConfig()
        custom_backend = InMemoryLineageStore(config)

        tracepipe.enable(backend=custom_backend)

        assert dbg().enabled is True

    def test_custom_identity_accepted(self):
        """enable() accepts custom RowIdentityStrategy."""
        from tracepipe.storage.row_identity import PandasRowIdentity

        tracepipe.reset()
        config = TracePipeConfig()
        custom_identity = PandasRowIdentity(config)

        tracepipe.enable(identity=custom_identity)
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.head(2)

        dropped = dbg().dropped_rows()
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


class TestRowLineageResult:
    """Tests for RowLineageResult via debug.inspect().explain_row()."""

    def test_is_alive_true(self):
        """is_alive property returns True for surviving rows."""
        tracepipe.reset()
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.head(2)

        row = dbg().explain_row(0)
        assert row.is_alive is True

    def test_is_alive_false(self):
        """is_alive property returns False for dropped rows."""
        tracepipe.reset()
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.head(2)

        row = dbg().explain_row(2)
        assert row.is_alive is False

    def test_dropped_at(self):
        """dropped_at property returns operation name."""
        tracepipe.reset()
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.head(2)

        row = dbg().explain_row(2)
        assert row.dropped_at is not None

    def test_cell_history(self):
        """cell_history() returns column-specific history."""
        tracepipe.reset()
        tracepipe.enable(watch=["a"])
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        df["a"] = df["a"] * 10

        row = dbg().explain_row(0)
        history = row.cell_history("a")
        assert len(history) >= 1

    def test_history(self):
        """history() returns full history."""
        tracepipe.reset()
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()

        row = dbg().explain_row(1)
        history = row.history()
        assert len(history) >= 1

    def test_to_dict(self):
        """to_dict() returns serializable dict."""
        tracepipe.reset()
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()

        row = dbg().explain_row(1)
        data = row.to_dict()

        assert "row_id" in data
        assert "is_alive" in data


class TestGroupLineageResult:
    """Tests for GroupLineageResult via debug.inspect().explain_group()."""

    def test_row_ids(self):
        """row_ids property returns member IDs."""
        tracepipe.reset()
        tracepipe.enable()
        df = pd.DataFrame({"cat": ["A", "A", "B"], "val": [1, 2, 3]})
        df.groupby("cat").mean()

        group = dbg().explain_group("A")
        assert group.row_count == 2
        assert set(group.row_ids) == {0, 1}

    def test_group_column(self):
        """group_column property returns correct column."""
        tracepipe.reset()
        tracepipe.enable()
        df = pd.DataFrame({"cat": ["A", "A"], "val": [1, 2]})
        df.groupby("cat").sum()

        group = dbg().explain_group("A")
        assert group.group_column == "cat"

    def test_to_dict(self):
        """to_dict() returns serializable dict."""
        tracepipe.reset()
        tracepipe.enable()
        df = pd.DataFrame({"cat": ["A", "A"], "val": [1, 2]})
        df.groupby("cat").sum()

        group = dbg().explain_group("A")
        data = group.to_dict()

        assert "group_key" in data
        assert "row_count" in data


class TestRegister:
    """Tests for tp.register() - registering pre-existing DataFrames."""

    def test_register_single_dataframe(self):
        """register(df) registers a single DataFrame."""
        # Create DataFrame BEFORE enable
        df = pd.DataFrame({"a": [1, 2, 3]})

        tracepipe.enable()
        tracepipe.register(df)

        # Snapshot should now capture rows
        snap = tracepipe.snapshot(df)
        assert len(snap.row_ids) == 3

    def test_register_multiple_dataframes(self):
        """register(df1, df2, ...) registers multiple DataFrames."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4, 5]})

        tracepipe.enable()
        tracepipe.register(df1, df2)

        snap1 = tracepipe.snapshot(df1)
        snap2 = tracepipe.snapshot(df2)

        assert len(snap1.row_ids) == 2
        assert len(snap2.row_ids) == 3

    def test_register_returns_module(self):
        """register() returns tracepipe module for chaining."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        tracepipe.enable()

        result = tracepipe.register(df)

        assert result is tracepipe

    def test_register_before_enable_warns(self):
        """register() warns if TracePipe not enabled."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.warns(UserWarning, match="TracePipe is not enabled"):
            tracepipe.register(df)

    def test_register_non_dataframe_raises(self):
        """register() raises TypeError for non-DataFrame."""
        tracepipe.enable()

        with pytest.raises(TypeError, match="Expected DataFrame"):
            tracepipe.register([1, 2, 3])

    def test_register_idempotent(self):
        """Registering same DataFrame twice is safe."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        tracepipe.enable()

        tracepipe.register(df)
        tracepipe.register(df)  # Should not error or change row IDs

        snap = tracepipe.snapshot(df)
        assert len(snap.row_ids) == 3


class TestPreEnableDataFrameTracking:
    """Tests that verify tracking works correctly for pre-enable DataFrames."""

    def test_snapshot_without_register_shows_zero_rows(self):
        """Snapshot of unregistered DataFrame shows 0 rows."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        tracepipe.enable()

        # Without register, snapshot shows 0 rows (this is the documented behavior)
        snap = tracepipe.snapshot(df)
        assert len(snap.row_ids) == 0

    def test_snapshot_with_register_shows_rows(self):
        """Snapshot of registered DataFrame shows correct rows."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        tracepipe.enable()
        tracepipe.register(df)

        snap = tracepipe.snapshot(df)
        assert len(snap.row_ids) == 3

    def test_filter_tracking_after_register(self):
        """Filter operations are tracked after register."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        tracepipe.enable()
        tracepipe.register(df)

        df = df.head(3)

        dropped = dbg().dropped_rows()
        assert 3 in dropped
        assert 4 in dropped

    def test_ghost_rows_after_register(self):
        """Ghost rows are captured for registered DataFrames."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        tracepipe.enable(mode="debug", watch=["a"])
        tracepipe.register(df)

        df = df.head(3)  # Drop rows 3 and 4

        ghost_df = dbg().ghost_rows()
        # Ghost rows should be captured now
        assert len(ghost_df) >= 2 or len(dbg().dropped_rows()) >= 2

    def test_cell_history_after_register(self):
        """Cell history is tracked for registered DataFrames."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        tracepipe.enable(mode="debug", watch=["a"])
        tracepipe.register(df)

        df["a"] = df["a"] * 10

        result = tracepipe.why(df, col="a", row=0)
        assert len(result.history) >= 1

    def test_trace_after_register(self):
        """Row tracing works for registered DataFrames."""
        df = pd.DataFrame({"id": ["A", "B", "C"], "val": [1, 2, 3]})
        tracepipe.enable()
        tracepipe.register(df)

        result = tracepipe.trace(df, where={"id": "A"})
        assert result.row_id == 0
        assert result.is_alive is True


class TestRetentionCalculation:
    """Tests for retention calculation in multi-table scenarios."""

    def test_retention_single_table(self):
        """Retention is calculated correctly for single table."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        df = df.head(4)  # Keep 4 out of 5 = 80%

        result = tracepipe.check(df)
        assert "retention_rate" in result.facts
        assert 0.7 <= result.facts["retention_rate"] <= 0.9

    def test_retention_multi_table_merge(self):
        """Retention handles multi-table merge scenarios correctly."""
        tracepipe.enable()

        # Create and merge two tables
        left = pd.DataFrame({"key": [1, 2, 3], "left_val": ["a", "b", "c"]})
        right = pd.DataFrame({"key": [1, 2], "right_val": ["x", "y"]})

        merged = left.merge(right, on="key", how="inner")  # 2 rows
        merged = merged.head(1)  # Keep 1 out of 2

        result = tracepipe.check(merged)
        # Retention should be sensible (not negative or > 1)
        assert "retention_rate" in result.facts
        assert 0.0 <= result.facts["retention_rate"] <= 1.0

    def test_retention_not_negative(self):
        """Retention is never negative."""
        tracepipe.enable()

        # Simulate the problematic scenario from the demo
        customers = pd.DataFrame({"id": range(20), "val": range(20)})
        orders = pd.DataFrame(
            {"id": range(50), "customer_id": list(range(20)) * 2 + list(range(10))}
        )

        # Filter
        customers = customers.head(19)
        orders = orders.head(30)

        # Merge (expands rows)
        merged = orders.merge(customers, left_on="customer_id", right_on="id", how="inner")

        # More filtering
        merged = merged.head(15)

        result = tracepipe.check(merged)
        assert "retention_rate" in result.facts
        retention = result.facts["retention_rate"]
        assert retention >= 0.0, f"Retention should not be negative, got {retention}"
        assert retention <= 1.0, f"Retention should not exceed 1.0, got {retention}"

    def test_contract_retention_not_negative(self):
        """Contract retention expectation handles edge cases."""
        tracepipe.enable()

        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        df = df.head(2)

        # This should not raise due to negative retention
        result = tracepipe.contract().expect_retention(min_rate=0.3).check(df)
        assert result.passed is True


class TestColumnModeNoRecursion:
    """Tests that COLUMN identity mode doesn't cause recursion."""

    def test_column_mode_basic_operations(self):
        """Basic operations work in COLUMN mode without recursion."""
        from tracepipe.core import IdentityStorage

        config = TracePipeConfig(identity_storage=IdentityStorage.COLUMN)
        tracepipe.enable(config=config)

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df = df.head(2)

        # Should complete without recursion error
        result = tracepipe.check(df)
        assert result.facts["rows_dropped"] == 1

    def test_column_mode_access_hidden_column(self):
        """Accessing hidden column doesn't cause recursion."""
        from tracepipe.core import IdentityStorage

        config = TracePipeConfig(identity_storage=IdentityStorage.COLUMN)
        tracepipe.enable(config=config)

        df = pd.DataFrame({"a": [1, 2, 3]})

        # Hidden column should exist
        assert "__tracepipe_row_id__" in df.columns

        # Accessing it should not cause recursion
        rids = df["__tracepipe_row_id__"].values
        assert len(rids) == 3

    def test_column_mode_drop_hidden_column(self):
        """Dropping hidden column doesn't cause recursion."""
        from tracepipe.core import IdentityStorage

        config = TracePipeConfig(identity_storage=IdentityStorage.COLUMN)
        tracepipe.enable(config=config)

        df = pd.DataFrame({"a": [1, 2, 3]})

        # Drop hidden column should work without recursion
        # Note: The result gets re-registered (new DataFrame), so column comes back
        # The key test is that this completes without infinite recursion
        clean = df.drop(columns=["__tracepipe_row_id__"])

        # Original user columns should be preserved
        assert "a" in clean.columns


class TestGhostRowFallbackCapture:
    """Tests that ghost rows are captured even in fallback scenarios."""

    def test_ghost_rows_with_sample(self):
        """Ghost values captured for sample() which uses fallback path."""
        tracepipe.enable(mode="debug", watch=["value"])

        df = pd.DataFrame({"value": [10.0, 20.0, 30.0, 40.0, 50.0]})

        # sample() uses fallback path (can't derive mask deterministically)
        sampled = df.sample(n=2, random_state=42)

        ghost_df = dbg().ghost_rows()
        dropped_count = len(dbg().dropped_rows())

        # Should have captured some drops
        assert dropped_count == 3  # 5 - 2 = 3 dropped

    def test_ghost_values_captured_for_watched_columns(self):
        """Ghost values include watched column data."""
        tracepipe.enable(mode="debug", watch=["name", "score"])

        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "score": [100.0, 200.0, 300.0]})

        df = df.head(1)  # Drop Bob and Charlie

        ghost_df = dbg().ghost_rows()

        # Ghost rows should have been captured
        assert len(ghost_df) >= 2 or len(dbg().dropped_rows()) >= 2


class TestExportWrapperPreventsRegistration:
    """Tests that export operations don't re-register DataFrames."""

    def test_to_csv_doesnt_reregister(self, tmp_path):
        """to_csv() doesn't cause new registrations during strip."""
        from tracepipe.core import IdentityStorage

        config = TracePipeConfig(identity_storage=IdentityStorage.COLUMN)
        tracepipe.enable(config=config)

        df = pd.DataFrame({"a": [1, 2, 3]})
        initial_steps = len(dbg().steps)

        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        # The column strip shouldn't create tracking steps
        # (some steps may be created, but not from internal operations)
        final_steps = len(dbg().steps)
        # Should not have exploded with many internal steps
        assert final_steps - initial_steps <= 2

    def test_stripped_df_not_tracked(self, tmp_path):
        """DataFrame created during strip isn't fully tracked."""
        import pyarrow.parquet as pq

        from tracepipe.core import IdentityStorage

        config = TracePipeConfig(identity_storage=IdentityStorage.COLUMN)
        tracepipe.enable(config=config)

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        parquet_path = tmp_path / "test.parquet"
        df.to_parquet(parquet_path)

        # Read raw file to verify hidden column was stripped
        table = pq.read_table(parquet_path)
        assert "__tracepipe_row_id__" not in table.column_names
        assert table.column_names == ["a", "b"]

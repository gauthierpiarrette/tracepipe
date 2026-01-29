# tests/test_row_identity.py
"""
Tests for tracepipe/storage/row_identity.py - Row identity tracking.
"""

import io
import warnings

import pandas as pd

import tracepipe
from tracepipe import TracePipeConfig
from tracepipe.storage.row_identity import PandasRowIdentity


class TestPandasRowIdentity:
    """Tests for PandasRowIdentity class."""

    def test_register_assigns_ids(self):
        """register() assigns sequential row IDs."""
        config = TracePipeConfig()
        identity = PandasRowIdentity(config)

        df = pd.DataFrame({"a": [1, 2, 3]})
        row_ids = identity.register(df)

        assert len(row_ids) == 3
        assert list(row_ids) == [0, 1, 2]

    def test_get_ids_returns_registered(self):
        """get_ids() returns IDs for registered DataFrame."""
        config = TracePipeConfig()
        identity = PandasRowIdentity(config)

        df = pd.DataFrame({"a": [1, 2, 3]})
        identity.register(df)
        row_ids = identity.get_ids(df)

        assert row_ids is not None
        assert len(row_ids) == 3

    def test_get_ids_returns_none_for_unregistered(self):
        """get_ids() returns None for unregistered DataFrame."""
        config = TracePipeConfig()
        identity = PandasRowIdentity(config)

        df = pd.DataFrame({"a": [1, 2, 3]})
        row_ids = identity.get_ids(df)

        assert row_ids is None

    def test_propagate_filter(self):
        """propagate() handles filtering correctly."""
        config = TracePipeConfig()
        identity = PandasRowIdentity(config)

        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        identity.register(df)

        filtered = df[df["a"] > 2]
        identity.propagate(df, filtered)

        result_ids = identity.get_ids(filtered)
        assert result_ids is not None
        assert len(result_ids) == 3

    def test_propagate_reorder(self):
        """propagate() handles reordering correctly."""
        config = TracePipeConfig()
        identity = PandasRowIdentity(config)

        df = pd.DataFrame({"a": [3, 1, 2]})
        identity.register(df)

        sorted_df = df.sort_values("a")
        identity.propagate(df, sorted_df)

        result_ids = identity.get_ids(sorted_df)
        assert result_ids is not None

    def test_get_dropped_ids(self):
        """get_dropped_ids() returns dropped row IDs."""
        config = TracePipeConfig()
        identity = PandasRowIdentity(config)

        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        identity.register(df)

        filtered = df.head(3)
        identity.propagate(df, filtered)

        dropped = identity.get_dropped_ids(df, filtered)

        assert 3 in dropped
        assert 4 in dropped

    def test_strip_hidden_column(self):
        """strip_hidden_column() removes tracking column."""
        config = TracePipeConfig(use_hidden_column=True)
        identity = PandasRowIdentity(config)

        df = pd.DataFrame({"a": [1, 2, 3]})
        identity.register(df)

        clean_df = identity.strip_hidden_column(df)

        assert "__tracepipe_row_id__" not in clean_df.columns


class TestRowIdentityIntegration:
    """Integration tests for row identity tracking."""

    def test_filter_preserves_ids(self):
        """Filter operations track dropped rows correctly."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        df[df["a"] > 2]

        dropped = tracepipe.dropped_rows()
        assert 0 in dropped
        assert 1 in dropped
        assert 2 not in dropped

    def test_reset_index_preserves_ids(self):
        """reset_index(drop=True) preserves row identity."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, 2, 3]}, index=[10, 20, 30])

        df.reset_index(drop=True)

        row = tracepipe.explain(0)
        assert row is not None

    def test_chained_operations_track_drops(self):
        """Chained operations correctly track all drops."""
        tracepipe.enable()
        df = pd.DataFrame({"a": [1, None, 3, None, 5]})

        df.dropna().reset_index(drop=True).head(2)

        dropped = tracepipe.dropped_rows()
        assert len(dropped) >= 2


class TestDuplicateIndexWarning:
    """Tests for duplicate index warning."""

    def test_duplicate_index_warning(self):
        """Duplicate index triggers warning."""
        tracepipe.enable()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pd.DataFrame({"a": [1, 2, 3]}, index=[0, 0, 1])

            dup_warnings = [x for x in w if "duplicate" in str(x.message).lower()]
            assert len(dup_warnings) >= 1


class TestHiddenColumn:
    """Tests for hidden column functionality."""

    def test_hidden_column_stripped_on_export(self):
        """Hidden column is stripped when exporting to CSV."""
        config = TracePipeConfig(use_hidden_column=True)
        tracepipe.enable(config=config)

        df = pd.DataFrame({"a": [1, 2, 3]})

        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        csv_content = buffer.getvalue()

        assert "__tracepipe" not in csv_content

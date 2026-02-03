"""
I/O operations tests: CSV, Parquet file handling.
Focused on critical paths: auto-registration, hidden column stripping, roundtrips.
"""

import pandas as pd
import pytest

import tracepipe as tp
from tracepipe.core import IdentityStorage


def dbg():
    """Helper to access debug inspector."""
    return tp.debug.inspect()


@pytest.fixture(autouse=True)
def reset_tp():
    """Reset TracePipe before each test."""
    tp.reset()
    yield
    try:
        tp.disable()
    except Exception:
        pass


class TestCSV:
    """CSV read/write operations."""

    def test_read_csv_auto_registers(self, tmp_path):
        """pd.read_csv() auto-registers and tracks operations."""
        tp.enable()
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a,b\n1,2\n3,4\n5,6\n")

        df = pd.read_csv(csv_path)
        df = df.head(2)

        dropped = dbg().dropped_rows()
        assert len(dropped) == 1

    def test_to_csv_strips_hidden_column(self, tmp_path):
        """to_csv() strips __tracepipe_row_id__ column."""
        config = tp.TracePipeConfig(identity_storage=IdentityStorage.COLUMN)
        tp.enable(config=config)

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        csv_path = tmp_path / "output.csv"
        df.to_csv(csv_path, index=False)

        content = csv_path.read_text()
        assert "__tracepipe" not in content
        assert "a,b" in content

    def test_roundtrip_tracking(self, tmp_path):
        """CSV roundtrip maintains tracking."""
        tp.enable()

        df1 = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        df1 = df1.head(2)

        csv_path = tmp_path / "roundtrip.csv"
        df1.to_csv(csv_path, index=False)

        df2 = pd.read_csv(csv_path)
        df2 = df2.head(1)

        dropped = dbg().dropped_rows()
        assert len(dropped) >= 1

    def test_empty_csv(self, tmp_path):
        """Reading empty CSV doesn't crash."""
        tp.enable()
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("a,b,c\n")

        df = pd.read_csv(csv_path)
        assert len(df) == 0


class TestParquet:
    """Parquet operations (requires pyarrow)."""

    @pytest.fixture
    def has_pyarrow(self):
        """Check if pyarrow is available."""
        try:
            import pyarrow  # noqa: F401

            return True
        except ImportError:
            pytest.skip("pyarrow not installed")

    def test_to_parquet_strips_hidden_column(self, tmp_path, has_pyarrow):
        """to_parquet() strips hidden column from the file."""
        import pyarrow.parquet as pq

        config = tp.TracePipeConfig(identity_storage=IdentityStorage.COLUMN)
        tp.enable(config=config)

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        parquet_path = tmp_path / "output.parquet"
        df.to_parquet(parquet_path)

        # Read raw parquet schema to verify hidden column was stripped
        # (don't use pd.read_parquet since that would re-add the column via __init__)
        table = pq.read_table(parquet_path)
        file_columns = table.column_names
        assert "__tracepipe_row_id__" not in file_columns
        assert file_columns == ["a", "b"]


class TestLargeFiles:
    """Large file handling."""

    def test_large_csv(self, tmp_path):
        """Large CSV file can be read and tracked."""
        tp.enable()

        csv_path = tmp_path / "large.csv"
        large_df = pd.DataFrame({"id": range(10_000), "value": range(10_000)})
        large_df.to_csv(csv_path, index=False)

        df = pd.read_csv(csv_path)
        df = df.head(1_000)

        dropped = dbg().dropped_rows()
        assert len(dropped) == 9_000

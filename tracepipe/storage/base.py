# tracepipe/storage/base.py
"""
Protocol definitions for TracePipe storage backends.

These protocols enable:
- Swappable storage backends (InMemory, SQLite, Delta Lake)
- Engine-specific row identity strategies (Pandas, Polars, Spark)
- Easy testing with mock implementations

To add a new backend, implement LineageBackend.
To support a new DataFrame engine, implement RowIdentityStrategy.
"""

from typing import Any, Optional, Protocol, runtime_checkable

from ..core import ChangeType, CompletenessLevel, LineageGaps, TracePipeConfig


@runtime_checkable
class LineageBackend(Protocol):
    """
    Protocol for lineage storage backends.

    Implementations:
    - InMemoryLineageStore (default, v0.2.0)
    - SQLiteBackend (future)
    - DeltaLakeBackend (future)
    """

    config: TracePipeConfig

    def append_diff(
        self,
        step_id: int,
        row_id: int,
        col: str,
        old_val: Any,
        new_val: Any,
        change_type: ChangeType,
    ) -> None:
        """Append a single cell diff."""
        ...

    def append_diff_batch(
        self, step_id: int, diffs: list[tuple], check_threshold: bool = True
    ) -> int:
        """Batch append diffs. Returns count appended."""
        ...

    def append_step(
        self,
        operation: str,
        stage: Optional[str],
        code_file: Optional[str],
        code_line: Optional[int],
        params: dict[str, Any],
        input_shape: Optional[tuple],
        output_shape: Optional[tuple],
        completeness: CompletenessLevel = CompletenessLevel.FULL,
        is_mass_update: bool = False,
        rows_affected: int = 0,
    ) -> int:
        """Append step metadata. Returns step_id."""
        ...

    def append_aggregation(
        self,
        step_id: int,
        group_column: str,
        membership: dict[str, list[int]],
        agg_functions: dict[str, str],
    ) -> None:
        """Record aggregation group membership."""
        ...

    def get_row_history(self, row_id: int) -> list[dict]:
        """Get all events for a specific row."""
        ...

    def get_dropped_rows(self, step_id: Optional[int] = None) -> list[int]:
        """Get dropped row IDs, optionally filtered by step."""
        ...

    def get_dropped_by_step(self) -> dict[str, int]:
        """Get count of dropped rows per operation."""
        ...

    def get_group_members(self, group_key: str) -> Optional[dict]:
        """Get rows that contributed to a group."""
        ...

    def compute_gaps(self, row_id: int) -> LineageGaps:
        """Compute lineage gaps for a row."""
        ...

    def should_track_cell_diffs(self, affected_count: int) -> bool:
        """Return False for mass updates exceeding threshold."""
        ...

    def to_json(self) -> str:
        """Export all data as JSON string."""
        ...

    @property
    def steps(self) -> list:
        """Access step metadata list."""
        ...

    @property
    def total_diff_count(self) -> int:
        """Total diffs including spilled."""
        ...

    @property
    def diff_count(self) -> int:
        """In-memory diff count."""
        ...


@runtime_checkable
class RowIdentityStrategy(Protocol):
    """
    Protocol for row identity tracking.

    Implementations:
    - PandasRowIdentity (default, v0.2.0)
    - PolarsRowIdentity (future)
    - SparkRowIdentity (future)
    """

    config: TracePipeConfig

    def register(
        self, df: Any, row_ids: Optional[Any] = None, warn_duplicate_index: bool = True
    ) -> Any:
        """Register a DataFrame and assign/return row IDs."""
        ...

    def get_ids(self, df: Any) -> Optional[Any]:
        """Get row IDs for a DataFrame, or None if not tracked."""
        ...

    def propagate(self, source_df: Any, result_df: Any) -> Optional[Any]:
        """Propagate row IDs from source to result DataFrame."""
        ...

    def get_dropped_ids(self, source_df: Any, result_df: Any) -> set:
        """Get row IDs that were dropped between source and result."""
        ...

    def strip_hidden_column(self, df: Any) -> Any:
        """Remove hidden column for export."""
        ...

    def cleanup(self) -> None:
        """Remove stale entries."""
        ...


# === FACTORY FUNCTIONS ===


def create_default_backend(config: TracePipeConfig) -> "LineageBackend":
    """Create the default in-memory backend."""
    from .lineage_store import InMemoryLineageStore

    return InMemoryLineageStore(config)


def create_default_identity(config: TracePipeConfig) -> "RowIdentityStrategy":
    """Create the default pandas row identity strategy."""
    from .row_identity import PandasRowIdentity

    return PandasRowIdentity(config)

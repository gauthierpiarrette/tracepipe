# tracepipe/core.py
"""
Core types, enums, and configuration for TracePipe.

Design Principles:
1. Pandas Execution is Authoritative: TracePipe never re-implements operations
2. Trust Over Features: Mark PARTIAL when uncertain; never lie about completeness
3. Don't Touch User Data: No DataFrame mutation by default
4. Modes for Adoption: CI mode (fast) vs Debug mode (deep)
5. NumPy-First: Vectorized operations; no Python loops over millions of rows
"""

import os
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Optional


class ChangeType(IntEnum):
    """Types of changes tracked by TracePipe."""

    MODIFIED = 0
    DROPPED = 1
    ADDED = 2
    REORDERED = 3


class CompletenessLevel(IntEnum):
    """
    Indicates how completely an operation's internals are tracked.

    FULL: Complete lineage captured (dropna, drop_duplicates, boolean indexing)
    PARTIAL: Output tracked, internals approximate (query with @var, merge in CI mode)
    UNKNOWN: Operation not instrumented (future: uninstrumented custom ops)
    """

    FULL = 0
    PARTIAL = 1
    UNKNOWN = 2


class TracePipeMode(Enum):
    """TracePipe operating modes."""

    CI = "ci"  # Fast: stats, drops, contracts
    DEBUG = "debug"  # Deep: merge provenance, ghost values, cell history


class IdentityStorage(Enum):
    """Row identity storage strategies."""

    REGISTRY = "registry"  # Default: WeakKeyDictionary, no data mutation
    COLUMN = "column"  # Opt-in: hidden column (for edge cases)
    ATTRS = "attrs"  # Alternative: df.attrs token


@dataclass
class TracePipeConfig:
    """Configuration with sensible defaults."""

    # Memory limits
    max_diffs_in_memory: int = 500_000
    max_diffs_per_step: int = 100_000
    max_group_membership_size: int = 100_000

    # Behavior options
    strict_mode: bool = False
    auto_watch: bool = False
    auto_watch_null_threshold: float = 0.01
    spillover_dir: str = ".tracepipe"
    warn_on_duplicate_index: bool = True
    cleanup_spillover_on_disable: bool = True

    # Mode system
    mode: TracePipeMode = TracePipeMode.CI

    # Identity storage (default to registry, not column)
    identity_storage: IdentityStorage = IdentityStorage.REGISTRY

    # Feature overrides (None = use mode default)
    merge_provenance: Optional[bool] = None
    ghost_row_values: Optional[bool] = None
    cell_history: Optional[bool] = None

    # Ghost row limits
    max_ghost_rows: int = 10_000

    @property
    def should_capture_merge_provenance(self) -> bool:
        if self.merge_provenance is not None:
            return self.merge_provenance
        return self.mode == TracePipeMode.DEBUG

    @property
    def should_capture_ghost_values(self) -> bool:
        if self.ghost_row_values is not None:
            return self.ghost_row_values
        return self.mode == TracePipeMode.DEBUG

    @property
    def should_capture_cell_history(self) -> bool:
        if self.cell_history is not None:
            return self.cell_history
        return self.mode == TracePipeMode.DEBUG

    @property
    def use_hidden_column(self) -> bool:
        return self.identity_storage == IdentityStorage.COLUMN

    @property
    def use_attrs_token(self) -> bool:
        return self.identity_storage == IdentityStorage.ATTRS

    @classmethod
    def from_env(cls) -> "TracePipeConfig":
        """Create config from environment variables."""
        mode_str = os.environ.get("TRACEPIPE_MODE", "ci")
        return cls(
            mode=TracePipeMode.DEBUG if mode_str == "debug" else TracePipeMode.CI,
            max_diffs_in_memory=int(os.environ.get("TRACEPIPE_MAX_DIFFS", 500_000)),
            max_diffs_per_step=int(os.environ.get("TRACEPIPE_MAX_DIFFS_PER_STEP", 100_000)),
            strict_mode=os.environ.get("TRACEPIPE_STRICT", "0") == "1",
            auto_watch=os.environ.get("TRACEPIPE_AUTO_WATCH", "0") == "1",
        )


@dataclass
class StepEvent:
    """
    Stable schema for pipeline step events.

    This schema is designed to be stable across versions.
    New fields should be added as Optional with defaults.
    """

    step_id: int
    operation: str
    timestamp: float = field(default_factory=time.time)

    # Context
    stage: Optional[str] = None
    code_file: Optional[str] = None
    code_line: Optional[int] = None

    # Shape tracking
    input_shape: Optional[tuple[int, ...]] = None
    output_shape: Optional[tuple[int, ...]] = None

    # Parameters (operation-specific)
    params: dict[str, Any] = field(default_factory=dict)

    # Completeness
    completeness: CompletenessLevel = CompletenessLevel.FULL

    # Mass update tracking
    is_mass_update: bool = False
    rows_affected: int = 0

    # Error tracking
    error: Optional[str] = None
    error_type: Optional[str] = None

    @property
    def code_location(self) -> Optional[str]:
        """Human-readable code location."""
        if self.code_file and self.code_line:
            return f"{self.code_file}:{self.code_line}"
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (for JSON export)."""
        return {
            "step_id": self.step_id,
            "operation": self.operation,
            "timestamp": self.timestamp,
            "stage": self.stage,
            "code_location": self.code_location,
            "code_file": self.code_file,
            "code_line": self.code_line,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "params": self.params,
            "completeness": self.completeness.name,
            "is_mass_update": self.is_mass_update,
            "rows_affected": self.rows_affected,
            "error": self.error,
            "error_type": self.error_type,
        }


# Backwards compatibility alias
StepMetadata = StepEvent


@dataclass
class AggregationMapping:
    """Tracks which rows contributed to an aggregation group."""

    step_id: int
    group_column: str
    membership: dict[str, list[int]]  # {group_key: [row_ids]}
    agg_functions: dict[str, str]


@dataclass
class LineageGap:
    """Represents a gap in lineage tracking."""

    step_id: int
    operation: str
    reason: str


@dataclass
class LineageGaps:
    """Collection of lineage gaps for a row."""

    gaps: list[LineageGap] = field(default_factory=list)

    @property
    def has_gaps(self) -> bool:
        """Return True if there are any gaps."""
        return len(self.gaps) > 0

    @property
    def is_fully_tracked(self) -> bool:
        """Return True if lineage is complete."""
        return len(self.gaps) == 0

    def summary(self) -> str:
        """Return a human-readable summary."""
        if self.is_fully_tracked:
            return "Fully tracked"
        elif len(self.gaps) == 1:
            return f"1 step has limited visibility: {self.gaps[0].operation}"
        else:
            return f"{len(self.gaps)} steps have limited visibility"


@dataclass
class GhostRowInfo:
    """Information about a dropped row."""

    row_id: int
    last_values: dict[str, Any]
    dropped_by: str
    dropped_step: int
    original_position: int


@dataclass
class MergeMapping:
    """
    Array-based merge mapping (memory efficient).

    Arrays are stored SORTED by out_rids to enable O(log n) lookup
    via binary search instead of O(n) linear scan.
    """

    step_id: int
    out_rids: Any  # numpy array, SORTED for binary search
    left_parent_rids: Any  # numpy array, -1 for no match, same order as out_rids
    right_parent_rids: Any  # numpy array, -1 for no match, same order as out_rids


@dataclass
class MergeStats:
    """Merge statistics."""

    left_rows: int
    right_rows: int
    result_rows: int
    expansion_ratio: float
    left_match_rate: float  # -1 if not computed
    right_match_rate: float  # -1 if not computed
    left_dup_rate: float  # -1 if not computed
    right_dup_rate: float  # -1 if not computed
    how: str

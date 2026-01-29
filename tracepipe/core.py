# tracepipe/core.py
"""
Core types, enums, and configuration for TracePipe.
"""

import os
from dataclasses import dataclass, field
from enum import IntEnum
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

    FULL: Completely tracked (e.g., fillna, dropna)
    PARTIAL: Output tracked, internals unknown (e.g., apply, pipe)
    UNKNOWN: Lineage reset (e.g., merge, concat)
    """

    FULL = 0
    PARTIAL = 1
    UNKNOWN = 2


@dataclass
class TracePipeConfig:
    """Configuration with sensible defaults."""

    max_diffs_in_memory: int = 500_000
    max_diffs_per_step: int = 100_000
    max_group_membership_size: int = 100_000  # Store count-only above this threshold
    strict_mode: bool = False
    auto_watch: bool = False
    auto_watch_null_threshold: float = 0.01
    spillover_dir: str = ".tracepipe"
    use_hidden_column: bool = False
    warn_on_duplicate_index: bool = True
    cleanup_spillover_on_disable: bool = True

    @classmethod
    def from_env(cls) -> "TracePipeConfig":
        """Create config from environment variables."""
        return cls(
            max_diffs_in_memory=int(os.environ.get("TRACEPIPE_MAX_DIFFS", 500_000)),
            max_diffs_per_step=int(os.environ.get("TRACEPIPE_MAX_DIFFS_PER_STEP", 100_000)),
            strict_mode=os.environ.get("TRACEPIPE_STRICT", "0") == "1",
            auto_watch=os.environ.get("TRACEPIPE_AUTO_WATCH", "0") == "1",
            use_hidden_column=os.environ.get("TRACEPIPE_HIDDEN_COL", "0") == "1",
        )


@dataclass
class StepMetadata:
    """Metadata for a single pipeline step."""

    step_id: int
    operation: str
    stage: Optional[str]
    timestamp: float
    code_file: Optional[str]
    code_line: Optional[int]
    params: dict[str, Any]
    input_shape: Optional[tuple]
    output_shape: Optional[tuple]
    is_mass_update: bool = False
    rows_affected: int = 0
    completeness: CompletenessLevel = CompletenessLevel.FULL


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

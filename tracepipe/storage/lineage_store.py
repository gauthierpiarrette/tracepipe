# tracepipe/storage/lineage_store.py
"""
In-memory lineage storage using Structure of Arrays (SoA) pattern.

Memory: ~40 bytes/diff vs ~150 bytes with dataclass

Features:
- Merge mapping storage with O(log n) lookup via binary search
- Sorted bulk drops for efficient drop event lookup
- Stable API for api/convenience/visualization layers
"""

import atexit
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..core import (
    AggregationMapping,
    ChangeType,
    CompletenessLevel,
    LineageGap,
    LineageGaps,
    MergeMapping,
    MergeStats,
    StepEvent,
    TracePipeConfig,
)
from ..utils.value_capture import capture_typed_value


class InMemoryLineageStore:
    """
    Columnar storage for lineage data using Structure of Arrays (SoA).

    Implements: LineageBackend protocol

    STABLE INTERNAL API (used by api.py, convenience.py, visualization):

    === ATTRIBUTES (read-only from outside) ===
    steps: list[StepEvent]                    # All recorded steps
    bulk_drops: dict[int, np.ndarray]         # step_id -> sorted dropped RIDs
    merge_mappings: list[MergeMapping]        # Merge parent mappings (debug mode)
    merge_stats: list[tuple[int, MergeStats]] # (step_id, stats) pairs

    === WRITE METHODS (called by instrumentation) ===
    append_step(...) -> int                   # Returns step_id
    append_bulk_drops(step_id, rids)          # rids will be sorted internally
    append_diff(step_id, row_id, col, ...)    # Cell-level diff

    === READ METHODS (called by api/convenience) ===
    get_drop_event(row_id) -> Optional[dict]  # {step_id, operation}
    get_dropped_rows() -> list[int]           # All dropped RIDs
    get_dropped_by_step() -> dict[str, int]   # operation -> count
    get_row_history(row_id) -> list[dict]     # Chronological events
    get_merge_stats(step_id=None) -> list[tuple[int, MergeStats]]
    get_merge_origin(row_id) -> Optional[dict]  # {left_parent, right_parent, step_id}
    """

    def __init__(self, config: TracePipeConfig):
        self.config = config
        self._spillover_dir = Path(config.spillover_dir)

        # === DIFF STORAGE (Columnar) ===
        self.diff_step_ids: list[int] = []
        self.diff_row_ids: list[int] = []
        self.diff_cols: list[str] = []
        self.diff_old_vals: list[Any] = []
        self.diff_old_types: list[str] = []
        self.diff_new_vals: list[Any] = []
        self.diff_new_types: list[str] = []
        self.diff_change_types: list[int] = []

        # === STEP METADATA ===
        self._steps: list[StepEvent] = []

        # === BULK DROPS (step_id -> SORTED numpy array) ===
        self.bulk_drops: dict[int, np.ndarray] = {}

        # === MERGE TRACKING ===
        self.merge_mappings: list[MergeMapping] = []
        self.merge_stats: list[tuple[int, MergeStats]] = []

        # === AGGREGATION MAPPINGS ===
        self.aggregation_mappings: list[AggregationMapping] = []

        # === SPILLOVER TRACKING ===
        self.spilled_files: list[str] = []

        # === COUNTERS ===
        self._step_counter: int = 0
        self._diff_count: int = 0
        self._total_diff_count: int = 0  # Including spilled

        # === STRING INTERNING ===
        self._col_intern: dict[str, str] = {}
        self._type_intern: dict[str, str] = {}

        # === ATEXIT HANDLER ===
        self._atexit_registered: bool = False
        self._register_atexit()

    @property
    def steps(self) -> list[StepEvent]:
        """Access step metadata list."""
        return self._steps

    def _intern_string(self, s: str, cache: dict[str, str]) -> str:
        """Intern string to avoid duplicate allocations."""
        if s not in cache:
            cache[s] = s
        return cache[s]

    def next_step_id(self) -> int:
        """Generate next step ID."""
        self._step_counter += 1
        return self._step_counter

    @property
    def diff_count(self) -> int:
        """In-memory diff count."""
        return self._diff_count

    @property
    def total_diff_count(self) -> int:
        """Total diffs including spilled."""
        return self._total_diff_count

    def append_diff(
        self,
        step_id: int,
        row_id: int,
        col: str,
        old_val: Any,
        new_val: Any,
        change_type: ChangeType,
    ) -> None:
        """Append a single diff in columnar format."""
        old_val, old_type = capture_typed_value(old_val)
        new_val, new_type = capture_typed_value(new_val)

        self.diff_step_ids.append(step_id)
        self.diff_row_ids.append(row_id)
        self.diff_cols.append(self._intern_string(col, self._col_intern))
        self.diff_old_vals.append(old_val)
        self.diff_old_types.append(self._intern_string(old_type, self._type_intern))
        self.diff_new_vals.append(new_val)
        self.diff_new_types.append(self._intern_string(new_type, self._type_intern))
        self.diff_change_types.append(int(change_type))

        self._diff_count += 1
        self._total_diff_count += 1

        # Check memory every 10k diffs
        if self._diff_count % 10_000 == 0:
            self._check_memory_and_spill()

    def append_diff_batch(
        self, step_id: int, diffs: list[tuple], check_threshold: bool = True
    ) -> int:
        """
        Batch append for performance.

        Args:
            step_id: Step ID for all diffs
            diffs: List of (row_id, col, old_val, new_val, change_type)
            check_threshold: If True, skip if too many diffs

        Returns:
            Number of diffs actually appended
        """
        if check_threshold and len(diffs) > self.config.max_diffs_per_step:
            return 0  # Caller should log as mass update

        for row_id, col, old_val, new_val, change_type in diffs:
            self.append_diff(step_id, row_id, col, old_val, new_val, change_type)

        return len(diffs)

    def append_bulk_drops(self, step_id: int, dropped_row_ids) -> int:
        """
        Bulk append dropped rows - optimized for filter operations.

        Stores dropped RIDs SORTED for O(log n) lookup via searchsorted.

        Args:
            step_id: Step ID for all drops
            dropped_row_ids: Array-like of row IDs that were dropped

        Returns:
            Number of drops recorded
        """
        n = len(dropped_row_ids)
        if n == 0:
            return 0

        # Convert to sorted numpy array
        if isinstance(dropped_row_ids, np.ndarray):
            sorted_rids = np.sort(dropped_row_ids.astype(np.int64))
        else:
            sorted_rids = np.sort(np.array(list(dropped_row_ids), dtype=np.int64))

        # Store sorted for O(log n) lookup
        self.bulk_drops[step_id] = sorted_rids

        # Also record in diff arrays for backwards compatibility
        row_ids_list = sorted_rids.tolist()

        # Pre-intern the constant strings once
        col_interned = self._intern_string("__row__", self._col_intern)
        old_type_interned = self._intern_string("str", self._type_intern)
        new_type_interned = self._intern_string("null", self._type_intern)

        # Bulk extend all arrays at once
        self.diff_step_ids.extend([step_id] * n)
        self.diff_row_ids.extend(row_ids_list)
        self.diff_cols.extend([col_interned] * n)
        self.diff_old_vals.extend(["present"] * n)
        self.diff_old_types.extend([old_type_interned] * n)
        self.diff_new_vals.extend([None] * n)
        self.diff_new_types.extend([new_type_interned] * n)
        self.diff_change_types.extend([int(ChangeType.DROPPED)] * n)

        self._diff_count += n
        self._total_diff_count += n

        # Check memory threshold
        if self._diff_count >= self.config.max_diffs_in_memory:
            self._check_memory_and_spill()

        return n

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
        """Append step metadata and return step_id."""
        step_id = self.next_step_id()
        self._steps.append(
            StepEvent(
                step_id=step_id,
                operation=operation,
                stage=stage,
                timestamp=time.time(),
                code_file=code_file,
                code_line=code_line,
                params=params,
                input_shape=input_shape,
                output_shape=output_shape,
                completeness=completeness,
                is_mass_update=is_mass_update,
                rows_affected=rows_affected,
            )
        )
        return step_id

    def append_aggregation(
        self,
        step_id: int,
        group_column: str,
        membership: dict[str, list[int]],
        agg_functions: dict[str, str],
    ) -> None:
        """Record aggregation group membership."""
        self.aggregation_mappings.append(
            AggregationMapping(
                step_id=step_id,
                group_column=group_column,
                membership=membership,
                agg_functions=agg_functions,
            )
        )

    def should_track_cell_diffs(self, affected_count: int) -> bool:
        """Return False for mass updates exceeding threshold."""
        return affected_count <= self.config.max_diffs_per_step

    # === DROP LOOKUP (O(log n) via searchsorted) ===

    def get_drop_event(self, row_id: int) -> Optional[dict]:
        """
        Get drop event for a row from bulk_drops.

        O(log n) per step via searchsorted.

        Returns:
            {step_id, operation} if dropped, else None.
        """
        for step_id, dropped_rids in self.bulk_drops.items():
            # Binary search in sorted array
            i = np.searchsorted(dropped_rids, row_id)
            if i < len(dropped_rids) and dropped_rids[i] == row_id:
                step = self._steps[step_id - 1] if step_id <= len(self._steps) else None
                return {
                    "step_id": step_id,
                    "operation": step.operation if step else "unknown",
                }
        return None

    def is_dropped(self, row_id: int) -> bool:
        """Fast check if row was dropped anywhere."""
        return self.get_drop_event(row_id) is not None

    # === MERGE LOOKUP (O(log n) via searchsorted) ===

    def get_merge_origin(self, row_id: int) -> Optional[dict]:
        """
        Get merge parent RIDs for a row.

        Uses binary search (O(log n)) instead of linear scan (O(n)).
        """
        for mapping in self.merge_mappings:
            # Binary search on sorted out_rids
            i = np.searchsorted(mapping.out_rids, row_id)
            if i < len(mapping.out_rids) and mapping.out_rids[i] == row_id:
                left_parent = mapping.left_parent_rids[i]
                right_parent = mapping.right_parent_rids[i]
                return {
                    "step_id": mapping.step_id,
                    "left_parent": int(left_parent) if left_parent >= 0 else None,
                    "right_parent": int(right_parent) if right_parent >= 0 else None,
                }
        return None

    def get_merge_stats(self, step_id: Optional[int] = None) -> list[tuple[int, MergeStats]]:
        """
        Get merge statistics.

        Returns:
            list of (step_id, MergeStats) tuples - ALWAYS this shape for consistency.
        """
        if step_id is not None:
            return [(sid, s) for sid, s in self.merge_stats if sid == step_id]
        return list(self.merge_stats)

    # === MEMORY MANAGEMENT ===

    def _check_memory_and_spill(self) -> None:
        """Spill to disk based on count threshold."""
        if self._diff_count < self.config.max_diffs_in_memory:
            return

        # Optional: use psutil for real memory check
        try:
            import psutil

            process = psutil.Process()
            mem_mb = process.memory_info().rss / (1024 * 1024)
            if mem_mb < 500:
                return
        except ImportError:
            pass

        self._spill_to_disk()

    def _spill_to_disk(self) -> None:
        """Spill current diffs to disk and clear memory."""
        self._spillover_dir.mkdir(exist_ok=True)

        filename = f"diffs_{int(time.time() * 1000)}_{self._diff_count}.json"
        filepath = self._spillover_dir / filename

        data = {
            "step_ids": self.diff_step_ids,
            "row_ids": self.diff_row_ids,
            "cols": self.diff_cols,
            "old_vals": self.diff_old_vals,
            "old_types": self.diff_old_types,
            "new_vals": self.diff_new_vals,
            "new_types": self.diff_new_types,
            "change_types": self.diff_change_types,
        }

        with open(filepath, "w") as f:
            json.dump(data, f)
        self.spilled_files.append(str(filepath))

        # Clear in-memory arrays
        self._clear_in_memory()

    def _clear_in_memory(self) -> None:
        """Clear in-memory diff arrays."""
        self.diff_step_ids.clear()
        self.diff_row_ids.clear()
        self.diff_cols.clear()
        self.diff_old_vals.clear()
        self.diff_old_types.clear()
        self.diff_new_vals.clear()
        self.diff_new_types.clear()
        self.diff_change_types.clear()
        self._diff_count = 0

    def _register_atexit(self) -> None:
        """Register cleanup handler if not already registered."""
        if not self._atexit_registered:
            atexit.register(self._cleanup_spillover)
            self._atexit_registered = True

    def _unregister_atexit(self) -> None:
        """Unregister cleanup handler."""
        if self._atexit_registered:
            try:
                atexit.unregister(self._cleanup_spillover)
            except Exception:
                pass
            self._atexit_registered = False

    def _cleanup_spillover(self) -> None:
        """Clean up spillover files on exit."""
        # Unregister to prevent multiple calls
        self._unregister_atexit()

        if not self.config.cleanup_spillover_on_disable:
            return

        for filepath in self.spilled_files:
            try:
                Path(filepath).unlink(missing_ok=True)
            except Exception:
                pass

        # Try to remove directory if empty
        try:
            if self._spillover_dir.exists() and not any(self._spillover_dir.iterdir()):
                self._spillover_dir.rmdir()
        except Exception:
            pass

    # === QUERY METHODS ===

    def _iter_all_diffs(self):
        """
        Iterate over all diffs (spilled + in-memory) without loading all into memory.

        Yields:
            dict with step_id, row_id, col, old_val, new_val, change_type, etc.
        """
        # Spilled files first (older data)
        for filepath in self.spilled_files:
            try:
                with open(filepath) as f:
                    data = json.load(f)
                for i in range(len(data["step_ids"])):
                    yield {
                        "step_id": data["step_ids"][i],
                        "row_id": data["row_ids"][i],
                        "col": data["cols"][i],
                        "old_val": data["old_vals"][i],
                        "old_type": data["old_types"][i],
                        "new_val": data["new_vals"][i],
                        "new_type": data["new_types"][i],
                        "change_type": data["change_types"][i],
                    }
            except Exception:
                continue

        # In-memory diffs
        for i in range(len(self.diff_step_ids)):
            yield {
                "step_id": self.diff_step_ids[i],
                "row_id": self.diff_row_ids[i],
                "col": self.diff_cols[i],
                "old_val": self.diff_old_vals[i],
                "old_type": self.diff_old_types[i],
                "new_val": self.diff_new_vals[i],
                "new_type": self.diff_new_types[i],
                "change_type": self.diff_change_types[i],
            }

    def get_row_history(self, row_id: int) -> list[dict]:
        """
        Get all events for a specific row in CHRONOLOGICAL order (oldest first).

        CONTRACT: Returned list has monotonically increasing step_id.
        Convenience layer may reverse for display.

        Note: This returns only direct events for this row_id.
        Use get_row_history_with_lineage() to include pre-merge parent history.
        """
        step_map = {s.step_id: s for s in self._steps}
        events = []

        # Collect from bulk_drops (sorted by step_id)
        for step_id in sorted(self.bulk_drops.keys()):
            dropped_rids = self.bulk_drops[step_id]
            i = np.searchsorted(dropped_rids, row_id)
            if i < len(dropped_rids) and dropped_rids[i] == row_id:
                step = step_map.get(step_id)
                events.append(
                    {
                        "step_id": step_id,
                        "operation": step.operation if step else "unknown",
                        "stage": step.stage if step else None,
                        "col": "__row__",
                        "old_val": "present",
                        "old_type": "str",
                        "new_val": None,
                        "new_type": "null",
                        "change_type": "DROPPED",
                        "timestamp": step.timestamp if step else None,
                        "completeness": step.completeness.name if step else "UNKNOWN",
                        "code_location": (
                            f"{step.code_file}:{step.code_line}"
                            if step and step.code_file
                            else None
                        ),
                    }
                )

        # Collect from diffs
        for diff in self._iter_all_diffs():
            if diff["row_id"] == row_id and diff["col"] != "__row__":
                step = step_map.get(diff["step_id"])
                events.append(
                    {
                        "step_id": diff["step_id"],
                        "operation": step.operation if step else "unknown",
                        "stage": step.stage if step else None,
                        "col": diff["col"],
                        "old_val": diff["old_val"],
                        "old_type": diff["old_type"],
                        "new_val": diff["new_val"],
                        "new_type": diff["new_type"],
                        "change_type": ChangeType(diff["change_type"]).name,
                        "timestamp": step.timestamp if step else None,
                        "completeness": step.completeness.name if step else "UNKNOWN",
                        "code_location": (
                            f"{step.code_file}:{step.code_line}"
                            if step and step.code_file
                            else None
                        ),
                    }
                )

        # ENFORCE: sort by step_id (chronological)
        events.sort(key=lambda e: e["step_id"])

        return events

    def get_row_history_with_lineage(self, row_id: int, max_depth: int = 10) -> list[dict]:
        """
        Get row history including pre-merge parent history.

        Follows merge lineage recursively to build complete cell provenance.
        This is essential for tracking changes that happened before merge operations.

        Args:
            row_id: Row ID to trace
            max_depth: Maximum merge depth to follow (prevents infinite loops)

        Returns:
            List of events in chronological order, including parent row events.
        """
        visited: set[int] = set()

        def _collect_history(rid: int, depth: int) -> list[dict]:
            if depth > max_depth or rid in visited:
                return []
            visited.add(rid)

            events = []

            # Check if this row came from a merge
            origin = self.get_merge_origin(rid)
            if origin and origin["left_parent"] is not None:
                # Recursively get parent's history first (chronological order)
                parent_events = _collect_history(origin["left_parent"], depth + 1)
                events.extend(parent_events)

            # Add this row's direct events
            events.extend(self.get_row_history(rid))

            return events

        all_events = _collect_history(row_id, 0)

        # Sort by step_id to ensure chronological order across lineage
        all_events.sort(key=lambda e: e["step_id"])

        return all_events

    def get_cell_history_with_lineage(
        self, row_id: int, column: str, max_depth: int = 10
    ) -> list[dict]:
        """
        Get cell history for a specific column, including pre-merge parent history.

        Args:
            row_id: Row ID to trace
            column: Column name to filter events for
            max_depth: Maximum merge depth to follow

        Returns:
            List of events for this column in chronological order.
        """
        all_events = self.get_row_history_with_lineage(row_id, max_depth)
        return [e for e in all_events if e["col"] == column]

    def get_dropped_rows(self, step_id: Optional[int] = None) -> list[int]:
        """Get all dropped row IDs, optionally filtered by step."""
        if step_id is not None:
            if step_id in self.bulk_drops:
                return self.bulk_drops[step_id].tolist()
            return []

        # Collect all dropped rows
        dropped = set()
        for rids in self.bulk_drops.values():
            dropped.update(rids.tolist())

        return sorted(dropped)

    def get_dropped_by_step(self) -> dict[str, int]:
        """Get count of dropped rows per operation."""
        step_map = {s.step_id: s.operation for s in self._steps}
        counts: dict[str, int] = {}

        for step_id, rids in self.bulk_drops.items():
            op = step_map.get(step_id, "unknown")
            counts[op] = counts.get(op, 0) + len(rids)

        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def get_group_members(self, group_key: str) -> Optional[dict]:
        """
        Get all rows that contributed to a group.

        Note: For large groups (exceeding max_group_membership_size),
        membership is stored as count-only: [-count]. In this case,
        row_ids will be empty and is_count_only will be True.
        """
        for mapping in self.aggregation_mappings:
            if group_key in mapping.membership:
                member_data = mapping.membership[group_key]

                # Check for count-only marker (negative count)
                if len(member_data) == 1 and member_data[0] < 0:
                    return {
                        "group_key": group_key,
                        "group_column": mapping.group_column,
                        "row_ids": [],
                        "row_count": abs(member_data[0]),
                        "is_count_only": True,
                        "agg_functions": mapping.agg_functions,
                    }
                else:
                    return {
                        "group_key": group_key,
                        "group_column": mapping.group_column,
                        "row_ids": member_data,
                        "row_count": len(member_data),
                        "is_count_only": False,
                        "agg_functions": mapping.agg_functions,
                    }
        return None

    def compute_gaps(self, row_id: int) -> LineageGaps:
        """Compute lineage gaps for a specific row."""
        gaps = []
        row_step_ids = set()

        # From bulk_drops
        for step_id, rids in self.bulk_drops.items():
            i = np.searchsorted(rids, row_id)
            if i < len(rids) and rids[i] == row_id:
                row_step_ids.add(step_id)

        # From diffs
        for diff in self._iter_all_diffs():
            if diff["row_id"] == row_id:
                row_step_ids.add(diff["step_id"])

        for step in self._steps:
            if step.step_id in row_step_ids:
                if step.completeness == CompletenessLevel.PARTIAL:
                    gaps.append(
                        LineageGap(
                            step_id=step.step_id,
                            operation=step.operation,
                            reason="Custom function - output tracked, internals unknown",
                        )
                    )
                elif step.completeness == CompletenessLevel.UNKNOWN:
                    gaps.append(
                        LineageGap(
                            step_id=step.step_id,
                            operation=step.operation,
                            reason="Operation resets lineage (merge/concat)",
                        )
                    )

        return LineageGaps(gaps=gaps)

    # === EXPORT METHODS ===

    def to_json(self) -> str:
        """Export all data as JSON string."""
        diffs = list(self._iter_all_diffs())

        data = {
            "tracepipe_version": "0.3.1",
            "export_timestamp": time.time(),
            "total_diffs": len(diffs),
            "total_steps": len(self._steps),
            "diffs": diffs,
            "steps": [s.to_dict() for s in self._steps],
            "aggregation_mappings": [
                {
                    "step_id": a.step_id,
                    "group_column": a.group_column,
                    "membership": a.membership,
                    "agg_functions": a.agg_functions,
                }
                for a in self.aggregation_mappings
            ],
            "merge_stats": [{"step_id": sid, **vars(stats)} for sid, stats in self.merge_stats],
        }

        return json.dumps(data)

    def to_arrow(self):
        """Convert to Arrow table (requires pyarrow)."""
        try:
            import pyarrow as pa
        except ImportError:
            raise ImportError("pyarrow required: pip install pyarrow")

        # Collect into columnar format
        step_ids, row_ids, cols = [], [], []
        old_vals, new_vals, change_types = [], [], []

        for diff in self._iter_all_diffs():
            step_ids.append(diff["step_id"])
            row_ids.append(diff["row_id"])
            cols.append(diff["col"])
            old_vals.append(str(diff["old_val"]))
            new_vals.append(str(diff["new_val"]))
            change_types.append(diff["change_type"])

        return pa.Table.from_pydict(
            {
                "step_id": pa.array(step_ids, type=pa.int32()),
                "row_id": pa.array(row_ids, type=pa.int64()),
                "col": pa.array(cols, type=pa.string()),
                "old_val": pa.array(old_vals, type=pa.string()),
                "new_val": pa.array(new_vals, type=pa.string()),
                "change_type": pa.array(change_types, type=pa.int8()),
            }
        )

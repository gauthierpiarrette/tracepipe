# tracepipe/storage/lineage_store.py
"""
In-memory lineage storage using Structure of Arrays (SoA) pattern.

Memory: ~40 bytes/diff vs ~150 bytes with dataclass
"""

import atexit
import json
import time
from pathlib import Path
from typing import Any, Optional

from ..core import (
    AggregationMapping,
    ChangeType,
    CompletenessLevel,
    LineageGap,
    LineageGaps,
    StepMetadata,
    TracePipeConfig,
)
from ..utils.value_capture import capture_typed_value


class InMemoryLineageStore:
    """
    Columnar storage for lineage data using Structure of Arrays (SoA).

    Implements: LineageBackend protocol

    Future alternatives:
    - SQLiteLineageStore: Persistent storage for long-running pipelines
    - DeltaLakeBackend: Distributed storage for big data
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
        self._steps: list[StepMetadata] = []

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

        # Register cleanup on exit
        atexit.register(self._cleanup_spillover)

    @property
    def steps(self) -> list[StepMetadata]:
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

        Uses list.extend() for O(1) amortized append instead of O(n) individual appends.
        Typically 10-50x faster than calling append_diff() in a loop.

        Args:
            step_id: Step ID for all drops
            dropped_row_ids: Array-like of row IDs that were dropped

        Returns:
            Number of drops recorded
        """
        import numpy as np

        n = len(dropped_row_ids)
        if n == 0:
            return 0

        # Convert to list if numpy array
        if isinstance(dropped_row_ids, np.ndarray):
            row_ids_list = dropped_row_ids.tolist()
        else:
            row_ids_list = list(dropped_row_ids)

        # Pre-intern the constant strings once
        col_interned = self._intern_string("__row__", self._col_intern)
        old_type_interned = self._intern_string("str", self._type_intern)
        new_type_interned = self._intern_string("null", self._type_intern)

        # Bulk extend all arrays at once (much faster than individual appends)
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
            StepMetadata(
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

    def _cleanup_spillover(self) -> None:
        """Clean up spillover files on exit."""
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
        """Get all events for a specific row."""
        step_map = {s.step_id: s for s in self._steps}
        events = []

        for diff in self._iter_all_diffs():
            if diff["row_id"] == row_id:
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

        return sorted(events, key=lambda e: e["step_id"])

    def get_dropped_rows(self, step_id: Optional[int] = None) -> list[int]:
        """Get all dropped row IDs, optionally filtered by step."""
        dropped = set()

        for diff in self._iter_all_diffs():
            if diff["change_type"] == ChangeType.DROPPED:
                if step_id is None or diff["step_id"] == step_id:
                    dropped.add(diff["row_id"])

        return sorted(dropped)

    def get_dropped_by_step(self) -> dict[str, int]:
        """Get count of dropped rows per operation."""
        step_map = {s.step_id: s.operation for s in self._steps}
        counts: dict[str, int] = {}

        for diff in self._iter_all_diffs():
            if diff["change_type"] == ChangeType.DROPPED:
                op = step_map.get(diff["step_id"], "unknown")
                counts[op] = counts.get(op, 0) + 1

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
            "tracepipe_version": "0.2.0",
            "export_timestamp": time.time(),
            "total_diffs": len(diffs),
            "total_steps": len(self._steps),
            "diffs": diffs,
            "steps": [
                {
                    "step_id": s.step_id,
                    "operation": s.operation,
                    "stage": s.stage,
                    "timestamp": s.timestamp,
                    "code_file": s.code_file,
                    "code_line": s.code_line,
                    "params": s.params,
                    "input_shape": s.input_shape,
                    "output_shape": s.output_shape,
                    "is_mass_update": s.is_mass_update,
                    "rows_affected": s.rows_affected,
                    "completeness": s.completeness.name,
                }
                for s in self._steps
            ],
            "aggregation_mappings": [
                {
                    "step_id": a.step_id,
                    "group_column": a.group_column,
                    "membership": a.membership,
                    "agg_functions": a.agg_functions,
                }
                for a in self.aggregation_mappings
            ],
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

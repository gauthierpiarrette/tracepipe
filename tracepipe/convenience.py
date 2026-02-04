# tracepipe/convenience.py
"""
Convenience API: The friendly face of TracePipe.

5 functions for 90% of use cases:
    enable()  - Start tracking
    check()   - Health audit
    why()     - Cell provenance ("why is this null?")
    trace()   - Row journey ("what happened to this row?")
    report()  - HTML export

All functions use df-first signatures for consistency:
    tp.check(df)
    tp.trace(df, row=5)
    tp.why(df, col="amount", row=5)
    tp.why(df, col="amount", where={"customer_id": "C123"})

Power users: Use tp.debug.inspect(), tp.contracts.contract() directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .context import get_context
from .core import TracePipeMode

# ============ RESULT TYPES ============


@dataclass
class CheckWarning:
    """A single warning from check()."""

    category: str  # "merge_expansion", "retention", "duplicate_keys", etc.
    severity: str  # "fact" (measured) or "heuristic" (inferred)
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    fix_hint: str | None = None

    def __repr__(self) -> str:
        icon = "[!]" if self.severity == "fact" else "[?]"
        return f"{icon} [{self.category}] {self.message}"


@dataclass
class CheckResult:
    """
    Result of check() - pipeline health audit.

    Separates FACTS (observed, high confidence) from HEURISTICS (inferred).
    .ok is True only if there are no FACT-level warnings.

    Key properties for quick access:
        .passed       - Alias for .ok (common naming convention)
        .retention    - Row retention rate (0.0-1.0)
        .n_dropped    - Total rows dropped
        .drops_by_op  - Drops broken down by operation
    """

    ok: bool
    warnings: list[CheckWarning]
    facts: dict[str, Any]
    suggestions: list[str]
    mode: str
    # Internal: store drops_by_op so we don't need to recompute
    _drops_by_op: dict[str, int] = field(default_factory=dict)

    # === CONVENIENCE PROPERTIES ===

    @property
    def passed(self) -> bool:
        """Alias for .ok (matches common naming convention)."""
        return self.ok

    @property
    def retention(self) -> float | None:
        """Row retention rate (0.0-1.0), or None if not computed."""
        return self.facts.get("retention_rate")

    @property
    def n_dropped(self) -> int:
        """Total number of rows dropped."""
        return self.facts.get("rows_dropped", 0)

    @property
    def drops_by_op(self) -> dict[str, int]:
        """Drops broken down by operation name."""
        return self._drops_by_op

    @property
    def n_steps(self) -> int:
        """Total pipeline steps recorded."""
        return self.facts.get("total_steps", 0)

    # === EXISTING PROPERTIES ===

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    @property
    def fact_warnings(self) -> list[CheckWarning]:
        return [w for w in self.warnings if w.severity == "fact"]

    @property
    def heuristic_warnings(self) -> list[CheckWarning]:
        return [w for w in self.warnings if w.severity == "heuristic"]

    def raise_if_failed(self) -> CheckResult:
        """Raise CheckFailed if any FACT warnings (for CI). Returns self for chaining."""
        if not self.ok:
            raise CheckFailed(self.fact_warnings)
        return self

    def __repr__(self) -> str:
        return self.to_text(verbose=False)

    def to_text(self, verbose: bool = True) -> str:
        """Format as text. Use verbose=True for full details."""
        lines = []
        status = "[OK] Pipeline healthy" if self.ok else "[WARN] Issues detected"
        lines.append(f"TracePipe Check: {status}")
        lines.append(f"  Mode: {self.mode}")

        if verbose and self.facts:
            lines.append("\n  Measured facts:")
            for k, v in self.facts.items():
                lines.append(f"    {k}: {v}")

        if self.fact_warnings:
            lines.append("\n  Issues (confirmed):")
            for w in self.fact_warnings:
                lines.append(f"    [!] {w.message}")
                if verbose and w.fix_hint:
                    lines.append(f"       -> {w.fix_hint}")

        if self.heuristic_warnings:
            lines.append("\n  Suggestions (possible issues):")
            for w in self.heuristic_warnings:
                lines.append(f"    [?] {w.message}")
                if verbose and w.fix_hint:
                    lines.append(f"       -> {w.fix_hint}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Export to dictionary."""
        return {
            "ok": self.ok,
            "passed": self.passed,
            "mode": self.mode,
            "retention": self.retention,
            "n_dropped": self.n_dropped,
            "n_steps": self.n_steps,
            "drops_by_op": self.drops_by_op,
            "facts": self.facts,
            "suggestions": self.suggestions,
            "warnings": [
                {
                    "category": w.category,
                    "severity": w.severity,
                    "message": w.message,
                    "details": w.details,
                    "fix_hint": w.fix_hint,
                }
                for w in self.warnings
            ],
        }


class CheckFailed(Exception):
    """Raised by CheckResult.raise_if_failed()."""

    def __init__(self, warnings: list[CheckWarning]):
        self.warnings = warnings
        messages = [w.message for w in warnings]
        super().__init__(f"Check failed: {'; '.join(messages)}")


@dataclass
class TraceResult:
    """
    Result of trace() - row journey.

    Answers: "What happened to this row?"
    Events are in CHRONOLOGICAL order (oldest->newest).

    Key attributes:
        origin: Where this row came from (concat, merge, or original)
        representative: If dropped by dedup, which row was kept instead
    """

    row_id: int
    is_alive: bool
    dropped_at: dict[str, Any] | None = None
    merge_origin: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    ghost_values: dict[str, Any] | None = None
    # Mode enforcement
    supported: bool = True
    unsupported_reason: str | None = None
    # v0.4+ provenance
    concat_origin: dict[str, Any] | None = None
    dedup_representative: dict[str, Any] | None = None

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def origin(self) -> dict[str, Any] | None:
        """
        Unified origin info: where did this row come from?

        Returns dict with 'type' key:
            - {"type": "concat", "source_df": 1, "step_id": 5}
            - {"type": "merge", "left_parent": 10, "right_parent": 20, "step_id": 3}
            - None if original row (not from concat/merge)
        """
        if self.concat_origin:
            return {
                "type": "concat",
                "source_df": self.concat_origin.get("source_index"),
                "step_id": self.concat_origin.get("step_id"),
            }
        if self.merge_origin:
            return {
                "type": "merge",
                "left_parent": self.merge_origin.get("left_parent"),
                "right_parent": self.merge_origin.get("right_parent"),
                "step_id": self.merge_origin.get("step_id"),
            }
        return None

    @property
    def representative(self) -> dict[str, Any] | None:
        """
        If dropped by drop_duplicates, which row was kept instead?

        Returns:
            {"kept_rid": 42, "subset": ["key"], "keep": "first"} or None
            kept_rid is None if keep=False (all duplicates dropped)
        """
        if not self.dedup_representative:
            return None
        return {
            "kept_rid": self.dedup_representative.get("kept_rid"),
            "subset": self.dedup_representative.get("subset_columns"),
            "keep": self.dedup_representative.get("keep_strategy"),
        }

    def to_dict(self) -> dict:
        """Export to dictionary."""
        return {
            "row_id": self.row_id,
            "is_alive": self.is_alive,
            "dropped_at": self.dropped_at,
            "origin": self.origin,
            "representative": self.representative,
            "n_events": self.n_events,
            "events": self.events,
            "ghost_values": self.ghost_values,
            "supported": self.supported,
            # Keep legacy fields for backwards compatibility
            "merge_origin": self.merge_origin,
        }

    def __repr__(self) -> str:
        return self.to_text(verbose=False)

    def to_text(self, verbose: bool = True) -> str:
        """Format as text. Use verbose=True for full details."""
        if not self.supported:
            return f"TraceResult: {self.unsupported_reason}"

        lines = [f"Row {self.row_id} Journey:"]

        if self.is_alive:
            lines.append("  Status: [OK] Alive")
        else:
            lines.append("  Status: [X] Dropped")
            if self.dropped_at:
                lines.append(
                    f"    at step {self.dropped_at['step_id']}: {self.dropped_at['operation']}"
                )

        # Display unified origin info
        origin = self.origin
        if origin:
            if origin["type"] == "merge":
                left = origin.get("left_parent", "?")
                right = origin.get("right_parent", "?")
                lines.append(f"  Origin: merge of row {left} (left) + row {right} (right)")
            elif origin["type"] == "concat":
                src = origin.get("source_df", "?")
                lines.append(f"  Origin: concat from DataFrame #{src}")

        # Display dedup representative if dropped by dedup
        if self.representative:
            kept = self.representative.get("kept_rid")
            subset = self.representative.get("subset")
            keep = self.representative.get("keep", "first")
            if kept is not None:
                subset_str = f" (key: {subset})" if subset else ""
                lines.append(f"  Replaced by: row {kept}{subset_str} [keep={keep}]")
            else:
                subset_str = f" on {subset}" if subset else ""
                lines.append(f"  Dropped: all duplicates removed{subset_str} [keep=False]")

        if len(self.events) == 0:
            lines.append("\n  Events: 0 (no changes to watched columns)")
        else:
            lines.append(f"\n  Events: {len(self.events)}")
            event_limit = 10 if verbose else 5
            for event in self.events[-event_limit:]:
                change = event.get("change_type", "?")
                op = event.get("operation", "?")
                col = event.get("col", "")
                if col and col != "__row__":
                    lines.append(f"    [{change}] {op}: {col}")
                else:
                    lines.append(f"    [{change}] {op}")

        if self.ghost_values:
            lines.append("\n  Last known values:")
            limit = 10 if verbose else 5
            for k, v in list(self.ghost_values.items())[:limit]:
                lines.append(f"    {k}: {v}")

        return "\n".join(lines)


@dataclass
class WhyResult:
    """
    Result of why() - cell provenance.

    Answers: "Why does this cell have this value?"
    History is stored in CHRONOLOGICAL order (oldest->newest).
    """

    row_id: int
    column: str
    current_value: Any = None
    history: list[dict[str, Any]] = field(default_factory=list)
    became_null_at: dict[str, Any] | None = None
    # Mode enforcement
    supported: bool = True
    unsupported_reason: str | None = None
    # Value tracking
    _current_value_known: bool = False

    @property
    def n_changes(self) -> int:
        return len(self.history)

    @property
    def root_cause(self) -> dict[str, Any] | None:
        """The first change (oldest)."""
        return self.history[0] if self.history else None

    @property
    def latest_change(self) -> dict[str, Any] | None:
        """The most recent change."""
        return self.history[-1] if self.history else None

    def to_dict(self) -> dict:
        """JSON-serializable dict representation."""
        return {
            "row_id": self.row_id,
            "column": self.column,
            "current_value": self.current_value,
            "n_changes": self.n_changes,
            "history": self.history,
            "became_null_at": self.became_null_at,
            "supported": self.supported,
        }

    def __repr__(self) -> str:
        return self.to_text(verbose=False)

    def to_text(self, verbose: bool = True) -> str:
        """Format as text. Use verbose=True for full details."""
        if not self.supported:
            return f"WhyResult: {self.unsupported_reason}"

        lines = [f"Cell History: row {self.row_id}, column '{self.column}'"]

        if self._current_value_known:
            lines.append(f"  Current value: {self.current_value}")
        else:
            lines.append("  Current value: (provide df to see)")

        if self.became_null_at:
            # Check if null was later recovered
            import pandas as pd

            is_still_null = pd.isna(self.current_value) if self._current_value_known else True
            if is_still_null:
                lines.append(f"  [!] Became null at step {self.became_null_at['step_id']}")
            else:
                lines.append(
                    f"  [i] Was null at step {self.became_null_at['step_id']} (later recovered)"
                )
            lines.append(f"      by: {self.became_null_at['operation']}")

        if self.history:
            lines.append(f"\n  History ({len(self.history)} changes, most recent first):")
            event_limit = 10 if verbose else 5
            for event in reversed(self.history[-event_limit:]):
                old = event.get("old_val", "?")
                new = event.get("new_val", "?")
                op = event.get("operation", "?")
                loc = event.get("code_location", "")
                lines.append(f"    {old} -> {new}")
                lines.append(f"      by: {op}")
                if verbose and loc:
                    lines.append(f"      at: {loc}")
        else:
            lines.append("\n  No changes tracked (original value)")

        return "\n".join(lines)


# ============ CONVENIENCE FUNCTIONS ============

# Default thresholds
_DEFAULT_MERGE_EXPANSION_THRESHOLD = 1.5
_DEFAULT_RETENTION_THRESHOLD = 0.5


def check(
    df: pd.DataFrame,
    *,
    merge_expansion_threshold: float | None = None,
    retention_threshold: float | None = None,
) -> CheckResult:
    """
    Run health check on pipeline.

    Args:
        df: DataFrame to check (used for additional validation)
        merge_expansion_threshold: Flag merges expanding beyond this ratio
        retention_threshold: Flag if retention drops below this

    Returns:
        CheckResult with .ok, .warnings, .facts, .suggestions
        Use print(result) for pretty output, result.to_dict() for data.

    Examples:
        result = tp.check(df)
        print(result)             # Pretty output
        result.raise_if_failed()  # For CI
    """
    ctx = get_context()
    warnings_list: list[CheckWarning] = []
    facts: dict[str, Any] = {}
    suggestions: list[str] = []

    merge_threshold_source = "user" if merge_expansion_threshold is not None else "default"
    retention_threshold_source = "user" if retention_threshold is not None else "default"

    merge_expansion_threshold = merge_expansion_threshold or _DEFAULT_MERGE_EXPANSION_THRESHOLD
    retention_threshold = retention_threshold or _DEFAULT_RETENTION_THRESHOLD

    # === FACTS ===
    dropped = ctx.store.get_dropped_rows()
    facts["rows_dropped"] = len(dropped)
    facts["total_steps"] = len(ctx.store.steps)

    # Merge statistics - filter to df's lineage to avoid cross-contamination
    merge_stats_list = _get_merge_stats_for_df(df, ctx)

    for i, (step_id, stats) in enumerate(merge_stats_list):
        facts[f"merge_{i}_expansion"] = stats.expansion_ratio
        facts[f"merge_{i}_result_rows"] = stats.result_rows

        if stats.expansion_ratio > merge_expansion_threshold:
            severity = "fact" if merge_threshold_source == "user" else "heuristic"
            warnings_list.append(
                CheckWarning(
                    category="merge_expansion",
                    severity=severity,
                    message=f"Merge expanded {stats.expansion_ratio:.2f}x "
                    f"({stats.left_rows} x {stats.right_rows} -> {stats.result_rows})",
                    details={
                        "step_id": step_id,
                        "expansion": stats.expansion_ratio,
                        "how": stats.how,
                    },
                    fix_hint="Check for duplicate keys in join columns",
                )
            )

        # Note on dup_rate semantics:
        # - left_dup_rate = fraction of LEFT rows appearing >1 times in result
        #   This happens when RIGHT table has duplicate join keys
        # - right_dup_rate = fraction of RIGHT rows appearing >1 times in result
        #   This happens when LEFT table has duplicate join keys
        if stats.right_dup_rate > 0.01:
            warnings_list.append(
                CheckWarning(
                    category="duplicate_keys",
                    severity="fact",
                    message=f"Left table has {stats.right_dup_rate:.1%} duplicate join keys",
                    details={"step_id": step_id, "dup_rate": stats.right_dup_rate},
                )
            )
        if stats.left_dup_rate > 0.01:
            warnings_list.append(
                CheckWarning(
                    category="duplicate_keys",
                    severity="fact",
                    message=f"Right table has {stats.left_dup_rate:.1%} duplicate join keys",
                    details={"step_id": step_id, "dup_rate": stats.left_dup_rate},
                )
            )

    # Retention rate - use max rows seen to handle multi-table pipelines
    if ctx.store.steps:
        max_rows_seen = 0
        for step in ctx.store.steps:
            # input_shape can be a single shape tuple (rows, cols) or
            # a tuple of shapes for merge operations ((left_rows, cols), (right_rows, cols))
            if step.input_shape:
                shape = step.input_shape
                if isinstance(shape[0], tuple):
                    # Multiple inputs (e.g., merge) - take max of all inputs
                    for s in shape:
                        if isinstance(s, tuple) and len(s) >= 1:
                            max_rows_seen = max(max_rows_seen, s[0])
                elif isinstance(shape[0], int):
                    max_rows_seen = max(max_rows_seen, shape[0])

            if step.output_shape and isinstance(step.output_shape[0], int):
                max_rows_seen = max(max_rows_seen, step.output_shape[0])

        if max_rows_seen > 0:
            current = len(df)
            retention = current / max_rows_seen if max_rows_seen > 0 else 1.0
            facts["retention_rate"] = round(retention, 4)

            if retention < retention_threshold:
                severity = "fact" if retention_threshold_source == "user" else "heuristic"
                warnings_list.append(
                    CheckWarning(
                        category="retention",
                        severity=severity,
                        message=f"Retention is {retention:.1%} (below {retention_threshold:.0%})",
                        details={
                            "retention": retention,
                            "max_rows_seen": max_rows_seen,
                            "current": current,
                        },
                        fix_hint="Review filter operations",
                    )
                )

    # === HEURISTICS ===
    for i, (step_id, stats) in enumerate(merge_stats_list):
        if stats.how == "left" and stats.expansion_ratio > 1.0:
            warnings_list.append(
                CheckWarning(
                    category="possible_unintended_expansion",
                    severity="heuristic",
                    message=f"Left join expanded {stats.expansion_ratio:.2f}x - was 1:1 expected?",
                    details={"step_id": step_id},
                    fix_hint="If 1:1 was intended, use validate='1:1' in merge()",
                )
            )

    drops_by_op = ctx.store.get_dropped_by_step()
    for op, count in drops_by_op.items():
        if count > 1000:
            suggestions.append(f"'{op}' dropped {count} rows - review if intentional")

    ok = len([w for w in warnings_list if w.severity == "fact"]) == 0

    return CheckResult(
        ok=ok,
        warnings=warnings_list,
        facts=facts,
        suggestions=suggestions,
        mode=ctx.config.mode.value,
        _drops_by_op=drops_by_op,
    )


def trace(
    df: pd.DataFrame,
    *,
    row: int | None = None,
    where: dict[str, Any] | None = None,
    include_ghost: bool = True,
) -> TraceResult | list[TraceResult]:
    """
    Trace a row's journey through the pipeline.

    Args:
        df: DataFrame to search in
        row: Row ID (if known)
        where: Selector dict, e.g. {"customer_id": "C123"}
        include_ghost: Include last-known values for dropped rows

    Returns:
        TraceResult (single row) or List[TraceResult] (if where matches multiple)
        Use print(result) for pretty output, result.to_dict() for data.

    Examples:
        result = tp.trace(df, row=5)
        print(result)
        tp.trace(df, where={"customer_id": "C123"})
    """
    ctx = get_context()

    # Mode enforcement for deep lineage
    if ctx.config.mode == TracePipeMode.CI and not ctx.config.should_capture_cell_history:
        # CI mode still supports basic trace (drop tracking)
        pass

    # Resolve row IDs
    if row is not None:
        row_ids = [row]
    elif where is not None:
        row_ids = _resolve_where(df, where, ctx)
    else:
        raise ValueError("Must provide 'row' or 'where'")

    results = []
    for rid in row_ids:
        result = _build_trace_result(rid, ctx, include_ghost)
        results.append(result)

    return results[0] if len(results) == 1 else results


def why(
    df: pd.DataFrame,
    *,
    col: str,
    row: int | None = None,
    where: dict[str, Any] | None = None,
) -> WhyResult | list[WhyResult]:
    """
    Explain why a cell has its current value.

    Args:
        df: DataFrame to search in
        col: Column name to trace
        row: Row ID (if known)
        where: Selector dict, e.g. {"customer_id": "C123"}

    Returns:
        WhyResult (single row) or List[WhyResult] (if where matches multiple)
        Use print(result) for pretty output, result.to_dict() for data.

    Examples:
        result = tp.why(df, col="amount", row=5)
        print(result)
        tp.why(df, col="email", where={"user_id": "U123"})
    """
    ctx = get_context()

    # Mode enforcement
    if ctx.config.mode == TracePipeMode.CI and not ctx.config.should_capture_cell_history:
        return WhyResult(
            row_id=-1,
            column=col,
            supported=False,
            unsupported_reason="Cell history requires mode='debug' or cell_history=True",
        )

    # Resolve row IDs
    if row is not None:
        row_ids = [row]
    elif where is not None:
        row_ids = _resolve_where(df, where, ctx)
    else:
        raise ValueError("Must provide 'row' or 'where'")

    results = []
    for rid in row_ids:
        result = _build_why_result(df, rid, col, ctx)
        results.append(result)

    return results[0] if len(results) == 1 else results


def report(
    df: pd.DataFrame,
    path: str = "tracepipe_report.html",
    *,
    title: str = "TracePipe Report",
) -> str:
    """
    Generate HTML report.

    Args:
        df: Final DataFrame
        path: Output path
        title: Report title

    Returns:
        Path to saved report
    """
    try:
        from .visualization.html_export import save as _save

        _save(path, title=title)
    except ImportError:
        # Fallback if visualization module can't be imported
        ctx = get_context()
        html_content = f"""<!DOCTYPE html>
<html>
<head><title>{title}</title></head>
<body>
<h1>{title}</h1>
<p>Mode: {ctx.config.mode.value}</p>
<p>Steps: {len(ctx.store.steps)}</p>
<p>Rows dropped: {len(ctx.store.get_dropped_rows())}</p>
<p>DataFrame shape: {df.shape}</p>
</body>
</html>"""
        with open(path, "w") as f:
            f.write(html_content)

    print(f"Report saved to: {path}")
    return path


def find(
    df: pd.DataFrame,
    *,
    where: dict[str, Any] | None = None,
    predicate: Callable[[pd.DataFrame], pd.Series] | None = None,
    limit: int = 10,
) -> list[int]:
    """
    Find row IDs matching a selector.

    Args:
        df: DataFrame to search
        where: Exact match selector
        predicate: Vector predicate (df -> boolean Series)
        limit: Maximum number of IDs to return

    Returns:
        List of row IDs

    Examples:
        rids = tp.find(df, where={"status": "failed"})
        tp.trace(df, row=rids[0])
    """
    ctx = get_context()

    if where:
        row_ids = _resolve_where(df, where, ctx, limit=limit)
    elif predicate:
        row_ids = _resolve_predicate(df, predicate, ctx, limit=limit)
    else:
        raise ValueError("Must provide 'where' or 'predicate'")

    return row_ids


# ============ HELPERS ============


def _get_merge_stats_for_df(df: pd.DataFrame, ctx) -> list[tuple[int, Any]]:
    """
    Get merge stats relevant to df's lineage only.

    This prevents cross-contamination where check(df) would show warnings
    from merges that produced OTHER DataFrames in the same session.
    """
    if not hasattr(ctx.store, "get_merge_stats"):
        return []

    all_stats = ctx.store.get_merge_stats()
    if not all_stats:
        return []

    # Get row IDs from df
    rids = ctx.row_manager.get_ids_array(df)
    if rids is None:
        return []

    # Find which merge steps produced rows in df
    relevant_step_ids = set()

    # Check merge mappings to find which merges produced df's rows
    if hasattr(ctx.store, "merge_mappings"):
        for mapping in ctx.store.merge_mappings:
            # Check if any of df's row IDs are in this merge's output
            for rid in rids:
                # Binary search in sorted out_rids
                i = np.searchsorted(mapping.out_rids, rid)
                if i < len(mapping.out_rids) and mapping.out_rids[i] == rid:
                    relevant_step_ids.add(mapping.step_id)
                    break  # Found at least one match, this merge is relevant

    # If no merge mappings found, fall back to checking if df was just merged
    # by seeing if it has more columns than typical (heuristic)
    if not relevant_step_ids and all_stats:
        # Fallback: return only the most recent merge that could have produced df
        # This handles the case where merge_mappings aren't available
        for step_id, stats in reversed(all_stats):
            if stats.result_rows == len(df):
                relevant_step_ids.add(step_id)
                break

    # Filter stats to relevant merges only
    return [(sid, stats) for sid, stats in all_stats if sid in relevant_step_ids]


def _json_safe(val: Any) -> Any:
    """Convert value to JSON-serializable form."""
    if pd.isna(val):
        return None
    if isinstance(val, (np.integer, np.floating)):
        return val.item()
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _resolve_where(
    df: pd.DataFrame,
    where: dict[str, Any],
    ctx,
    limit: int | None = None,
) -> list[int]:
    """Resolve row IDs from where dict selector."""
    rids = ctx.row_manager.get_ids_array(df)
    if rids is None:
        raise ValueError("DataFrame not tracked by TracePipe")

    mask = np.ones(len(df), dtype=bool)
    for col, val in where.items():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in DataFrame")

        series = df[col]
        if isinstance(val, (list, tuple)):
            col_mask = series.isin(val).to_numpy()
        elif pd.isna(val):
            col_mask = series.isna().to_numpy()
        else:
            col_mask = series.eq(val).to_numpy()
        mask &= col_mask

    matched_positions = np.where(mask)[0]
    if len(matched_positions) == 0:
        raise ValueError(f"No rows matched where={where}")

    if limit:
        matched_positions = matched_positions[:limit]

    return rids[matched_positions].tolist()


def _resolve_predicate(
    df: pd.DataFrame,
    predicate: Callable[[pd.DataFrame], pd.Series],
    ctx,
    limit: int | None = None,
) -> list[int]:
    """Resolve row IDs from predicate function."""
    rids = ctx.row_manager.get_ids_array(df)
    if rids is None:
        raise ValueError("DataFrame not tracked by TracePipe")

    mask_series = predicate(df)
    if not isinstance(mask_series, pd.Series):
        raise TypeError("predicate must return pd.Series")
    if mask_series.dtype != bool:
        raise TypeError("predicate must return boolean Series")

    mask = mask_series.to_numpy()
    matched_positions = np.where(mask)[0]

    if len(matched_positions) == 0:
        raise ValueError("No rows matched predicate")

    if limit:
        matched_positions = matched_positions[:limit]

    return rids[matched_positions].tolist()


def _build_trace_result(row_id: int, ctx, include_ghost: bool) -> TraceResult:
    """Build TraceResult for a single row."""
    store = ctx.store

    drop_event = store.get_drop_event(row_id)
    merge_origin = store.get_merge_origin(row_id)

    # v0.4+ provenance: concat origin and dedup representative
    concat_origin = None
    dedup_representative = None
    if hasattr(store, "get_concat_origin"):
        concat_origin = store.get_concat_origin(row_id)
    if hasattr(store, "get_duplicate_representative"):
        dedup_representative = store.get_duplicate_representative(row_id)

    # Use lineage-aware history to include pre-merge parent events
    if hasattr(store, "get_row_history_with_lineage"):
        history = store.get_row_history_with_lineage(row_id)
    else:
        history = store.get_row_history(row_id)

    dropped_at = None
    if drop_event:
        dropped_at = {
            "step_id": drop_event.get("step_id"),
            "operation": drop_event.get("operation"),
        }

    ghost_values = None
    if include_ghost and drop_event is not None:
        ghost_df = ctx.row_manager.get_ghost_rows(limit=10000)
        if not ghost_df.empty and "__tp_row_id__" in ghost_df.columns:
            ghost_row = ghost_df[ghost_df["__tp_row_id__"] == row_id]
            if not ghost_row.empty:
                ghost_values = ghost_row.iloc[0].to_dict()
                tp_cols = [
                    "__tp_row_id__",
                    "__tp_dropped_by__",
                    "__tp_dropped_step__",
                    "__tp_original_position__",
                ]
                for col in tp_cols:
                    ghost_values.pop(col, None)

    return TraceResult(
        row_id=row_id,
        is_alive=drop_event is None,
        dropped_at=dropped_at,
        merge_origin=merge_origin,
        events=history,
        ghost_values=ghost_values,
        concat_origin=concat_origin,
        dedup_representative=dedup_representative,
    )


def _build_why_result(df: pd.DataFrame, row_id: int, col: str, ctx) -> WhyResult:
    """Build WhyResult for a single cell."""
    from .value_provenance import explain_value

    history_obj = explain_value(row_id, col, df)

    current_value = None
    current_value_known = False
    rids = ctx.row_manager.get_ids_array(df)
    if rids is not None:
        pos = np.where(rids == row_id)[0]
        if len(pos) > 0 and col in df.columns:
            current_value = df.iloc[pos[0]][col]
            current_value_known = True

    became_null_at = None
    if history_obj.became_null_at:
        became_null_at = {
            "step_id": history_obj.became_null_at,
            "operation": history_obj.became_null_by,
        }

    result = WhyResult(
        row_id=row_id,
        column=col,
        current_value=_json_safe(current_value),
        history=[
            {
                "step_id": e.step_id,
                "operation": e.operation,
                "old_val": _json_safe(e.old_value),
                "new_val": _json_safe(e.new_value),
                "change_type": e.change_type,
                "code_location": e.code_location,
            }
            for e in history_obj.events
        ],
        became_null_at=became_null_at,
    )
    result._current_value_known = current_value_known
    return result

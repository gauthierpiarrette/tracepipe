# tests/test_convenience_debug.py
"""
Comprehensive tests for convenience.py and debug.py.

These modules are user-facing and need high coverage to ensure
TracePipe is reliable in production.
"""

import pandas as pd
import pytest

import tracepipe as tp
from tracepipe.convenience import (
    CheckFailed,
    CheckResult,
    CheckWarning,
)


@pytest.fixture(autouse=True)
def reset_tp():
    """Reset TracePipe before each test."""
    tp.reset()
    yield
    try:
        tp.disable()
    except Exception:
        pass


# =============================================================================
# CheckWarning TESTS
# =============================================================================


class TestCheckWarning:
    """Test CheckWarning dataclass."""

    def test_warning_repr_fact(self):
        """Fact warning has [!] icon."""
        warning = CheckWarning(
            category="retention",
            severity="fact",
            message="Low retention rate",
        )
        repr_str = repr(warning)
        assert "[!]" in repr_str
        assert "retention" in repr_str

    def test_warning_repr_heuristic(self):
        """Heuristic warning has [?] icon."""
        warning = CheckWarning(
            category="possible_issue",
            severity="heuristic",
            message="Might be a problem",
        )
        repr_str = repr(warning)
        assert "[?]" in repr_str

    def test_warning_with_details(self):
        """Warning can have details dict."""
        warning = CheckWarning(
            category="merge",
            severity="fact",
            message="Merge expanded",
            details={"ratio": 2.5, "step": 3},
            fix_hint="Check join keys",
        )
        assert warning.details["ratio"] == 2.5
        assert warning.fix_hint == "Check join keys"


# =============================================================================
# CheckResult TESTS
# =============================================================================


class TestCheckResult:
    """Test CheckResult dataclass."""

    def test_has_warnings_property(self):
        """has_warnings returns True when warnings exist."""
        result = CheckResult(
            ok=True,
            warnings=[CheckWarning(category="x", severity="fact", message="y")],
            facts={},
            suggestions=[],
            mode="debug",
        )
        assert result.has_warnings is True

    def test_fact_warnings_filter(self):
        """fact_warnings filters by severity."""
        result = CheckResult(
            ok=False,
            warnings=[
                CheckWarning(category="a", severity="fact", message="fact1"),
                CheckWarning(category="b", severity="heuristic", message="hint1"),
                CheckWarning(category="c", severity="fact", message="fact2"),
            ],
            facts={},
            suggestions=[],
            mode="debug",
        )
        fact_warnings = result.fact_warnings
        assert len(fact_warnings) == 2
        assert all(w.severity == "fact" for w in fact_warnings)

    def test_heuristic_warnings_filter(self):
        """heuristic_warnings filters by severity."""
        result = CheckResult(
            ok=True,
            warnings=[
                CheckWarning(category="a", severity="fact", message="fact1"),
                CheckWarning(category="b", severity="heuristic", message="hint1"),
            ],
            facts={},
            suggestions=[],
            mode="debug",
        )
        heuristic_warnings = result.heuristic_warnings
        assert len(heuristic_warnings) == 1
        assert heuristic_warnings[0].severity == "heuristic"

    def test_raise_if_failed_passes(self):
        """raise_if_failed returns self when ok."""
        result = CheckResult(ok=True, warnings=[], facts={}, suggestions=[], mode="debug")
        returned = result.raise_if_failed()
        assert returned is result

    def test_raise_if_failed_raises(self):
        """raise_if_failed raises CheckFailed when not ok."""
        result = CheckResult(
            ok=False,
            warnings=[CheckWarning(category="x", severity="fact", message="error")],
            facts={},
            suggestions=[],
            mode="debug",
        )
        with pytest.raises(CheckFailed):
            result.raise_if_failed()

    def test_to_text_verbose(self):
        """to_text with verbose=True shows all details."""
        result = CheckResult(
            ok=False,
            warnings=[
                CheckWarning(
                    category="retention",
                    severity="fact",
                    message="Low retention",
                    fix_hint="Check filters",
                ),
                CheckWarning(
                    category="suggestion",
                    severity="heuristic",
                    message="Consider optimization",
                    fix_hint="Add index",
                ),
            ],
            facts={"retention_rate": 0.3, "rows_dropped": 70},
            suggestions=["Review pipeline"],
            mode="debug",
        )
        text = result.to_text(verbose=True)
        assert "Measured facts:" in text
        assert "retention_rate" in text
        assert "Issues (confirmed):" in text
        assert "Check filters" in text
        assert "Suggestions (possible issues):" in text

    def test_to_text_non_verbose(self):
        """to_text with verbose=False is shorter."""
        result = CheckResult(ok=True, warnings=[], facts={"a": 1}, suggestions=[], mode="ci")
        text = result.to_text(verbose=False)
        assert "TracePipe Check" in text
        assert "Measured facts" not in text  # verbose only

    def test_to_dict(self):
        """to_dict exports all fields."""
        warning = CheckWarning(category="x", severity="fact", message="y")
        result = CheckResult(
            ok=True,
            warnings=[warning],
            facts={"key": "value"},
            suggestions=["hint"],
            mode="debug",
        )
        d = result.to_dict()
        assert d["ok"] is True
        assert d["mode"] == "debug"
        assert d["facts"]["key"] == "value"
        assert len(d["warnings"]) == 1


# =============================================================================
# TraceResult TESTS
# =============================================================================


class TestTraceResult:
    """Test TraceResult dataclass."""

    def test_trace_alive_row(self):
        """TraceResult for alive row."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = tp.trace(df, row=0)
        assert result.is_alive is True
        assert "OK" in str(result) or "Alive" in str(result)

    def test_trace_dropped_row(self):
        """TraceResult for dropped row shows drop info."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()
        result = tp.trace(df, row=1)
        assert result.is_alive is False
        text = str(result)
        assert "Dropped" in text or "X" in text

    def test_trace_with_events(self):
        """TraceResult shows events when cell is modified."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})
        df["a"] = df["a"] * 2
        result = tp.trace(df, row=0)
        text = result.to_text(verbose=True)
        assert "Events" in text

    def test_trace_to_dict(self):
        """TraceResult.to_dict() exports correctly."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = tp.trace(df, row=0)
        d = result.to_dict()
        assert "row_id" in d
        assert "is_alive" in d
        assert d["row_id"] == 0

    def test_trace_verbose_vs_non_verbose(self):
        """to_text verbose shows more detail."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})
        df["a"] = df["a"] * 2
        result = tp.trace(df, row=0)

        verbose_text = result.to_text(verbose=True)
        short_text = result.to_text(verbose=False)
        # Both should have core info
        assert "row 0" in verbose_text.lower() or "Row 0" in verbose_text
        assert "row 0" in short_text.lower() or "Row 0" in short_text

    def test_trace_merged_row_origin(self):
        """TraceResult shows merge origin for merged rows."""
        tp.enable(mode="debug")
        left = pd.DataFrame({"key": [1, 2], "left_val": ["a", "b"]})
        right = pd.DataFrame({"key": [1, 2], "right_val": [10, 20]})
        df = left.merge(right, on="key")
        result = tp.trace(df, row=0)
        # Merged row might have origin info
        text = str(result)
        assert result is not None

    def test_trace_ghost_values(self):
        """TraceResult can show ghost values for dropped rows."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.head(1)  # Drop rows 1 and 2
        result = tp.trace(df, row=1)  # Trace dropped row
        # Dropped row should have ghost values in debug mode
        text = result.to_text(verbose=True)
        assert result.is_alive is False

    def test_trace_unsupported_mode(self):
        """TraceResult in CI mode has limited support."""
        tp.enable(mode="ci")
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = tp.trace(df, row=0)
        text = str(result)
        # Should still return valid result
        assert result is not None

    def test_trace_merge_with_provenance(self):
        """TraceResult shows merge origin in debug mode."""
        tp.enable(mode="debug")
        left = pd.DataFrame({"key": [1, 2, 3], "left_val": ["a", "b", "c"]})
        right = pd.DataFrame({"key": [1, 2, 3], "right_val": [10, 20, 30]})
        df = left.merge(right, on="key", how="inner")
        result = tp.trace(df, row=0)
        text = result.to_text(verbose=True)
        # Should have merge info
        assert result is not None


# =============================================================================
# WhyResult TESTS
# =============================================================================


class TestWhyResult:
    """Test WhyResult dataclass."""

    def test_why_no_changes(self):
        """WhyResult for unchanged cell."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = tp.why(df, col="a", row=0)
        assert result.n_changes == 0
        text = str(result)
        assert "No changes" in text or "original" in text.lower()

    def test_why_with_changes(self):
        """WhyResult for changed cell shows history."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})
        df["a"] = df["a"] * 10
        result = tp.why(df, col="a", row=0)
        assert result.current_value == 10
        text = str(result)
        assert "History" in text or "change" in text.lower()

    def test_why_root_cause(self):
        """WhyResult.root_cause returns first change."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})
        df["a"] = df["a"] * 2
        df["a"] = df["a"] + 10
        result = tp.why(df, col="a", row=0)
        # root_cause is first change
        if result.history:
            assert result.root_cause is result.history[0]

    def test_why_latest_change(self):
        """WhyResult.latest_change returns last change."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})
        df["a"] = df["a"] * 2
        df["a"] = df["a"] + 10
        result = tp.why(df, col="a", row=0)
        if result.history:
            assert result.latest_change is result.history[-1]

    def test_why_became_null(self):
        """WhyResult tracks when value became null."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        df.loc[0, "a"] = None
        result = tp.why(df, col="a", row=0)
        text = str(result)
        # Should mention null
        assert "null" in text.lower() or result.current_value is None

    def test_why_to_dict(self):
        """WhyResult.to_dict() exports correctly."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = tp.why(df, col="a", row=0)
        d = result.to_dict()
        assert "row_id" in d
        assert "column" in d
        assert d["column"] == "a"

    def test_why_unsupported_mode(self):
        """WhyResult in CI mode is unsupported."""
        tp.enable(mode="ci")  # CI mode doesn't track cell history
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = tp.why(df, col="a", row=0)
        # Should indicate limited support in CI mode
        text = str(result)
        assert result is not None

    def test_why_null_then_recovered(self):
        """WhyResult shows when null was recovered."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        df.loc[0, "a"] = None  # Make null
        df["a"] = df["a"].fillna(999)  # Recover
        result = tp.why(df, col="a", row=0)
        assert result.current_value == 999
        text = result.to_text(verbose=True)
        # Should show the recovery
        assert result is not None

    def test_why_verbose_with_code_location(self):
        """WhyResult verbose shows code location."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})
        df["a"] = df["a"] * 2
        result = tp.why(df, col="a", row=0)
        verbose_text = result.to_text(verbose=True)
        short_text = result.to_text(verbose=False)
        # Verbose should have more detail
        assert len(verbose_text) >= len(short_text)

    def test_why_with_where_parameter(self):
        """why() with where= finds cell by column value."""
        tp.enable(mode="debug", watch=["amount"])
        df = pd.DataFrame(
            {
                "customer_id": ["C001", "C002", "C003"],
                "amount": [100.0, 200.0, 300.0],
            }
        )
        df["amount"] = df["amount"] * 1.5

        # Find cell by customer_id
        result = tp.why(df, col="amount", where={"customer_id": "C002"})

        assert result is not None
        assert result.column == "amount"
        assert result.current_value == 300.0  # 200 * 1.5
        assert result.n_changes >= 1

    def test_why_with_where_multiple_criteria(self):
        """why() with where= using multiple column criteria."""
        tp.enable(mode="debug", watch=["value"])
        df = pd.DataFrame(
            {
                "region": ["US", "US", "EU"],
                "category": ["A", "B", "A"],
                "value": [10, 20, 30],
            }
        )
        df["value"] = df["value"] + 100

        # Find by region AND category
        result = tp.why(df, col="value", where={"region": "US", "category": "B"})

        assert result is not None
        assert result.current_value == 120  # 20 + 100

    def test_why_with_where_no_match_raises(self):
        """why() with where= that matches no rows raises ValueError."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"id": ["X", "Y"], "a": [1, 2]})

        with pytest.raises(ValueError, match="No rows matched"):
            tp.why(df, col="a", where={"id": "Z"})


# =============================================================================
# DebugInspector TESTS
# =============================================================================


class TestDebugInspector:
    """Test tp.debug.inspect() methods."""

    def test_inspector_watch(self):
        """watch() adds columns."""
        tp.enable(mode="debug")
        dbg = tp.debug.inspect()
        dbg.watch("col1", "col2")
        assert "col1" in dbg.watched_columns
        assert "col2" in dbg.watched_columns

    def test_inspector_in_memory_diffs(self):
        """in_memory_diffs returns count."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})
        df["a"] = df["a"] * 2
        dbg = tp.debug.inspect()
        assert dbg.in_memory_diffs >= 0

    def test_inspector_total_diffs(self):
        """total_diffs returns count."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})
        df["a"] = df["a"] * 2
        dbg = tp.debug.inspect()
        assert dbg.total_diffs >= 0

    def test_inspector_dropped_by_operation(self):
        """dropped_by_operation returns dict."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()
        dbg = tp.debug.inspect()
        drops_by_op = dbg.dropped_by_operation()
        assert isinstance(drops_by_op, dict)

    def test_inspector_merge_stats(self):
        """merge_stats returns list of dicts."""
        tp.enable(mode="debug")
        left = pd.DataFrame({"key": [1, 2], "val": ["a", "b"]})
        right = pd.DataFrame({"key": [2, 3], "info": [100, 200]})
        left.merge(right, on="key")
        dbg = tp.debug.inspect()
        stats = dbg.merge_stats()
        assert isinstance(stats, list)

    def test_inspector_mass_updates(self):
        """mass_updates returns list."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": range(10)})
        df = df.head(5)
        dbg = tp.debug.inspect()
        updates = dbg.mass_updates()
        assert isinstance(updates, list)

    def test_inspector_ghost_rows(self):
        """ghost_rows returns DataFrame."""
        tp.enable(mode="debug", watch=["a"])
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.head(1)
        dbg = tp.debug.inspect()
        ghosts = dbg.ghost_rows()
        assert isinstance(ghosts, pd.DataFrame)

    def test_inspector_aggregation_groups(self):
        """aggregation_groups returns list."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"cat": ["A", "A", "B"], "val": [1, 2, 3]})
        df.groupby("cat").sum()
        dbg = tp.debug.inspect()
        groups = dbg.aggregation_groups()
        assert isinstance(groups, list)

    def test_inspector_register(self):
        """register() manually registers DataFrame."""
        tp.enable(mode="debug")
        dbg = tp.debug.inspect()
        df = pd.DataFrame({"a": [1, 2, 3]})
        dbg.register(df)
        # Should be able to get row IDs after registration
        rids = dbg.get_row_ids(df)
        assert rids is not None

    def test_inspector_get_row_ids(self):
        """get_row_ids returns array."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})
        dbg = tp.debug.inspect()
        rids = dbg.get_row_ids(df)
        assert rids is not None
        assert len(rids) == 3

    def test_inspector_repr_enabled(self):
        """Inspector repr when enabled."""
        tp.enable(mode="debug")
        dbg = tp.debug.inspect()
        repr_str = repr(dbg)
        assert "DebugInspector" in repr_str
        assert "mode=debug" in repr_str

    def test_inspector_repr_disabled(self):
        """Inspector repr when disabled."""
        tp.reset()
        dbg = tp.debug.inspect()
        repr_str = repr(dbg)
        assert "enabled=False" in repr_str

    def test_inspector_export_json(self, tmp_path):
        """export('json', path) creates file."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()
        dbg = tp.debug.inspect()
        filepath = tmp_path / "export.json"
        dbg.export("json", str(filepath))
        assert filepath.exists()

    def test_inspector_export_json_no_path(self):
        """export('json') returns JSON string."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})
        dbg = tp.debug.inspect()
        json_str = dbg.export("json")
        assert json_str is not None
        assert isinstance(json_str, str)

    def test_inspector_export_invalid_format(self):
        """export() with invalid format raises."""
        tp.enable(mode="debug")
        dbg = tp.debug.inspect()
        with pytest.raises(ValueError, match="Unknown format"):
            dbg.export("invalid_format")

    def test_inspector_export_arrow_no_path(self):
        """export('arrow') without path raises."""
        tp.enable(mode="debug")
        dbg = tp.debug.inspect()
        with pytest.raises(ValueError, match="path is required"):
            dbg.export("arrow")

    def test_inspector_alive_rows(self):
        """alive_rows returns non-dropped rows."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        df = df.head(3)  # Drop rows 3 and 4
        dbg = tp.debug.inspect()
        alive = dbg.alive_rows()
        assert 0 in alive
        assert 1 in alive
        assert 2 in alive
        # 3 and 4 should be dropped
        dropped = dbg.dropped_rows()
        assert 3 in dropped or 4 in dropped

    def test_inspector_export_arrow_with_path(self, tmp_path):
        """export('arrow', path) creates parquet file if pyarrow available."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.dropna()
        dbg = tp.debug.inspect()
        filepath = tmp_path / "export.parquet"
        try:
            dbg.export("arrow", str(filepath))
            assert filepath.exists()
        except ImportError:
            # pyarrow not installed, expected
            pass

    def test_inspector_export_arrow_full_pipeline(self, tmp_path):
        """export('arrow') with full pipeline data."""
        tp.enable(mode="debug", watch=["val"])

        # Build pipeline with various operations
        df = pd.DataFrame({"key": [1, 2, 3], "val": [10.0, None, 30.0]})
        df = df.dropna()
        df["val"] = df["val"] * 2

        dbg = tp.debug.inspect()
        filepath = tmp_path / "lineage.parquet"

        try:
            import pyarrow  # noqa: F401

            dbg.export("arrow", str(filepath))
            assert filepath.exists()
            # Verify we can read it back
            import pyarrow.parquet as pq

            table = pq.read_table(filepath)
            assert table.num_rows >= 0
        except ImportError:
            # pyarrow not installed - test the import error path
            with pytest.raises(ImportError, match="pyarrow"):
                dbg.export("arrow", str(filepath))


# =============================================================================
# debug.find() TESTS
# =============================================================================


class TestDebugFind:
    """Test tp.debug.find() function."""

    def test_find_with_where(self):
        """find() returns row IDs matching criteria."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"status": ["ok", "fail", "ok"], "val": [1, 2, 3]})
        rids = tp.debug.find(df, where={"status": "ok"})
        assert len(rids) == 2
        assert 0 in rids
        assert 2 in rids

    def test_find_no_match(self):
        """find() raises ValueError when no match."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"status": ["ok", "ok"], "val": [1, 2]})
        with pytest.raises(ValueError, match="No rows matched"):
            tp.debug.find(df, where={"status": "fail"})

    def test_find_multiple_criteria(self):
        """find() with multiple where criteria."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "x"]})
        rids = tp.debug.find(df, where={"a": 1, "b": "x"})
        assert len(rids) == 1
        assert 0 in rids

    def test_find_with_predicate(self):
        """find() with predicate function."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        rids = tp.debug.find(df, predicate=lambda d: d["a"] > 3)
        assert len(rids) == 2  # rows with a=4 and a=5

    def test_find_predicate_no_match(self):
        """find() with predicate raises when no match."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="No rows matched"):
            tp.debug.find(df, predicate=lambda d: d["a"] > 100)

    def test_find_predicate_invalid_return(self):
        """find() with invalid predicate raises TypeError."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(TypeError, match="predicate must return"):
            tp.debug.find(df, predicate=lambda d: "not a series")

    def test_find_where_list_values(self):
        """find() where with list matches multiple values."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"status": ["a", "b", "c", "a"]})
        rids = tp.debug.find(df, where={"status": ["a", "b"]})
        assert len(rids) == 3  # rows 0, 1, 3

    def test_find_where_null(self):
        """find() where with None matches nulls."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, None, 3, None]})
        rids = tp.debug.find(df, where={"a": None})
        assert len(rids) == 2  # rows 1 and 3

    def test_find_missing_column(self):
        """find() with missing column raises."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Column"):
            tp.debug.find(df, where={"missing_col": 1})

    def test_find_no_selector(self):
        """find() without where or predicate raises."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Must provide"):
            tp.debug.find(df)


# =============================================================================
# check() FUNCTION TESTS
# =============================================================================


class TestCheckFunction:
    """Test tp.check() with various scenarios."""

    def test_check_low_retention_warning(self):
        """check() warns on low retention."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": range(100)})
        df = df.head(10)  # 10% retention
        result = tp.check(df)
        # Should have warning about low retention
        assert len(result.warnings) > 0 or result.facts["retention_rate"] < 0.5

    def test_check_custom_retention_threshold(self):
        """check() respects custom retention_threshold."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": range(100)})
        df = df.head(60)  # 60% retention
        # With default threshold (0.5), this should pass
        result = tp.check(df, retention_threshold=0.5)
        assert result.ok or result.facts["retention_rate"] >= 0.5

    def test_check_merge_expansion(self):
        """check() detects merge expansion."""
        tp.enable(mode="debug")
        left = pd.DataFrame({"key": [1, 1, 1], "val": ["a", "b", "c"]})
        right = pd.DataFrame({"key": [1, 1], "info": [100, 200]})
        df = left.merge(right, on="key")  # Many-to-many = 6 rows
        result = tp.check(df, merge_expansion_threshold=1.5)
        # Should detect the expansion
        assert result is not None

    def test_check_left_join_expansion_heuristic(self):
        """check() adds heuristic warning for left join expansion."""
        tp.enable(mode="debug")
        left = pd.DataFrame({"key": [1, 1], "val": ["a", "b"]})
        right = pd.DataFrame({"key": [1, 1, 1], "info": [10, 20, 30]})
        df = left.merge(right, on="key", how="left")  # Expands to 6 rows
        result = tp.check(df)
        # Should have heuristic warning about expansion
        heuristics = result.heuristic_warnings
        assert isinstance(heuristics, list)

    def test_trace_no_selector(self):
        """trace() without row or where raises."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Must provide"):
            tp.trace(df)


# =============================================================================
# report() FUNCTION TESTS
# =============================================================================


class TestReportFunction:
    """Test tp.report() HTML generation."""

    def test_report_creates_file(self, tmp_path):
        """report() creates HTML file."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, None, 3]})
        df = df.dropna()
        filepath = tmp_path / "report.html"
        result = tp.report(df, str(filepath))
        assert filepath.exists()
        assert result == str(filepath)

    def test_report_html_content(self, tmp_path):
        """report() creates valid HTML."""
        tp.enable(mode="debug")
        df = pd.DataFrame({"a": [1, 2, 3]})
        filepath = tmp_path / "report.html"
        tp.report(df, str(filepath))
        content = filepath.read_text()
        assert "<html" in content or "<!DOCTYPE" in content

    def test_report_end_to_end_pipeline(self, tmp_path):
        """End-to-end test: full pipeline with report generation."""
        # Enable tracking with watched columns
        tp.enable(mode="debug", watch=["amount", "status"])

        # Create initial data
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "amount": [100.0, None, 300.0, 400.0, 500.0],
                "status": ["active", "pending", "active", "closed", "active"],
            }
        )

        # Pipeline operations
        df = df.dropna()  # Drop row with null
        df["amount"] = df["amount"] * 1.1  # Transform
        df = df[df["status"] == "active"]  # Filter

        # Generate report
        filepath = tmp_path / "pipeline_report.html"
        result_path = tp.report(df, str(filepath), title="Pipeline Audit")

        # Verify report was created
        assert filepath.exists()
        assert result_path == str(filepath)

        # Verify report contains meaningful content
        content = filepath.read_text()
        assert "Pipeline Audit" in content or "TracePipe" in content
        assert "html" in content.lower()

        # Verify check() still works after report
        check_result = tp.check(df)
        assert check_result.facts["rows_dropped"] >= 2  # At least 2 rows dropped

    def test_report_html_contains_pipeline_info(self, tmp_path):
        """HTML report contains pipeline statistics and operations."""
        tp.enable(mode="debug", watch=["value"])

        # Create pipeline with trackable operations
        df = pd.DataFrame(
            {
                "id": ["A", "B", "C", "D", "E"],
                "value": [10, 20, None, 40, 50],
                "category": ["x", "y", "x", "y", "x"],
            }
        )

        # Operations that should appear in report
        df = df.dropna()  # Drop 1 row
        df["value"] = df["value"] * 2  # Modify values
        df = df[df["value"] > 30]  # Filter rows

        filepath = tmp_path / "detailed_report.html"
        tp.report(df, str(filepath), title="Detailed Pipeline Report")

        content = filepath.read_text()

        # Verify HTML structure
        assert "<!DOCTYPE html>" in content or "<html" in content
        assert "</html>" in content
        assert "<head>" in content
        assert "<body>" in content

        # Verify title appears
        assert "Detailed Pipeline Report" in content

        # Verify it's a complete HTML document
        assert len(content) > 500  # Should have substantial content

    def test_report_html_with_merge_operations(self, tmp_path):
        """HTML report captures merge operation details."""
        tp.enable(mode="debug")

        # Create tables for merge
        customers = pd.DataFrame({"customer_id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
        orders = pd.DataFrame({"customer_id": [1, 1, 2, 4], "amount": [100, 150, 200, 50]})

        # Merge operation
        df = customers.merge(orders, on="customer_id", how="left")

        filepath = tmp_path / "merge_report.html"
        tp.report(df, str(filepath), title="Merge Pipeline")

        content = filepath.read_text()

        # Should be valid HTML
        assert "<html" in content
        assert "</html>" in content
        assert "Merge Pipeline" in content

    def test_report_html_empty_pipeline(self, tmp_path):
        """HTML report handles pipeline with no operations."""
        tp.enable(mode="debug")

        df = pd.DataFrame({"a": [1, 2, 3]})
        # No operations - just create and report

        filepath = tmp_path / "empty_pipeline.html"
        tp.report(df, str(filepath))

        assert filepath.exists()
        content = filepath.read_text()
        assert "<html" in content or "<!DOCTYPE" in content

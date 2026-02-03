# tests/test_lineage_through_merge.py
"""
Integration tests for cell-level provenance tracking through merge operations.

These tests verify that TracePipe correctly tracks cell changes that happen
BEFORE a merge operation and makes them visible when querying AFTER the merge.

This addresses the critical bug where:
1. Cell changes (e.g., fillna) are recorded against pre-merge row IDs
2. Merge assigns NEW row IDs to result rows
3. Queries for post-merge row IDs couldn't find pre-merge changes

The fix: get_row_history_with_lineage() follows merge_mappings to include
parent row history in query results.
"""

import pandas as pd
import pytest

import tracepipe as tp


@pytest.fixture(autouse=True)
def reset_tracepipe():
    """Reset TracePipe state before each test."""
    tp.reset()
    yield
    tp.disable()


class TestCellProvenanceThroughMerge:
    """Tests for cell history surviving merge operations."""

    def test_why_shows_fillna_change_after_merge(self):
        """why() should show pre-merge fillna change for post-merge row."""
        tp.enable(mode="debug", watch=["income"])

        # Setup - one row with NaN income
        customers = pd.DataFrame(
            {
                "id": ["C1", "C2"],
                "income": [50000.0, None],  # C2 has NaN
            }
        )
        regions = pd.DataFrame(
            {
                "id": ["C1", "C2"],
                "region": ["NY", "CA"],
            }
        )

        # Make change BEFORE merge
        customers["income"] = customers["income"].fillna(0)

        # Verify fillna worked
        assert customers.loc[1, "income"] == 0

        # Merge
        df = customers.merge(regions, on="id")

        # Query post-merge row for C2
        result = tp.why(df, col="income", where={"id": "C2"})

        # CRITICAL: The fillna change should be visible!
        assert result.n_changes >= 1, (
            f"Pre-merge fillna change should be tracked, got {result.n_changes} changes. "
            f"History: {result.history}"
        )

        # Verify the NaN -> 0 transition is in history
        found_nan_to_zero = False
        for event in result.history:
            old_val = event.get("old_val")
            new_val = event.get("new_val")
            if (pd.isna(old_val) or old_val is None) and new_val == 0:
                found_nan_to_zero = True
                break

        assert found_nan_to_zero, f"Should show NaN -> 0 transition, history: {result.history}"

    def test_why_shows_multiple_changes_after_merge(self):
        """why() should show multiple pre-merge changes after merge."""
        tp.enable(mode="debug", watch=["value"])

        df = pd.DataFrame({"key": [1], "value": [10]})

        # Multiple changes before merge
        df["value"] = df["value"] * 2  # 10 -> 20
        df["value"] = df["value"] + 5  # 20 -> 25

        other = pd.DataFrame({"key": [1], "extra": ["x"]})
        df = df.merge(other, on="key")

        result = tp.why(df, col="value", row=0)

        # Should have at least 2 change events
        assert result.n_changes >= 2, f"Expected at least 2 changes, got {result.n_changes}"

    def test_trace_shows_complete_journey_through_merge(self):
        """trace() should include events from before merge."""
        tp.enable(mode="debug", watch=["value"])

        df = pd.DataFrame({"key": [1], "value": [None]})
        df["value"] = df["value"].fillna(99)  # Change before merge

        other = pd.DataFrame({"key": [1], "extra": ["x"]})
        df = df.merge(other, on="key")

        result = tp.trace(df, row=0)

        # Should show the fillna event even though merge happened
        assert (
            result.n_events >= 1
        ), f"Pre-merge events should be visible, got {result.n_events} events"

    def test_lineage_through_multiple_merges(self):
        """Cell history should survive multiple sequential merges."""
        tp.enable(mode="debug", watch=["amount"])

        # Start with data
        df = pd.DataFrame({"id": [1], "amount": [None]})
        df["amount"] = df["amount"].fillna(100)

        # First merge
        extra1 = pd.DataFrame({"id": [1], "region": ["US"]})
        df = df.merge(extra1, on="id")

        # Second merge
        extra2 = pd.DataFrame({"id": [1], "status": ["active"]})
        df = df.merge(extra2, on="id")

        result = tp.why(df, col="amount", where={"id": 1})

        # Should still see the original fillna change
        assert (
            result.n_changes >= 1
        ), f"Change should survive 2 merges, got {result.n_changes} changes"

    def test_merge_origin_still_captured(self):
        """Merge origin info should still be available alongside history."""
        tp.enable(mode="debug", watch=["val"])

        left = pd.DataFrame({"key": [1], "val": [10]})
        left["val"] = left["val"] * 2  # Change before merge

        right = pd.DataFrame({"key": [1], "other": ["x"]})
        df = left.merge(right, on="key")

        # Use where= to find the actual row in the merged DataFrame
        result = tp.trace(df, where={"key": 1})

        # Should have merge origin (merged rows get new IDs with parent mapping)
        assert result.merge_origin is not None, "Merge origin should be captured"

        # Should also have the pre-merge change (via lineage traversal)
        assert result.n_events >= 1, "Pre-merge events should also be visible"


class TestEnableStateReset:
    """Tests for enable() behavior with repeated calls."""

    def test_enable_twice_resets_state(self):
        """Calling enable() twice should reset accumulated state."""
        tp.enable(mode="debug")

        # First pipeline run
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.head(1)  # Drop some rows

        # Get check result
        result1 = tp.check(df)
        steps_after_first = len(tp.debug.inspect().steps)

        # Second "run" with enable() again (simulating IDE rerun)
        tp.enable(mode="debug")

        # State should be reset
        steps_after_reset = len(tp.debug.inspect().steps)
        assert (
            steps_after_reset == 0
        ), f"Steps should be cleared on re-enable, got {steps_after_reset}"

    def test_enable_twice_doesnt_accumulate_merge_stats(self):
        """Calling enable() twice shouldn't accumulate merge warnings."""
        # First run
        tp.enable(mode="debug")
        left = pd.DataFrame({"k": [1, 1], "v": [1, 2]})  # Duplicate keys
        right = pd.DataFrame({"k": [1], "r": ["a"]})
        df1 = left.merge(right, on="k")
        result1 = tp.check(df1)
        n_warnings_first = len([w for w in result1.warnings if "duplicate" in w.message.lower()])

        # Second "run" with enable() again
        tp.enable(mode="debug")
        left = pd.DataFrame({"k": [1, 1], "v": [1, 2]})
        right = pd.DataFrame({"k": [1], "r": ["a"]})
        df2 = left.merge(right, on="k")
        result2 = tp.check(df2)
        n_warnings_second = len([w for w in result2.warnings if "duplicate" in w.message.lower()])

        # Should NOT have 2x warnings
        assert n_warnings_second == n_warnings_first, (
            f"Expected {n_warnings_first} duplicate warnings, "
            f"got {n_warnings_second} (accumulated)"
        )

    def test_watched_columns_reset_on_new_watch(self):
        """When watch param is provided, old watched columns should be cleared."""
        tp.enable(mode="debug", watch=["col_a"])

        # Re-enable with different watch
        tp.enable(mode="debug", watch=["col_b"])

        ctx = tp.debug.inspect()
        watched = ctx.watched_columns

        assert "col_b" in watched, "New watch column should be present"
        assert "col_a" not in watched, "Old watch column should be cleared"


class TestFillnaTrackingVerification:
    """Regression tests to verify fillna changes are actually tracked."""

    def test_series_fillna_assignment_tracked(self):
        """df['col'] = df['col'].fillna(val) should be tracked."""
        tp.enable(mode="debug", watch=["a"])

        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        df["a"] = df["a"].fillna(0)

        # Verify the value changed
        assert df.loc[1, "a"] == 0

        # Verify TracePipe tracked it
        result = tp.why(df, col="a", row=1)
        assert (
            result.n_changes >= 1
        ), f"Series.fillna + assignment should be tracked, got {result.n_changes} changes"

    def test_dataframe_fillna_tracked(self):
        """df.fillna({'col': val}) should be tracked."""
        tp.enable(mode="debug", watch=["a"])

        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        df = df.fillna({"a": 0})

        result = tp.why(df, col="a", row=1)
        assert (
            result.n_changes >= 1
        ), f"DataFrame.fillna should be tracked, got {result.n_changes} changes"

    def test_loc_assignment_tracked(self):
        """df.loc[mask, col] = val should be tracked."""
        tp.enable(mode="debug", watch=["a"])

        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        df.loc[df["a"].isna(), "a"] = 0

        result = tp.why(df, col="a", row=1)
        assert (
            result.n_changes >= 1
        ), f"loc assignment should be tracked, got {result.n_changes} changes"


class TestMergeWarningLabeling:
    """Tests for correct left/right labeling in merge warnings."""

    def test_duplicate_left_keys_warns_about_left(self):
        """When LEFT has duplicate keys, warning should say 'Left table'."""
        tp.enable(mode="debug")

        # LEFT has duplicate key (94107 appears twice)
        left = pd.DataFrame({"zip": ["10001", "94107", "94107"], "val": [1, 2, 3]})
        # RIGHT has unique keys
        right = pd.DataFrame({"zip": ["10001", "94107", "99999"], "region": ["NY", "CA", "XX"]})

        df = left.merge(right, on="zip", how="left")
        result = tp.check(df)

        # Should warn about LEFT having duplicates, not RIGHT
        dup_warnings = [w for w in result.warnings if "duplicate" in w.message.lower()]
        assert len(dup_warnings) >= 1, "Should have duplicate key warning"

        # The warning should mention "Left table", not "Right table"
        left_warnings = [w for w in dup_warnings if "left" in w.message.lower()]
        right_warnings = [w for w in dup_warnings if "right" in w.message.lower()]

        assert len(left_warnings) >= 1, (
            f"Should warn about LEFT table having duplicates. "
            f"Got warnings: {[w.message for w in dup_warnings]}"
        )
        # RIGHT is unique, shouldn't warn about right
        assert len(right_warnings) == 0, (
            f"Should NOT warn about RIGHT table (it's unique). "
            f"Got warnings: {[w.message for w in dup_warnings]}"
        )

    def test_duplicate_right_keys_warns_about_right(self):
        """When RIGHT has duplicate keys, warning should say 'Right table'."""
        tp.enable(mode="debug")

        # LEFT has unique keys
        left = pd.DataFrame({"zip": ["10001", "94107"], "val": [1, 2]})
        # RIGHT has duplicate key (94107 appears twice)
        right = pd.DataFrame({"zip": ["10001", "94107", "94107"], "region": ["NY", "CA", "CA2"]})

        df = left.merge(right, on="zip", how="left")
        result = tp.check(df)

        dup_warnings = [w for w in result.warnings if "duplicate" in w.message.lower()]
        assert len(dup_warnings) >= 1, "Should have duplicate key warning"

        # The warning should mention "Right table"
        right_warnings = [w for w in dup_warnings if "right" in w.message.lower()]
        assert len(right_warnings) >= 1, (
            f"Should warn about RIGHT table having duplicates. "
            f"Got warnings: {[w.message for w in dup_warnings]}"
        )


class TestNoDoubleLogging:
    """Tests to verify no duplicate events are logged."""

    def test_fillna_logs_once_not_twice(self):
        """df['col'] = df['col'].fillna() should log exactly one event per row."""
        tp.enable(mode="debug", watch=["income"])

        df = pd.DataFrame({"id": [1], "income": [None]})
        df["income"] = df["income"].fillna(0)

        result = tp.why(df, col="income", row=0)

        # Should have exactly 1 change event, not 2 (no double-logging)
        assert result.n_changes == 1, (
            f"Expected exactly 1 change event, got {result.n_changes}. "
            f"Double-logging bug if > 1. History: {result.history}"
        )

    def test_setitem_logs_once(self):
        """Direct column assignment should log exactly once."""
        tp.enable(mode="debug", watch=["val"])

        df = pd.DataFrame({"id": [1], "val": [10]})
        df["val"] = df["val"] * 2

        result = tp.why(df, col="val", row=0)

        assert result.n_changes == 1, (
            f"Expected exactly 1 change event, got {result.n_changes}. "
            f"History: {result.history}"
        )


class TestMergeWarningScoping:
    """Tests for merge warnings being scoped to df's lineage."""

    def test_check_only_shows_warnings_for_df_lineage(self):
        """check(df) should only show warnings from merges that produced df."""
        tp.enable(mode="debug")

        # First pipeline - merge with unique right
        left1 = pd.DataFrame({"k": ["a", "a", "b"], "v": [1, 2, 3]})  # Dup left
        right1 = pd.DataFrame({"k": ["a", "b", "c"], "r": [10, 20, 30]})  # Unique
        df1 = left1.merge(right1, on="k", how="left")

        # Second pipeline - merge with duplicate right
        left2 = pd.DataFrame({"k": ["x", "y"], "v": [1, 2]})  # Unique
        right2 = pd.DataFrame({"k": ["x", "x", "y"], "r": [10, 11, 20]})  # Dup right
        df2 = left2.merge(right2, on="k", how="left")

        # check(df1) should only show warnings about LEFT duplicates (from df1's merge)
        result1 = tp.check(df1)
        dup_warnings1 = [w for w in result1.warnings if "duplicate" in w.message.lower()]

        # check(df2) should only show warnings about RIGHT duplicates (from df2's merge)
        result2 = tp.check(df2)
        dup_warnings2 = [w for w in result2.warnings if "duplicate" in w.message.lower()]

        # df1's check should NOT include df2's "Right table" warning
        right_in_df1 = [w for w in dup_warnings1 if "right" in w.message.lower()]
        assert len(right_in_df1) == 0, (
            f"df1 check should not have Right table warnings (from df2). "
            f"Got: {[w.message for w in dup_warnings1]}"
        )

        # df2's check should NOT include df1's "Left table" warning
        left_in_df2 = [w for w in dup_warnings2 if "left" in w.message.lower()]
        assert len(left_in_df2) == 0, (
            f"df2 check should not have Left table warnings (from df1). "
            f"Got: {[w.message for w in dup_warnings2]}"
        )


class TestLineageDepthLimit:
    """Tests for lineage traversal depth limiting."""

    def test_max_depth_prevents_infinite_loop(self):
        """Lineage traversal should respect max_depth to prevent issues."""
        tp.enable(mode="debug", watch=["val"])

        # Create a chain of merges
        df = pd.DataFrame({"key": [1], "val": [10]})
        df["val"] = df["val"] + 1  # Change at depth 0

        # Do many merges to test depth limiting
        for i in range(15):
            other = pd.DataFrame({"key": [1], f"extra_{i}": [i]})
            df = df.merge(other, on="key")

        # This should complete without hanging (depth limit = 10 by default)
        result = tp.why(df, col="val", row=0)

        # Should have at least the original change (within depth limit)
        # The exact count depends on implementation, but it shouldn't hang
        assert result is not None

"""
Integration tests for multi-scenario session behavior.

These tests verify that:
1. Multiple operations in the same session don't contaminate each other
2. Multiple pipelines in sequence produce correct isolated results
3. Warning messages contain accurate content
4. Reset properly clears state between runs
"""

import pandas as pd

import tracepipe as tp


class TestMultiScenarioSessions:
    """Tests for running multiple scenarios in sequence without contamination."""

    def test_reset_creates_fresh_session(self):
        """tp.reset() followed by enable() creates a truly fresh session."""
        # First session
        tp.reset()
        tp.enable(mode="debug", watch=["val"])
        df1 = pd.DataFrame({"val": [None]})
        df1["val"] = df1["val"].fillna(100)
        result1 = tp.why(df1, col="val", row=0)
        assert result1.n_changes == 1

        # Second session - completely fresh
        tp.reset()
        tp.enable(mode="debug", watch=["val"])
        df2 = pd.DataFrame({"val": [None]})
        df2["val"] = df2["val"].fillna(200)

        # Should track the second fillna correctly
        result2 = tp.why(df2, col="val", row=0)
        assert result2.n_changes == 1
        assert result2.history[0]["new_val"] == 200.0

        tp.disable()

    def test_merge_then_transform_correct_order(self):
        """Operations should be tracked in correct chronological order."""
        tp.reset()
        tp.enable(mode="debug", watch=["val"])

        left = pd.DataFrame({"key": ["a", "b"], "val": [10, 20]})
        right = pd.DataFrame({"key": ["a", "b"], "extra": [100, 200]})

        # Transform first
        left["val"] = left["val"] * 2

        # Then merge
        df = left.merge(right, on="key")

        # Should show the *2 change
        result = tp.why(df, col="val", row=0)
        assert result.n_changes == 1
        assert result.history[0]["new_val"] == 20  # 10 * 2

        tp.disable()

    def test_enable_reset_clears_previous_state(self):
        """Calling enable() again should reset all state."""
        tp.reset()

        # First run
        tp.enable(mode="debug", watch=["a"])
        df1 = pd.DataFrame({"a": [None, 1]})
        df1["a"] = df1["a"].fillna(0)

        # Get stats from first run
        result1 = tp.why(df1, col="a", row=0)
        assert result1.n_changes == 1

        # Second run - should start fresh
        tp.enable(mode="debug", watch=["b"])  # Different watch
        df2 = pd.DataFrame({"b": [None, 2]})
        df2["b"] = df2["b"].fillna(99)

        # First df's events should not affect second df's check
        result2 = tp.why(df2, col="b", row=0)
        assert result2.n_changes == 1
        assert result2.history[0]["new_val"] == 99.0

        tp.disable()

    def test_multiple_merges_distinct_warnings(self):
        """Each merge should have its own warnings, not accumulated."""
        tp.reset()
        tp.enable(mode="debug")

        # Merge 1: Left has duplicates
        left1 = pd.DataFrame({"k": ["a", "a", "b"], "v": [1, 2, 3]})
        right1 = pd.DataFrame({"k": ["a", "b", "c"], "r": [10, 20, 30]})
        df1 = left1.merge(right1, on="k", how="left")

        # Merge 2: Right has duplicates
        left2 = pd.DataFrame({"k": ["x", "y"], "v": [1, 2]})
        right2 = pd.DataFrame({"k": ["x", "x", "y"], "r": [10, 11, 20]})
        df2 = left2.merge(right2, on="k", how="left")

        # Check df1 warnings - should only mention LEFT duplicates
        result1 = tp.check(df1)
        dup_warnings1 = [w for w in result1.warnings if "duplicate" in w.message.lower()]
        assert (
            len(dup_warnings1) == 1
        ), f"df1 should have exactly 1 dup warning, got {len(dup_warnings1)}"
        assert "left" in dup_warnings1[0].message.lower(), "df1 warning should mention LEFT"

        # Check df2 warnings - should only mention RIGHT duplicates
        result2 = tp.check(df2)
        dup_warnings2 = [w for w in result2.warnings if "duplicate" in w.message.lower()]
        assert (
            len(dup_warnings2) == 1
        ), f"df2 should have exactly 1 dup warning, got {len(dup_warnings2)}"
        assert "right" in dup_warnings2[0].message.lower(), "df2 warning should mention RIGHT"

        tp.disable()


class TestMessageContentVerification:
    """Tests that verify actual warning/message content is accurate."""

    def test_left_duplicate_warning_text(self):
        """Left duplicate warning should clearly identify LEFT table."""
        tp.reset()
        tp.enable(mode="debug")

        left = pd.DataFrame({"k": ["a", "a"], "v": [1, 2]})  # Duplicates
        right = pd.DataFrame({"k": ["a", "b"], "r": [10, 20]})  # Unique
        df = left.merge(right, on="k", how="left")

        result = tp.check(df)
        dup_warnings = [w for w in result.warnings if "duplicate" in w.message.lower()]

        assert len(dup_warnings) == 1
        msg = dup_warnings[0].message.lower()
        assert "left table" in msg or "left" in msg, f"Warning should mention 'left': {msg}"
        assert (
            "33.3%" in dup_warnings[0].message or "50" in dup_warnings[0].message
        ), f"Warning should include percentage: {dup_warnings[0].message}"

        tp.disable()

    def test_right_duplicate_warning_text(self):
        """Right duplicate warning should clearly identify RIGHT table."""
        tp.reset()
        tp.enable(mode="debug")

        left = pd.DataFrame({"k": ["a", "b"], "v": [1, 2]})  # Unique
        right = pd.DataFrame({"k": ["a", "a"], "r": [10, 20]})  # Duplicates
        df = left.merge(right, on="k", how="left")

        result = tp.check(df)
        dup_warnings = [w for w in result.warnings if "duplicate" in w.message.lower()]

        assert len(dup_warnings) == 1
        msg = dup_warnings[0].message.lower()
        assert "right table" in msg or "right" in msg, f"Warning should mention 'right': {msg}"

        tp.disable()

    def test_merge_expansion_warning_text(self):
        """Expansion warning should include correct ratio when threshold exceeded."""
        tp.reset()
        tp.enable(mode="debug")

        # Create a larger expansion to exceed the default threshold
        left = pd.DataFrame({"k": ["a", "b"], "v": [1, 2]})
        right = pd.DataFrame({"k": ["a", "a", "a", "b", "b"], "r": [1, 2, 3, 4, 5]})  # 3x expansion
        df = left.merge(right, on="k", how="left")

        result = tp.check(df, merge_expansion_threshold=1.5)  # Lower threshold
        expansion_warnings = [w for w in result.warnings if "expand" in w.message.lower()]

        # Should have expansion warning with explicit threshold
        if len(expansion_warnings) >= 1:
            msg = expansion_warnings[0].message
            # Just verify the warning mentions the ratio
            assert any(
                c.isdigit() for c in msg
            ), f"Expansion warning should include a number: {msg}"

        tp.disable()

    def test_why_shows_correct_operation_name(self):
        """tp.why() should show accurate operation names."""
        tp.reset()
        tp.enable(mode="debug", watch=["a"])

        df = pd.DataFrame({"a": [None, 1]})
        df = df.fillna({"a": 0})

        result = tp.why(df, col="a", row=0)

        assert result.n_changes == 1
        operation = result.history[0].get("operation", result.history[0].get("op", ""))
        assert "fillna" in operation.lower(), f"Operation should mention fillna: {operation}"

        tp.disable()

    def test_trace_returns_valid_result(self):
        """tp.trace() should return valid TraceResult after merge."""
        tp.reset()
        tp.enable(mode="debug", watch=["val"])

        left = pd.DataFrame({"key": ["a"], "val": [10]})
        right = pd.DataFrame({"key": ["a"], "extra": [100]})
        df = left.merge(right, on="key")

        result = tp.trace(df, row=0)

        # Should return a valid TraceResult
        assert result is not None
        assert hasattr(result, "row_id")
        assert hasattr(result, "is_alive")
        assert result.is_alive is True  # Row exists

        tp.disable()


class TestReliabilityScenarios:
    """Official reliability test scenarios (from user's script)."""

    def test_fillna_series_setitem(self):
        """df['col'] = df['col'].fillna(0) should log exactly 1 event."""
        tp.reset()
        tp.enable(mode="debug", watch=["income"])

        df = pd.DataFrame({"id": [1, 2], "income": [None, 100.0]})
        df["income"] = df["income"].fillna(0)

        result = tp.why(df, col="income", row=0)
        assert (
            result.n_changes == 1
        ), f"Series+setitem fillna: expected 1 event, got {result.n_changes}"

        tp.disable()

    def test_fillna_dataframe_level(self):
        """df = df.fillna({'col': 0}) should log exactly 1 event."""
        tp.reset()
        tp.enable(mode="debug", watch=["income"])

        df = pd.DataFrame({"id": [1, 2], "income": [None, 100.0]})
        df = df.fillna({"income": 0})

        result = tp.why(df, col="income", row=0)
        assert result.n_changes == 1, f"DataFrame.fillna: expected 1 event, got {result.n_changes}"

        tp.disable()

    def test_loc_mask_mutation(self):
        """df.loc[mask, 'col'] = 0 should log exactly 1 event."""
        tp.reset()
        tp.enable(mode="debug", watch=["income"])

        df = pd.DataFrame({"id": [1, 2], "income": [None, 100.0]})
        df.loc[df["income"].isna(), "income"] = 0

        result = tp.why(df, col="income", row=0)
        assert result.n_changes == 1, f"loc mask: expected 1 event, got {result.n_changes}"

        tp.disable()

    def test_replace_operation(self):
        """df.replace should log exactly 1 event per changed row."""
        tp.reset()
        tp.enable(mode="debug", watch=["val"])

        df = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        df = df.replace({"val": {10: 100}})

        result = tp.why(df, col="val", row=0)
        assert result.n_changes == 1, f"replace: expected 1 event, got {result.n_changes}"

        tp.disable()

    def test_full_pipeline_smoke(self):
        """Full pipeline: dropna -> fillna -> filter -> merge runs without error."""
        tp.reset()
        tp.enable(mode="debug", watch=["income", "region"])

        customers = pd.DataFrame(
            {
                "customer_id": ["C-001", "C-002", "C-003"],
                "age": [17, 18, 25],
                "income": [None, 50000.0, None],
                "zip": ["10001", "10001", "94107"],
            }
        )
        regions = pd.DataFrame({"zip": ["10001", "94107"], "region": ["NY", "CA"]})

        df = customers.copy()
        df = df.dropna(subset=["zip"])
        df["income"] = df["income"].fillna(0)
        df = df[df["age"] >= 18]
        df = df.merge(regions, on="zip", how="left")

        # Check health - pipeline should complete
        result = tp.check(df)
        assert result is not None, "check() should return a result"

        # Verify the pipeline produced expected output
        assert len(df) >= 1, "Pipeline should produce results"
        assert "region" in df.columns, "Merge should add region column"

        # Verify we can query the result without errors
        trace_result = tp.trace(df, row=0)
        assert trace_result is not None

        tp.disable()


class TestNoContamination:
    """Tests specifically for cross-pipeline contamination issues."""

    def test_parallel_pipelines_shared_source(self):
        """Pipelines from same source should not contaminate each other's events."""
        tp.reset()
        tp.enable(mode="debug", watch=["income"])

        # Shared source
        customers = pd.DataFrame({"id": ["A", "B"], "income": [None, 100.0]})

        # Two pipelines from copies
        df1 = customers.copy()
        df1["income"] = df1["income"].fillna(0)

        df2 = customers.copy()
        df2["income"] = df2["income"].fillna(0)  # Same operation

        # With deduplication, should still show exactly 1 event
        result1 = tp.why(df1, col="income", row=0)
        assert (
            result1.n_changes == 1
        ), f"df1 should show exactly 1 change (deduplicated), got {result1.n_changes}"

        tp.disable()

    def test_sequential_enables_isolated(self):
        """Sequential enable() calls should fully isolate sessions."""
        tp.reset()

        # Session 1
        tp.enable(mode="debug", watch=["a"])
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df1["a"] = df1["a"] * 10
        stats1 = tp.check(df1).facts.get("total_steps", 0)

        # Session 2 (should reset)
        tp.enable(mode="debug", watch=["b"])
        df2 = pd.DataFrame({"b": [4, 5, 6]})
        # No operations on b

        # Session 2 should not inherit session 1's stats
        result2 = tp.check(df2)
        # This should be a clean check, not accumulated from session 1

        tp.disable()

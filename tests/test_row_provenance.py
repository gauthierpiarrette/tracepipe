# tests/test_row_provenance.py
"""
Comprehensive tests for row provenance tracking in concat and drop_duplicates.

Tests the new ConcatMapping and DuplicateDropMapping features added in 0.4.0.
"""

import numpy as np
import pandas as pd

import tracepipe as tp
from tracepipe.context import get_context
from tracepipe.core import CompletenessLevel

# ============================================================================
# CONCAT PROVENANCE TESTS
# ============================================================================


class TestConcatAxis0Provenance:
    """Tests for pd.concat(axis=0) row provenance."""

    def test_concat_preserves_rids_basic(self, debug_tracepipe):
        """Basic axis=0 concat should preserve row IDs from both sources."""
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})

        # Get source RIDs before concat
        ctx = get_context()
        rids1 = ctx.row_manager.get_ids_array(df1)
        rids2 = ctx.row_manager.get_ids_array(df2)

        result = pd.concat([df1, df2], ignore_index=True)

        # Result should have RIDs from both sources, concatenated
        result_rids = ctx.row_manager.get_ids_array(result)
        assert result_rids is not None
        assert len(result_rids) == 4
        # RIDs should be preserved (same as source RIDs concatenated)
        expected_rids = np.concatenate([rids1, rids2])
        np.testing.assert_array_equal(result_rids, expected_rids)

    def test_concat_stores_mapping(self, debug_tracepipe):
        """Concat should store ConcatMapping for provenance queries."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})

        result = pd.concat([df1, df2])

        ctx = get_context()
        assert len(ctx.store.concat_mappings) == 1

        mapping = ctx.store.concat_mappings[0]
        assert len(mapping.out_rids) == 4
        assert len(mapping.source_indices) == 4
        # First two rows came from df1 (source_index=0)
        assert mapping.source_indices[0] == 0
        assert mapping.source_indices[1] == 0
        # Last two rows came from df2 (source_index=1)
        assert mapping.source_indices[2] == 1
        assert mapping.source_indices[3] == 1

    def test_concat_origin_lookup(self, debug_tracepipe):
        """get_concat_origin should return source info for each row."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})

        ctx = get_context()
        rids1 = ctx.row_manager.get_ids_array(df1)
        rids2 = ctx.row_manager.get_ids_array(df2)

        result = pd.concat([df1, df2])

        # Check origin for row from df1
        origin1 = ctx.store.get_concat_origin(rids1[0])
        assert origin1 is not None
        assert origin1["source_index"] == 0

        # Check origin for row from df2
        origin2 = ctx.store.get_concat_origin(rids2[0])
        assert origin2 is not None
        assert origin2["source_index"] == 1

    def test_concat_preserves_lineage_after_transform(self):
        """Concat should preserve lineage from pre-concat transforms."""
        tp.enable(mode="debug", watch=["a"])

        df1 = pd.DataFrame({"a": [1.0, None]})
        df1["a"] = df1["a"].fillna(0)  # Transform tracked

        df2 = pd.DataFrame({"a": [3, 4]})

        # Get RID for row 1 (the one that was filled) before concat
        ctx = get_context()
        df1_rids = ctx.row_manager.get_ids_array(df1).copy()
        filled_rid = df1_rids[1]  # Row 1 was None -> 0

        result = pd.concat([df1, df2], ignore_index=True)

        # The filled row's RID should be preserved in result
        result_rids = ctx.row_manager.get_ids_array(result)
        assert filled_rid in result_rids

        # Check history for the filled row using original RID
        history = ctx.store.get_row_history_with_lineage(filled_rid)

        # Should have setitem event (fillna is recorded as __setitem__)
        # df["a"] = df["a"].fillna(0) is recorded as DataFrame.__setitem__[a]
        setitem_events = [e for e in history if "setitem" in e["operation"].lower()]
        assert (
            len(setitem_events) >= 1
        ), f"Expected setitem in history, got: {[e['operation'] for e in history]}"

    def test_concat_ignore_index_still_preserves_rids(self, debug_tracepipe):
        """ignore_index=True changes pandas index but should preserve RIDs."""
        df1 = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
        df2 = pd.DataFrame({"a": [3, 4]}, index=["x", "y"])

        ctx = get_context()
        rids1 = ctx.row_manager.get_ids_array(df1).copy()
        rids2 = ctx.row_manager.get_ids_array(df2).copy()

        result = pd.concat([df1, df2], ignore_index=True)

        result_rids = ctx.row_manager.get_ids_array(result)
        expected = np.concatenate([rids1, rids2])
        np.testing.assert_array_equal(result_rids, expected)

    def test_concat_after_sort_preserves_rids(self, debug_tracepipe):
        """Concat after sort_values should preserve correct RIDs."""
        df1 = pd.DataFrame({"a": [3, 1, 2]})
        df1 = df1.sort_values("a")  # Reorders rows

        df2 = pd.DataFrame({"a": [4, 5]})

        ctx = get_context()
        rids1 = ctx.row_manager.get_ids_array(df1).copy()
        rids2 = ctx.row_manager.get_ids_array(df2).copy()

        result = pd.concat([df1, df2], ignore_index=True)

        result_rids = ctx.row_manager.get_ids_array(result)
        expected = np.concatenate([rids1, rids2])
        np.testing.assert_array_equal(result_rids, expected)

    def test_concat_with_empty_df(self, debug_tracepipe):
        """Concat with empty DataFrame should handle gracefully."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": []})
        df3 = pd.DataFrame({"a": [3]})

        result = pd.concat([df1, df2, df3], ignore_index=True)

        ctx = get_context()
        result_rids = ctx.row_manager.get_ids_array(result)
        assert len(result_rids) == 3  # 2 from df1 + 0 from df2 + 1 from df3

    def test_repeated_concat_chains_correctly(self, debug_tracepipe):
        """Multiple concats should chain provenance correctly."""
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [2]})
        df3 = pd.DataFrame({"a": [3]})

        # First concat
        temp = pd.concat([df1, df2])
        # Second concat
        result = pd.concat([temp, df3])

        ctx = get_context()
        assert len(ctx.store.concat_mappings) == 2

        # All three original rows should have preserved RIDs
        result_rids = ctx.row_manager.get_ids_array(result)
        assert len(result_rids) == 3

    def test_concat_step_has_full_completeness(self, debug_tracepipe):
        """Axis=0 concat step should be marked FULL."""
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [2]})

        pd.concat([df1, df2])

        ctx = get_context()
        concat_steps = [s for s in ctx.store.steps if s.operation == "pd.concat"]
        assert len(concat_steps) == 1
        assert concat_steps[0].completeness == CompletenessLevel.FULL


class TestConcatAxis1Provenance:
    """Tests for pd.concat(axis=1) row provenance."""

    def test_concat_axis1_same_rids_propagates(self, debug_tracepipe):
        """Axis=1 concat with same RIDs should propagate them."""
        df1 = pd.DataFrame({"a": [1, 2]})

        ctx = get_context()
        rids1 = ctx.row_manager.get_ids_array(df1).copy()

        # Create df2 with same structure (will have different RIDs initially)
        # But after axis=1 concat with same df, RIDs should align
        result = pd.concat([df1, df1.rename(columns={"a": "b"})], axis=1)

        # Should preserve original RIDs since both inputs have same RIDs
        result_rids = ctx.row_manager.get_ids_array(result)
        np.testing.assert_array_equal(result_rids, rids1)

    def test_concat_axis1_different_rids_partial(self, debug_tracepipe):
        """Axis=1 concat with different RIDs should be PARTIAL."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})  # Different RIDs

        pd.concat([df1, df2], axis=1)

        ctx = get_context()
        concat_steps = [s for s in ctx.store.steps if s.operation == "pd.concat"]
        assert len(concat_steps) == 1
        assert concat_steps[0].completeness == CompletenessLevel.PARTIAL


# ============================================================================
# DROP_DUPLICATES PROVENANCE TESTS
# ============================================================================


class TestDropDuplicatesProvenance:
    """Tests for drop_duplicates provenance tracking (debug mode only)."""

    def test_dedup_stores_mapping_debug_mode(self, debug_tracepipe):
        """drop_duplicates should store DuplicateDropMapping in debug mode."""
        df = pd.DataFrame({"a": [1, 1, 2, 2, 3], "b": [1, 2, 3, 4, 5]})

        df.drop_duplicates(subset=["a"], keep="first")

        ctx = get_context()
        assert len(ctx.store.duplicate_drop_mappings) == 1

    def test_dedup_no_mapping_ci_mode(self, enabled_tracepipe):
        """drop_duplicates should NOT store mapping in CI mode."""
        df = pd.DataFrame({"a": [1, 1, 2]})

        df.drop_duplicates(subset=["a"], keep="first")

        ctx = get_context()
        assert len(ctx.store.duplicate_drop_mappings) == 0

    def test_dedup_keep_first_maps_to_first(self, debug_tracepipe):
        """keep='first' should map dropped rows to first occurrence."""
        df = pd.DataFrame({"a": [1, 1, 1], "b": ["first", "second", "third"]})

        ctx = get_context()
        source_rids = ctx.row_manager.get_ids_array(df).copy()

        df.drop_duplicates(subset=["a"], keep="first")

        # Row 1 and 2 (duplicates) should map to row 0 (first)
        rep1 = ctx.store.get_duplicate_representative(source_rids[1])
        rep2 = ctx.store.get_duplicate_representative(source_rids[2])

        assert rep1 is not None
        assert rep1["kept_rid"] == source_rids[0]
        assert rep2 is not None
        assert rep2["kept_rid"] == source_rids[0]

    def test_dedup_keep_last_maps_to_last(self, debug_tracepipe):
        """keep='last' should map dropped rows to last occurrence."""
        df = pd.DataFrame({"a": [1, 1, 1], "b": ["first", "second", "third"]})

        ctx = get_context()
        source_rids = ctx.row_manager.get_ids_array(df).copy()

        df.drop_duplicates(subset=["a"], keep="last")

        # Row 0 and 1 (dropped) should map to row 2 (last)
        rep0 = ctx.store.get_duplicate_representative(source_rids[0])
        rep1 = ctx.store.get_duplicate_representative(source_rids[1])

        assert rep0 is not None
        assert rep0["kept_rid"] == source_rids[2]
        assert rep1 is not None
        assert rep1["kept_rid"] == source_rids[2]

    def test_dedup_keep_false_no_representative(self, debug_tracepipe):
        """keep=False should have no representative (kept_rid=None)."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3]})

        ctx = get_context()
        source_rids = ctx.row_manager.get_ids_array(df).copy()

        df.drop_duplicates(subset=["a"], keep=False)

        # Both duplicate rows (0 and 1) should have no representative
        rep0 = ctx.store.get_duplicate_representative(source_rids[0])
        rep1 = ctx.store.get_duplicate_representative(source_rids[1])

        assert rep0 is not None
        assert rep0["kept_rid"] is None
        assert rep0["keep_strategy"] == "False"

        assert rep1 is not None
        assert rep1["kept_rid"] is None

    def test_dedup_with_nan_values(self, debug_tracepipe):
        """drop_duplicates with NaN should handle NaN equality correctly."""
        df = pd.DataFrame({"a": [1, np.nan, np.nan, 2], "b": [1, 2, 3, 4]})

        ctx = get_context()
        source_rids = ctx.row_manager.get_ids_array(df).copy()

        result = df.drop_duplicates(subset=["a"], keep="first")

        # pandas treats NaN == NaN for duplicates
        # So row 2 (second NaN) should be dropped, mapping to row 1 (first NaN)
        rep2 = ctx.store.get_duplicate_representative(source_rids[2])
        assert rep2 is not None
        assert rep2["kept_rid"] == source_rids[1]

    def test_dedup_with_string_index(self, debug_tracepipe):
        """drop_duplicates should work correctly with non-integer index."""
        df = pd.DataFrame({"a": [1, 1, 2]}, index=["x", "y", "z"])

        ctx = get_context()
        source_rids = ctx.row_manager.get_ids_array(df).copy()

        df.drop_duplicates(subset=["a"], keep="first")

        # Row 'y' (second duplicate) should map to row 'x' (first)
        rep = ctx.store.get_duplicate_representative(source_rids[1])
        assert rep is not None
        assert rep["kept_rid"] == source_rids[0]

    def test_dedup_subset_columns_recorded(self, debug_tracepipe):
        """Subset columns should be recorded in the mapping."""
        df = pd.DataFrame({"a": [1, 1], "b": [2, 2], "c": [3, 4]})

        df.drop_duplicates(subset=["a", "b"], keep="first")

        ctx = get_context()
        mapping = ctx.store.duplicate_drop_mappings[0]
        assert mapping.subset_columns == ("a", "b")

    def test_dedup_no_duplicates_no_mapping(self, debug_tracepipe):
        """If no duplicates exist, no mapping should be stored."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        df.drop_duplicates(subset=["a"])

        ctx = get_context()
        # No mapping stored because no rows were dropped
        assert len(ctx.store.duplicate_drop_mappings) == 0


# ============================================================================
# INTEGRATION TESTS: CONCAT + OTHER OPERATIONS
# ============================================================================


class TestConcatIntegration:
    """Integration tests combining concat with other operations."""

    def test_concat_then_merge_preserves_lineage(self):
        """Concat then merge should preserve full lineage."""
        tp.enable(mode="debug", watch=["value"])

        # Create and transform df1
        df1 = pd.DataFrame({"key": [1], "value": [None]})
        df1["value"] = df1["value"].fillna(0)

        # Create df2
        df2 = pd.DataFrame({"key": [2], "value": [20]})

        # Concat
        combined = pd.concat([df1, df2], ignore_index=True)

        # Merge with another df
        other = pd.DataFrame({"key": [1, 2], "extra": ["a", "b"]})
        result = combined.merge(other, on="key")

        # The row from df1 should still have fillna history
        ctx = get_context()

        # Use tp.trace to verify history is accessible
        trace_result = tp.trace(result, row=0)
        # trace_result.events is a list
        assert len(trace_result.events) >= 1

    def test_filter_then_concat(self, debug_tracepipe):
        """Filter then concat should preserve correct lineage."""
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df1 = df1[df1["a"] > 1]  # Keep rows with a > 1

        df2 = pd.DataFrame({"a": [4, 5]})

        result = pd.concat([df1, df2])

        ctx = get_context()
        result_rids = ctx.row_manager.get_ids_array(result)
        assert len(result_rids) == 4  # 2 from df1 + 2 from df2

    def test_dedup_then_fillna_lineage(self):
        """drop_duplicates then fillna should track both operations."""
        tp.enable(mode="debug", watch=["b"])

        df = pd.DataFrame({"a": [1, 1, 2], "b": [None, 2.0, None]})

        df = df.drop_duplicates(subset=["a"], keep="first")
        df["b"] = df["b"].fillna(0)

        # Check that operations are tracked via context
        ctx = get_context()

        # Should have drop_duplicates step
        dedup_steps = [s for s in ctx.store.steps if "drop_duplicates" in s.operation]
        assert len(dedup_steps) >= 1

        # Should have setitem step (fillna via assignment is recorded as setitem)
        setitem_steps = [s for s in ctx.store.steps if "setitem" in s.operation.lower()]
        assert len(setitem_steps) >= 1


# ============================================================================
# PARITY TESTS
# ============================================================================


class TestDedupProvParity:
    """Parity tests to verify our mapping matches pandas behavior."""

    def test_dedup_mapping_matches_pandas_keep_first(self, debug_tracepipe):
        """Verify our hash-based grouping matches pandas duplicated (keep='first')."""
        df = pd.DataFrame(
            {
                "a": [1, 1, 2, np.nan, np.nan, 3],
                "b": ["x", "x", "y", "z", "z", "w"],
            }
        )
        subset = ["a", "b"]

        ctx = get_context()
        source_rids = ctx.row_manager.get_ids_array(df).copy()

        result = df.drop_duplicates(subset=subset, keep="first")

        # Get pandas ground truth
        kept_mask = ~df.duplicated(subset=subset, keep="first")
        dropped_pos = np.where(~kept_mask.values)[0]

        # Verify each dropped row maps to a row with the same key (NaN-safe)
        for pos in dropped_pos:
            rep = ctx.store.get_duplicate_representative(source_rids[pos])
            assert rep is not None, f"No mapping for dropped row at position {pos}"

            if rep["kept_rid"] is not None:
                # Find the position of the kept row
                kept_pos = np.where(source_rids == rep["kept_rid"])[0][0]

                # Verify keys match (NaN-safe comparison using .equals())
                dropped_key = df.loc[df.index[pos], subset]
                kept_key = df.loc[df.index[kept_pos], subset]
                assert dropped_key.equals(kept_key), (
                    f"Row {pos} mapped to wrong representative: "
                    f"{dropped_key.values} != {kept_key.values}"
                )


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_concat_single_df(self, debug_tracepipe):
        """Concat with single DataFrame should work."""
        df = pd.DataFrame({"a": [1, 2]})

        result = pd.concat([df])

        ctx = get_context()
        result_rids = ctx.row_manager.get_ids_array(result)
        assert result_rids is not None
        assert len(result_rids) == 2

    def test_empty_result_concat(self, debug_tracepipe):
        """Concat resulting in empty DataFrame should handle gracefully."""
        df1 = pd.DataFrame({"a": []})
        df2 = pd.DataFrame({"a": []})

        result = pd.concat([df1, df2])

        ctx = get_context()
        result_rids = ctx.row_manager.get_ids_array(result)
        assert result_rids is not None
        assert len(result_rids) == 0

    def test_dedup_all_unique(self, debug_tracepipe):
        """drop_duplicates with all unique rows should not create mapping."""
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        result = df.drop_duplicates()

        ctx = get_context()
        # No mapping because no duplicates
        assert len(ctx.store.duplicate_drop_mappings) == 0
        # Result should have same length
        assert len(result) == 5

    def test_dedup_all_same(self, debug_tracepipe):
        """drop_duplicates where all rows are same should keep one."""
        df = pd.DataFrame({"a": [1, 1, 1, 1, 1]})

        ctx = get_context()
        source_rids = ctx.row_manager.get_ids_array(df).copy()

        result = df.drop_duplicates(keep="first")

        assert len(result) == 1

        # All but first should map to first
        for i in range(1, 5):
            rep = ctx.store.get_duplicate_representative(source_rids[i])
            assert rep is not None
            assert rep["kept_rid"] == source_rids[0]

    def test_binary_search_nonexistent_rid(self, debug_tracepipe):
        """Searching for non-existent RID should return None."""
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [2]})

        pd.concat([df1, df2])

        ctx = get_context()
        # Search for RID that doesn't exist
        result = ctx.store.get_concat_origin(99999)
        assert result is None

        result = ctx.store.get_duplicate_representative(99999)
        assert result is None


# ============================================================================
# TRACE RESULT UX TESTS (v0.4+ clean API)
# ============================================================================


class TestTraceResultOriginProperty:
    """Tests for TraceResult.origin unified provenance property."""

    def test_trace_concat_origin(self, debug_tracepipe):
        """tp.trace() should include concat origin via .origin property."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})

        result = pd.concat([df1, df2], ignore_index=True)

        # Trace row that came from df2 (row 2 in result)
        trace = tp.trace(result, row=2)

        # Should have origin info
        assert trace.origin is not None
        assert trace.origin["type"] == "concat"
        assert trace.origin["source_df"] == 1  # Second DataFrame (0-indexed)
        assert "step_id" in trace.origin

    def test_trace_merge_origin(self, debug_tracepipe):
        """tp.trace() should include merge origin via .origin property."""
        df1 = pd.DataFrame({"key": [1, 2], "val": ["a", "b"]})
        df2 = pd.DataFrame({"key": [1, 2], "other": ["x", "y"]})

        result = df1.merge(df2, on="key")

        # Use row=0 to trace the first row in the result DataFrame
        trace = tp.trace(result, row=0)

        # Should have merge origin
        assert trace.origin is not None
        assert trace.origin["type"] == "merge"
        assert "left_parent" in trace.origin
        assert "right_parent" in trace.origin

    def test_trace_original_row_no_origin(self, debug_tracepipe):
        """Original rows (not from concat/merge) should have origin=None."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        df = df.dropna()  # Simple filter, no concat/merge

        trace = tp.trace(df, row=0)

        assert trace.origin is None

    def test_trace_origin_in_to_dict(self, debug_tracepipe):
        """TraceResult.to_dict() should include origin."""
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [2]})

        result = pd.concat([df1, df2])

        trace = tp.trace(result, row=1)
        d = trace.to_dict()

        assert "origin" in d
        assert d["origin"]["type"] == "concat"

    def test_trace_origin_in_text_output(self, debug_tracepipe):
        """TraceResult text output should mention origin."""
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [2]})

        result = pd.concat([df1, df2])

        trace = tp.trace(result, row=1)
        text = str(trace)

        assert "Origin: concat from DataFrame #1" in text


class TestTraceResultRepresentativeProperty:
    """Tests for TraceResult.representative dedup provenance property."""

    def test_trace_dedup_representative(self, debug_tracepipe):
        """tp.trace() should include representative for dedup-dropped rows."""
        df = pd.DataFrame({"key": ["A", "A", "B"], "val": [1, 2, 3]})

        ctx = get_context()
        source_rids = ctx.row_manager.get_ids_array(df).copy()

        df.drop_duplicates(subset=["key"], keep="first")

        # Trace the dropped row (row 1 in original, val=2)
        trace = tp.trace(df, row=source_rids[1])

        # Should have representative info
        assert trace.representative is not None
        assert trace.representative["kept_rid"] == source_rids[0]
        # Subset may be tuple or list depending on how it was stored
        assert "key" in trace.representative["subset"]
        assert trace.representative["keep"] == "first"

    def test_trace_dedup_keep_false_no_representative(self, debug_tracepipe):
        """keep=False should have representative with kept_rid=None."""
        df = pd.DataFrame({"key": ["A", "A", "B"], "val": [1, 2, 3]})

        ctx = get_context()
        source_rids = ctx.row_manager.get_ids_array(df).copy()

        df.drop_duplicates(subset=["key"], keep=False)

        # Trace a dropped duplicate
        trace = tp.trace(df, row=source_rids[0])

        # Should have representative but kept_rid is None
        assert trace.representative is not None
        assert trace.representative["kept_rid"] is None

    def test_trace_non_dedup_dropped_no_representative(self, debug_tracepipe):
        """Rows dropped by non-dedup operations should have representative=None."""
        df = pd.DataFrame({"a": [1, 2, None, 4]})

        ctx = get_context()
        source_rids = ctx.row_manager.get_ids_array(df).copy()

        df.dropna()

        # Trace the row dropped by dropna (row 2)
        trace = tp.trace(df, row=source_rids[2])

        # Should not have representative (not a dedup drop)
        assert trace.representative is None

    def test_trace_representative_in_text_output(self, debug_tracepipe):
        """TraceResult text should show representative for dedup drops."""
        df = pd.DataFrame({"key": ["A", "A"], "val": [1, 2]})

        ctx = get_context()
        source_rids = ctx.row_manager.get_ids_array(df).copy()

        df.drop_duplicates(subset=["key"], keep="first")

        trace = tp.trace(df, row=source_rids[1])
        text = str(trace)

        assert "Replaced by:" in text
        assert "[keep=first]" in text

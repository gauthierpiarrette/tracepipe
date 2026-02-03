#!/usr/bin/env python3
"""
Red Team Test Suite for TracePipe v0.4.0

These tests are designed to exploit weaknesses in monkey-patching strategies.

Run: python examples/red_team_test.py
"""

import gc
import sys
import traceback

import numpy as np
import pandas as pd

import tracepipe as tp

# Results tracking
results = {"passed": [], "failed": [], "partial": []}


def test(name: str, category: str):
    """Decorator for test functions."""

    def decorator(fn):
        def wrapper():
            print(f"\n{'=' * 60}")
            print(f"TEST: {name}")
            print(f"Category: {category}")
            print("=" * 60)

            # Reset TracePipe for each test
            tp.reset()
            tp.enable(mode="debug")

            try:
                result, notes = fn()
                if result == "PASS":
                    print(f"✅ PASS: {notes}")
                    results["passed"].append((name, notes))
                elif result == "PARTIAL":
                    print(f"⚠️  PARTIAL: {notes}")
                    results["partial"].append((name, notes))
                else:
                    print(f"❌ FAIL: {notes}")
                    results["failed"].append((name, notes))
            except Exception as e:
                print(f"❌ FAIL (exception): {e}")
                traceback.print_exc()
                results["failed"].append((name, str(e)))
            finally:
                tp.disable()

        return wrapper

    return decorator


# ============================================================================
# CATEGORY 1: BACKDOOR ATTACKS
# ============================================================================


@test("NumPy Bypass", "Backdoor Attacks")
def test_numpy_bypass():
    """Modify raw NumPy array - bypasses all Pandas methods."""
    tp.enable(mode="debug", watch=["A"])
    df = pd.DataFrame({"A": [1, 2, 3]})

    arr = df["A"].values  # Get raw numpy array

    # Pandas 2.x with CoW makes arrays read-only - the backdoor is closed!
    try:
        arr[0] = 99  # Modify raw memory
    except ValueError as e:
        if "read-only" in str(e):
            return "PASS", "Pandas 2.x CoW prevents NumPy bypass (array is read-only)"
        raise

    # Check if TracePipe detected the change (older Pandas)
    dbg = tp.debug.inspect()
    stats = dbg.stats()

    if df["A"][0] == 99 and stats.get("total_diffs", 0) == 0:
        return "FAIL", "NumPy bypass invisible (expected - documented limitation)"
    elif stats.get("total_diffs", 0) > 0:
        return "PASS", "Somehow detected NumPy modification!"
    return "PARTIAL", f"Unexpected state: df[A][0]={df['A'][0]}, diffs={stats}"


@test("at/iat Scalar Accessors", "Backdoor Attacks")
def test_at_iat():
    """Test fast scalar accessors .at and .iat."""
    tp.enable(mode="debug", watch=["A"])
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    df.at[0, "A"] = 10  # Fast scalar access
    df.iat[1, 0] = 20  # Fast integer scalar access

    dbg = tp.debug.inspect()
    stats = dbg.stats()
    steps = dbg.steps

    if stats.get("total_diffs", 0) >= 2:
        return "PASS", f"Caught both at/iat: {stats.get('total_diffs')} diffs"
    elif stats.get("total_diffs", 0) == 1:
        return "PARTIAL", "Caught one accessor but not both"
    return "FAIL", f"Missed scalar accessors. Steps={len(steps)}, diffs={stats}"


# ============================================================================
# CATEGORY 2: GHOST REFERENCES (View vs Copy)
# ============================================================================


@test("Chained Assignment", "Ghost References")
def test_chained_assignment():
    """The famous SettingWithCopyWarning scenario."""
    tp.enable(mode="debug", watch=["B"])
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    original_b = df["B"].copy()

    # This may or may not modify df depending on Pandas version
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df[df["A"] > 1]["B"] = 99

    # Check if df actually changed
    df_changed = not (df["B"] == original_b).all()

    dbg = tp.debug.inspect()
    stats = dbg.stats()

    if not df_changed:
        # Pandas didn't modify df (created temp copy) - correct to ignore
        return "PASS", "Correctly ignored chained assignment (no-op in this Pandas version)"
    elif df_changed and stats.get("total_diffs", 0) > 0:
        return "PASS", "Detected chained assignment modification"
    return "FAIL", f"df changed but TracePipe missed it. diffs={stats}"


@test("View Mutation", "Ghost References")
def test_view_mutation():
    """Modifying a view also modifies the original."""
    tp.enable(mode="debug", watch=["A"])
    df = pd.DataFrame({"A": [1, 2, 3]})

    sub = df.iloc[0:2]  # Often a VIEW
    sub.loc[0, "A"] = 55  # Modifying 'sub' may modify 'df'

    df_changed = df.loc[0, "A"] == 55

    dbg = tp.debug.inspect()
    stats = dbg.stats()

    if df_changed and stats.get("total_diffs", 0) > 0:
        return "PASS", "Detected view mutation propagating to original"
    elif not df_changed:
        return "PARTIAL", "Pandas made a copy (CoW?), view didn't propagate"
    return "FAIL", "df changed via view but TracePipe missed it"


# ============================================================================
# CATEGORY 3: IN-PLACE MUTATORS
# ============================================================================


@test("inplace Operations", "In-Place Mutators")
def test_inplace():
    """Methods with inplace=True return None."""
    tp.enable(mode="debug", watch=["A"])
    df = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [4, 5, 6]})
    _original_len = len(df)

    df.fillna(0, inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.sort_values("A", inplace=True)

    dbg = tp.debug.inspect()
    stats = dbg.stats()

    # Check if row identity still works
    try:
        trace = tp.trace(df, row=0)
        has_identity = trace.supported
    except Exception:
        has_identity = False

    if stats.get("total_steps", 0) >= 2 and has_identity:
        return "PASS", f"Tracked inplace ops. Steps={stats.get('total_steps')}"
    elif stats.get("total_steps", 0) > 0:
        return "PARTIAL", f"Some tracking, identity may be broken. Steps={stats.get('total_steps')}"
    return "FAIL", "inplace ops broke tracking entirely"


@test("drop inplace", "In-Place Mutators")
def test_drop_inplace():
    """drop(inplace=True) is particularly tricky."""
    tp.enable(mode="debug")
    df = pd.DataFrame({"A": [1, 2, 3]})

    df.drop(0, inplace=True)

    dbg = tp.debug.inspect()
    dropped = dbg.dropped_rows()

    if len(dropped) >= 1:
        return "PASS", f"Caught inplace drop. Dropped rows: {dropped}"
    return "FAIL", "inplace drop not tracked"


# ============================================================================
# CATEGORY 4: DATA TYPE STRESS TEST
# ============================================================================


@test("Mutable Object in Cell", "Data Type Stress")
def test_mutable_cell():
    """Mutable objects (lists/dicts) in cells - invisible modification."""
    tp.enable(mode="debug", watch=["A"])
    df = pd.DataFrame({"A": [[1], [2]]})

    val = df.loc[0, "A"]
    val.append(99)  # Modify list in place

    dbg = tp.debug.inspect()
    stats = dbg.stats()

    # This is a known limitation
    if stats.get("total_diffs", 0) == 0:
        return "FAIL", "Mutable cell modification invisible (expected - documented limitation)"
    return "PASS", "Somehow detected mutable cell modification!"


@test("Categorical Operations", "Data Type Stress")
def test_categorical():
    """Categorical data uses integer codes internally."""
    tp.enable(mode="debug", watch=["cat"])
    df = pd.DataFrame({"val": ["a", "b", "a"]})

    df["cat"] = df["val"].astype("category")
    df["cat"] = df["cat"].cat.rename_categories({"a": "A", "b": "B"})

    dbg = tp.debug.inspect()
    stats = dbg.stats()

    if stats.get("total_steps", 0) >= 1:
        return "PASS", f"Tracked categorical ops. Steps={stats.get('total_steps')}"
    return "PARTIAL", "Categorical operations may not be fully tracked"


# ============================================================================
# CATEGORY 5: STRUCTURAL & LOGIC BREAKS
# ============================================================================


@test("Index Alignment", "Structural Breaks")
def test_index_alignment():
    """Pandas aligns by index labels, not position."""
    tp.enable(mode="debug", watch=["A"])
    df1 = pd.DataFrame({"A": [1]}, index=[10])
    df2 = pd.DataFrame({"A": [2]}, index=[10])

    res = df1 + df2

    # Result should be 3 at index 10
    if res.loc[10, "A"] == 3:
        dbg = tp.debug.inspect()
        stats = dbg.stats()
        return (
            "PARTIAL",
            f"Index alignment works, lineage tracking: steps={stats.get('total_steps', 0)}",
        )
    return "FAIL", "Index alignment broke something"


@test("MultiIndex Operations", "Structural Breaks")
def test_multiindex():
    """MultiIndex uses tuple-based lookups."""
    tp.enable(mode="debug", watch=["val"])
    df = pd.DataFrame(
        {"val": [1, 2, 3, 4]},
        index=pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2)]),
    )

    df.loc[("a", 1), "val"] = 99

    dbg = tp.debug.inspect()
    stats = dbg.stats()

    if stats.get("total_diffs", 0) >= 1:
        return "PASS", f"MultiIndex assignment tracked. Diffs={stats.get('total_diffs')}"
    return "FAIL", "MultiIndex assignment not tracked"


# ============================================================================
# CATEGORY 6: PERFORMANCE & RECURSION
# ============================================================================


@test("Apply with Lambda", "Performance & Recursion")
def test_apply_lambda():
    """apply() with lambda - tests recursion safety."""
    tp.enable(mode="debug", watch=["A"])
    df = pd.DataFrame({"A": [1, 2, 3]})

    _result = df.apply(lambda x: x + 1)

    # Should not crash or infinite loop
    dbg = tp.debug.inspect()
    stats = dbg.stats()

    if stats.get("total_steps", 0) >= 1:
        return "PASS", f"apply() tracked safely. Steps={stats.get('total_steps')}"
    return "PARTIAL", "apply() completed but may not be fully tracked"


@test("Memory Stability (1000 iterations)", "Performance & Recursion")
def test_memory_stability():
    """Check for memory leaks with repeated operations."""
    import tracemalloc

    tp.enable(mode="debug", watch=["A"])
    df = pd.DataFrame({"A": [1]})

    tracemalloc.start()
    initial = tracemalloc.get_traced_memory()[0]

    for i in range(1000):
        df.loc[0, "A"] = i

    gc.collect()
    final = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    growth_mb = (final - initial) / (1024 * 1024)

    # Allow up to 10MB growth for 1000 ops (generous)
    if growth_mb < 10:
        return "PASS", f"Memory growth: {growth_mb:.2f}MB for 1000 ops"
    elif growth_mb < 50:
        return "PARTIAL", f"Memory growth high: {growth_mb:.2f}MB"
    return "FAIL", f"Memory leak detected: {growth_mb:.2f}MB"


@test("Large DataFrame (100k rows)", "Performance & Recursion")
def test_large_dataframe():
    """Performance with large DataFrames."""
    import time

    tp.enable(mode="debug", watch=["A"])
    df = pd.DataFrame({"A": range(100_000)})

    start = time.time()
    df = df[df["A"] > 50_000]  # Filter half
    elapsed = time.time() - start

    dbg = tp.debug.inspect()
    dropped = dbg.dropped_rows()

    if len(dropped) >= 50_000 and elapsed < 5.0:
        return "PASS", f"Handled 100k rows, dropped {len(dropped)}, time={elapsed:.2f}s"
    elif elapsed >= 5.0:
        return "PARTIAL", f"Slow: {elapsed:.2f}s for 100k rows"
    return "FAIL", f"Large DataFrame handling broken. Dropped={len(dropped)}"


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("\n" + "=" * 60)
    print("🔴 RED TEAM TEST SUITE - TracePipe v0.4.0")
    print("=" * 60)
    print("Testing edge cases that break monkey-patching strategies...")

    # Run all tests
    tests = [
        test_numpy_bypass,
        test_at_iat,
        test_chained_assignment,
        test_view_mutation,
        test_inplace,
        test_drop_inplace,
        test_mutable_cell,
        test_categorical,
        test_index_alignment,
        test_multiindex,
        test_apply_lambda,
        test_memory_stability,
        test_large_dataframe,
    ]

    for t in tests:
        t()

    # Summary
    print("\n" + "=" * 60)
    print("📊 RED TEAM RESULTS")
    print("=" * 60)

    total = len(results["passed"]) + len(results["partial"]) + len(results["failed"])

    print(f"\n✅ PASSED:  {len(results['passed'])}/{total}")
    for name, notes in results["passed"]:
        print(f"   • {name}")

    print(f"\n⚠️  PARTIAL: {len(results['partial'])}/{total}")
    for name, notes in results["partial"]:
        print(f"   • {name}: {notes}")

    print(f"\n❌ FAILED:  {len(results['failed'])}/{total}")
    for name, notes in results["failed"]:
        print(f"   • {name}: {notes}")

    # Verdict
    failed_count = len(results["failed"])
    print("\n" + "=" * 60)
    if failed_count <= 3:
        print("🏆 VERDICT: EXCELLENT (0-3 failed) - Production ready!")
    elif failed_count <= 7:
        print("✓ VERDICT: GOOD (4-7 failed) - Document the limitations")
    else:
        print("⚠️  VERDICT: NEEDS WORK (8+ failed) - Architecture review needed")
    print("=" * 60)

    return failed_count


if __name__ == "__main__":
    failed = main()
    sys.exit(0 if failed <= 7 else 1)

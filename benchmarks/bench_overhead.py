#!/usr/bin/env python3
"""
Overhead Benchmark - Measures TracePipe overhead on common operations.

Compares execution time with and without TracePipe enabled.

Usage:
    python benchmarks/bench_overhead.py
    python benchmarks/bench_overhead.py --iterations 100
"""

import argparse
import gc
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import tracepipe as tp


@dataclass
class BenchResult:
    name: str
    baseline_ms: float
    tracepipe_ms: float

    @property
    def overhead(self) -> float:
        return self.tracepipe_ms / self.baseline_ms if self.baseline_ms > 0 else float("inf")

    @property
    def overhead_pct(self) -> float:
        return (self.overhead - 1) * 100

    @property
    def delta_ms(self) -> float:
        return self.tracepipe_ms - self.baseline_ms

    @property
    def is_acceptable(self) -> bool:
        # Accept if either:
        # 1. Overhead < 2x, OR
        # 2. Absolute delta < 1ms (micro-operations)
        return self.overhead < 2.0 or self.delta_ms < 1.0

    def __str__(self) -> str:
        status = "✅" if self.is_acceptable else "⚠️" if self.overhead < 5.0 else "❌"
        return (
            f"{status} {self.name:30s} "
            f"baseline={self.baseline_ms:8.2f}ms  "
            f"tp={self.tracepipe_ms:8.2f}ms  "
            f"overhead={self.overhead:.2f}x ({self.overhead_pct:+.0f}%)"
        )


def timeit(func, iterations: int = 10) -> float:
    """Time a function, return average ms."""
    gc.collect()
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)
    return np.median(times)


def make_df(n_rows: int = 10000) -> pd.DataFrame:
    """Create a test DataFrame."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "a": np.random.randn(n_rows),
            "b": np.random.randn(n_rows),
            "c": np.random.choice(["X", "Y", "Z"], n_rows),
            "d": np.random.randint(0, 100, n_rows),
            "e": np.where(np.random.random(n_rows) < 0.1, np.nan, np.random.randn(n_rows)),
        }
    )


def bench_dropna(df: pd.DataFrame, iterations: int) -> BenchResult:
    """Benchmark dropna()."""

    def run():
        df.dropna()

    baseline = timeit(run, iterations)

    tp.enable(mode="debug", watch=["e"])
    tp_time = timeit(run, iterations)
    tp.disable()

    return BenchResult("dropna", baseline, tp_time)


def bench_query(df: pd.DataFrame, iterations: int) -> BenchResult:
    """Benchmark query()."""

    def run():
        df.query("a > 0 and b < 0.5")

    baseline = timeit(run, iterations)

    tp.enable(mode="debug")
    tp_time = timeit(run, iterations)
    tp.disable()

    return BenchResult("query", baseline, tp_time)


def bench_boolean_mask(df: pd.DataFrame, iterations: int) -> BenchResult:
    """Benchmark boolean mask filtering."""

    def run():
        df[df["a"] > 0]

    baseline = timeit(run, iterations)

    tp.enable(mode="debug")
    tp_time = timeit(run, iterations)
    tp.disable()

    return BenchResult("df[mask]", baseline, tp_time)


def bench_fillna(df: pd.DataFrame, iterations: int) -> BenchResult:
    """Benchmark fillna()."""

    def run():
        df.fillna(0)

    baseline = timeit(run, iterations)

    tp.enable(mode="debug", watch=["e"])
    tp_time = timeit(run, iterations)
    tp.disable()

    return BenchResult("fillna", baseline, tp_time)


def bench_replace(df: pd.DataFrame, iterations: int) -> BenchResult:
    """Benchmark replace()."""

    def run():
        df.replace({"c": {"X": "XX"}})

    baseline = timeit(run, iterations)

    tp.enable(mode="debug", watch=["c"])
    tp_time = timeit(run, iterations)
    tp.disable()

    return BenchResult("replace", baseline, tp_time)


def bench_groupby_sum(df: pd.DataFrame, iterations: int) -> BenchResult:
    """Benchmark groupby().sum()."""

    def run():
        df.groupby("c")["a"].sum()

    baseline = timeit(run, iterations)

    tp.enable(mode="debug")
    tp_time = timeit(run, iterations)
    tp.disable()

    return BenchResult("groupby.sum", baseline, tp_time)


def bench_merge(df: pd.DataFrame, iterations: int) -> BenchResult:
    """Benchmark merge()."""
    df2 = pd.DataFrame({"c": ["X", "Y", "Z"], "label": ["Label X", "Label Y", "Label Z"]})

    def run():
        df.merge(df2, on="c")

    baseline = timeit(run, iterations)

    tp.enable(mode="debug")
    tp_time = timeit(run, iterations)
    tp.disable()

    return BenchResult("merge", baseline, tp_time)


def bench_loc_filter(df: pd.DataFrame, iterations: int) -> BenchResult:
    """Benchmark loc[] filter."""

    def run():
        df.loc[df["a"] > 0]

    baseline = timeit(run, iterations)

    tp.enable(mode="debug")
    tp_time = timeit(run, iterations)
    tp.disable()

    return BenchResult("loc[mask]", baseline, tp_time)


def bench_loc_setitem(df: pd.DataFrame, iterations: int) -> BenchResult:
    """Benchmark loc[] setitem."""
    df_copy = df.copy()

    def run():
        df_copy.loc[0:10, "a"] = 999

    baseline = timeit(run, iterations)

    tp.enable(mode="debug", watch=["a"])
    tp_time = timeit(run, iterations)
    tp.disable()

    return BenchResult("loc[]=", baseline, tp_time)


def bench_at_setitem(df: pd.DataFrame, iterations: int) -> BenchResult:
    """Benchmark at[] setitem - single scalar assignment."""
    df_copy = df.copy()

    def run():
        # Single scalar assignment (realistic use case)
        df_copy.at[0, "a"] = 999

    baseline = timeit(run, iterations * 10)  # More iterations for stable timing

    tp.enable(mode="debug", watch=["a"])
    tp_time = timeit(run, iterations * 10)
    tp.disable()

    return BenchResult("at[]=", baseline, tp_time)


def bench_sort_values(df: pd.DataFrame, iterations: int) -> BenchResult:
    """Benchmark sort_values()."""

    def run():
        df.sort_values("a")

    baseline = timeit(run, iterations)

    tp.enable(mode="debug")
    tp_time = timeit(run, iterations)
    tp.disable()

    return BenchResult("sort_values", baseline, tp_time)


def bench_drop_inplace(df: pd.DataFrame, iterations: int) -> BenchResult:
    """Benchmark drop(inplace=True)."""

    def run():
        df_copy = df.copy()
        df_copy.drop(index=df_copy.index[:10], inplace=True)

    baseline = timeit(run, iterations)

    tp.enable(mode="debug")
    tp_time = timeit(run, iterations)
    tp.disable()

    return BenchResult("drop(inplace)", baseline, tp_time)


def run_benchmarks(n_rows: int = 10000, iterations: int = 10) -> list[BenchResult]:
    """Run all overhead benchmarks."""
    df = make_df(n_rows)

    benchmarks = [
        bench_dropna,
        bench_query,
        bench_boolean_mask,
        bench_fillna,
        bench_replace,
        bench_groupby_sum,
        bench_merge,
        bench_loc_filter,
        bench_loc_setitem,
        bench_at_setitem,
        bench_sort_values,
        bench_drop_inplace,
    ]

    results = []
    for bench in benchmarks:
        tp.reset()
        result = bench(df, iterations)
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="TracePipe overhead benchmarks")
    parser.add_argument("--rows", type=int, default=10000, help="Number of rows")
    parser.add_argument("--iterations", type=int, default=10, help="Iterations per benchmark")
    parser.add_argument("--ci", action="store_true", help="CI mode (smaller dataset)")
    args = parser.parse_args()

    if args.ci:
        args.rows = 1000
        args.iterations = 5

    print("=" * 70)
    print(f"TracePipe Overhead Benchmark (n={args.rows:,}, iterations={args.iterations})")
    print("=" * 70)

    results = run_benchmarks(args.rows, args.iterations)

    print("\nResults:")
    print("-" * 70)
    for r in results:
        print(r)

    # Summary
    overheads = [r.overhead for r in results]
    avg_overhead = np.mean(overheads)
    max_overhead = max(overheads)
    worst = max(results, key=lambda r: r.overhead)

    print("-" * 70)
    print(f"Average overhead: {avg_overhead:.2f}x")
    print(f"Max overhead: {max_overhead:.2f}x ({worst.name})")

    # Pass if all results are acceptable
    all_acceptable = all(r.is_acceptable for r in results)
    if all_acceptable:
        print("\n✅ PASS: All operations within acceptable limits")
        return 0
    else:
        failed = [r.name for r in results if not r.is_acceptable]
        print(f"\n⚠️  WARNING: Some operations slow: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    exit(main())

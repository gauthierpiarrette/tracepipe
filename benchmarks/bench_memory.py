#!/usr/bin/env python3
"""
Memory Benchmark - Measures TracePipe memory usage patterns.

Tests memory overhead and checks for leaks.

Usage:
    python benchmarks/bench_memory.py
    python benchmarks/bench_memory.py --iterations 1000
"""

import argparse
import gc
import tracemalloc
from dataclasses import dataclass

import numpy as np
import pandas as pd

import tracepipe as tp


@dataclass
class MemoryResult:
    name: str
    baseline_mb: float
    tracepipe_mb: float

    @property
    def overhead(self) -> float:
        return self.tracepipe_mb / self.baseline_mb if self.baseline_mb > 0 else float("inf")

    @property
    def delta_mb(self) -> float:
        return self.tracepipe_mb - self.baseline_mb

    def __str__(self) -> str:
        status = "✅" if self.overhead < 2.0 else "⚠️" if self.overhead < 5.0 else "❌"
        return (
            f"{status} {self.name:30s} "
            f"baseline={self.baseline_mb:8.2f}MB  "
            f"tp={self.tracepipe_mb:8.2f}MB  "
            f"delta={self.delta_mb:+.2f}MB  "
            f"overhead={self.overhead:.2f}x"
        )


def measure_memory(func) -> float:
    """Measure peak memory of a function in MB."""
    gc.collect()
    tracemalloc.start()

    func()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()

    return peak / (1024 * 1024)


def make_df(n_rows: int = 10000) -> pd.DataFrame:
    """Create a test DataFrame."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "a": np.random.randn(n_rows),
            "b": np.random.randn(n_rows),
            "c": np.random.choice(["X", "Y", "Z"], n_rows),
            "d": np.where(np.random.random(n_rows) < 0.1, np.nan, np.random.randn(n_rows)),
        }
    )


def bench_dataframe_creation(n_rows: int) -> MemoryResult:
    """Memory overhead of DataFrame creation."""

    def baseline():
        df = make_df(n_rows)
        return df

    def with_tp():
        tp.enable(mode="debug")
        df = make_df(n_rows)
        tp.disable()
        return df

    baseline_mb = measure_memory(baseline)
    tp_mb = measure_memory(with_tp)

    return MemoryResult(f"DataFrame({n_rows:,})", baseline_mb, tp_mb)


def bench_filter_pipeline(n_rows: int) -> MemoryResult:
    """Memory overhead of filter pipeline."""

    def baseline():
        df = make_df(n_rows)
        df = df[df["a"] > 0]
        df = df[df["b"] < 0.5]
        df = df.dropna()
        return df

    def with_tp():
        tp.enable(mode="debug")
        df = make_df(n_rows)
        df = df[df["a"] > 0]
        df = df[df["b"] < 0.5]
        df = df.dropna()
        tp.disable()
        return df

    baseline_mb = measure_memory(baseline)
    tp_mb = measure_memory(with_tp)

    return MemoryResult(f"filter_pipeline({n_rows:,})", baseline_mb, tp_mb)


def bench_transform_pipeline(n_rows: int) -> MemoryResult:
    """Memory overhead of transform pipeline."""

    def baseline():
        df = make_df(n_rows)
        df = df.fillna(0)
        df["e"] = df["a"] * 2
        df["f"] = df["b"] + df["d"]
        return df

    def with_tp():
        tp.enable(mode="debug", watch=["d", "e", "f"])
        df = make_df(n_rows)
        df = df.fillna(0)
        df["e"] = df["a"] * 2
        df["f"] = df["b"] + df["d"]
        tp.disable()
        return df

    baseline_mb = measure_memory(baseline)
    tp_mb = measure_memory(with_tp)

    return MemoryResult(f"transform_pipeline({n_rows:,})", baseline_mb, tp_mb)


def bench_repeated_operations(n_iterations: int) -> MemoryResult:
    """Memory stability over repeated operations."""

    def baseline():
        df = make_df(100)
        for i in range(n_iterations):
            df.loc[0, "a"] = i
        return df

    def with_tp():
        tp.enable(mode="debug", watch=["a"])
        df = make_df(100)
        for i in range(n_iterations):
            df.loc[0, "a"] = i
        tp.disable()
        return df

    baseline_mb = measure_memory(baseline)
    tp_mb = measure_memory(with_tp)

    return MemoryResult(f"repeated_ops({n_iterations:,})", baseline_mb, tp_mb)


def bench_many_drops(n_rows: int) -> MemoryResult:
    """Memory for tracking many dropped rows."""

    def baseline():
        df = make_df(n_rows)
        df = df[df["a"] > 0]  # Drops ~50%
        return df

    def with_tp():
        tp.enable(mode="debug")
        df = make_df(n_rows)
        df = df[df["a"] > 0]
        tp.disable()
        return df

    baseline_mb = measure_memory(baseline)
    tp_mb = measure_memory(with_tp)

    return MemoryResult(f"many_drops({n_rows:,})", baseline_mb, tp_mb)


def check_memory_leak(iterations: int) -> tuple[bool, float]:
    """Check for memory leaks over many iterations."""
    tp.enable(mode="debug", watch=["a"])
    df = make_df(100)

    tracemalloc.start()
    initial = tracemalloc.get_traced_memory()[0]

    for i in range(iterations):
        df.loc[0, "a"] = i

    gc.collect()
    final = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    tp.disable()

    growth_mb = (final - initial) / (1024 * 1024)
    # Allow 10KB per iteration (TracePipe stores diffs)
    # This is expected behavior, not a leak
    max_expected_mb = iterations * 0.01
    is_leaking = growth_mb > max_expected_mb

    return is_leaking, growth_mb


def run_benchmarks(n_rows: int = 10000, iterations: int = 100) -> list[MemoryResult]:
    """Run all memory benchmarks."""
    results = []

    tp.reset()
    results.append(bench_dataframe_creation(n_rows))

    tp.reset()
    results.append(bench_filter_pipeline(n_rows))

    tp.reset()
    results.append(bench_transform_pipeline(n_rows))

    tp.reset()
    results.append(bench_repeated_operations(iterations))

    tp.reset()
    results.append(bench_many_drops(n_rows))

    return results


def main():
    parser = argparse.ArgumentParser(description="TracePipe memory benchmarks")
    parser.add_argument("--rows", type=int, default=10000, help="Number of rows")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations for leak test")
    parser.add_argument("--ci", action="store_true", help="CI mode (smaller)")
    args = parser.parse_args()

    if args.ci:
        args.rows = 1000
        args.iterations = 50

    print("=" * 80)
    print(f"TracePipe Memory Benchmark (n={args.rows:,})")
    print("=" * 80)

    results = run_benchmarks(args.rows, args.iterations)

    print("\nResults:")
    print("-" * 80)
    for r in results:
        print(r)

    # Check for memory leaks
    print("\n" + "-" * 80)
    print("Memory Leak Check:")
    is_leaking, growth_mb = check_memory_leak(args.iterations)
    if is_leaking:
        print(f"❌ FAIL: Memory grew {growth_mb:.2f}MB over {args.iterations} iterations")
    else:
        print(f"✅ PASS: Memory stable ({growth_mb:.2f}MB growth over {args.iterations} iterations)")

    # Summary
    print("-" * 80)
    overheads = [r.overhead for r in results]
    avg_overhead = np.mean(overheads)
    max_overhead = max(overheads)
    worst = max(results, key=lambda r: r.overhead)

    print(f"Average memory overhead: {avg_overhead:.2f}x")
    print(f"Max memory overhead: {max_overhead:.2f}x ({worst.name})")

    if avg_overhead < 2.0 and not is_leaking:
        print("\n✅ PASS: Memory usage acceptable")
        return 0
    else:
        print("\n⚠️  WARNING: Memory concerns detected")
        return 1


if __name__ == "__main__":
    exit(main())

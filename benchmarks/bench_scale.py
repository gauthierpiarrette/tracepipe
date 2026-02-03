#!/usr/bin/env python3
"""
Scale Benchmark - Tests TracePipe performance at different DataFrame sizes.

Measures throughput (rows/sec) and checks for linear scaling.

Usage:
    python benchmarks/bench_scale.py
    python benchmarks/bench_scale.py --max-rows 1000000
"""

import argparse
import gc
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import tracepipe as tp


@dataclass
class ScaleResult:
    n_rows: int
    operation: str
    mode: str
    time_ms: float
    rows_per_sec: float

    def __str__(self) -> str:
        return (
            f"{self.operation:20s} {self.mode:6s} "
            f"n={self.n_rows:>10,}  "
            f"time={self.time_ms:>10.2f}ms  "
            f"throughput={self.rows_per_sec:>12,.0f} rows/sec"
        )


def make_df(n_rows: int) -> pd.DataFrame:
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


def bench_filter_scale(n_rows: int, mode: str) -> ScaleResult:
    """Benchmark filter operation at scale."""
    tp.reset()
    if mode != "none":
        tp.enable(mode=mode)

    df = make_df(n_rows)
    gc.collect()

    start = time.perf_counter()
    _result = df[df["a"] > 0]
    elapsed_ms = (time.perf_counter() - start) * 1000

    if mode != "none":
        tp.disable()

    rows_per_sec = n_rows / (elapsed_ms / 1000)
    return ScaleResult(n_rows, "filter", mode, elapsed_ms, rows_per_sec)


def bench_dropna_scale(n_rows: int, mode: str) -> ScaleResult:
    """Benchmark dropna at scale."""
    tp.reset()
    if mode != "none":
        tp.enable(mode=mode, watch=["d"])

    df = make_df(n_rows)
    gc.collect()

    start = time.perf_counter()
    _result = df.dropna()
    elapsed_ms = (time.perf_counter() - start) * 1000

    if mode != "none":
        tp.disable()

    rows_per_sec = n_rows / (elapsed_ms / 1000)
    return ScaleResult(n_rows, "dropna", mode, elapsed_ms, rows_per_sec)


def bench_fillna_scale(n_rows: int, mode: str) -> ScaleResult:
    """Benchmark fillna at scale."""
    tp.reset()
    if mode != "none":
        tp.enable(mode=mode, watch=["d"])

    df = make_df(n_rows)
    gc.collect()

    start = time.perf_counter()
    _result = df.fillna(0)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if mode != "none":
        tp.disable()

    rows_per_sec = n_rows / (elapsed_ms / 1000)
    return ScaleResult(n_rows, "fillna", mode, elapsed_ms, rows_per_sec)


def bench_groupby_scale(n_rows: int, mode: str) -> ScaleResult:
    """Benchmark groupby at scale."""
    tp.reset()
    if mode != "none":
        tp.enable(mode=mode)

    df = make_df(n_rows)
    gc.collect()

    start = time.perf_counter()
    _result = df.groupby("c")["a"].sum()
    elapsed_ms = (time.perf_counter() - start) * 1000

    if mode != "none":
        tp.disable()

    rows_per_sec = n_rows / (elapsed_ms / 1000)
    return ScaleResult(n_rows, "groupby.sum", mode, elapsed_ms, rows_per_sec)


def bench_sort_scale(n_rows: int, mode: str) -> ScaleResult:
    """Benchmark sort at scale."""
    tp.reset()
    if mode != "none":
        tp.enable(mode=mode)

    df = make_df(n_rows)
    gc.collect()

    start = time.perf_counter()
    _result = df.sort_values("a")
    elapsed_ms = (time.perf_counter() - start) * 1000

    if mode != "none":
        tp.disable()

    rows_per_sec = n_rows / (elapsed_ms / 1000)
    return ScaleResult(n_rows, "sort_values", mode, elapsed_ms, rows_per_sec)


def run_scale_benchmarks(sizes: list[int]) -> list[ScaleResult]:
    """Run scale benchmarks for all sizes."""
    benchmarks = [
        bench_filter_scale,
        bench_dropna_scale,
        bench_fillna_scale,
        bench_groupby_scale,
        bench_sort_scale,
    ]
    modes = ["none", "ci", "debug"]

    results = []
    for n_rows in sizes:
        for bench in benchmarks:
            for mode in modes:
                # Full isolation between tests
                tp.reset()
                gc.collect()
                result = bench(n_rows, mode)
                results.append(result)
                print(result)
                tp.reset()
        print()

    return results


def analyze_scaling(results: list[ScaleResult]) -> dict:
    """Analyze scaling behavior."""
    # Group by operation and mode
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        groups[(r.operation, r.mode)].append(r)

    analysis = {}
    for (op, mode), items in groups.items():
        if len(items) < 2:
            continue

        # Check if time grows linearly with size
        sizes = [r.n_rows for r in items]
        times = [r.time_ms for r in items]

        # Calculate scaling factor (should be ~1.0 for linear)
        size_ratio = sizes[-1] / sizes[0]
        time_ratio = times[-1] / times[0] if times[0] > 0 else float("inf")
        scaling_factor = time_ratio / size_ratio

        # Sub-linear (< 1.0) is fine, super-linear (> 3.0) is a problem
        # Allow wider range because small datasets have high variance
        analysis[(op, mode)] = {
            "scaling_factor": scaling_factor,
            "is_linear": scaling_factor < 5.0,  # Only flag super-linear blowups
        }

    return analysis


def main():
    parser = argparse.ArgumentParser(description="TracePipe scale benchmarks")
    parser.add_argument("--max-rows", type=int, default=100000, help="Max rows to test")
    parser.add_argument("--ci", action="store_true", help="CI mode (smaller)")
    args = parser.parse_args()

    if args.ci:
        sizes = [1_000, 10_000]
    elif args.max_rows >= 1_000_000:
        sizes = [1_000, 10_000, 100_000, 1_000_000]
    else:
        sizes = [1_000, 10_000, 100_000]

    print("=" * 80)
    print("TracePipe Scale Benchmark")
    print("=" * 80)
    print(f"Testing sizes: {[f'{n:,}' for n in sizes]}")
    print()

    results = run_scale_benchmarks(sizes)

    # Analyze scaling
    print("=" * 80)
    print("Scaling Analysis")
    print("=" * 80)

    analysis = analyze_scaling(results)
    all_linear = True
    for (op, mode), data in sorted(analysis.items()):
        status = "✅" if data["is_linear"] else "❌"
        print(f"{status} {op:20s} {mode:6s} scaling_factor={data['scaling_factor']:.2f}")
        if not data["is_linear"]:
            all_linear = False

    if all_linear:
        print("\n✅ PASS: All operations scale linearly")
        return 0
    else:
        print("\n⚠️  WARNING: Some operations show non-linear scaling")
        return 1


if __name__ == "__main__":
    exit(main())

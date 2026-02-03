#!/usr/bin/env python3
"""
Run All Benchmarks - Execute all TracePipe benchmarks and generate a report.

Usage:
    python benchmarks/run_all.py
    python benchmarks/run_all.py --ci          # CI mode (smaller, faster)
    python benchmarks/run_all.py --full        # Full benchmark (larger datasets)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_benchmark(script: str, args: list[str] = None) -> tuple[int, float]:
    """Run a benchmark script, return (exit_code, duration_seconds)."""
    args = args or []
    cmd = [sys.executable, script] + args

    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    start = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    duration = time.time() - start

    return result.returncode, duration


def main():
    parser = argparse.ArgumentParser(description="Run all TracePipe benchmarks")
    parser.add_argument("--ci", action="store_true", help="CI mode (smaller, faster)")
    parser.add_argument("--full", action="store_true", help="Full benchmark (larger datasets)")
    args = parser.parse_args()

    benchmarks_dir = Path(__file__).parent
    benchmarks = [
        "bench_overhead.py",
        "bench_scale.py",
        "bench_memory.py",
    ]

    extra_args = []
    if args.ci:
        extra_args = ["--ci"]
    elif args.full:
        extra_args = ["--rows", "100000", "--iterations", "20"]

    print("=" * 60)
    print("TracePipe Benchmark Suite")
    print("=" * 60)
    print(f"Mode: {'CI' if args.ci else 'Full' if args.full else 'Standard'}")

    results = {}
    total_start = time.time()

    for bench in benchmarks:
        script = benchmarks_dir / bench
        if script.exists():
            exit_code, duration = run_benchmark(str(script), extra_args)
            results[bench] = {"exit_code": exit_code, "duration": duration}
        else:
            print(f"⚠️  Skipping {bench}: not found")

    total_duration = time.time() - total_start

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, data in results.items():
        status = "✅ PASS" if data["exit_code"] == 0 else "❌ FAIL"
        print(f"{status}  {name:30s}  {data['duration']:.1f}s")
        if data["exit_code"] != 0:
            all_passed = False

    print("-" * 60)
    print(f"Total time: {total_duration:.1f}s")

    if all_passed:
        print("\n🏆 All benchmarks passed!")
        return 0
    else:
        print("\n⚠️  Some benchmarks failed - review results above")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
TracePipe Overhead Benchmark
============================

Compares execution time WITH vs WITHOUT TracePipe to measure actual overhead.
"""

import time

import numpy as np
import pandas as pd

import tracepipe


def generate_data(n_rows=50_000):
    """Generate test dataset."""
    np.random.seed(42)

    data = {
        "customer_id": [f"CUST_{i:06d}" for i in range(n_rows)],
        "age": np.random.randint(18, 80, n_rows),
        "income": np.random.lognormal(10.5, 0.8, n_rows),
        "credit_score": np.random.normal(680, 80, n_rows),
        "account_balance": np.random.exponential(5000, n_rows),
    }

    # Add missing values
    data["age"] = data["age"].astype(float)
    data["income"] = data["income"].astype(float)
    data["credit_score"] = data["credit_score"].astype(float)

    missing_indices_age = np.random.choice(n_rows, int(n_rows * 0.02), replace=False)
    missing_indices_income = np.random.choice(n_rows, int(n_rows * 0.03), replace=False)
    missing_indices_credit = np.random.choice(n_rows, int(n_rows * 0.015), replace=False)

    for idx in missing_indices_age:
        data["age"][idx] = np.nan
    for idx in missing_indices_income:
        data["income"][idx] = np.nan
    for idx in missing_indices_credit:
        data["credit_score"][idx] = np.nan

    # Add duplicates
    n_dupes = int(n_rows * 0.003)
    dupe_indices = np.random.choice(n_rows, n_dupes, replace=False)
    for idx in dupe_indices:
        data["customer_id"][idx] = f"CUST_{np.random.randint(0, n_rows):06d}"

    # Add negative values
    error_indices = np.random.choice(n_rows, int(n_rows * 0.005), replace=False)
    for idx in error_indices:
        data["account_balance"][idx] = -data["account_balance"][idx]

    return pd.DataFrame(data)


def run_pipeline_without_tracepipe(df):
    """Run pipeline WITHOUT TracePipe."""
    df = df.copy()

    # 1. Drop duplicates
    df = df.drop_duplicates(subset=["customer_id"], keep="first")

    # 2. Filter negative balances
    df = df[df["account_balance"] >= 0]

    # 3. Impute age
    age_median = df["age"].median()
    df["age"] = df["age"].fillna(age_median)

    # 4. Impute income
    income_median = df["income"].median()
    df["income"] = df["income"].fillna(income_median)

    # 5. Impute credit score
    credit_median = df["credit_score"].median()
    df["credit_score"] = df["credit_score"].fillna(credit_median)

    # 6. Drop remaining NaN
    df = df.dropna()

    return df


def run_pipeline_with_tracepipe(df):
    """Run pipeline WITH TracePipe."""
    df = df.copy()

    tracepipe.reset()
    tracepipe.enable()
    tracepipe.watch("age", "income", "credit_score", "customer_id", "account_balance")

    # 1. Drop duplicates
    df = df.drop_duplicates(subset=["customer_id"], keep="first")

    # 2. Filter negative balances
    df = df[df["account_balance"] >= 0]

    # 3. Impute age
    age_median = df["age"].median()
    df["age"] = df["age"].fillna(age_median)

    # 4. Impute income
    income_median = df["income"].median()
    df["income"] = df["income"].fillna(income_median)

    # 5. Impute credit score
    credit_median = df["credit_score"].median()
    df["credit_score"] = df["credit_score"].fillna(credit_median)

    # 6. Drop remaining NaN
    df = df.dropna()

    tracepipe.disable()

    return df


def benchmark_operation(name, func, *args, n_runs=5):
    """Benchmark a single operation."""
    times = []

    for _ in range(n_runs):
        start = time.time()
        _result = func(*args)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    return avg_time, min_time, max_time


print("=" * 80)
print("TracePipe Overhead Benchmark")
print("=" * 80)
print()

# Generate test data
print("📊 Generating 50,000 row dataset...")
df_template = generate_data(50_000)
print(f"   ✓ {len(df_template):,} rows, {len(df_template.columns)} columns")
print(f"   ✓ Missing values: {df_template.isna().sum().sum():,}")
print()

# Benchmark WITHOUT TracePipe
print("⚡ Benchmarking WITHOUT TracePipe (5 runs)...")
without_avg, without_min, without_max = benchmark_operation(
    "without", run_pipeline_without_tracepipe, df_template
)
print(f"   Average: {without_avg:.3f}s")
print(f"   Min:     {without_min:.3f}s")
print(f"   Max:     {without_max:.3f}s")
print()

# Benchmark WITH TracePipe
print("🔍 Benchmarking WITH TracePipe (5 runs)...")
with_avg, with_min, with_max = benchmark_operation("with", run_pipeline_with_tracepipe, df_template)
print(f"   Average: {with_avg:.3f}s")
print(f"   Min:     {with_min:.3f}s")
print(f"   Max:     {with_max:.3f}s")
print()

# Calculate overhead
overhead_abs = with_avg - without_avg
overhead_pct = (with_avg / without_avg - 1) * 100 if without_avg > 0 else 0
slowdown_factor = with_avg / without_avg if without_avg > 0 else 0

print("=" * 80)
print("📈 Results")
print("=" * 80)
print()
print(f"WITHOUT TracePipe:  {without_avg:.3f}s")
print(f"WITH TracePipe:     {with_avg:.3f}s")
print()
print(f"Absolute Overhead:  +{overhead_abs:.3f}s")
print(f"Relative Overhead:  +{overhead_pct:.1f}%")
print(f"Slowdown Factor:    {slowdown_factor:.2f}x")
print()

if slowdown_factor < 1.5:
    verdict = "✅ EXCELLENT - Minimal overhead"
elif slowdown_factor < 3:
    verdict = "✅ GOOD - Acceptable overhead for debugging"
elif slowdown_factor < 5:
    verdict = "⚠️  MODERATE - Noticeable but usable"
else:
    verdict = "❌ HIGH - Significant performance impact"

print(f"Verdict: {verdict}")
print()

# Per-operation breakdown
print("=" * 80)
print("🔬 Per-Operation Breakdown")
print("=" * 80)
print()

operations = [
    ("drop_duplicates", lambda df: df.drop_duplicates(subset=["customer_id"], keep="first")),
    ("filter (boolean mask)", lambda df: df[df["account_balance"] >= 0]),
    ("fillna (age)", lambda df: df.assign(age=df["age"].fillna(df["age"].median()))),
    ("fillna (income)", lambda df: df.assign(income=df["income"].fillna(df["income"].median()))),
    ("dropna", lambda df: df.dropna()),
]

for op_name, op_func in operations:
    # Without TracePipe
    test_df = df_template.copy()
    times_without = []
    for _ in range(5):
        start = time.time()
        _ = op_func(test_df)
        times_without.append(time.time() - start)
    avg_without = sum(times_without) / len(times_without)

    # With TracePipe
    test_df = df_template.copy()
    tracepipe.reset()
    tracepipe.enable()
    tracepipe.watch("age", "income", "credit_score", "customer_id", "account_balance")
    times_with = []
    for _ in range(5):
        start = time.time()
        _ = op_func(test_df)
        times_with.append(time.time() - start)
    avg_with = sum(times_with) / len(times_with)
    tracepipe.disable()

    overhead = (avg_with / avg_without - 1) * 100 if avg_without > 0 else 0

    print(
        f"{op_name:25} | Without: {avg_without*1000:6.1f}ms | With: {avg_with*1000:6.1f}ms | Overhead: {overhead:+6.1f}%"
    )

print()
print("=" * 80)

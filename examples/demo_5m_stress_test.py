"""
TracePipe Stress Test: 5 Million Rows
=====================================

Tests TracePipe's performance at scale with a realistic ML preprocessing pipeline.
"""

import os
import time

import numpy as np
import pandas as pd
import psutil

import tracepipe
from tracepipe.context import get_context


def get_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def print_memory(label):
    """Print current memory usage."""
    mem = get_memory_mb()
    print(f"   💾 {label}: {mem:.0f} MB")

def generate_data(n_rows=5_000_000):
    """Generate synthetic customer data."""
    print(f"📁 Generating {n_rows:,} customer records...")
    start = time.time()

    np.random.seed(42)

    data = {
        'customer_id': [f'CUST_{i:08d}' for i in range(n_rows)],
        'age': np.random.randint(18, 80, n_rows).astype(float),
        'income': np.random.lognormal(10.5, 0.8, n_rows).astype(float),
        'credit_score': np.random.normal(680, 80, n_rows).astype(float),
        'account_balance': np.random.exponential(5000, n_rows).astype(float),
    }

    # Add missing values (~2% per column)
    for col in ['age', 'income', 'credit_score']:
        n_missing = int(n_rows * 0.02)
        missing_idx = np.random.choice(n_rows, n_missing, replace=False)
        data[col][missing_idx] = np.nan

    # Add some negative balances (errors, ~0.5%)
    error_idx = np.random.choice(n_rows, int(n_rows * 0.005), replace=False)
    data['account_balance'][error_idx] *= -1

    # Add duplicates (~0.3%)
    n_dupes = int(n_rows * 0.003)
    dupe_idx = np.random.choice(n_rows, n_dupes, replace=False)
    for idx in dupe_idx:
        data['customer_id'][idx] = f'CUST_{np.random.randint(0, n_rows):08d}'

    df = pd.DataFrame(data)
    elapsed = time.time() - start
    print(f"   ✓ Generated {len(df):,} rows in {elapsed:.2f}s")
    return df

def run_pipeline(df, with_tracepipe=True):
    """Run the ML preprocessing pipeline."""
    if with_tracepipe:
        tracepipe.reset()
        tracepipe.enable()
        tracepipe.watch('age', 'income', 'credit_score', 'account_balance')

    start_total = time.time()

    # 1. Drop duplicates
    print("   → Drop duplicates...")
    start = time.time()
    len_before = len(df)
    df = df.drop_duplicates(subset=['customer_id'], keep='first')
    elapsed = time.time() - start
    print(f"      Removed {len_before - len(df):,} rows in {elapsed:.2f}s")
    print_memory("After dedup")

    # 2. Filter negative balances
    print("   → Filter negative balances...")
    start = time.time()
    len_before = len(df)
    df = df[df['account_balance'] >= 0]
    elapsed = time.time() - start
    print(f"      Removed {len_before - len(df):,} rows in {elapsed:.2f}s")
    print_memory("After filter")

    # 3. Impute age
    print("   → Impute missing age...")
    start = time.time()
    age_median = df['age'].median()
    df['age'] = df['age'].fillna(age_median)
    elapsed = time.time() - start
    print(f"      Filled {(df['age'] == age_median).sum():,} values in {elapsed:.2f}s")
    print_memory("After age imputation")

    # 4. Impute income
    print("   → Impute missing income...")
    start = time.time()
    income_median = df['income'].median()
    df['income'] = df['income'].fillna(income_median)
    elapsed = time.time() - start
    print(f"      Filled values in {elapsed:.2f}s")
    print_memory("After income imputation")

    # 5. Impute credit score
    print("   → Impute missing credit_score...")
    start = time.time()
    credit_median = df['credit_score'].median()
    df['credit_score'] = df['credit_score'].fillna(credit_median)
    elapsed = time.time() - start
    print(f"      Filled values in {elapsed:.2f}s")
    print_memory("After credit imputation")

    # 6. Drop remaining NaN
    print("   → Drop remaining NaN...")
    start = time.time()
    len_before = len(df)
    df = df.dropna()
    elapsed = time.time() - start
    print(f"      Removed {len_before - len(df):,} rows in {elapsed:.2f}s")
    print_memory("After dropna")

    total_time = time.time() - start_total

    if with_tracepipe:
        ctx = get_context()
        stats = {
            'total_steps': len(ctx.store.steps),
            'total_diffs': ctx.store.total_diff_count,
        }
        tracepipe.disable()
    else:
        stats = None

    return df, total_time, stats

print("=" * 80)
print("TracePipe Stress Test: 5 Million Rows")
print("=" * 80)
print()

baseline_memory = get_memory_mb()
print(f"📊 Baseline memory: {baseline_memory:.0f} MB")
print()

# Generate data
df = generate_data(5_000_000)
print_memory("After data generation")
print()

# Test WITHOUT TracePipe
print("⚡ Running pipeline WITHOUT TracePipe...")
print("-" * 80)
df_copy = df.copy()
df_without, time_without, _ = run_pipeline(df_copy, with_tracepipe=False)
print(f"\n✅ Completed in {time_without:.2f}s")
print(f"   Final: {len(df_without):,} rows")
print()

# Test WITH TracePipe
print("🔍 Running pipeline WITH TracePipe...")
print("-" * 80)
df_copy = df.copy()
df_with, time_with, stats = run_pipeline(df_copy, with_tracepipe=True)
print(f"\n✅ Completed in {time_with:.2f}s")
print(f"   Final: {len(df_with):,} rows")
print()

# Export lineage
print("💾 Exporting lineage...")
start = time.time()
tracepipe.reset()
tracepipe.enable()
tracepipe.watch('age', 'income', 'credit_score', 'account_balance')
df_copy = df.copy()
df_final, _, _ = run_pipeline(df_copy, with_tracepipe=True)
print()

try:
    print("   → Exporting JSON...")
    start_export = time.time()
    tracepipe.export_json("stress_test_5m_lineage.json")
    json_time = time.time() - start_export
    json_size_mb = os.path.getsize("stress_test_5m_lineage.json") / 1024 / 1024
    print(f"      ✓ Exported in {json_time:.2f}s (size: {json_size_mb:.1f} MB)")
except Exception as e:
    print(f"      ✗ JSON export failed: {e}")

try:
    print("   → Exporting HTML...")
    start_export = time.time()
    tracepipe.save("stress_test_5m_report.html")
    html_time = time.time() - start_export
    html_size_mb = os.path.getsize("stress_test_5m_report.html") / 1024 / 1024
    print(f"      ✓ Exported in {html_time:.2f}s (size: {html_size_mb:.1f} MB)")
except Exception as e:
    print(f"      ✗ HTML export failed: {e}")

export_time = time.time() - start
print_memory("After export")
print()

# Summary
print("=" * 80)
print("📈 Performance Summary")
print("=" * 80)
print()

final_memory = get_memory_mb()
memory_overhead = final_memory - baseline_memory

print("Execution Time:")
print(f"  WITHOUT TracePipe:  {time_without:.2f}s")
print(f"  WITH TracePipe:     {time_with:.2f}s")
print(f"  Absolute Overhead:  +{time_with - time_without:.2f}s")
print(f"  Relative Overhead:  +{((time_with / time_without - 1) * 100):.1f}%")
print(f"  Slowdown Factor:    {time_with / time_without:.2f}x")
print()

if time_with / time_without < 2:
    verdict = "✅ EXCELLENT - Minimal overhead"
elif time_with / time_without < 5:
    verdict = "✅ GOOD - Acceptable overhead"
elif time_with / time_without < 10:
    verdict = "⚠️  MODERATE - Noticeable but usable"
else:
    verdict = "❌ HIGH - Consider sampling mode"

print(f"Verdict: {verdict}")
print()

print("Memory Usage:")
print(f"  Peak memory:        {final_memory:.0f} MB")
print(f"  Overhead:           {memory_overhead:.0f} MB")
print(f"  Per-row cost:       {(memory_overhead * 1024) / 5_000_000:.3f} KB/row")
print()

if stats:
    print("Lineage Statistics:")
    print(f"  Total steps:        {stats['total_steps']}")
    print(f"  Total diffs:        {stats['total_diffs']:,}")
    print()

print("=" * 80)
print("💡 Recommendations")
print("=" * 80)
print()

if time_with / time_without > 10:
    print("⚠️  High overhead detected!")
    print()
    print("Consider using SAMPLING MODE for better performance:")
    print("  tracepipe.enable(sample_rate=0.01)  # Track 1% of rows")
    print("  - 100x faster tracking")
    print("  - Statistical validity maintained")
    print("  - Same insights, much lower overhead")
elif time_with / time_without > 5:
    print("Moderate overhead detected.")
    print()
    print("For production use, consider:")
    print("  • Sampling mode for very large datasets")
    print("  • Selective column watching (only critical columns)")
    print("  • Periodic tracking (enable/disable as needed)")
else:
    print("✅ Performance is good for 5M rows!")
    print()
    print("TracePipe can handle datasets of this size with acceptable overhead.")
    print("For even larger datasets (10M+), consider sampling mode.")

print()
print("=" * 80)


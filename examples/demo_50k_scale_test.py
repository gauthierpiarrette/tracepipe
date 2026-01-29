"""
TracePipe Scale Test: 50K Rows
==============================

This demo tests TracePipe's performance and memory usage with a realistic-sized dataset.

Expected behavior:
- Memory usage will grow significantly (100s of MB)
- Processing will slow down noticeably
- HTML export may struggle

This demonstrates the need for sampling mode and streaming backends.
"""

import os
import time

import numpy as np
import pandas as pd
import psutil

import tracepipe
from tracepipe.context import get_context


# Track memory usage
def get_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def print_memory(label):
    """Print current memory usage."""
    mem = get_memory_mb()
    print(f"   💾 {label}: {mem:.1f} MB")


print("=" * 70)
print("TracePipe Scale Test: 50,000 Rows")
print("=" * 70)
print()

# Baseline memory
baseline_memory = get_memory_mb()
print(f"📊 Baseline memory: {baseline_memory:.1f} MB")
print()

# ============================================================================
# Step 1: Enable TracePipe
# ============================================================================
print("🔍 Enabling TracePipe...")
start_time = time.time()

tracepipe.enable()
tracepipe.watch('customer_id', 'age', 'income', 'credit_score', 'account_balance')

enable_time = time.time() - start_time
print(f"   ✓ Enabled in {enable_time:.2f}s")
print_memory("After enable")
print()

# ============================================================================
# Step 2: Generate 50K rows of synthetic data
# ============================================================================
print("📁 Generating 50,000 customer records...")
start_time = time.time()

np.random.seed(42)
n_rows = 50_000

data = {
    'customer_id': [f'CUST_{i:06d}' for i in range(n_rows)],
    'age': np.random.randint(18, 80, n_rows),
    'income': np.random.lognormal(10.5, 0.8, n_rows),  # ~$50k average
    'credit_score': np.random.normal(680, 80, n_rows),
    'account_balance': np.random.exponential(5000, n_rows),
    'num_transactions': np.random.poisson(25, n_rows),
    'days_since_signup': np.random.randint(1, 3650, n_rows),
    'has_premium': np.random.choice([True, False], n_rows, p=[0.3, 0.7]),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_rows),
    'churned': np.random.choice([0, 1], n_rows, p=[0.85, 0.15])
}

# Add data quality issues (realistic for 50K rows)
# Missing values (~2% of data)
missing_indices_age = np.random.choice(n_rows, int(n_rows * 0.02), replace=False)
missing_indices_income = np.random.choice(n_rows, int(n_rows * 0.03), replace=False)
missing_indices_credit = np.random.choice(n_rows, int(n_rows * 0.015), replace=False)

data['age'] = data['age'].astype(float)
data['income'] = data['income'].astype(float)
data['credit_score'] = data['credit_score'].astype(float)

for idx in missing_indices_age:
    data['age'][idx] = np.nan
for idx in missing_indices_income:
    data['income'][idx] = np.nan
for idx in missing_indices_credit:
    data['credit_score'][idx] = np.nan

# Negative values (data errors, ~0.5%)
error_indices = np.random.choice(n_rows, int(n_rows * 0.005), replace=False)
for idx in error_indices:
    data['account_balance'][idx] = -data['account_balance'][idx]

# Duplicate IDs (~0.3%)
n_dupes = int(n_rows * 0.003)
dupe_indices = np.random.choice(n_rows, n_dupes, replace=False)
for idx in dupe_indices:
    data['customer_id'][idx] = f'CUST_{np.random.randint(0, n_rows):06d}'

df = pd.DataFrame(data)

gen_time = time.time() - start_time
print(f"   ✓ Generated {len(df):,} rows in {gen_time:.2f}s")
print(f"   Columns: {list(df.columns)}")
print_memory("After data generation")
print()

# ============================================================================
# Step 3: Data Cleaning Pipeline
# ============================================================================
print("🧹 Stage: Data Cleaning")

# Count issues
missing_count = df.isna().sum().sum()
negative_balance_count = (df['account_balance'] < 0).sum()
dupe_count = df['customer_id'].duplicated().sum()

print("   Data quality issues:")
print(f"   - Missing values: {missing_count:,}")
print(f"   - Negative balances: {negative_balance_count:,}")
print(f"   - Duplicate customer IDs: {dupe_count:,}")
print()

# Remove duplicates
print("   → Removing duplicate customer IDs...")
start_time = time.time()
len_before = len(df)

df = df.drop_duplicates(subset=['customer_id'], keep='first')

step_time = time.time() - start_time
print(f"      Removed {len_before - len(df):,} duplicates in {step_time:.2f}s")
print_memory("After dedup")

# Remove negative balances
print("   → Removing negative account balances...")
start_time = time.time()
len_before = len(df)

tracepipe.stage("cleaning")
df = df[df['account_balance'] >= 0]

step_time = time.time() - start_time
print(f"      Removed {len_before - len(df):,} rows in {step_time:.2f}s")
print_memory("After filter")

# Remove outliers (extreme ages)
print("   → Removing age outliers (>100 or <18)...")
start_time = time.time()
len_before = len(df)

df = df[(df['age'].isna()) | ((df['age'] >= 18) & (df['age'] <= 100))]

step_time = time.time() - start_time
print(f"      Removed {len_before - len(df):,} rows in {step_time:.2f}s")
print_memory("After outlier removal")
print()

# ============================================================================
# Step 4: Feature Engineering (Imputation)
# ============================================================================
print("⚙️ Stage: Feature Engineering")

tracepipe.stage("feature_engineering")

# Impute age
print("   → Imputing missing age with median...")
start_time = time.time()
age_median = df['age'].median()
df['age'] = df['age'].fillna(age_median)

step_time = time.time() - start_time
print(f"      Filled with median: {age_median:.1f} in {step_time:.2f}s")
print_memory("After age imputation")

# Impute income
print("   → Imputing missing income with median...")
start_time = time.time()
income_median = df['income'].median()
df['income'] = df['income'].fillna(income_median)

step_time = time.time() - start_time
print(f"      Filled with median: ${income_median:,.0f} in {step_time:.2f}s")
print_memory("After income imputation")

# Impute credit score
print("   → Imputing missing credit_score with median...")
start_time = time.time()
credit_median = df['credit_score'].median()
df['credit_score'] = df['credit_score'].fillna(credit_median)

step_time = time.time() - start_time
print(f"      Filled with median: {credit_median:.0f} in {step_time:.2f}s")
print_memory("After credit score imputation")
print()

# ============================================================================
# Step 5: Final Filtering
# ============================================================================
print("✂️ Stage: Final Filtering")
tracepipe.stage("final_filter")

print("   → Dropping rows with any remaining NaN...")
start_time = time.time()
len_before = len(df)

df = df.dropna()

step_time = time.time() - start_time
print(f"      Removed {len_before - len(df):,} rows in {step_time:.2f}s")
print_memory("After final dropna")
print()

print(f"✅ Final dataset: {len(df):,} rows, {len(df.columns)} columns")
print()

# ============================================================================
# Step 6: Export Lineage
# ============================================================================
print("=" * 70)
print("📊 TracePipe Lineage Analysis")
print("=" * 70)
print()

print("💾 Exporting lineage reports...")
print_memory("Before export")

# JSON export
print("   → Exporting JSON...")
start_time = time.time()
try:
    tracepipe.export_json("scale_test_50k_lineage.json")
    json_time = time.time() - start_time

    # Get file size
    json_size_mb = os.path.getsize("scale_test_50k_lineage.json") / 1024 / 1024
    print(f"      ✓ JSON exported in {json_time:.2f}s (size: {json_size_mb:.1f} MB)")
except Exception as e:
    print(f"      ✗ JSON export failed: {e}")

print_memory("After JSON export")

# HTML export
print("   → Exporting HTML...")
start_time = time.time()
try:
    tracepipe.save("scale_test_50k_report.html")
    html_time = time.time() - start_time

    # Get file size
    html_size_mb = os.path.getsize("scale_test_50k_report.html") / 1024 / 1024
    print(f"      ✓ HTML exported in {html_time:.2f}s (size: {html_size_mb:.1f} MB)")
except Exception as e:
    print(f"      ✗ HTML export failed: {e}")

print_memory("After HTML export")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 70)
print("📈 Performance Summary")
print("=" * 70)
print()

final_memory = get_memory_mb()
memory_overhead = final_memory - baseline_memory

print("Memory Usage:")
print(f"  Baseline:       {baseline_memory:.1f} MB")
print(f"  Final:          {final_memory:.1f} MB")
print(f"  Overhead:       {memory_overhead:.1f} MB")
print(f"  Per-row cost:   {(memory_overhead * 1024) / n_rows:.2f} KB/row")
print()

# Get lineage stats
ctx = get_context()
print("Lineage Statistics:")
print(f"  Total steps:    {len(ctx.store.steps)}")
print(f"  Total diffs:    {ctx.store.total_diff_count:,}")
print(f"  Rows dropped:   {sum(ctx.store.get_dropped_by_step().values()):,}")
print()

print("=" * 70)
print("💡 Observations")
print("=" * 70)
print()
print("With 50K rows, you likely noticed:")
print("  • Significant memory overhead (100-500+ MB)")
print("  • Operations noticeably slower than without tracking")
print("  • Large export file sizes")
print()
print("This demonstrates the need for:")
print("  ✓ Sampling mode (track 1% = 500 rows)")
print("  ✓ Streaming backend (SQLite/disk instead of memory)")
print("  ✓ Optimized storage (columnar format, compression)")
print()
print("=" * 70)


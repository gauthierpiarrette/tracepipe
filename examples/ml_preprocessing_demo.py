#!/usr/bin/env python3
"""
TracePipe Demo: Realistic ML Preprocessing Pipeline

Simulates a data scientist's preprocessing workflow for a customer churn prediction model.
This demonstrates how TracePipe helps debug data transformations in real ML pipelines.

Scenario: You're building a churn prediction model for a telecom company.
Your stakeholder asks: "Why did customer X get flagged as high-risk? What happened to their data?"

TracePipe lets you answer that question by tracking every row through the pipeline.
"""

import numpy as np
import pandas as pd

import tracepipe

# Reproducibility
np.random.seed(42)

print("=" * 70)
print("TracePipe Demo: Customer Churn ML Preprocessing Pipeline")
print("=" * 70)

# =============================================================================
# STEP 1: Enable TracePipe FIRST (before any DataFrames are created)
# =============================================================================
print("\n🔍 Enabling TracePipe lineage tracking...")
tracepipe.enable()

# Watch columns that are critical for model features
tracepipe.watch("tenure_months", "monthly_charges", "total_charges", "num_complaints", "churned")

# =============================================================================
# STEP 2: Load Raw Data (simulating a messy real-world dataset)
# =============================================================================
# NOTE: Because TracePipe is enabled, DataFrames are AUTO-REGISTERED!
print("\n📁 Loading raw customer data...")

n_customers = 500

# Generate realistic messy data (auto-registered because enable() was called first)
df = pd.DataFrame(
    {
        "customer_id": [f"CUST_{i:05d}" for i in range(n_customers)],
        "tenure_months": np.random.exponential(24, n_customers).astype(int),
        "monthly_charges": np.random.normal(70, 25, n_customers).round(2),
        "total_charges": None,  # Will calculate, but some will be missing
        "contract_type": np.random.choice(
            ["Month-to-month", "One year", "Two year"], n_customers, p=[0.5, 0.3, 0.2]
        ),
        "payment_method": np.random.choice(
            ["Credit card", "Bank transfer", "Electronic check", "Mailed check"], n_customers
        ),
        "internet_service": np.random.choice(
            ["DSL", "Fiber optic", "No", None], n_customers, p=[0.3, 0.4, 0.25, 0.05]
        ),
        "tech_support": np.random.choice(
            ["Yes", "No", "No internet service", None], n_customers, p=[0.3, 0.4, 0.25, 0.05]
        ),
        "online_security": np.random.choice(["Yes", "No", "No internet service"], n_customers),
        "num_complaints": np.random.poisson(1.5, n_customers),
        "last_interaction_days": np.random.exponential(30, n_customers).astype(int),
        "churned": np.random.choice([0, 1], n_customers, p=[0.73, 0.27]),
    }
)
# DataFrame is AUTO-REGISTERED! No manual register() needed.
print(f"Auto-registered {len(df)} rows for tracking")

# Add realistic data quality issues (these modify df in-place, IDs preserved)
# 1. Some negative values (data entry errors)
df.loc[np.random.choice(n_customers, 15, replace=False), "monthly_charges"] = -np.random.uniform(
    10, 50, 15
).round(2)

# 2. Some extreme outliers
df.loc[np.random.choice(n_customers, 8, replace=False), "tenure_months"] = np.random.randint(
    200, 500, 8
)

# 3. Missing values scattered throughout
df.loc[np.random.choice(n_customers, 30, replace=False), "tenure_months"] = np.nan
df.loc[np.random.choice(n_customers, 25, replace=False), "monthly_charges"] = np.nan
df.loc[np.random.choice(n_customers, 20, replace=False), "num_complaints"] = np.nan

# 4. Calculate total_charges with some missing
df["total_charges"] = df["tenure_months"] * df["monthly_charges"]
df.loc[np.random.choice(n_customers, 10, replace=False), "total_charges"] = np.nan

# 5. Duplicate rows (common ETL issue)
duplicate_indices = np.random.choice(n_customers, 12, replace=False)
duplicates = df.iloc[duplicate_indices].copy()
df = pd.concat([df, duplicates], ignore_index=True)

# 6. Inconsistent string values
df.loc[df["contract_type"] == "Month-to-month", "contract_type"] = np.where(
    np.random.random(sum(df["contract_type"] == "Month-to-month")) < 0.1,
    "month-to-month",  # lowercase variant
    "Month-to-month",
)

print(f"Data shape after adding issues: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nData quality issues:")
print(f"  - Missing values: {df.isna().sum().sum()}")
print(f"  - Negative charges: {(df['monthly_charges'] < 0).sum()}")
print(f"  - Duplicate customer IDs: {df['customer_id'].duplicated().sum()}")

# =============================================================================
# STEP 3: Data Cleaning Stage
# =============================================================================
print("\n🧹 Stage: Data Cleaning")

with tracepipe.stage("cleaning"):
    initial_count = len(df)

    # 3.1 Remove exact duplicates
    print("  → Removing duplicate rows...")
    df = df.drop_duplicates(subset=["customer_id"], keep="first")
    print(f"     Removed {initial_count - len(df)} duplicates")

    # 3.2 Remove rows with negative charges (invalid data)
    # Note: Use notna() to keep rows with missing values for imputation later
    print("  → Removing invalid negative charges...")
    before = len(df)
    df = df[(df["monthly_charges"] >= 0) | (df["monthly_charges"].isna())]
    print(f"     Removed {before - len(df)} rows with negative charges")

    # 3.3 Remove extreme tenure outliers (> 120 months = 10 years)
    # Note: Keep rows with missing tenure for imputation later
    print("  → Removing tenure outliers (>10 years)...")
    before = len(df)
    df = df[(df["tenure_months"] <= 120) | (df["tenure_months"].isna())]
    print(f"     Removed {before - len(df)} outlier rows")

    # 3.4 Handle missing values in critical columns
    print("  → Dropping rows with missing churned labels...")
    before = len(df)
    df = df.dropna(subset=["churned"])
    print(f"     Removed {before - len(df)} rows without labels")

print(f"After cleaning: {len(df)} rows ({initial_count - len(df)} removed)")

# =============================================================================
# STEP 4: Feature Engineering Stage
# =============================================================================
print("\n⚙️ Stage: Feature Engineering")

with tracepipe.stage("feature_engineering"):
    # 4.1 Impute missing tenure with median
    print("  → Imputing missing tenure_months with median...")
    median_tenure = df["tenure_months"].median()
    df["tenure_months"] = df["tenure_months"].fillna(median_tenure)
    print(f"     Filled with median: {median_tenure}")

    # 4.2 Impute missing monthly_charges with median by contract type
    print("  → Imputing missing monthly_charges by contract type...")
    df["monthly_charges"] = df.groupby("contract_type")["monthly_charges"].transform(
        lambda x: x.fillna(x.median())
    )

    # 4.3 Recalculate total_charges where missing
    # Use fillna with calculated values (tracked) instead of .loc (not tracked)
    print("  → Recalculating missing total_charges...")
    calculated_total = df["tenure_months"] * df["monthly_charges"]
    df["total_charges"] = df["total_charges"].fillna(calculated_total)

    # 4.4 Impute missing complaints with 0 (assume no complaints recorded)
    print("  → Imputing missing complaints as 0...")
    df["num_complaints"] = df["num_complaints"].fillna(0)

    # 4.5 Create derived features
    print("  → Creating derived features...")

    # Average monthly spend
    df["avg_monthly_spend"] = df["total_charges"] / df["tenure_months"].replace(0, 1)

    # Complaint rate per month
    df["complaint_rate"] = df["num_complaints"] / df["tenure_months"].replace(0, 1)

    # Customer value score
    df["customer_value"] = df["tenure_months"] * df["monthly_charges"] / 100

    # Engagement flag (recent interaction)
    df["is_engaged"] = (df["last_interaction_days"] < 30).astype(int)

# =============================================================================
# STEP 5: Categorical Encoding Stage
# =============================================================================
print("\n🔢 Stage: Categorical Encoding")

with tracepipe.stage("encoding"):
    # 5.1 Standardize contract_type (fix inconsistent casing)
    print("  → Standardizing contract_type values...")
    df["contract_type"] = df["contract_type"].str.title()

    # 5.2 Binary encoding for internet service
    print("  → Encoding internet_service as binary...")
    df["has_internet"] = (df["internet_service"] != "No").astype(int)

    # 5.3 Risk score based on multiple factors
    print("  → Calculating risk_score...")
    df["risk_score"] = (
        (df["contract_type"] == "Month-To-Month").astype(int) * 2
        + (df["payment_method"] == "Electronic check").astype(int) * 1
        + (df["num_complaints"] > 2).astype(int) * 2
        + (df["tenure_months"] < 12).astype(int) * 1
    )

# =============================================================================
# STEP 6: Final Filtering Stage
# =============================================================================
print("\n✂️ Stage: Final Filtering")

with tracepipe.stage("final_filter"):
    # Remove any remaining rows with NaN in key features
    print("  → Dropping rows with any remaining NaN in features...")
    feature_cols = [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "num_complaints",
        "risk_score",
    ]
    before = len(df)
    df = df.dropna(subset=feature_cols)
    print(f"     Removed {before - len(df)} rows")

    print(f"\nFinal dataset: {len(df)} rows, {len(df.columns)} columns")

# =============================================================================
# STEP 7: Analyze Pipeline with TracePipe
# =============================================================================
print("\n" + "=" * 70)
print("📊 TracePipe Lineage Analysis")
print("=" * 70)

# Summary statistics
stats = tracepipe.stats()
print("\nPipeline Statistics:")
print(f"  Total steps tracked: {stats['total_steps']}")
print(f"  Total cell changes: {stats['total_diffs']}")
print(f"  Watched columns: {stats['watched_columns']}")

# Dropped rows analysis
dropped = tracepipe.dropped_rows()
dropped_by_step = tracepipe.dropped_rows_by_step()

print("\nData Loss Analysis:")
print(f"  Total rows dropped: {len(dropped)}")
print("  Drop breakdown by operation:")
for op, count in sorted(dropped_by_step.items(), key=lambda x: -x[1]):
    print(f"    - {op}: {count} rows")

# Sample specific row journeys
print("\n🔎 Sample Row Journeys:")

# Find a row that was dropped
if dropped:
    dropped_id = dropped[0]
    dropped_row = tracepipe.explain(dropped_id)
    print(f"\n  Dropped Row #{dropped_id}:")
    print(f"    Status: {'Alive' if dropped_row.is_alive else 'DROPPED'}")
    print(f"    Dropped at: {dropped_row.dropped_at}")

# Find a row that survived with changes
all_row_ids = set(range(n_customers))
survived_with_changes = []
for rid in list(all_row_ids - set(dropped))[:50]:  # Check first 50 survivors
    row = tracepipe.explain(rid)
    if row.is_alive and len(row.history()) > 0:
        # Filter to only actual value changes (not __row__ events)
        value_changes = [h for h in row.history() if h["col"] not in ("__row__", "__position__")]
        if value_changes:
            survived_with_changes.append((rid, value_changes))

if survived_with_changes:
    rid, changes = survived_with_changes[0]
    print(f"\n  Survived Row #{rid} (with {len(changes)} value changes):")
    for change in changes[:5]:  # Show first 5 changes
        print(
            f"    - {change['operation']}: {change['col']} = {change['old_val']} → {change['new_val']}"
        )

# =============================================================================
# STEP 8: Export Reports
# =============================================================================
print("\n💾 Exporting lineage reports...")

tracepipe.export_json("ml_pipeline_lineage.json")
print("  ✓ JSON: ml_pipeline_lineage.json")

tracepipe.save("ml_pipeline_report.html")
print("  ✓ HTML: ml_pipeline_report.html")

# =============================================================================
# STEP 9: Answer Business Questions
# =============================================================================
print("\n" + "=" * 70)
print("💡 Answering Business Questions with TracePipe")
print("=" * 70)

# Question 1: "Why were so many customers excluded from the model?"
print("\n❓ Q1: Why were customers excluded from the model?")
print(f"   A: {len(dropped)} customers were excluded:")
for reason, count in sorted(dropped_by_step.items(), key=lambda x: -x[1])[:3]:
    pct = count / len(dropped) * 100 if dropped else 0
    print(f"      • {reason}: {count} ({pct:.1f}%)")

# Question 2: "Which features had the most data quality issues?"
print("\n❓ Q2: Which features had the most imputations?")
changes_summary = {}
for rid in list(all_row_ids - set(dropped))[:100]:
    row = tracepipe.explain(rid)
    for event in row.history():
        if event["col"] not in ("__row__", "__position__"):
            col = event["col"]
            changes_summary[col] = changes_summary.get(col, 0) + 1

for col, count in sorted(changes_summary.items(), key=lambda x: -x[1])[:5]:
    print(f"      • {col}: {count} cells modified")

# Question 3: "Can I trace a specific high-risk customer?"
print("\n❓ Q3: Trace a specific customer's data journey")
# Find a high-risk customer that survived
high_risk = df[df["risk_score"] >= 4]
if len(high_risk) > 0:
    sample_idx = high_risk.index[0]
    # Get the original row ID from tracepipe
    sample_customer = high_risk.loc[sample_idx, "customer_id"]
    print(f"   Customer: {sample_customer} (Risk Score: {high_risk.loc[sample_idx, 'risk_score']})")

    # Find this customer's row ID
    for rid in range(min(n_customers, 100)):
        row = tracepipe.explain(rid)
        if row.is_alive:
            history = row.history()
            if history:
                print(f"   Row {rid} transformations:")
                for h in history[:3]:
                    if h["col"] not in ("__row__", "__position__"):
                        print(
                            f"      → {h['col']}: {h['old_val']} → {h['new_val']} ({h['operation']})"
                        )
            break

# =============================================================================
# Cleanup
# =============================================================================
tracepipe.disable()

print("\n" + "=" * 70)
print("✅ Demo Complete!")
print("=" * 70)
print("\nOpen ml_pipeline_report.html to explore the full lineage interactively.")
print("Try searching for specific row IDs to see their complete transformation history.")

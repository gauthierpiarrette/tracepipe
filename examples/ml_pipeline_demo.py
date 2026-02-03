#!/usr/bin/env python3
"""
TracePipe Demo: ML Preprocessing Pipeline

A realistic example showing how TracePipe helps debug data transformations
in machine learning pipelines.

Scenario: Building a customer churn prediction model.
Question: "Why did customer X get flagged? What happened to their data?"

Run: PYTHONPATH=. python examples/ml_pipeline_demo.py
"""

import numpy as np
import pandas as pd

import tracepipe as tp

np.random.seed(42)

print("=" * 60)
print("TracePipe: Customer Churn ML Pipeline")
print("=" * 60)

# =============================================================================
# STEP 1: Enable TracePipe
# =============================================================================
tp.enable(mode="debug", watch=["tenure_months", "monthly_charges", "churned"])

# =============================================================================
# STEP 2: Load Raw Data (simulating messy real-world data)
# =============================================================================
print("\n📁 Loading raw customer data...")

n_customers = 200

df = pd.DataFrame(
    {
        "customer_id": [f"CUST_{i:04d}" for i in range(n_customers)],
        "tenure_months": np.random.exponential(24, n_customers).astype(int),
        "monthly_charges": np.random.normal(70, 25, n_customers).round(2),
        "contract_type": np.random.choice(
            ["Month-to-month", "One year", "Two year"], n_customers, p=[0.5, 0.3, 0.2]
        ),
        "churned": np.random.choice([0, 1], n_customers, p=[0.73, 0.27]),
    }
)

# Add data quality issues
df.loc[np.random.choice(n_customers, 10, replace=False), "monthly_charges"] = np.nan
df.loc[np.random.choice(n_customers, 15, replace=False), "tenure_months"] = np.nan
df.loc[np.random.choice(n_customers, 5, replace=False), "monthly_charges"] = -50  # Invalid

print(f"Loaded {len(df)} rows")
print(f"Missing values: {df.isna().sum().sum()}")
print(f"Invalid charges: {(df['monthly_charges'] < 0).sum()}")

# =============================================================================
# STEP 3: Data Cleaning
# =============================================================================
print("\n🧹 Cleaning data...")

with tp.stage("cleaning"):
    before = len(df)

    # Remove invalid negative charges
    df = df[df["monthly_charges"] >= 0]
    print(f"  Removed {before - len(df)} rows with invalid charges")

    # Drop rows missing critical values
    before = len(df)
    df = df.dropna(subset=["churned"])
    print(f"  Removed {before - len(df)} rows missing labels")

# =============================================================================
# STEP 4: Feature Engineering
# =============================================================================
print("\n⚙️ Engineering features...")

with tp.stage("features"):
    # Impute missing values
    df["tenure_months"] = df["tenure_months"].fillna(df["tenure_months"].median())
    df["monthly_charges"] = df["monthly_charges"].fillna(df["monthly_charges"].median())

    # Create derived features
    df["total_value"] = df["tenure_months"] * df["monthly_charges"]
    df["is_new_customer"] = (df["tenure_months"] < 6).astype(int)

print(f"\nFinal dataset: {len(df)} rows")

# =============================================================================
# STEP 5: Analyze with TracePipe
# =============================================================================
print("\n" + "=" * 60)
print("📊 TracePipe Analysis")
print("=" * 60)

# Health check
print("\n🔍 Pipeline Health Check:")
result = tp.check(df)
print(result)

# Debug inspector
dbg = tp.debug.inspect()

# Dropped rows
dropped = dbg.dropped_rows()
print(f"\n❌ Dropped {len(dropped)} rows")
for op, count in dbg.dropped_by_operation().items():
    print(f"   • {op}: {count}")

# Trace a surviving row
print("\n📖 Sample row journey:")
if len(df) > 0:
    trace = tp.trace(df, row=0)
    print(trace)

# Why analysis for a specific cell
print("\n❓ Why did a cell change?")
why = tp.why(df, col="tenure_months", row=0)
print(why)

# =============================================================================
# STEP 6: Export
# =============================================================================
print("\n💾 Exporting reports...")
dbg.export("json", "churn_lineage.json")
print("  ✓ JSON: churn_lineage.json")

tp.report(df, "churn_report.html")
print("  ✓ HTML: churn_report.html")

tp.disable()

print("\n" + "=" * 60)
print("✅ Demo complete! Open churn_report.html to explore.")
print("=" * 60)

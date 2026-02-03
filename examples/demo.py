#!/usr/bin/env python
"""
TracePipe Demo - Complete ML Pipeline with Lineage Tracking

Run: python examples/demo.py
"""

import pandas as pd

import tracepipe as tp

# Enable with watched columns
tp.enable(mode="debug", watch=["age", "income", "purchase_amount"])

print("=" * 60)
print("TracePipe v0.4.0 Demo - ML Pipeline Lineage Tracking")
print("=" * 60)

df = pd.DataFrame(
    {
        "user_id": [1, 2, 3, 4, 5],
        "age": [25, None, 35, 45, 28],
        "income": [50000, 60000, None, 80000, 55000],
        "category": ["A", "B", "A", "B", "A"],
        "purchase_amount": [100, 250, None, 400, 150],
    }
)

print("\n📥 Raw data loaded:")
print(df)

with tp.stage("data_cleaning"):
    df = df.fillna(df.mean(numeric_only=True))
    df = df.dropna()
    print(f"\n🧹 After cleaning: {len(df)} rows")

with tp.stage("feature_engineering"):
    df["age_bucket"] = pd.cut(
        df["age"], bins=[0, 30, 50, 100], labels=["young", "middle", "senior"]
    )
    df["income_normalized"] = df["income"] / df["income"].max()
    df["purchase_ratio"] = df["purchase_amount"] / df["income"]
    print(f"\n🔧 Features added: {list(df.columns)}")

with tp.stage("aggregation"):
    summary = (
        df.groupby("category")
        .agg({"income": "mean", "purchase_amount": "sum", "age": "mean"})
        .reset_index()
    )
    print("\n📊 Summary by category:")
    print(summary)

# === Analyze with TracePipe API ===
print(f"\n{'=' * 60}")
print("📈 Lineage Analysis")
print(f"{'=' * 60}")

# Health check
result = tp.check(df)
print(result)

# Debug inspector for detailed info
dbg = tp.debug.inspect()
print(f"\nSteps tracked: {len(dbg.steps)}")
print(f"Total diffs: {dbg.total_diffs}")
print(f"Dropped rows: {dbg.dropped_rows()}")

# Trace a specific row
print("\n📖 Row 0 journey:")
trace = tp.trace(df, row=0)
print(trace)

# Export
print("\n💾 Exporting...")
dbg.export("json", "lineage_export.json")
print("  ✓ JSON exported to lineage_export.json")

tp.report(df, "lineage_report.html")
print("  ✓ HTML report saved to lineage_report.html")

tp.disable()

print("\n" + "=" * 60)
print("Demo complete! Open lineage_report.html to explore the lineage.")
print("=" * 60)

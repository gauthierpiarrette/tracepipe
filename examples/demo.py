#!/usr/bin/env python
"""
TracePipe Demo - Complete ML Pipeline with Lineage Tracking

Run: python examples/demo.py
"""
import tracepipe
import pandas as pd
import numpy as np

tracepipe.enable()

print("=" * 60)
print("TracePipe v1.0.0 Demo - ML Pipeline Lineage Tracking")
print("=" * 60)

df = pd.DataFrame({
    "user_id": [1, 2, 3, 4, 5],
    "age": [25, None, 35, 45, 28],
    "income": [50000, 60000, None, 80000, 55000],
    "category": ["A", "B", "A", "B", "A"],
    "purchase_amount": [100, 250, None, 400, 150],
})

print("\n📥 Raw data loaded:")
print(df)

with tracepipe.stage("data_cleaning"):
    df = df.fillna(df.mean(numeric_only=True))
    df = df.dropna()
    print(f"\n🧹 After cleaning: {len(df)} rows")

with tracepipe.stage("feature_engineering"):
    df["age_bucket"] = pd.cut(df["age"], bins=[0, 30, 50, 100], labels=["young", "middle", "senior"])
    df["income_normalized"] = df["income"] / df["income"].max()
    df["purchase_ratio"] = df["purchase_amount"] / df["income"]
    print(f"\n🔧 Features added: {list(df.columns)}")

with tracepipe.stage("aggregation"):
    summary = df.groupby("category").agg({
        "income": "mean",
        "purchase_amount": "sum",
        "age": "mean"
    }).reset_index()
    print(f"\n📊 Summary by category:")
    print(summary)

lineage = tracepipe.explain()
print(f"\n{'=' * 60}")
print(f"📈 Lineage captured: {len(lineage)} operations")
print(f"{'=' * 60}")

lineage.print_summary()

filepath = lineage.show(open_browser=False)
print(f"\n✅ HTML visualization saved to: {filepath}")

json_output = tracepipe.export_to_json()
print(f"\n📤 JSON export ready ({len(json_output)} chars)")

print("\n" + "=" * 60)
print("Demo complete! Open the HTML file to explore the lineage.")
print("=" * 60)

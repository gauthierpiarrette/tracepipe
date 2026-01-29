#!/usr/bin/env python3
"""
TracePipe Demo - Row-Level Data Lineage

This demo shows:
1. Basic row tracking (filter, transform)
2. Cell-level change tracking
3. Aggregation lineage (groupby)
4. Querying lineage data
5. Export and visualization
"""

import pandas as pd

import tracepipe

print("=" * 60)
print("TracePipe Demo - Row-Level Data Lineage")
print("=" * 60)

# === SETUP ===
tracepipe.enable()
tracepipe.watch("age", "salary")  # Track cell-level changes for these columns

# === CREATE DATA ===
print("\n📊 Creating sample employee dataset...")
df = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "department": ["Engineering", "Sales", "Engineering", "Sales", "Engineering"],
        "age": [28, None, 35, 42, None],
        "salary": [75000, 65000, 85000, None, 70000],
    }
)
print(df)
print(f"Initial shape: {df.shape}")

# === CLEANING STAGE ===
print("\n🧹 Stage 1: Data Cleaning")
with tracepipe.stage("cleaning"):
    # Drop rows with missing values
    df_clean = df.dropna()
    print(f"After dropna: {df_clean.shape}")

    # Fill remaining nulls in specific columns (if any)
    df_clean = df_clean.fillna({"salary": df_clean["salary"].median()})

# === TRANSFORMATION STAGE ===
print("\n🔄 Stage 2: Transformations")
with tracepipe.stage("transforms"):
    # Give everyone a 10% raise
    df_clean["salary"] = df_clean["salary"] * 1.1
    print("Applied 10% salary increase")

    # Add seniority based on age
    df_clean["seniority"] = df_clean["age"].apply(lambda x: "Senior" if x >= 35 else "Junior")

# === AGGREGATION STAGE ===
print("\n📈 Stage 3: Aggregation")
with tracepipe.stage("aggregation"):
    dept_stats = df_clean.groupby("department").agg({"salary": ["mean", "count"], "age": "mean"})
    print("\nDepartment statistics:")
    print(dept_stats)

# === QUERY LINEAGE ===
print("\n🔍 Querying Lineage Data")
print("-" * 40)

# 1. Which rows were dropped?
dropped = tracepipe.dropped_rows()
print(f"\n❌ Dropped row IDs: {dropped}")

# 2. Where were they dropped?
dropped_by_step = tracepipe.dropped_rows(by_step=True)
print(f"Dropped by operation: {dropped_by_step}")

# 3. Examine a specific row's journey
print("\n📖 Row 0 (Alice) journey:")
row0 = tracepipe.explain(0)
print(f"  Status: {'✓ alive' if row0.is_alive else '✗ dropped'}")
print(f"  Events: {len(row0.history())}")
for event in row0.history():
    if event["col"] == "__row__":
        print(f"    Step {event['step_id']}: {event['operation']} - ROW {event['change_type']}")
    elif event["col"] == "__position__":
        print(
            f"    Step {event['step_id']}: {event['operation']} - Position {event['old_val']} → {event['new_val']}"
        )
    else:
        print(
            f"    Step {event['step_id']}: {event['operation']} - {event['col']}: {event['old_val']} → {event['new_val']}"
        )

# 4. What happened to Bob (row 1)?
print("\n📖 Row 1 (Bob) journey:")
row1 = tracepipe.explain(1)
print(f"  Status: {'✓ alive' if row1.is_alive else '✗ dropped at ' + str(row1.dropped_at)}")

# 5. Aggregation group lineage
print("\n👥 Engineering group membership:")
groups = tracepipe.aggregation_groups()
if "Engineering" in groups:
    eng_group = tracepipe.explain_group("Engineering")
    print(f"  Row count: {eng_group.row_count}")
    print(f"  Member row IDs: {eng_group.row_ids}")
    print(f"  Aggregation functions: {eng_group.aggregation_functions}")

# === SUMMARY STATS ===
print("\n📊 TracePipe Statistics:")
stats = tracepipe.stats()
for key, value in stats.items():
    print(f"  {key}: {value}")

# === PIPELINE STEPS ===
print("\n📋 Pipeline Steps:")
for step in tracepipe.steps():
    stage_info = f" [{step['stage']}]" if step["stage"] else ""
    completeness = f" ({step['completeness']})" if step["completeness"] != "FULL" else ""
    print(f"  {step['step_id']}. {step['operation']}{stage_info}{completeness}")

# === EXPORT ===
print("\n💾 Exporting lineage data...")
tracepipe.export_json("lineage_export.json")
print("  ✓ JSON exported to lineage_export.json")

tracepipe.save("lineage_report.html")
print("  ✓ HTML report saved to lineage_report.html")

# === CLEANUP ===
tracepipe.disable()

print("\n" + "=" * 60)
print("Demo complete! Check lineage_report.html for visualization.")
print("=" * 60)

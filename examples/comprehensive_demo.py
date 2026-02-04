#!/usr/bin/env python
"""
TracePipe Comprehensive Demo - E-Commerce Data Pipeline

This demo simulates a realistic e-commerce data processing pipeline,
exercising most TracePipe features:

Features Demonstrated:
- Multiple data sources and merges (inner, left)
- Complex transformations (fillna, replace, astype, cut)
- GroupBy with multiple aggregations
- Chained operations
- Contracts with failure handling
- Full API: enable, check, trace, why, report, snapshot, diff, contract
- Both CI and DEBUG modes
- Pipeline stages
- Ghost row inspection
- Cell history tracking
- Error recovery workflows

Run: python examples/comprehensive_demo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import tracepipe as tp

# ============================================================
# SECTION 1: Setup and Data Generation
# ============================================================


def create_sample_data():
    """Generate realistic e-commerce sample data with intentional issues."""

    # Customers table - some with missing data
    customers = pd.DataFrame(
        {
            "customer_id": [f"C{i:03d}" for i in range(1, 21)],
            "name": [
                "Alice",
                "Bob",
                "Charlie",
                "Diana",
                "Eve",
                "Frank",
                "Grace",
                "Henry",
                "Ivy",
                "Jack",
                "Kate",
                "Liam",
                "Mia",
                "Noah",
                "Olivia",
                "Peter",
                "Quinn",
                "Rose",
                "Sam",
                "Tina",
            ],
            "email": [
                "alice@email.com",
                None,
                "charlie@email.com",
                "diana@email.com",
                "eve@email.com",
                "frank@email.com",
                "grace@email.com",
                None,
                "ivy@email.com",
                "jack@email.com",
                "kate@email.com",
                "liam@email.com",
                "mia@email.com",
                None,
                "olivia@email.com",
                "peter@email.com",
                "quinn@email.com",
                "rose@email.com",
                "sam@email.com",
                "tina@email.com",
            ],
            "region_code": [
                "US-W",
                "US-E",
                "US-W",
                "EU-N",
                "EU-S",
                "US-E",
                "APAC",
                "EU-N",
                "APAC",
                "US-W",
                "EU-S",
                "US-E",
                "APAC",
                None,
                "EU-N",
                "US-W",
                "US-E",
                "APAC",
                "EU-S",
                "US-W",
            ],
            "signup_date": pd.to_datetime(
                [
                    "2023-01-15",
                    "2023-02-20",
                    "2023-01-10",
                    "2023-03-05",
                    "2023-02-28",
                    "2023-04-12",
                    "2023-01-22",
                    "2023-05-18",
                    "2023-03-30",
                    "2023-02-14",
                    "2023-06-01",
                    "2023-04-25",
                    "2023-07-10",
                    "2023-05-05",
                    "2023-03-15",
                    "2023-08-20",
                    "2023-06-30",
                    "2023-04-08",
                    "2023-07-22",
                    "2023-05-28",
                ]
            ),
            "lifetime_value": [
                1500.0,
                2300.0,
                None,
                800.0,
                3500.0,
                1200.0,
                4500.0,
                950.0,
                2800.0,
                None,
                1800.0,
                3200.0,
                2100.0,
                1600.0,
                900.0,
                5000.0,
                750.0,
                2400.0,
                1100.0,
                3800.0,
            ],
            "is_premium": [
                True,
                True,
                False,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                False,
                True,
                True,
                False,
                False,
                True,
                False,
                True,
                False,
                True,
            ],
        }
    )

    # Orders table - multiple orders per customer
    np.random.seed(42)
    orders = pd.DataFrame(
        {
            "order_id": [f"ORD{i:04d}" for i in range(1, 51)],
            "customer_id": np.random.choice([f"C{i:03d}" for i in range(1, 21)], 50),
            "product_id": np.random.choice([f"P{i:02d}" for i in range(1, 11)], 50),
            "quantity": np.random.randint(1, 10, 50),
            "unit_price": np.round(np.random.uniform(10, 500, 50), 2),
            "discount_pct": np.where(
                np.random.random(50) > 0.7, np.round(np.random.uniform(0.05, 0.30, 50), 2), 0.0
            ),
            "order_date": pd.to_datetime(
                [
                    f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
                    for _ in range(50)
                ]
            ),
            "status": np.random.choice(
                ["completed", "completed", "completed", "pending", "cancelled", "refunded"], 50
            ),
        }
    )

    # Products table
    products = pd.DataFrame(
        {
            "product_id": [f"P{i:02d}" for i in range(1, 11)],
            "product_name": [
                "Laptop",
                "Mouse",
                "Keyboard",
                "Monitor",
                "Headphones",
                "Webcam",
                "USB Hub",
                "Desk Lamp",
                "Chair",
                "Desk",
            ],
            "category": [
                "Electronics",
                "Accessories",
                "Accessories",
                "Electronics",
                "Audio",
                "Electronics",
                "Accessories",
                "Home",
                "Furniture",
                "Furniture",
            ],
            "cost": [800, 25, 75, 300, 150, 80, 35, 45, 250, 400],
        }
    )

    # Regions table
    regions = pd.DataFrame(
        {
            "region_code": ["US-W", "US-E", "EU-N", "EU-S", "APAC"],
            "region_name": [
                "US West",
                "US East",
                "Northern Europe",
                "Southern Europe",
                "Asia Pacific",
            ],
            "tax_rate": [0.08, 0.07, 0.20, 0.22, 0.10],
            "currency": ["USD", "USD", "EUR", "EUR", "USD"],
        }
    )

    return customers, orders, products, regions


def print_section(title, emoji="📌"):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"{emoji} {title}")
    print(f"{'=' * 70}")


# ============================================================
# SECTION 2: Main Pipeline with Full Feature Coverage
# ============================================================


def main():
    print_section("TracePipe Comprehensive Demo", "🚀")
    print("Simulating a realistic e-commerce data pipeline...")

    # --------------------------------------------------------
    # FEATURE 1: Enable BEFORE data creation (best practice)
    # --------------------------------------------------------
    print_section("FEATURE 1: Enable Tracking (DEBUG Mode)", "⚡")

    # Enable BEFORE creating DataFrames so they're auto-registered
    # Watch columns that exist early AND get modified/dropped
    tp.enable(mode="debug", watch=["lifetime_value", "email", "discount_pct", "total_amount"])

    dbg = tp.debug.inspect()
    print(f"Mode: {dbg.mode}")
    print(f"Watched columns: {dbg.watched_columns}")

    # Now generate data - DataFrames will be auto-registered!
    customers, orders, products, regions = create_sample_data()

    print("\n📊 Data sources loaded (auto-registered):")
    print(f"   • Customers: {len(customers)} rows")
    print(f"   • Orders: {len(orders)} rows")
    print(f"   • Products: {len(products)} rows")
    print(f"   • Regions: {len(regions)} rows")

    # --------------------------------------------------------
    # FEATURE 2: Snapshot BEFORE processing
    # --------------------------------------------------------
    print_section("FEATURE 2: Snapshot Before Processing", "📸")

    snap_before = tp.snapshot(customers, include_values=True)
    print(f"Snapshot captured: {len(snap_before.row_ids)} rows")
    print(f"Columns tracked: {list(snap_before.column_stats.keys())}")

    # --------------------------------------------------------
    # FEATURE 3: Pipeline Stages
    # --------------------------------------------------------
    print_section("FEATURE 3: Pipeline with Stages", "🔄")

    # STAGE 1: Data Cleaning
    with tp.stage("data_cleaning"):
        print("\n[Stage: data_cleaning]")

        # Track initial state
        print(f"  Before cleaning: {len(customers)} customers")

        # Fill missing lifetime_value with median
        n_null_ltv = customers["lifetime_value"].isna().sum()
        median_ltv = customers["lifetime_value"].median()
        customers["lifetime_value"] = customers["lifetime_value"].fillna(median_ltv)
        print(f"  Filled {n_null_ltv} null lifetime_values with median={median_ltv}")

        # Drop customers without region (can't calculate tax)
        customers = customers.dropna(subset=["region_code"])
        print(f"  After dropping null regions: {len(customers)} customers")

        # Filter only completed orders
        completed_orders = orders[orders["status"] == "completed"]
        print(f"  Completed orders: {len(completed_orders)} / {len(orders)}")

    # STAGE 2: Merge Operations (Multiple merges!)
    with tp.stage("data_enrichment"):
        print("\n[Stage: data_enrichment]")

        # MERGE 1: Orders + Products (inner join)
        order_details = completed_orders.merge(products, on="product_id", how="inner")
        print(f"  Orders + Products (inner): {len(order_details)} rows")

        # MERGE 2: Order Details + Customers (left join - some orders may have invalid customers)
        order_customer = order_details.merge(customers, on="customer_id", how="left")
        print(f"  + Customers (left): {len(order_customer)} rows")

        # MERGE 3: Add Region data (left join)
        enriched = order_customer.merge(regions, on="region_code", how="left")
        print(f"  + Regions (left): {len(enriched)} rows")

    # STAGE 3: Feature Engineering
    with tp.stage("feature_engineering"):
        print("\n[Stage: feature_engineering]")

        # Calculate total amount
        enriched["gross_amount"] = enriched["quantity"] * enriched["unit_price"]
        enriched["discount_amount"] = enriched["gross_amount"] * enriched["discount_pct"]
        enriched["total_amount"] = enriched["gross_amount"] - enriched["discount_amount"]

        # Calculate tax
        enriched["tax_amount"] = enriched["total_amount"] * enriched["tax_rate"].fillna(0)
        enriched["final_amount"] = enriched["total_amount"] + enriched["tax_amount"]

        # Profit margin
        enriched["profit"] = enriched["total_amount"] - (enriched["quantity"] * enriched["cost"])
        enriched["profit_margin"] = enriched["profit"] / enriched["total_amount"]

        # Customer segment based on lifetime value
        enriched["customer_segment"] = pd.cut(
            enriched["lifetime_value"],
            bins=[0, 1000, 2500, 5000, float("inf")],
            labels=["Bronze", "Silver", "Gold", "Platinum"],
        )

        print(
            "  Added columns: gross_amount, discount_amount, total_amount, tax_amount, final_amount, profit, profit_margin, customer_segment"
        )

    # STAGE 4: Data Quality Filtering
    with tp.stage("quality_filtering"):
        print("\n[Stage: quality_filtering]")

        before_filter = len(enriched)

        # Remove negative profit orders (data quality issue)
        enriched = enriched[enriched["profit"] >= 0]
        print(f"  Removed negative profit: {before_filter - len(enriched)} rows")

        # Remove orders with no customer match
        before_customer_filter = len(enriched)
        enriched = enriched.dropna(subset=["name"])
        print(f"  Removed unmatched customers: {before_customer_filter - len(enriched)} rows")

        print(f"  Final dataset: {len(enriched)} rows")

    # STAGE 5: Aggregation
    with tp.stage("aggregation"):
        print("\n[Stage: aggregation]")

        # Customer-level summary
        customer_summary = (
            enriched.groupby("customer_id")
            .agg(
                {"total_amount": ["sum", "mean", "count"], "profit": "sum", "discount_pct": "mean"}
            )
            .reset_index()
        )
        customer_summary.columns = [
            "customer_id",
            "total_revenue",
            "avg_order_value",
            "order_count",
            "total_profit",
            "avg_discount",
        ]
        print(f"  Customer summary: {len(customer_summary)} customers")

        # Category-level summary
        category_summary = (
            enriched.groupby("category")
            .agg({"total_amount": "sum", "profit": "sum", "quantity": "sum"})
            .reset_index()
        )
        print(f"  Category summary: {len(category_summary)} categories")

        # Region-level summary
        region_summary = (
            enriched.groupby("region_name")
            .agg({"final_amount": "sum", "tax_amount": "sum", "order_id": "count"})
            .reset_index()
        )
        region_summary.columns = ["region", "total_sales", "total_tax", "order_count"]
        print(f"  Region summary: {len(region_summary)} regions")

    # --------------------------------------------------------
    # FEATURE 4: Health Check with check()
    # --------------------------------------------------------
    print_section("FEATURE 4: Pipeline Health Check", "🔍")

    check_result = tp.check(enriched)
    print(check_result.to_text(verbose=True))

    # --------------------------------------------------------
    # FEATURE 5: Contracts with Failure Handling
    # --------------------------------------------------------
    print_section("FEATURE 5: Data Quality Contracts", "📋")

    # Contract that should PASS
    print("\n[Contract 1: Should PASS]")
    result_pass = (
        tp.contract()
        .expect_row_count(min_rows=10)
        .expect_columns_exist("customer_id", "total_amount", "profit")
        .expect_no_duplicates()
        .check(enriched)
    )
    print(result_pass)

    # Contract that might FAIL (demonstrating failure handling)
    print("\n[Contract 2: Testing potential failures]")
    result_check = (
        tp.contract()
        .expect_no_nulls("email")  # Some emails are null!
        .expect_retention(min_rate=0.9)  # May fail due to filtering
        .expect_unique("order_id")
        .check(enriched)
    )
    print(result_check)

    # Demonstrate error handling
    if not result_check.passed:
        print("\n⚠️  Contract failed! Handling gracefully...")
        print(f"   Failures: {[f.name for f in result_check.failures]}")
        print("   (In production, you might: log, alert, or take corrective action)")

    # --------------------------------------------------------
    # FEATURE 6: Row Tracing with trace()
    # --------------------------------------------------------
    print_section("FEATURE 6: Row Tracing", "🔎")

    # Trace a dropped row (row 13 was customer C014 with null region_code)
    print("\n[Trace dropped row - customer with null region]")
    trace_dropped = tp.trace(enriched, row=13)  # This row was dropped
    print(trace_dropped.to_text(verbose=True))

    # Trace a row that had lifetime_value modified (row 2 had null -> filled)
    print("\n[Trace row with modified lifetime_value]")
    trace_modified = tp.trace(enriched, row=2)
    print(trace_modified.to_text(verbose=True))

    # Trace by business key - find a customer that made it through
    print("\n[Trace by business key (customer_id)]")
    try:
        # Find customers in the enriched data
        available_customers = enriched["customer_id"].unique()[:3]
        if len(available_customers) > 0:
            cust_id = available_customers[0]
            trace_by_key = tp.trace(enriched, where={"customer_id": cust_id})
            if isinstance(trace_by_key, list):
                print(f"Found {len(trace_by_key)} rows for {cust_id}")
                print(trace_by_key[0].to_text(verbose=False))
            else:
                print(trace_by_key.to_text(verbose=False))
    except ValueError as e:
        print(f"   Customer not found: {e}")

    # --------------------------------------------------------
    # FEATURE 7: Cell Provenance with why()
    # --------------------------------------------------------
    print_section("FEATURE 7: Cell Provenance (why())", "❓")

    # Row 2 (C003) had null lifetime_value that was filled with median
    print("\n[Why does lifetime_value have this value? (row 2 was null -> filled)]")
    try:
        why_ltv = tp.why(enriched, col="lifetime_value", row=2)
        print(why_ltv.to_text(verbose=True))
    except Exception as e:
        print(f"   Could not trace: {e}")

    # Also show discount_pct which exists in original orders
    print("\n[Why does discount_pct have this value?]")
    try:
        why_discount = tp.why(enriched, col="discount_pct", row=0)
        print(why_discount.to_text(verbose=True))
    except Exception as e:
        print(f"   Could not trace: {e}")

    # --------------------------------------------------------
    # FEATURE 8: Ghost Row Inspection (Dropped Rows)
    # --------------------------------------------------------
    print_section("FEATURE 8: Ghost Rows (Dropped Row Values)", "👻")

    # Refresh inspector to get latest state
    dbg = tp.debug.inspect()

    # Ghost rows capture the last values of dropped rows for watched columns
    ghost_df = dbg.ghost_rows(limit=10)
    if not ghost_df.empty:
        print(f"Captured last values of {len(ghost_df)} dropped rows:")
        # Show relevant columns
        display_cols = ["__tp_row_id__", "__tp_dropped_by__"]
        for col in ["lifetime_value", "email", "discount_pct"]:
            if col in ghost_df.columns:
                display_cols.append(col)
        print(ghost_df[display_cols].head(5).to_string())
    else:
        print("No ghost rows captured.")
        print("  Note: Ghost values are captured when watched columns exist at drop time.")
        print(f"  Watched: {dbg.watched_columns}")
        print(f"  Total drops: {len(dbg.dropped_rows())}")

    # --------------------------------------------------------
    # FEATURE 9: Snapshot AFTER and Diff
    # --------------------------------------------------------
    print_section("FEATURE 9: Snapshot Diff", "📊")

    snap_after = tp.snapshot(enriched, include_values=True)
    print(f"Before: {len(snap_before.row_ids)} rows")
    print(f"After: {len(snap_after.row_ids)} rows")

    diff_result = tp.diff(snap_before, snap_after)
    print("\nDiff result:")
    print(diff_result)

    # --------------------------------------------------------
    # FEATURE 10: Debug Inspector Deep Dive
    # --------------------------------------------------------
    print_section("FEATURE 10: Debug Inspector", "🔬")

    print(f"Steps recorded: {len(dbg.steps)}")
    print(f"Total diffs tracked: {dbg.total_diffs}")
    print(f"In-memory diffs: {dbg.in_memory_diffs}")
    print(f"Dropped rows: {len(dbg.dropped_rows())}")

    print("\n[Steps by stage]:")
    for step in dbg.steps[:10]:  # First 10 steps
        stage = step.stage or "no_stage"
        print(f"  {step.step_id:3d}. [{stage:20s}] {step.operation}")
    if len(dbg.steps) > 10:
        print(f"  ... and {len(dbg.steps) - 10} more steps")

    print("\n[Merge statistics]:")
    for stat in dbg.merge_stats():
        print(
            f"  Step {stat['step_id']}: {stat['left_rows']}x{stat['right_rows']} -> {stat['result_rows']} ({stat['how']})"
        )

    print("\n[Aggregation groups]:")
    groups = dbg.aggregation_groups()
    print(f"  {len(groups)} groups tracked")
    if groups:
        sample_group = groups[0]
        group_info = dbg.explain_group(sample_group)
        print(f"  Example: group '{sample_group}' has {group_info.row_count} members")

    # --------------------------------------------------------
    # FEATURE 11: Export Results
    # --------------------------------------------------------
    print_section("FEATURE 11: Export Results", "💾")

    output_dir = Path(".")

    # JSON export
    json_path = output_dir / "comprehensive_lineage.json"
    dbg.export("json", str(json_path))
    print(f"  ✓ JSON lineage: {json_path}")

    # HTML report
    html_path = output_dir / "comprehensive_report.html"
    tp.report(enriched, str(html_path))
    print(f"  ✓ HTML report: {html_path}")

    # Save snapshot for future comparison
    snap_path = output_dir / "pipeline_snapshot.json"
    snap_after.save(str(snap_path))
    print(f"  ✓ Snapshot: {snap_path}")

    # --------------------------------------------------------
    # FEATURE 12: CI Mode Comparison
    # --------------------------------------------------------
    print_section("FEATURE 12: CI Mode vs DEBUG Mode", "⚖️")

    tp.disable()
    tp.reset()

    # Re-run in CI mode
    tp.enable(mode="ci")

    # Quick pipeline
    customers_ci, orders_ci, products_ci, regions_ci = create_sample_data()
    customers_ci = customers_ci.dropna(subset=["region_code"])
    merged_ci = orders_ci.merge(products_ci, on="product_id")

    dbg_ci = tp.debug.inspect()
    ci_stats = dbg_ci.stats()

    print("\nCI Mode Statistics:")
    print(f"  Mode: {ci_stats['mode']}")
    print(f"  Steps: {ci_stats['total_steps']}")
    print(f"  Diffs: {ci_stats['total_diffs']}")
    print(f"  Merge provenance: {ci_stats['features']['merge_provenance']}")
    print(f"  Ghost values: {ci_stats['features']['ghost_row_values']}")
    print(f"  Cell history: {ci_stats['features']['cell_history']}")

    print("\n[CI mode is faster but captures less detail]")
    print("  Use CI mode for production pipelines")
    print("  Use DEBUG mode when investigating issues")

    # --------------------------------------------------------
    # FEATURE 13: Find utility
    # --------------------------------------------------------
    print_section("FEATURE 13: Find Rows by Predicate", "🔍")

    # Use find to locate rows
    print("\n[Find rows where status = 'completed']")
    try:
        rids = tp.debug.find(merged_ci, where={"category": "Electronics"})
        print(f"  Found {len(rids)} rows with category='Electronics'")
        if rids:
            print(f"  Row IDs: {rids[:5]}{'...' if len(rids) > 5 else ''}")
    except Exception as e:
        print(f"  Error: {e}")

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------
    print_section("Demo Complete!", "✅")

    tp.disable()

    print(
        """
Summary of features demonstrated:
  ✓ DEBUG and CI modes
  ✓ Watched columns
  ✓ Pipeline stages
  ✓ Multiple merges (inner, left)
  ✓ Transformations (fillna, dropna, filter)
  ✓ GroupBy aggregations
  ✓ check() health inspection
  ✓ trace() row journey
  ✓ why() cell provenance
  ✓ contract() data quality
  ✓ snapshot() and diff()
  ✓ Ghost row inspection
  ✓ Debug inspector
  ✓ JSON and HTML export
  ✓ find() row lookup

Output files:
  • comprehensive_lineage.json - Raw lineage data
  • comprehensive_report.html - Interactive report
  • pipeline_snapshot.json - Snapshot for regression testing

Open comprehensive_report.html in a browser to explore the lineage visually!
"""
    )


if __name__ == "__main__":
    main()

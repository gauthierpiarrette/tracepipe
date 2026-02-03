# ML Pipeline Example

A complete example of using TracePipe to audit an ML data pipeline.

## The Scenario

You're building a customer churn prediction model. Your pipeline:

1. Loads customer data
2. Cleans missing values
3. Filters valid customers
4. Engineers features
5. Merges with additional data

You need to understand what happens to your training data.

## Full Example

```python
import tracepipe as tp
import pandas as pd
import numpy as np

# Enable debug tracking
tp.enable(mode="debug", watch=["income", "age", "churn_score"])

# =============================================================================
# Stage 1: Load Data
# =============================================================================
tp.stage("load")

customers = pd.DataFrame({
    "customer_id": [f"C-{i}" for i in range(1000)],
    "age": np.random.randint(18, 80, 1000),
    "income": np.random.normal(50000, 20000, 1000),
    "tenure_months": np.random.randint(1, 120, 1000),
    "region": np.random.choice(["US", "EU", "APAC"], 1000),
    "churn_score": np.random.random(1000),
})

# Inject some realistic issues
customers.loc[50:100, "income"] = None
customers.loc[200:220, "age"] = -1  # Invalid ages
customers.loc[300:310, "churn_score"] = None

print(f"Loaded {len(customers)} customers")

# =============================================================================
# Stage 2: Clean Data
# =============================================================================
tp.stage("clean")

# Drop rows with missing income
customers = customers.dropna(subset=["income"])
print(f"After dropna: {len(customers)}")

# Fill missing churn scores with median
median_score = customers["churn_score"].median()
customers["churn_score"] = customers["churn_score"].fillna(median_score)

# Remove invalid ages
customers = customers[customers["age"] > 0]
print(f"After age filter: {len(customers)}")

# =============================================================================
# Stage 3: Feature Engineering
# =============================================================================
tp.stage("features")

# Normalize income
customers["income_normalized"] = (
    customers["income"] - customers["income"].mean()
) / customers["income"].std()

# Age buckets
customers["age_bucket"] = pd.cut(
    customers["age"],
    bins=[0, 25, 35, 50, 65, 100],
    labels=["18-25", "26-35", "36-50", "51-65", "65+"]
)

# Log transform
customers["log_tenure"] = np.log1p(customers["tenure_months"])

# =============================================================================
# Stage 4: Enrich with Region Data
# =============================================================================
tp.stage("enrich")

region_data = pd.DataFrame({
    "region": ["US", "EU", "APAC"],
    "market_size": [1.0, 0.8, 1.2],
    "growth_rate": [0.05, 0.03, 0.08],
})

customers = customers.merge(region_data, on="region", how="left")

# =============================================================================
# Audit the Pipeline
# =============================================================================

print("\n" + "="*60)
print("PIPELINE AUDIT")
print("="*60)

# Health check
result = tp.check(customers)
print(result)

# Contract validation
print("\n--- Contract Check ---")
contract_result = (tp.contract()
    .expect_no_nulls(["customer_id", "income", "churn_score"])
    .expect_unique("customer_id")
    .expect_retention(min_rate=0.8)
    .expect_range("age", min_val=0, max_val=150)
    .check(customers))

print(contract_result)

# Trace a specific customer
print("\n--- Customer Journey ---")
trace = tp.trace(customers, where={"customer_id": "C-55"})
print(trace)

# Why did churn_score change?
print("\n--- Cell Provenance ---")
why = tp.why(customers, col="churn_score", where={"customer_id": "C-305"})
print(why)

# Generate report
tp.report(customers, "ml_pipeline_audit.html", title="Churn Model Pipeline Audit")
print("\n✓ Report saved to ml_pipeline_audit.html")

# Debug inspection
print("\n--- Debug Stats ---")
dbg = tp.debug.inspect()
stats = dbg.stats()
print(f"Steps recorded: {stats['steps_recorded']}")
print(f"Rows dropped: {stats['rows_dropped']}")

# Export lineage
dbg.export("json", "ml_pipeline_lineage.json")
print("✓ Lineage exported to ml_pipeline_lineage.json")

tp.disable()
```

## Expected Output

```
Loaded 1000 customers
After dropna: 949
After age filter: 928

============================================================
PIPELINE AUDIT
============================================================
TracePipe Check: [OK] Pipeline healthy
  Mode: debug

Retention: 928/1000 (92.8%)
Dropped: 72 rows
  • DataFrame.dropna: 51
  • DataFrame.__getitem__[mask]: 21

Value changes: 11 cells modified
  • DataFrame.fillna: 11 (churn_score)

--- Contract Check ---
Contract: [PASSED] All 5 expectations met
  ✓ no_nulls(customer_id, income, churn_score)
  ✓ unique(customer_id)
  ✓ retention >= 80.0%
  ✓ range(age): 0 <= x <= 150

--- Customer Journey ---
Row 55 Journey:
  Status: [DROPPED]
  Dropped by: DataFrame.dropna (step 2)

  Events: 1
    [DROPPED] DataFrame.dropna

--- Cell Provenance ---
Cell History: row 294, column 'churn_score'
  Current value: 0.4821
  [i] Was null at step 1 (later recovered)
      by: DataFrame.fillna

  History (1 change):
    None -> 0.4821
      by: DataFrame.fillna

✓ Report saved to ml_pipeline_audit.html

--- Debug Stats ---
Steps recorded: 12
Rows dropped: 72
✓ Lineage exported to ml_pipeline_lineage.json
```

## Key Takeaways

1. **Visibility**: You can see exactly where rows are dropped and why
2. **Null Recovery**: TracePipe flags when nulls are filled, showing the imputation
3. **Contract Validation**: Ensure your training data meets quality standards
4. **Reproducibility**: Export lineage for documentation and debugging

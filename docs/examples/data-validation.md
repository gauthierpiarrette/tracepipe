# Data Validation Example

Using TracePipe contracts for data quality validation in CI/CD.

## The Scenario

You have an ETL pipeline that runs daily. You need to:

1. Validate incoming data meets expectations
2. Ensure transformations don't drop too many rows
3. Gate deployments on data quality
4. Alert on anomalies

## CI/CD Integration

### Basic Validation Script

```python
#!/usr/bin/env python3
"""validate_pipeline.py - Run as part of CI/CD"""

import sys
import tracepipe as tp
import pandas as pd

def main():
    tp.enable(mode="ci")  # Lightweight for CI

    # Load and process data
    df = pd.read_csv("data/daily_extract.csv")
    df = df.dropna(subset=["id", "amount"])
    df = df[df["amount"] > 0]
    df = df.drop_duplicates(subset=["id"])

    # Define contract
    result = (tp.contract()
        # Schema validation
        .expect_columns(["id", "amount", "timestamp", "category"])
        .expect_dtypes({
            "id": "object",
            "amount": "float64",
        })

        # Data quality
        .expect_unique("id")
        .expect_no_nulls(["id", "amount"])
        .expect_range("amount", min_val=0)
        .expect_values("category", ["A", "B", "C", "D"])

        # Pipeline health
        .expect_retention(min_rate=0.9)  # At least 90% retained

        .check(df))

    # Report results
    print(result)

    if not result.passed:
        print("\n❌ Data validation FAILED")
        print("\nFailures:")
        for failure in result.failures:
            print(f"  • {failure}")
        sys.exit(1)

    print("\n✅ Data validation PASSED")
    tp.disable()
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### GitHub Actions Workflow

```yaml
# .github/workflows/data-validation.yml
name: Data Validation

on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM
  workflow_dispatch:

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install tracepipe pandas

      - name: Download data
        run: ./scripts/download_daily_data.sh

      - name: Validate pipeline
        run: python validate_pipeline.py

      - name: Upload lineage report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: lineage-report
          path: pipeline_audit.html
```

## Multi-Stage Validation

### Checkpoint Validation

```python
import tracepipe as tp
import pandas as pd

def validate_checkpoint(df, checkpoint_name, contract):
    """Validate a pipeline checkpoint"""
    result = contract.check(df)

    if not result.passed:
        raise ValueError(f"Checkpoint '{checkpoint_name}' failed: {result.failures}")

    print(f"✓ Checkpoint '{checkpoint_name}' passed")
    return df

tp.enable(mode="ci")

# Stage 1: Raw data
df = pd.read_csv("raw_data.csv")
validate_checkpoint(df, "raw_load",
    tp.contract()
        .expect_columns(["id", "value", "timestamp"]))

# Stage 2: Cleaned data
tp.stage("clean")
df = df.dropna()
df = df[df["value"] > 0]
validate_checkpoint(df, "cleaned",
    tp.contract()
        .expect_no_nulls(["id", "value"])
        .expect_retention(min_rate=0.95))

# Stage 3: Enriched data
tp.stage("enrich")
df = df.merge(lookup_table, on="id", how="left")
validate_checkpoint(df, "enriched",
    tp.contract()
        .expect_columns(["id", "value", "category", "region"]))

# Final validation
result = tp.contract().expect_retention(min_rate=0.9).check(df)
result.raise_if_failed()

tp.disable()
```

## Alerting on Anomalies

### Slack Integration

```python
import tracepipe as tp
import pandas as pd
import requests

SLACK_WEBHOOK = "https://hooks.slack.com/services/..."

def send_slack_alert(message):
    requests.post(SLACK_WEBHOOK, json={"text": message})

def run_pipeline_with_alerting():
    tp.enable(mode="ci")

    df = pd.read_csv("data.csv")
    df = df.dropna()
    df = df[df["status"] == "active"]

    result = tp.check(df)

    # Alert on low retention
    if result.retention < 0.8:
        send_slack_alert(
            f"⚠️ Pipeline Alert: Low retention ({result.retention:.1%})\n"
            f"Dropped {result.n_dropped} rows"
        )

    # Alert on contract failures
    contract_result = (tp.contract()
        .expect_retention(min_rate=0.9)
        .expect_unique("id")
        .check(df))

    if not contract_result.passed:
        failures = "\n".join(f"• {f}" for f in contract_result.failures)
        send_slack_alert(
            f"🚨 Data Contract Violation!\n{failures}"
        )

    tp.disable()
    return df
```

## Regression Testing

### Baseline Comparison

```python
import tracepipe as tp
import pandas as pd
import json

def save_baseline(df, path="baseline_stats.json"):
    """Save current stats as baseline"""
    result = tp.check(df)
    baseline = {
        "retention": result.retention,
        "n_rows": len(df),
        "n_dropped": result.n_dropped,
        "drops_by_op": result.drops_by_op,
    }
    with open(path, "w") as f:
        json.dump(baseline, f)
    print(f"Baseline saved to {path}")

def compare_to_baseline(df, baseline_path="baseline_stats.json"):
    """Compare current run to baseline"""
    with open(baseline_path) as f:
        baseline = json.load(f)

    result = tp.check(df)

    # Check for regressions
    issues = []

    retention_diff = result.retention - baseline["retention"]
    if retention_diff < -0.05:  # 5% regression threshold
        issues.append(
            f"Retention dropped: {baseline['retention']:.1%} → {result.retention:.1%}"
        )

    row_diff = len(df) - baseline["n_rows"]
    if abs(row_diff) > baseline["n_rows"] * 0.1:  # 10% change
        issues.append(
            f"Row count changed significantly: {baseline['n_rows']} → {len(df)}"
        )

    if issues:
        print("⚠️ Regressions detected:")
        for issue in issues:
            print(f"  • {issue}")
        return False

    print("✓ No regressions detected")
    return True
```

## Best Practices

### 1. Use CI Mode in Production

```python
# Production: lightweight tracking
tp.enable(mode="ci")

# Development: full debugging
tp.enable(mode="debug", watch=["important_col"])
```

### 2. Set Appropriate Thresholds

```python
# Strict for critical pipelines
.expect_retention(min_rate=0.99)

# Lenient for exploratory analysis
.expect_retention(min_rate=0.5)
```

### 3. Layer Contracts

```python
# Schema first (fast fail)
tp.contract().expect_columns([...]).check(df).raise_if_failed()

# Then data quality
tp.contract().expect_unique(...).expect_no_nulls(...).check(df).raise_if_failed()

# Finally pipeline health
tp.contract().expect_retention(...).check(df).raise_if_failed()
```

### 4. Document Expectations

```python
# Clear contract with comments
contract = (tp.contract()
    .expect_unique("order_id")        # Business rule: no duplicate orders
    .expect_no_nulls("customer_id")   # Required for downstream joins
    .expect_retention(min_rate=0.95)  # Historical average is 97%
)
```

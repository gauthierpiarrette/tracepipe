# Data Contracts

Define and validate expectations on your pipeline output.

## Basic Usage

```python
result = (tp.contract()
    .expect_unique("customer_id")
    .expect_no_nulls("email")
    .expect_retention(min_rate=0.8)
    .check(df))

print(result)
```

Output (if passing):

```
Contract: [PASSED] All 3 expectations met
  ✓ unique(customer_id)
  ✓ no_nulls(email)
  ✓ retention >= 80.0%
```

Output (if failing):

```
Contract: [FAILED] 1 of 3 expectations failed
  ✓ unique(customer_id)
  ✗ no_nulls(email): 5 nulls found
  ✓ retention >= 80.0%
```

## The ContractResult Object

```python
result = tp.contract().expect_unique("id").check(df)

# Access fields
result.passed          # bool: all expectations met
result.expectations    # list[Expectation]: all expectations
result.failures        # list[Expectation]: failed expectations

# Raise on failure
result.raise_if_failed()  # Raises ContractViolationError
```

## Available Expectations

### `expect_unique(column)`

Ensures no duplicate values in a column:

```python
.expect_unique("order_id")
.expect_unique("email")
```

### `expect_no_nulls(column)`

Ensures no null values:

```python
.expect_no_nulls("customer_id")
.expect_no_nulls(["name", "email"])  # Multiple columns
```

### `expect_retention(min_rate)`

Ensures minimum row retention:

```python
.expect_retention(min_rate=0.9)   # At least 90% retained
.expect_retention(min_rate=0.5)   # At least 50% retained
```

### `expect_no_drops()`

Ensures no rows were dropped:

```python
.expect_no_drops()  # Fails if any row was dropped
```

### `expect_columns(columns)`

Ensures specific columns exist:

```python
.expect_columns(["id", "name", "email"])
```

### `expect_dtypes(dtypes)`

Ensures column data types:

```python
.expect_dtypes({
    "id": "int64",
    "price": "float64",
    "name": "object"
})
```

### `expect_range(column, min_val, max_val)`

Ensures values are within a range:

```python
.expect_range("age", min_val=0, max_val=150)
.expect_range("price", min_val=0)  # Just minimum
```

### `expect_values(column, allowed)`

Ensures values are from an allowed set:

```python
.expect_values("status", ["active", "inactive", "pending"])
.expect_values("country", ["US", "CA", "UK", "DE"])
```

## Chaining Expectations

Build complex contracts with method chaining:

```python
contract = (tp.contract()
    # Schema validation
    .expect_columns(["id", "email", "status", "amount"])
    .expect_dtypes({"amount": "float64"})

    # Data quality
    .expect_unique("id")
    .expect_no_nulls(["id", "email"])
    .expect_values("status", ["active", "inactive"])
    .expect_range("amount", min_val=0)

    # Pipeline health
    .expect_retention(min_rate=0.8)
)

result = contract.check(df)
```

## Using in CI/CD

```python
import sys

result = (tp.contract()
    .expect_unique("id")
    .expect_retention(min_rate=0.9)
    .check(df))

if not result.passed:
    print("Data contract violated!")
    for failure in result.failures:
        print(f"  ✗ {failure}")
    sys.exit(1)
```

Or use the exception-based approach:

```python
try:
    (tp.contract()
        .expect_unique("id")
        .expect_retention(min_rate=0.9)
        .check(df)
        .raise_if_failed())
except tp.ContractViolationError as e:
    print(f"Contract failed: {e}")
    sys.exit(1)
```

## Custom Expectations

For custom validation logic:

```python
def validate_email_format(df):
    """Check that all emails contain @"""
    invalid = df[~df["email"].str.contains("@", na=False)]
    if len(invalid) > 0:
        return False, f"{len(invalid)} invalid emails"
    return True, None

# Use with expect_custom (if available) or validate manually
result = tp.contract().expect_no_nulls("email").check(df)
if result.passed:
    valid, msg = validate_email_format(df)
    if not valid:
        print(f"Custom validation failed: {msg}")
```

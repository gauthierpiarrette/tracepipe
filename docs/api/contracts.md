# Contracts API

Data quality contracts for validation.

## Creating Contracts

### contract

```python
tp.contract() -> ContractBuilder
```

Create a new contract builder.

**Returns:** `ContractBuilder` with chainable methods.

**Example:**

```python
result = (tp.contract()
    .expect_unique("id")
    .expect_no_nulls("email")
    .check(df))
```

---

## Contract Methods

### expect_unique

```python
.expect_unique(column: str) -> ContractBuilder
```

Expect no duplicate values in column.

**Example:**

```python
.expect_unique("order_id")
```

---

### expect_no_nulls

```python
.expect_no_nulls(columns: str | list[str]) -> ContractBuilder
```

Expect no null values.

**Example:**

```python
.expect_no_nulls("customer_id")
.expect_no_nulls(["name", "email"])
```

---

### expect_retention

```python
.expect_retention(min_rate: float) -> ContractBuilder
```

Expect minimum row retention rate.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `min_rate` | `float` | Minimum retention (0-1) |

**Example:**

```python
.expect_retention(min_rate=0.9)  # At least 90%
```

---

### expect_no_drops

```python
.expect_no_drops() -> ContractBuilder
```

Expect no rows were dropped.

**Example:**

```python
.expect_no_drops()
```

---

### expect_columns

```python
.expect_columns(columns: list[str]) -> ContractBuilder
```

Expect columns to exist.

**Example:**

```python
.expect_columns(["id", "name", "email"])
```

---

### expect_dtypes

```python
.expect_dtypes(dtypes: dict[str, str]) -> ContractBuilder
```

Expect column data types.

**Example:**

```python
.expect_dtypes({
    "id": "int64",
    "price": "float64",
})
```

---

### expect_range

```python
.expect_range(
    column: str,
    min_val: float | None = None,
    max_val: float | None = None,
) -> ContractBuilder
```

Expect values within range.

**Example:**

```python
.expect_range("age", min_val=0, max_val=150)
.expect_range("price", min_val=0)  # No max
```

---

### expect_values

```python
.expect_values(column: str, allowed: list[Any]) -> ContractBuilder
```

Expect values from allowed set.

**Example:**

```python
.expect_values("status", ["active", "inactive"])
```

---

### check

```python
.check(df: pd.DataFrame) -> ContractResult
```

Execute the contract against a DataFrame.

**Returns:** `ContractResult`

| Attribute | Type | Description |
|-----------|------|-------------|
| `.passed` | `bool` | All expectations met |
| `.expectations` | `list` | All expectations |
| `.failures` | `list` | Failed expectations |

---

## ContractResult Methods

### raise_if_failed

```python
result.raise_if_failed() -> None
```

Raise `ContractViolationError` if contract failed.

**Example:**

```python
result = tp.contract().expect_unique("id").check(df)
result.raise_if_failed()  # Raises if duplicates found
```

---

## Complete Example

```python
import tracepipe as tp

tp.enable(mode="ci")

df = process_pipeline(raw_data)

# Define and check contract
result = (tp.contract()
    # Schema
    .expect_columns(["id", "email", "amount"])
    .expect_dtypes({"amount": "float64"})

    # Quality
    .expect_unique("id")
    .expect_no_nulls(["id", "email"])
    .expect_range("amount", min_val=0)
    .expect_values("status", ["pending", "complete", "failed"])

    # Pipeline health
    .expect_retention(min_rate=0.8)

    .check(df))

# Handle result
if result.passed:
    print("✓ All contracts passed")
else:
    print("✗ Contract violations:")
    for failure in result.failures:
        print(f"  - {failure}")
    result.raise_if_failed()
```

# tracepipe/contracts.py
"""
Data quality contracts for TracePipe pipelines.

Contracts provide a fluent API for defining data quality expectations
that can be checked against pipeline output.

Usage:
    result = (tp.contract()
        .expect_merge_expansion(max_ratio=2.0)
        .expect_retention(min_rate=0.9)
        .expect_no_nulls("user_id", "email")
        .expect_unique("transaction_id")
        .check(df))

    # Fail fast
    result.raise_if_failed()
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import pandas as pd

from .context import get_context


@dataclass
class ExpectationResult:
    """Result of a single expectation check."""

    name: str
    passed: bool
    actual_value: Any
    expected: str
    message: str


@dataclass
class ContractResult:
    """Result of contract validation."""

    passed: bool
    expectations: list[ExpectationResult]

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Contract {status}"]
        for exp in self.expectations:
            mark = "[OK]" if exp.passed else "[FAIL]"
            lines.append(f"  {mark} {exp.name}: {exp.message}")
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        """Raise ContractViolation if any expectation failed."""
        if not self.passed:
            failed = [e for e in self.expectations if not e.passed]
            raise ContractViolation(failed)

    @property
    def failures(self) -> list[ExpectationResult]:
        """Get list of failed expectations."""
        return [e for e in self.expectations if not e.passed]

    def to_dict(self) -> dict:
        """Export to dictionary."""
        return {
            "passed": self.passed,
            "expectations": [
                {
                    "name": e.name,
                    "passed": e.passed,
                    "actual_value": e.actual_value,
                    "expected": e.expected,
                    "message": e.message,
                }
                for e in self.expectations
            ],
        }


class ContractViolation(Exception):
    """Raised when contract expectations fail."""

    def __init__(self, failures: list[ExpectationResult]):
        self.failures = failures
        messages = [f"{f.name}: {f.message}" for f in failures]
        super().__init__(f"Contract violated: {'; '.join(messages)}")


class ContractBuilder:
    """
    Fluent API for defining data quality contracts.

    Usage:
        result = (tp.contract()
            .expect_merge_expansion(max_ratio=2.0)
            .expect_retention(min_rate=0.9)
            .expect_no_nulls("user_id", "email")
            .expect_unique("transaction_id")
            .check(df))
    """

    def __init__(self):
        self._expectations: list[Callable[[pd.DataFrame], ExpectationResult]] = []

    def expect_merge_expansion(self, max_ratio: float = 2.0) -> "ContractBuilder":
        """Fail if any merge expanded rows beyond ratio."""

        def check(df: pd.DataFrame) -> ExpectationResult:
            ctx = get_context()
            stats = ctx.store.get_merge_stats()

            # Extract MergeStats objects from (step_id, stats) tuples
            violations = [s for _, s in stats if s.expansion_ratio > max_ratio]

            if violations:
                worst = max(violations, key=lambda s: s.expansion_ratio)
                return ExpectationResult(
                    name="merge_expansion",
                    passed=False,
                    actual_value=worst.expansion_ratio,
                    expected=f"<= {max_ratio}",
                    message=f"Merge expanded {worst.expansion_ratio:.1f}x (max: {max_ratio}x)",
                )

            max_actual = max((s.expansion_ratio for _, s in stats), default=0)
            return ExpectationResult(
                name="merge_expansion",
                passed=True,
                actual_value=max_actual,
                expected=f"<= {max_ratio}",
                message=f"All merges within {max_ratio}x expansion limit",
            )

        self._expectations.append(check)
        return self

    def expect_retention(self, min_rate: float = 0.8) -> "ContractBuilder":
        """Fail if too many rows were dropped."""

        def check(df: pd.DataFrame) -> ExpectationResult:
            ctx = get_context()
            dropped = len(ctx.store.get_dropped_rows())
            current = len(df)

            # Estimate original count: use max input rows seen across all steps
            # This handles multi-table pipelines where merges can expand rows
            max_input_rows = 0
            for step in ctx.store.steps:
                # input_shape can be a single shape tuple (rows, cols) or
                # a tuple of shapes for merge operations
                if step.input_shape:
                    shape = step.input_shape
                    if isinstance(shape[0], tuple):
                        # Multiple inputs (e.g., merge) - take max of all inputs
                        for s in shape:
                            if isinstance(s, tuple) and len(s) >= 1:
                                max_input_rows = max(max_input_rows, s[0])
                    elif isinstance(shape[0], int):
                        max_input_rows = max(max_input_rows, shape[0])

                if step.output_shape and isinstance(step.output_shape[0], int):
                    max_input_rows = max(max_input_rows, step.output_shape[0])

            # Fall back to current + dropped if no steps recorded
            if max_input_rows == 0:
                max_input_rows = current + dropped

            # Retention = final rows / peak rows seen
            # This gives a sensible answer for multi-table pipelines
            retention = current / max_input_rows if max_input_rows > 0 else 1.0

            if retention < min_rate:
                return ExpectationResult(
                    name="retention",
                    passed=False,
                    actual_value=retention,
                    expected=f">= {min_rate}",
                    message=f"Retention {retention:.1%} below minimum {min_rate:.1%}",
                )

            return ExpectationResult(
                name="retention",
                passed=True,
                actual_value=retention,
                expected=f">= {min_rate}",
                message=f"Retention {retention:.1%} meets minimum {min_rate:.1%}",
            )

        self._expectations.append(check)
        return self

    def expect_no_nulls(self, *columns: str) -> "ContractBuilder":
        """Fail if specified columns contain nulls."""

        def check(df: pd.DataFrame) -> ExpectationResult:
            null_cols = []
            for col in columns:
                if col in df.columns and df[col].isna().any():
                    null_count = df[col].isna().sum()
                    null_cols.append(f"{col}({null_count})")

            if null_cols:
                return ExpectationResult(
                    name="no_null",
                    passed=False,
                    actual_value=null_cols,
                    expected="no nulls",
                    message=f"Nulls found in: {', '.join(null_cols)}",
                )

            return ExpectationResult(
                name="no_null",
                passed=True,
                actual_value=[],
                expected="no nulls",
                message=f"No nulls in {', '.join(columns)}",
            )

        self._expectations.append(check)
        return self

    # Alias for backwards compatibility
    expect_no_null_in = expect_no_nulls

    def expect_unique(self, *columns: str) -> "ContractBuilder":
        """Fail if columns have duplicate values."""

        def check(df: pd.DataFrame) -> ExpectationResult:
            cols = [c for c in columns if c in df.columns]
            if not cols:
                return ExpectationResult(
                    name="unique",
                    passed=True,
                    actual_value=0,
                    expected="unique",
                    message="Columns not present",
                )

            dup_count = df.duplicated(subset=cols).sum()

            if dup_count > 0:
                return ExpectationResult(
                    name="unique",
                    passed=False,
                    actual_value=dup_count,
                    expected="0 duplicates",
                    message=f"{dup_count} duplicate rows on {cols}",
                )

            return ExpectationResult(
                name="unique",
                passed=True,
                actual_value=0,
                expected="0 duplicates",
                message=f"All rows unique on {cols}",
            )

        self._expectations.append(check)
        return self

    def expect_row_count(
        self, min_rows: int = 0, max_rows: Optional[int] = None
    ) -> "ContractBuilder":
        """Fail if row count outside bounds."""

        def check(df: pd.DataFrame) -> ExpectationResult:
            n = len(df)

            if n < min_rows:
                return ExpectationResult(
                    name="row_count",
                    passed=False,
                    actual_value=n,
                    expected=f">= {min_rows}",
                    message=f"Only {n} rows, minimum is {min_rows}",
                )

            if max_rows is not None and n > max_rows:
                return ExpectationResult(
                    name="row_count",
                    passed=False,
                    actual_value=n,
                    expected=f"<= {max_rows}",
                    message=f"{n} rows exceeds maximum {max_rows}",
                )

            max_str = str(max_rows) if max_rows is not None else "inf"
            return ExpectationResult(
                name="row_count",
                passed=True,
                actual_value=n,
                expected=f"{min_rows}-{max_str}",
                message=f"{n} rows within bounds",
            )

        self._expectations.append(check)
        return self

    def expect_columns_exist(self, *columns: str) -> "ContractBuilder":
        """Fail if any specified columns are missing."""

        def check(df: pd.DataFrame) -> ExpectationResult:
            missing = [c for c in columns if c not in df.columns]

            if missing:
                return ExpectationResult(
                    name="columns_exist",
                    passed=False,
                    actual_value=missing,
                    expected="all present",
                    message=f"Missing columns: {', '.join(missing)}",
                )

            return ExpectationResult(
                name="columns_exist",
                passed=True,
                actual_value=[],
                expected="all present",
                message=f"All {len(columns)} columns present",
            )

        self._expectations.append(check)
        return self

    def expect_no_duplicates(self) -> "ContractBuilder":
        """Fail if DataFrame has duplicate rows."""

        def check(df: pd.DataFrame) -> ExpectationResult:
            dup_count = df.duplicated().sum()

            if dup_count > 0:
                return ExpectationResult(
                    name="no_duplicates",
                    passed=False,
                    actual_value=dup_count,
                    expected="0",
                    message=f"{dup_count} duplicate rows",
                )

            return ExpectationResult(
                name="no_duplicates",
                passed=True,
                actual_value=0,
                expected="0",
                message="No duplicate rows",
            )

        self._expectations.append(check)
        return self

    def expect_dtype(self, column: str, expected_dtype: str) -> "ContractBuilder":
        """Fail if column dtype doesn't match expected."""

        def check(df: pd.DataFrame) -> ExpectationResult:
            if column not in df.columns:
                return ExpectationResult(
                    name="dtype",
                    passed=False,
                    actual_value="column missing",
                    expected=expected_dtype,
                    message=f"Column '{column}' not found",
                )

            actual = str(df[column].dtype)

            # Allow partial matches (e.g., "int" matches "int64")
            if expected_dtype in actual or actual in expected_dtype:
                return ExpectationResult(
                    name="dtype",
                    passed=True,
                    actual_value=actual,
                    expected=expected_dtype,
                    message=f"Column '{column}' dtype is {actual}",
                )

            return ExpectationResult(
                name="dtype",
                passed=False,
                actual_value=actual,
                expected=expected_dtype,
                message=f"Column '{column}' dtype is {actual}, expected {expected_dtype}",
            )

        self._expectations.append(check)
        return self

    def expect_values_in(self, column: str, allowed_values: list) -> "ContractBuilder":
        """Fail if column contains values not in allowed set."""

        def check(df: pd.DataFrame) -> ExpectationResult:
            if column not in df.columns:
                return ExpectationResult(
                    name="values_in",
                    passed=False,
                    actual_value="column missing",
                    expected=f"in {allowed_values}",
                    message=f"Column '{column}' not found",
                )

            # Get unique values not in allowed set
            actual_values = df[column].dropna().unique()
            invalid = [v for v in actual_values if v not in allowed_values]

            if invalid:
                # Show up to 5 invalid values
                invalid_str = str(invalid[:5])
                if len(invalid) > 5:
                    invalid_str += f"... ({len(invalid)} total)"
                return ExpectationResult(
                    name="values_in",
                    passed=False,
                    actual_value=invalid_str,
                    expected=f"in {allowed_values}",
                    message=f"Column '{column}' has invalid values: {invalid_str}",
                )

            return ExpectationResult(
                name="values_in",
                passed=True,
                actual_value=list(actual_values),
                expected=f"in {allowed_values}",
                message=f"All values in '{column}' are valid",
            )

        self._expectations.append(check)
        return self

    def expect(
        self,
        predicate: Callable[[pd.DataFrame], bool],
        name: str,
        message: str = "",
    ) -> "ContractBuilder":
        """Add custom expectation."""

        def check(df: pd.DataFrame) -> ExpectationResult:
            try:
                passed = predicate(df)
            except Exception as e:
                message_with_error = f"{message}: {e}" if message else str(e)
                return ExpectationResult(
                    name=name,
                    passed=False,
                    actual_value=str(e),
                    expected="pass",
                    message=message_with_error,
                )

            return ExpectationResult(
                name=name,
                passed=passed,
                actual_value=passed,
                expected="pass",
                message=message if message else ("passed" if passed else "failed"),
            )

        self._expectations.append(check)
        return self

    def check(self, df: pd.DataFrame) -> ContractResult:
        """Run all expectations and return result."""
        results = [exp(df) for exp in self._expectations]
        return ContractResult(
            passed=all(r.passed for r in results),
            expectations=results,
        )


def contract() -> ContractBuilder:
    """Create a new contract builder."""
    return ContractBuilder()

# Changelog

All notable changes to TracePipe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.3.5 - 2026-02-03

### Fixed
- **DataFrame.fillna double-logging**: `df.fillna({"col": 0})` now logs exactly 1 event
  - Previously logged both `DataFrame.fillna` and internal `__setitem__` for same change
  - Added `wrap_pandas_transform_method` with `_in_transform_op` flag to suppress nested setitem
  - Works for both `fillna` and `replace` operations, including `inplace=True`

### Added
- Known Limitations section in README documenting concat/dedup tracking gaps
- Test for `DataFrame.fillna` single-event logging

## 0.3.4 - 2026-02-03

### Fixed
- **Event deduplication**: Identical events from parallel pipelines are now deduplicated
  - When multiple DataFrames share row IDs (e.g., from `df.copy()`), same changes are recorded once
  - Events deduplicated by `(col, old_val, new_val, operation)` signature
  - Prevents "4 events" when only 1 logical change occurred

### Added
- `_stable_repr()` helper for robust value comparison in deduplication
- Tests for cross-pipeline event deduplication behavior

## 0.3.3 - 2026-02-03

### Fixed
- **Double-logging bug**: `df['col'] = df['col'].fillna()` now logs exactly one event, not two
  - Fixed duplicate capture from both `_wrap_setitem` and `wrap_series_assignment`
- **Merge warning scoping**: `tp.check(df)` now only shows warnings for merges in df's lineage
  - Previously showed warnings from ALL merges in the session (cross-contamination)
  - Now filters by tracking which merge steps produced the queried DataFrame's rows

### Added
- `_get_merge_stats_for_df()` helper to scope merge warnings to df's lineage
- Tests for double-logging prevention and merge warning scoping

## 0.3.2 - 2026-02-03

### Fixed
- Merge duplicate key warnings now correctly identify which table (left/right) has duplicates
- Previously `right_dup_rate` was mislabeled as "Right table" when it actually indicates LEFT table duplicates

## 0.3.1 - 2026-02-03

### Fixed
- Cell history now correctly chains through merge operations via lineage traversal
- `tp.why()` and `tp.trace()` show pre-merge changes for post-merge rows
- `enable()` resets accumulated state when called multiple times (fixes duplicate warnings in notebooks/IDEs)

### Added
- `get_row_history_with_lineage()` and `get_cell_history_with_lineage()` methods for lineage-aware queries
- `follow_lineage` parameter in `explain_value()` for opt-out of lineage traversal
- Integration tests for cell provenance through merge operations

## 0.3.0 - 2026-02-03

### Added
- MkDocs documentation site with Material theme
- Comprehensive API reference documentation
- Getting started guides and tutorials
- `tp.register()` API for manually registering DataFrames created before `enable()`
- Configurable retention threshold in `tp.check()`
- Ghost row capture for fallback filter paths
- Comprehensive test coverage for COLUMN identity mode
- Data quality contracts with fluent API (`tp.contract().expect_*()`)
- HTML report generation with `tp.report()`
- Snapshot and diff functionality
- Debug mode with cell-level tracking
- `tp.why()` for cell provenance
- `tp.trace()` for row journey
- Watched columns for selective tracking
- Ghost values capture
- Basic row-level lineage tracking
- Support for filter operations (dropna, query, boolean indexing)
- Support for transform operations (fillna, replace, setitem)
- Support for merge and join operations
- CI and Debug modes

### Fixed
- Recursion bug when accessing hidden `__tracepipe_row_id__` column in COLUMN mode
- Config propagation to `row_manager` and `store` components in `enable()`
- Retention rate calculation for multi-table pipelines with merges
- Export wrappers (`to_csv`, `to_parquet`) now correctly strip hidden column
- `_filter_op_depth` cleanup in error scenarios

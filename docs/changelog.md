# Changelog

All notable changes to TracePipe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.2] - 2026-02-04

### Fixed
- **`CheckResult` change tracking**: Added `n_changes` and `changes_by_op` properties in debug mode to track value changes across pipeline steps
- **`TraceResult` status fields**: Added `status`, `dropped_by`, and `dropped_at_step` properties for clearer dropped row analysis
- **`DiffResult` completeness**: Added `cells_changed`, `changes_by_column`, `rows_unchanged`, and `changed_rows` for detailed snapshot comparison
- **Ghost value API**: Implemented `dbg.get_ghost_values(row_id)` for retrieving last known values of dropped rows
- **Merge provenance**: `trace.origin` and `trace.merge_origin` now properly populated for merged rows
- **Documentation alignment**: All documented APIs now match actual implementation with comprehensive test coverage

### Changed
- **`tp.trace()` API enhancement**: Added `row_id=` parameter for explicit internal row ID tracking
  - `row=` now strictly refers to DataFrame positional index
  - `row_id=` refers to TracePipe's internal row identifier (stable across operations)
  - Supports tracing dropped rows by ID: `tp.trace(df, row_id=42)`
- **`tp.why()` API enhancement**: Added `row_id=` parameter matching `tp.trace()` signature

### Added
- Comprehensive test suite (`test_doc_api_alignment.py`) with 27 tests validating documented API features
- Better error messages for out-of-bounds row access

## [0.4.1] - 2026-02-04

### Fixed
- Fully implemented `CheckResult` convenience properties (`.passed`, `.retention`, `.n_dropped`, `.n_steps`, `.drops_by_op`)
- Added comprehensive tests for `CheckResult` API to ensure properties work correctly
- Properties now properly access underlying `.facts` dictionary for all metrics

### Changed
- Cleaned up example files and test scripts

## [0.4.0] - 2026-02-04

### Added

- **Full row provenance for `pd.concat(axis=0)`**: Row IDs are now preserved through concatenation
  - Each result row maintains its original RID from the source DataFrame
  - `ConcatMapping` tracks which source DataFrame each row came from
  - Concat steps are now marked `FULL` completeness

- **Duplicate drop provenance in debug mode**: `drop_duplicates` now tracks which row "won"
  - `DuplicateDropMapping` maps dropped rows to their kept representative
  - Supports `keep='first'`, `keep='last'`, and `keep=False`
  - Uses `hash_pandas_object` for fast, NaN-safe key comparison

- **Clean `TraceResult` API for provenance**:
  - `trace.origin` — Unified origin: `{"type": "concat", "source_df": 1}` or `{"type": "merge", ...}`
  - `trace.representative` — For dedup drops: `{"kept_rid": 42, "subset": ["key"], "keep": "first"}`
  - No need to access internal `.store` methods

- **Clean `CheckResult` API**:
  - `result.passed` — Alias for `.ok`
  - `result.retention` — Row retention rate (0.0-1.0)
  - `result.n_dropped`, `result.n_steps`, `result.drops_by_op`
  - All properties discoverable via autocomplete

- **Comprehensive test suite**: 38 new tests covering concat, dedup, and TraceResult API

### Changed

- `wrap_concat_with_lineage` rewritten for full provenance tracking
- `axis=1` concat propagates RIDs if all inputs match, otherwise PARTIAL
- `TraceResult` enhanced with `.origin` and `.representative` properties

## [0.3.5] - 2026-02-03

### Fixed

- **DataFrame.fillna double-logging**: `df.fillna({"col": 0})` now logs exactly 1 event
- Added `wrap_pandas_transform_method` with `_in_transform_op` flag

### Added

- Known Limitations section in README documenting concat/dedup tracking gaps

### Changed

- Test suite hardened with exact count assertions and multi-scenario tests

## [0.3.4] - 2026-02-03

### Fixed

- **Event deduplication**: Identical events from parallel pipelines are now deduplicated

## [0.3.3] - 2026-02-03

### Fixed

- **Double-logging bug**: `df['col'] = df['col'].fillna()` now logs exactly one event
- **Merge warning scoping**: `tp.check(df)` now only shows warnings for merges in df's lineage

## [0.3.2] - 2026-02-03

### Fixed

- Merge duplicate key warnings now correctly identify which table (left/right) has duplicates

## [0.3.1] - 2026-02-03

### Fixed

- Cell history now correctly chains through merge operations via lineage traversal
- `tp.why()` and `tp.trace()` show pre-merge changes for post-merge rows
- `enable()` resets accumulated state when called multiple times

### Added

- `get_row_history_with_lineage()` and `get_cell_history_with_lineage()` methods

## [0.3.0] - 2026-02-03

### Added

- MkDocs documentation site with Material theme
- Comprehensive API reference documentation
- Getting started guides and tutorials
- `tp.register()` API for manually registering DataFrames
- Configurable retention threshold in `tp.check()`
- Ghost row capture for fallback filter paths
- Data quality contracts with fluent API
- HTML report generation
- Snapshot and diff functionality
- Debug mode with cell-level tracking
- `tp.why()` for cell provenance
- `tp.trace()` for row journey
- Support for all major pandas operations

### Fixed

- Recursion bug when accessing hidden column in COLUMN mode
- Config propagation issues
- Retention rate calculation for multi-table pipelines
- Export wrappers correctly strip hidden column

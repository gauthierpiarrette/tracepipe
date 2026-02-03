# Changelog

All notable changes to TracePipe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-02-03

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

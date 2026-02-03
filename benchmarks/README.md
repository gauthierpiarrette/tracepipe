# TracePipe Benchmarks

Performance benchmarks for TracePipe

## Running Benchmarks

```bash
# Run all benchmarks
python benchmarks/run_all.py

# Run individual benchmarks
python benchmarks/bench_overhead.py
python benchmarks/bench_scale.py
python benchmarks/bench_memory.py
```

## Benchmark Categories

### 1. Overhead (`bench_overhead.py`)
Measures the overhead TracePipe adds to common pandas operations:
- Filter operations (dropna, query, boolean mask)
- Transform operations (fillna, replace)
- Aggregation (groupby)
- Merge/join
- Scalar access (at/iat, loc/iloc)

**Target:** < 2x overhead for most operations

### 2. Scale (`bench_scale.py`)
Tests performance with varying DataFrame sizes:
- 1K, 10K, 100K, 1M rows
- Measures throughput (rows/sec)
- Tests both CI and DEBUG modes

**Target:** Linear scaling, no exponential blowup

### 3. Memory (`bench_memory.py`)
Measures memory usage patterns:
- Baseline vs TracePipe-enabled memory
- Memory growth over repeated operations
- Spillover threshold testing

**Target:** < 2x memory overhead, no leaks

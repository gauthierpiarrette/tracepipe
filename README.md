# 🔍 TracePipe

**Automatic data lineage & debugging for ML pipelines**

TracePipe captures the complete data transformation history of your ML pipelines with zero code changes. Point at any output and instantly see the full transformation graph.

## Installation

```bash
pip install tracepipe
```

With optional dependencies:
```bash
pip install tracepipe[sklearn]  # For scikit-learn support
pip install tracepipe[all]      # All optional dependencies
```

## Quick Start

```python
import tracepipe
import pandas as pd

# Enable automatic tracking
tracepipe.enable()

# Your normal ML pipeline code
df = pd.DataFrame({
    "age": [25, None, 35, 45],
    "income": [50000, 60000, None, 80000],
    "category": ["A", "B", "A", "B"]
})

# Named pipeline stages (optional)
with tracepipe.stage("cleaning"):
    df = df.fillna(df.mean(numeric_only=True))
    df = df.dropna()

with tracepipe.stage("feature_engineering"):
    df["age_bucket"] = pd.cut(df["age"], bins=[0, 30, 50, 100])
    df["income_normalized"] = df["income"] / df["income"].max()

# Explain what happened
lineage = tracepipe.explain(df)
lineage.show()  # Opens interactive HTML visualization
```

## Features

### 🔄 Automatic Instrumentation
TracePipe automatically instruments pandas, numpy, and scikit-learn operations:

```python
tracepipe.enable()  # That's it!

# All these are automatically tracked:
df = df.fillna(0)
df = df.merge(other_df, on="key")
df = df.groupby("category").sum()
arr = np.concatenate([arr1, arr2])
X_scaled = scaler.fit_transform(X)
predictions = model.predict(X)
```

### 📊 Named Stages
Organize your pipeline into logical stages:

```python
with tracepipe.stage("preprocessing"):
    df = df.fillna(0)
    df = df.drop_duplicates()

with tracepipe.stage("feature_engineering"):
    df["new_feature"] = df["a"] * df["b"]
```

### 🔍 Point-in-Time Lineage
Trace any output back to its origins:

```python
# Get lineage for specific data
lineage = tracepipe.explain(output=predictions)

# Or by node ID
lineage = tracepipe.explain(node_id="abc123")

# Compare stages
diff = lineage.diff("raw_input", "final_features")
```

### 📈 Interactive Visualization
Beautiful HTML visualization of your data flow:

```python
lineage = tracepipe.explain()
lineage.show()           # Opens in browser
lineage.save("lineage.html")  # Save to file
```

### 📤 Export Formats

**JSON Export:**
```python
# Export to JSON string
json_str = tracepipe.export_to_json()

# Export to file
tracepipe.export_to_json("audit_trail.json")
```

**OpenLineage Format:**
```python
# For enterprise integration (Airflow, Marquez, etc.)
events = tracepipe.export_to_openlineage(
    namespace="my_company",
    job_name="daily_pipeline"
)
```

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `tracepipe.enable()` | Enable automatic lineage tracking |
| `tracepipe.disable()` | Disable tracking (preserves existing data) |
| `tracepipe.reset()` | Clear all lineage data |
| `tracepipe.stage(name)` | Context manager for named pipeline stages |

### Query Functions

| Function | Description |
|----------|-------------|
| `tracepipe.explain(output=None)` | Get lineage for data or entire pipeline |
| `tracepipe.summary()` | Get pipeline summary statistics |
| `tracepipe.print_summary()` | Print human-readable summary |

### Export Functions

| Function | Description |
|----------|-------------|
| `tracepipe.export_to_json(filepath=None)` | Export to JSON format |
| `tracepipe.export_to_openlineage()` | Export to OpenLineage format |

### LineageResult Methods

| Method | Description |
|--------|-------------|
| `.show()` | Open interactive HTML visualization |
| `.save(filepath)` | Save HTML to file |
| `.to_json()` | Convert to JSON string |
| `.to_dict()` | Convert to dictionary |
| `.diff(from_stage, to_stage)` | Compare two stages |
| `.print_summary()` | Print text summary |

## Supported Libraries

### Pandas
All major DataFrame operations are tracked:
- Transforms: `fillna`, `dropna`, `replace`, `astype`, `rename`, etc.
- Filters: `query`, `filter`, `[]` (getitem), `head`, `tail`
- Joins: `merge`, `join`, `concat`
- Aggregations: `groupby`, `agg`, `sum`, `mean`, etc.
- Assignments: `assign`, `__setitem__`

### NumPy
Array operations are tracked:
- Creation: `array`, `zeros`, `ones`, `arange`, etc.
- Transforms: `reshape`, `transpose`, `concatenate`, etc.
- Math: `dot`, `matmul`, `sum`, `mean`, etc.

### Scikit-learn
Transformer and model operations:
- Preprocessing: `StandardScaler`, `OneHotEncoder`, etc.
- Models: `fit`, `transform`, `predict`, `predict_proba`
- Pipelines: Full pipeline tracking

## Performance

TracePipe is designed for minimal overhead:
- **< 5% runtime overhead** for typical pipelines
- Lazy provenance graphs (metadata only during execution)
- Columnar diff compression for memory efficiency

## Use Cases

### Debugging Model Drift
```python
# Find what changed when predictions went wrong
lineage = tracepipe.explain(predictions)
lineage.show()  # Visualize the entire data flow
```

### Compliance & Audit
```python
# Export GDPR-ready audit trail
tracepipe.export_to_json("audit_trail.json")
```

### Onboarding & Documentation
```python
# Auto-generate pipeline documentation
lineage.save("pipeline_documentation.html")
```

## License

MIT License - see LICENSE file for details.

"""Test different fillna patterns to find the slow one."""

import time

import numpy as np
import pandas as pd

import tracepipe

# Generate data
np.random.seed(42)
n_rows = 50_000

data = {
    "age": np.random.randint(18, 80, n_rows).astype(float),
    "income": np.random.lognormal(10.5, 0.8, n_rows).astype(float),
}

# Add missing values
missing_indices = np.random.choice(n_rows, int(n_rows * 0.02), replace=False)
for idx in missing_indices:
    data["age"][idx] = np.nan

df_template = pd.DataFrame(data)
print(f"Test dataset: {len(df_template):,} rows, {df_template['age'].isna().sum()} missing values")
print()

# Pattern 1: df['col'] = df['col'].fillna(value) - MOST COMMON
print("Pattern 1: df['col'] = df['col'].fillna(value)")
print("-" * 60)

df = df_template.copy()
start = time.time()
median_val = df["age"].median()
df["age"] = df["age"].fillna(median_val)
time_without = time.time() - start
print(f"  WITHOUT TracePipe: {time_without*1000:.1f}ms")

df = df_template.copy()
tracepipe.reset()
tracepipe.enable()
tracepipe.watch("age")
start = time.time()
median_val = df["age"].median()
df["age"] = df["age"].fillna(median_val)
time_with = time.time() - start
tracepipe.disable()
print(f"  WITH TracePipe:    {time_with*1000:.1f}ms")
print(
    f"  Overhead:          {(time_with/time_without-1)*100:+.1f}% ({time_with/time_without:.1f}x)"
)
print()

# Pattern 2: df = df.fillna({'col': value})
print("Pattern 2: df = df.fillna({'col': value})")
print("-" * 60)

df = df_template.copy()
start = time.time()
median_val = df["age"].median()
df = df.fillna({"age": median_val})
time_without = time.time() - start
print(f"  WITHOUT TracePipe: {time_without*1000:.1f}ms")

df = df_template.copy()
tracepipe.reset()
tracepipe.enable()
tracepipe.watch("age")
start = time.time()
median_val = df["age"].median()
df = df.fillna({"age": median_val})
time_with = time.time() - start
tracepipe.disable()
print(f"  WITH TracePipe:    {time_with*1000:.1f}ms")
print(
    f"  Overhead:          {(time_with/time_without-1)*100:+.1f}% ({time_with/time_without:.1f}x)"
)
print()

# Pattern 3: df.fillna(value, inplace=True)
print("Pattern 3: df.fillna(value, inplace=True)")
print("-" * 60)

df = df_template.copy()
start = time.time()
median_val = df["age"].median()
df.fillna({"age": median_val}, inplace=True)
time_without = time.time() - start
print(f"  WITHOUT TracePipe: {time_without*1000:.1f}ms")

df = df_template.copy()
tracepipe.reset()
tracepipe.enable()
tracepipe.watch("age")
start = time.time()
median_val = df["age"].median()
df.fillna({"age": median_val}, inplace=True)
time_with = time.time() - start
tracepipe.disable()
print(f"  WITH TracePipe:    {time_with*1000:.1f}ms")
print(
    f"  Overhead:          {(time_with/time_without-1)*100:+.1f}% ({time_with/time_without:.1f}x)"
)

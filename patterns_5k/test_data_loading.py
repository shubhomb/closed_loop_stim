#!/usr/bin/env python3
"""Quick smoke test for data loading pipeline (read_pattern_json + preprocess).

Run from patterns_5k/:
    python test_data_loading.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils import load_raw_data
import numpy as np

cfg = {
    "datadir": "data/oracle_ICMS_150/",
    "problematic_neurons": [],
}

print("=== Testing load_raw_data ===")
raw = load_raw_data(cfg)

# Basic sanity checks
assert raw["n_stim_channels"] == 42, f"Expected 42 stim channels, got {raw['n_stim_channels']}"
assert raw["n_neurons"] > 0, f"Expected >0 neurons, got {raw['n_neurons']}"
print(f"  n_stim_channels={raw['n_stim_channels']}, n_neurons={raw['n_neurons']}")

pdf = raw["pattern_df"]
print(f"\npattern_df shape: {pdf.shape}")
print(f"Columns: {list(pdf.columns)}")

# Check no NaN in critical timestamp columns
for col in ['pattern_flag_start_timestamp', 'pattern_end_timestamp', 'step_start_timestamp']:
    n_nan = pdf[col].isna().sum()
    assert n_nan == 0, f"{col} has {n_nan} NaN values!"
    print(f"  {col}: dtype={pdf[col].dtype}, NaN={n_nan} ✓")

# Check timestamp dtypes are integer-compatible
for col in ['pattern_flag_start_timestamp', 'pattern_end_timestamp']:
    assert pdf[col].dtype in [np.int64, np.int32, int], f"{col} dtype is {pdf[col].dtype}, expected int"

# Check oracle patterns exist
n_oracle = pdf['is_oracle'].sum()
assert n_oracle > 0, "No oracle patterns found!"
print(f"  Oracle rows: {n_oracle} ✓")

# Check trial assignment
oracle_trials = pdf.loc[pdf['is_oracle'], 'trial'].unique()
print(f"  Oracle trial values: {sorted(oracle_trials)}")

# Check spike responses
n_trials = len(raw["spike_responses"])
print(f"\nSpike responses: {n_trials} trials")
sample_key = list(raw["spike_responses"].keys())[0]
sample_shape = raw["spike_responses"][sample_key].shape
print(f"  Sample shape: {sample_shape} (neurons × time_ms)")
assert sample_shape[0] == raw["n_neurons"], f"Expected {raw['n_neurons']} neurons, got {sample_shape[0]}"

# Check pattern stims
n_patterns = len(raw["pattern_stims"])
print(f"Pattern stims: {n_patterns} unique patterns")

print("\n=== All checks passed ✓ ===")

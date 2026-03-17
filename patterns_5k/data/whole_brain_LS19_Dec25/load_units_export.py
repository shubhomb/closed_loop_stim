"""
Quick-start script for loading and inspecting units_export.pkl.
Run this to verify the file structure and browse its contents.
"""
import pickle
from pathlib import Path
import os
import numpy as np
# ── Point this to your pkl file ───────────────────────────────────────────────
pkl_path = Path("/users/shubhom/Downloads/LS19_Dec25/units_export.pkl")

# ── Load ──────────────────────────────────────────────────────────────────────
with open(pkl_path, "rb") as f:
    units = pickle.load(f)

# ── Overview ──────────────────────────────────────────────────────────────────
print(f"Total units: {len(units)}")
print(f"Keys (unit IDs): {list(units.keys())[:10]} ...")
print()

# ── Label counts ──────────────────────────────────────────────────────────────
labels = [u["label"] for u in units.values()]
for lbl in sorted(set(labels)):
    print(f"  {lbl}: {labels.count(lbl)} units")
print()

# ── Recordings represented ────────────────────────────────────────────────────
recordings = sorted(set(u["recording"] for u in units.values()))
print(f"Recordings ({len(recordings)}):")
for rec in recordings:
    rec_units = [uid for uid, u in units.items() if u["recording"] == rec]
    print(f"  {rec}: {len(rec_units)} units")
print()

# ── Inspect first unit as example ─────────────────────────────────────────────
uid = 0
u = units[uid]
print(f"Example — unit {uid}:")
print(f"  recording:        {u['recording']}")
print(f"  original_unit_id: {u['original_unit_id']}")
print(f"  label:            {u['label']}")
print(f"  position:         {u['position']}  (shape {u['position'].shape})")
print(f"  spike_train:      {u['spike_train'].shape[0]} spikes, dtype={u['spike_train'].dtype}")
print(f"  first 5 samples:  {u['spike_train'][:5]}")

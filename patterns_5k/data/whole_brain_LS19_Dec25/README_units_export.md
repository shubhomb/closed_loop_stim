# units_export.pkl

Curated spike-sorted units from LS19_12182025, with noise units removed.

## How to load

```python
import pickle

with open("units_export.pkl", "rb") as f:
    units = pickle.load(f)
```

## Structure

The pkl file is a **dict** where each key is a unique integer unit ID:

```
len(units)  # total number of non-noise units

units[0] = {
    "spike_train":      np.array([999, 1493, 1557, ...]),  # sample indices (int64)
    "label":            "good",                             # "good" or "mua"
    "position":         np.array([-7750., -900., 20.]),     # [x, y, z] in µm
    "recording":        "20251218_152916_stim3_sh0.nwb",    # source recording
    "original_unit_id": "1",                                # unit ID within that recording
}
```

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `spike_train` | `np.ndarray` (int64) | Spike times as sample indices (fs = 30000 Hz) |
| `label` | `str` | `"good"` (single unit) or `"mua"` (multi-unit activity) |
| `position` | `np.ndarray` (float64, shape 3) | Estimated unit position [x, y, z] in µm |
| `recording` | `str` | Name of the source recording/shank |
| `original_unit_id` | `str` | Original unit ID in that recording's sorting result |

## Quick inspection

```bash
python load_units_export.py
```

This prints unit counts, label breakdown, recordings, and an example unit.

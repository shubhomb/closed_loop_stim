"""Bootstrap a _checkpoint.json for an interrupted sweep from existing trial results."""
import os
import json
import glob
import hashlib
import yaml

SWEEP_DIR = "results/base_cnn_kernel_sweep_2026-02-15_11-28-56"
SWEEP_CONFIG = "sweep_base_cnn_hp.yaml"

# Compute config hash
with open(SWEEP_CONFIG, "rb") as f:
    config_hash = hashlib.sha256(f.read()).hexdigest()

results = []
for d in sorted(glob.glob(os.path.join(SWEEP_DIR, "trial_*"))):
    name = os.path.basename(d)
    summary_path = os.path.join(d, "summary_metrics.json")
    config_path = os.path.join(d, "config.yaml")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            s = json.load(f)
        s["experiment"] = name
        s["status"] = "completed"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        for k in ("kernel_sizes", "learning_rate", "fc_dims", "weight_decay", "conv_channels"):
            if k in cfg:
                s[k] = cfg[k]
        results.append(s)
    else:
        print(f"  Skipping {name} (no summary_metrics.json)")

print(f"Found {len(results)} completed trials")
if results:
    print(f"Last completed: {results[-1]['experiment']}")

ckpt = {"config_hash": config_hash, "results": results}
ckpt_path = os.path.join(SWEEP_DIR, "_checkpoint.json")
with open(ckpt_path, "w") as f:
    json.dump(ckpt, f, indent=2, default=str)
print(f"Checkpoint saved to {ckpt_path} ({len(results)} trials, hash={config_hash[:12]}...)")

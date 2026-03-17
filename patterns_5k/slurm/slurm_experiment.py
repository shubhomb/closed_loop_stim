#!/usr/bin/env python3
"""
SLURM-aware single-experiment runner for the NOTS cluster at Rice.

Wraps ``run_experiment.run_experiment`` so that it can be launched as a
standalone job that receives its configuration via CLI arguments.  The caller
(typically ``slurm_sweep.py`` or a shell script) supplies:

    * ``--config``     : path to a YAML config (base or per-trial)
    * ``--output-dir`` : where to write artifacts
    * ``--overrides``  : JSON string of config overrides
    * ``--seed``       : override the random seed (for multi-seed runs)
    * ``--run-id``     : human readable label (used in logging)

This module can also be imported and called programmatically::

    from slurm_experiment import slurm_experiment
    summary = slurm_experiment(cfg, run_dir)

Usage from the command line::

    python slurm_experiment.py --config config.yaml \\
        --output-dir results/trial_0001 \\
        --overrides '{"learning_rate": 0.001, "kernel_sizes": [5]}' \\
        --seed 42 --run-id trial_0001_seed42


NOTS reference: https://kb.rice.edu/page.php?id=148046
Workflow:
python slurm_sweep.py --sweep-config slurm_sweep_base_cnn.yaml --dry-run --tasks-per-job 50 2>&1 | head -30
bash launch_all_sweeps.sh


# 1. Check if jobs are done
squeue -u $USER

# 2. Aggregate (produces CSVs + leaderboard printout)
bash aggregate_sweeps.sh $SHARED_SCRATCH/$USER/patterns5k_results/slurm_base_cnn_* $SHARED_SCRATCH/$USER/patterns5k_results/slurm_hist_cnn_*

# 3. Download the CSVs to your laptop
scp nots:$SHARED_SCRATCH/$USER/patterns5k_results/slurm_base_cnn_*/sweep_summary.csv ./

# 4. Or grab just the best model
best_dir=$(python -c "
import pandas as pd
df = pd.read_csv('sweep_results.csv')
print(df.loc[df['all_test_fve'].idxmax(), 'run_dir'])
")
scp -r "nots:$best_dir/best_model.pt" ./

Attribution: this work was supported in part by the Big-Data Private-Cloud Research Cyberinfrastructure MRI-award funded by NSF under grant CNS-1338099 and by Rice University's Center for Research Computing (CRC)

"""

import argparse
import json
import os
import sys
import socket
import time
from datetime import datetime

# Ensure the project root (parent of slurm/) is on sys.path so we can
# import run_experiment, models, utils, etc.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import yaml

# ---------------------------------------------------------------------------
# Core wrapper
# ---------------------------------------------------------------------------

def slurm_experiment(cfg: dict, run_dir: str, preloaded_data=None) -> dict:
    """Run a single experiment, identical to ``run_experiment.run_experiment``.

    This thin wrapper exists so that the SLURM entry-point and the
    programmatic API share the same function signature.
    """
    from run_experiment import run_experiment
    return run_experiment(cfg, run_dir, preloaded_data=preloaded_data)


# ---------------------------------------------------------------------------
# CLI entry-point (called by sbatch)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a single experiment inside a SLURM job")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to base YAML config")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for experiment artifacts")
    parser.add_argument("--overrides", type=str, default=None,
                        help="JSON string of config overrides")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Human-readable run identifier (for logs)")
    args = parser.parse_args()

    # Load base config --------------------------------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply JSON overrides ----------------------------------------------
    if args.overrides:
        overrides = json.loads(args.overrides)
        cfg.update(overrides)
        # Unpack bundled dict-valued overrides (e.g. _arch → kernel_sizes …)
        for k in list(cfg.keys()):
            if isinstance(cfg[k], dict) and k.startswith("_"):
                cfg.update(cfg.pop(k))

    # Seed override ------------------------------------------------------
    if args.seed is not None:
        cfg["seed"] = args.seed

    # Environment info ---------------------------------------------------
    run_id = args.run_id or os.path.basename(args.output_dir)
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    hostname = socket.gethostname()

    print(f"[slurm_experiment] run_id={run_id}  SLURM_JOB_ID={slurm_job_id}  host={hostname}")
    print(f"[slurm_experiment] output_dir={args.output_dir}")
    print(f"[slurm_experiment] config keys: {sorted(cfg.keys())}")
    sys.stdout.flush()

    t0 = time.time()
    summary = slurm_experiment(cfg, args.output_dir)
    elapsed = time.time() - t0

    # Append SLURM metadata to summary ----------------------------------
    summary["slurm_job_id"] = slurm_job_id
    summary["hostname"] = hostname
    summary["wall_time_s"] = round(elapsed, 1)
    summary["run_id"] = run_id

    # Overwrite summary_metrics.json with enriched version ---------------
    metrics_path = os.path.join(args.output_dir, "summary_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[slurm_experiment] Finished in {elapsed:.1f}s")
    key_metrics = {k: summary.get(k) for k in
                   ["all_test_correlation", "all_test_fve", "AR_FVE", "AR_test_corr"]}
    print(f"[slurm_experiment] {json.dumps(key_metrics)}")


if __name__ == "__main__":
    main()

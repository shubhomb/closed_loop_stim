#!/usr/bin/env python3
"""SLURM single-experiment entry point for vis_stim.

Mirror of ``patterns_5k/slurm/slurm_experiment.py`` but wired to
``vis_stim/run_experiment.py``.  Each SLURM array task invokes this
script with ``--config <per-trial.yaml> --output-dir <trial_dir>``.

Programmatic API::

    from slurm_experiment import slurm_experiment
    summary = slurm_experiment(cfg, run_dir)
"""

import argparse
import json
import os
import socket
import sys
import time

# Make the vis_stim project root importable
_VIS_STIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _VIS_STIM_ROOT not in sys.path:
    sys.path.insert(0, _VIS_STIM_ROOT)

import yaml


def slurm_experiment(cfg: dict, run_dir: str, preloaded_data=None,
                     prebuilt_datasets=None) -> dict:
    """Thin wrapper around vis_stim.run_experiment.run_experiment."""
    from run_experiment import run_experiment
    return run_experiment(cfg, run_dir, preloaded_data=preloaded_data,
                          prebuilt_datasets=prebuilt_datasets)


def main():
    parser = argparse.ArgumentParser(
        description="Run a single vis_stim experiment inside a SLURM job")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--overrides", type=str, default=None,
                        help="JSON string of config overrides")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.overrides:
        overrides = json.loads(args.overrides)
        cfg.update(overrides)
        for k in list(cfg.keys()):
            if isinstance(cfg[k], dict) and k.startswith("_"):
                cfg.update(cfg.pop(k))

    if args.seed is not None:
        cfg["seed"] = args.seed

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

    summary["slurm_job_id"] = slurm_job_id
    summary["hostname"] = hostname
    summary["wall_time_s"] = round(elapsed, 1)
    summary["run_id"] = run_id

    metrics_path = os.path.join(args.output_dir, "summary_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[slurm_experiment] Finished in {elapsed:.1f}s")
    key = {k: summary.get(k) for k in ("best_val_loss", "best_val_corr",
                                        "test_loss", "test_corr")}
    print(f"[slurm_experiment] {json.dumps(key)}")


if __name__ == "__main__":
    main()

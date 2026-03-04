#!/usr/bin/env python3
"""
SLURM-based hyperparameter sweep orchestrator for the NOTS cluster at Rice.

Instead of running all trials sequentially in a single process (as
``run_sweep.py`` does), this script **generates one SLURM job-array per
sweep**, where each array task corresponds to a single (config, seed) trial
calling ``slurm_experiment.py``.  After all jobs finish, a separate
aggregation step collects ``summary_metrics.json`` from every trial directory
and produces the final ``sweep_results.csv`` with per-config mean ± SEM
across seeds.

Workflow
--------
1. ``python slurm_sweep.py --sweep-config <yaml>``
   → Expands the search space  ×  seeds, writes per-trial config YAMLs
     and a manifest file, then submits a SLURM job array.
2. Each array task runs ``slurm_experiment.py`` independently.
3. ``python slurm_sweep.py --sweep-config <yaml> --aggregate <sweep_dir>``
   → Collects results, computes mean ± SEM grouped by hyperparameter
     config, and writes ``sweep_results.csv`` + ``sweep_summary.csv``.

Flags
-----
--sweep-config   : path to sweep YAML (same format as run_sweep.py)
--dry-run        : print sbatch commands without submitting
--seeds N        : number of random seeds for each config (default: 5)
--start-seed S   : first seed value (default: 0)
--slurm-template : path to the .slurm job template (default: nots_gpu.slurm)
--aggregate DIR  : skip submission; aggregate results from DIR
--output-dir     : override the sweep output directory
--dependency     : SLURM dependency string (e.g. afterok:12345)
--max-array-size : max tasks per array job (default: 500, splits into chunks)
--concurrent     : max simultaneously running array tasks (default: 50)
"""

import argparse
import copy
import itertools
import json
import hashlib
import logging
import math
import os
import random
import subprocess
import sys
from datetime import datetime
from typing import Optional

import yaml

# numpy and pandas are only needed for aggregation, not job submission.
# They may not be installed in the base Python on NOTS login nodes.
# Import them lazily inside aggregate_results() instead.


# ---------------------------------------------------------------------------
# Grid / random helpers (inlined to avoid importing run_sweep, which
# pulls in numpy, torch, etc. that aren't on the login node)
# ---------------------------------------------------------------------------

def _flatten_grid(search_space: dict) -> list:
    """Expand a grid search space into a list of flat config dicts."""
    keys, value_lists = [], []
    for k, spec in search_space.items():
        keys.append(k)
        if isinstance(spec, list):
            value_lists.append(spec)
        elif isinstance(spec, dict):
            if "values" in spec:
                value_lists.append(spec["values"])
            else:
                stype = spec.get("type", "categorical")
                lo, hi = spec.get("low", 0), spec.get("high", 1)
                n_pts = spec.get("n_grid_points", 3)
                if stype in ("log_uniform", "float_log_uniform"):
                    pts = [lo * ((hi / lo) ** (i / (n_pts - 1))) for i in range(n_pts)] if lo > 0 and n_pts > 1 else [lo]
                    value_lists.append([round(p, 8) for p in pts])
                elif stype == "float_uniform":
                    pts = [lo + (hi - lo) * i / (n_pts - 1) for i in range(n_pts)] if n_pts > 1 else [lo]
                    value_lists.append(pts)
                elif stype == "int_uniform":
                    value_lists.append(list(range(lo, hi + 1)))
                else:
                    value_lists.append([lo])
        else:
            value_lists.append([spec])
    combos = list(itertools.product(*value_lists))
    return [dict(zip(keys, combo)) for combo in combos]


def _sample_random(search_space: dict, rng: random.Random) -> dict:
    """Draw one random config from the search space."""
    cfg = {}
    for k, spec in search_space.items():
        if isinstance(spec, list):
            cfg[k] = rng.choice(spec)
        elif isinstance(spec, dict):
            stype = spec.get("type", "categorical")
            if stype == "categorical":
                cfg[k] = rng.choice(spec["values"])
            elif stype == "int_uniform":
                cfg[k] = rng.randint(spec["low"], spec["high"])
            elif stype == "float_uniform":
                cfg[k] = rng.uniform(spec["low"], spec["high"])
            elif stype in ("log_uniform", "float_log_uniform"):
                log_val = rng.uniform(math.log(spec["low"]), math.log(spec["high"]))
                cfg[k] = math.exp(log_val)
            else:
                cfg[k] = rng.choice(spec["values"])
        else:
            cfg[k] = spec
    return cfg


def _make_experiment_name(trial_idx: int, overrides: dict) -> str:
    """Create a readable experiment sub-folder name."""
    parts = [f"trial_{trial_idx:04d}"]
    flat = {}
    for k, v in overrides.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    for key in ("model_type", "learning_rate", "dropout", "init_state", "history",
                "kernel_sizes", "conv_channels", "fc_dims"):
        if key in flat:
            val = flat[key]
            if isinstance(val, float):
                parts.append(f"{key}={val:.1e}" if val < 0.01 else f"{key}={val}")
            elif isinstance(val, list):
                parts.append(f"{key}={'x'.join(str(v) for v in val)}")
            else:
                parts.append(f"{key}={val}")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Config / directory setup
# ---------------------------------------------------------------------------

def _write_trial_config(cfg: dict, path: str):
    """Serialise a single-trial config to YAML."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def _overrides_to_json(overrides: dict) -> str:
    """Safely serialise overrides (may contain lists / nested dicts)."""
    return json.dumps(overrides, default=str)


def _config_fingerprint(overrides: dict) -> str:
    """Deterministic short hash of a config dict (for grouping seeds)."""
    canonical = json.dumps(overrides, sort_keys=True, default=str)
    return hashlib.md5(canonical.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Job submission (array-based)
# ---------------------------------------------------------------------------

def submit_array_job(
    slurm_template: str,
    manifest_path: str,
    n_tasks: int,
    tasks_per_job: int = 1,
    job_name: str = "sweep",
    max_array_size: int = 500,
    concurrent: int = 50,
    dependency: Optional[str] = None,
    dry_run: bool = False,
) -> list:
    """Submit one or more SLURM job arrays covering *n_tasks* trials.

    When *tasks_per_job* > 1, multiple manifest lines are packed into a
    single array task (run sequentially), reducing the total number of
    array tasks.  The SLURM template reads lines
    ``[TASK_ID * tasks_per_job .. TASK_ID * tasks_per_job + tasks_per_job - 1]``
    from the manifest.

    If the resulting number of array tasks exceeds *max_array_size*, the
    work is split into multiple sbatch submissions.

    The ``%concurrent`` throttle (e.g. ``--array=0-499%50``) limits how
    many tasks run simultaneously so the scheduler stays happy.

    Returns a list of submitted job-ID strings (empty on dry-run).
    """
    # Total array tasks after packing
    n_array_tasks = -(-n_tasks // tasks_per_job)  # ceil division

    job_ids = []
    start = 0
    chunk_idx = 0

    while start < n_array_tasks:
        end = min(start + max_array_size, n_array_tasks) - 1  # inclusive
        array_spec = f"{start}-{end}%{concurrent}"
        chunk_name = f"{job_name}_chunk{chunk_idx}" if end < n_array_tasks - 1 or start > 0 else job_name

        sbatch_cmd = ["sbatch", f"--array={array_spec}", f"--job-name={chunk_name}"]
        if dependency:
            sbatch_cmd.append(f"--dependency={dependency}")

        # Redirect SLURM logs to $SHARED_SCRATCH to avoid home quota
        scratch = os.environ.get("SHARED_SCRATCH", "")
        user = os.environ.get("USER", "unknown")
        if scratch:
            log_dir = os.path.join(scratch, user, "patterns5k_logs")
            os.makedirs(log_dir, exist_ok=True)
            sbatch_cmd.append(f"--output={log_dir}/patterns5k_%A_%a.out")
            sbatch_cmd.append(f"--error={log_dir}/patterns5k_%A_%a.err")

        # Pass manifest_path as $1, tasks_per_job as $2 to the SLURM template
        sbatch_cmd += [slurm_template, manifest_path, str(tasks_per_job)]

        n_chunk_tasks = end - start + 1
        n_chunk_trials = min(n_chunk_tasks * tasks_per_job, n_tasks - start * tasks_per_job)

        if dry_run:
            print(f"  [dry-run] {' '.join(sbatch_cmd)}")
            print(f"            ({n_chunk_tasks} array tasks × {tasks_per_job} trials/task "
                  f"= {n_chunk_trials} trials)")
        else:
            result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  sbatch FAILED: {result.stderr.strip()}", file=sys.stderr)
            else:
                parts = result.stdout.strip().split()
                jid = parts[-1] if parts else "unknown"
                job_ids.append(jid)
                print(f"  Submitted array job {jid}: {chunk_name}  "
                      f"({n_chunk_tasks} array tasks × {tasks_per_job} trials/task)")

        start = end + 1
        chunk_idx += 1

    return job_ids


# ---------------------------------------------------------------------------
# Sweep expansion
# ---------------------------------------------------------------------------

def expand_sweep(sweep_cfg: dict, n_seeds: int, start_seed: int):
    """Expand the sweep config into a list of (overrides, seed, name) tuples.

    Each unique hyperparameter combination is repeated ``n_seeds`` times
    with different random seeds.
    """
    strategy = sweep_cfg.get("strategy", "grid")
    search_space = sweep_cfg.get("search_space", {})
    base_seed = sweep_cfg.get("seed", 42)
    rng = random.Random(base_seed)

    if strategy == "grid":
        override_list = _flatten_grid(search_space)
    elif strategy == "random":
        n_trials = sweep_cfg.get("n_trials", 20)
        override_list = [_sample_random(search_space, rng) for _ in range(n_trials)]
    else:
        raise ValueError(f"Unsupported strategy for SLURM sweep: {strategy}. "
                         "Use 'grid' or 'random'. (Optuna requires sequential control.)")

    experiments = []
    for idx, overrides in enumerate(override_list):
        config_id = _config_fingerprint(overrides)
        for seed_offset in range(n_seeds):
            seed = start_seed + seed_offset
            exp_name = _make_experiment_name(idx, overrides) + f"_seed{seed}"
            experiments.append({
                "overrides": overrides,
                "seed": seed,
                "name": exp_name,
                "config_id": config_id,
                "trial_idx": idx,
            })

    return experiments


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(sweep_dir: str, sweep_cfg: dict):
    """Walk ``sweep_dir`` and collect all ``summary_metrics.json`` files.

    Returns a DataFrame with one row per (config, seed) and an additional
    ``sweep_summary.csv`` with mean ± SEM grouped by ``config_id``.
    """
    import pandas as pd  # lazy import — only needed for aggregation
    rows = []
    for entry in sorted(os.listdir(sweep_dir)):
        metrics_path = os.path.join(sweep_dir, entry, "summary_metrics.json")
        if not os.path.isfile(metrics_path):
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        # Try to recover config_id from the trial config
        trial_cfg_path = os.path.join(sweep_dir, entry, "config.yaml")
        if os.path.isfile(trial_cfg_path):
            with open(trial_cfg_path) as f:
                tcfg = yaml.safe_load(f)
            # Remove keys that vary per-seed
            tcfg_core = {k: v for k, v in tcfg.items() if k != "seed"}
            m["config_id"] = _config_fingerprint(tcfg_core)
        m["experiment"] = entry
        rows.append(m)

    if not rows:
        print("No results found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Per-trial CSV
    csv_path = os.path.join(sweep_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} trial results to {csv_path}")

    # Grouped summary (mean ± SEM)
    metric_cols = [c for c in df.columns if df[c].dtype in ("float64", "float32", "int64")]
    if "config_id" in df.columns and metric_cols:
        grouped = df.groupby("config_id")[metric_cols]
        summary = grouped.agg(["mean", "sem", "count"])
        summary.columns = ["_".join(c) for c in summary.columns]
        summary_path = os.path.join(sweep_dir, "sweep_summary.csv")
        summary.to_csv(summary_path)
        print(f"Wrote grouped summary to {summary_path}")

        # Also produce a human-readable leaderboard
        main_metric = sweep_cfg.get("metric", "all_test_corr")
        mean_col = f"{main_metric}_mean"
        sem_col = f"{main_metric}_sem"
        if mean_col in summary.columns:
            ascending = sweep_cfg.get("direction", "maximize") == "minimize"
            leaderboard = summary.sort_values(mean_col, ascending=ascending)
            print(f"\n{'='*70}")
            print(f"LEADERBOARD (by {main_metric})")
            print(f"{'='*70}")
            for cid, row in leaderboard.head(15).iterrows():
                n = int(row.get(f"{main_metric}_count", 0))
                print(f"  {cid}: {mean_col}={row[mean_col]:.6f} ± {row[sem_col]:.6f}  (n={n})")
            print(f"{'='*70}\n")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SLURM-based hyperparameter sweep for NOTS (job-array mode)")
    parser.add_argument("--sweep-config", type=str, required=True,
                        help="Path to sweep YAML")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sbatch commands without submitting")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of seeds per config (default: 5)")
    parser.add_argument("--start-seed", type=int, default=0,
                        help="First seed value (default: 0)")
    parser.add_argument("--slurm-template", type=str, default="slurm/nots_gpu.slurm",
                        help="Path to the .slurm job script template")
    parser.add_argument("--aggregate", type=str, default=None, metavar="DIR",
                        help="Aggregate results from a finished sweep directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override the sweep output directory")
    parser.add_argument("--dependency", type=str, default=None,
                        help="SLURM dependency (e.g. afterok:12345)")
    parser.add_argument("--max-array-size", type=int, default=500,
                        help="Max tasks per array job submission (default: 500)")
    parser.add_argument("--concurrent", type=int, default=50,
                        help="Max simultaneously running array tasks (default: 50)")
    parser.add_argument("--tasks-per-job", type=int, default=50,
                        help="Number of trials packed into each array task "
                             "(run sequentially). Increase to stay under QOS "
                             "job-count limits. (default: 50)")
    args = parser.parse_args()

    # Load sweep config --------------------------------------------------
    with open(args.sweep_config) as f:
        sweep_cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Aggregation mode
    # ------------------------------------------------------------------
    if args.aggregate:
        aggregate_results(args.aggregate, sweep_cfg)
        return

    # ------------------------------------------------------------------
    # Submission mode — job arrays
    # ------------------------------------------------------------------
    sweep_name = sweep_cfg.get("sweep_name", "sweep")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Default to $SHARED_SCRATCH to avoid home-directory quota issues on NOTS
    scratch = os.environ.get("SHARED_SCRATCH", "")
    user = os.environ.get("USER", "unknown")
    if scratch:
        default_results = os.path.join(scratch, user, "patterns5k_results")
    else:
        default_results = "results"
    sweep_dir = args.output_dir or os.path.join(default_results, f"{sweep_name}_{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)

    # Copy the sweep config for reproducibility
    import shutil
    shutil.copy(args.sweep_config, os.path.join(sweep_dir, "sweep_config.yaml"))

    base_config = sweep_cfg.get("base_config", {})
    experiments = expand_sweep(sweep_cfg, args.seeds, args.start_seed)

    n_configs = len(experiments) // args.seeds
    n_array_tasks = -(-len(experiments) // args.tasks_per_job)
    print(f"Sweep: {sweep_name}")
    print(f"Strategy: {sweep_cfg.get('strategy', 'grid')}")
    print(f"Configs: {n_configs}  ×  {args.seeds} seeds  =  {len(experiments)} trials")
    print(f"Tasks per job: {args.tasks_per_job}  →  {n_array_tasks} array tasks")
    print(f"Output: {sweep_dir}")
    print(f"SLURM template: {args.slurm_template}")
    print(f"Array submissions: ceil({n_array_tasks} / {args.max_array_size}) = "
          f"{-(-n_array_tasks // args.max_array_size)}")
    print(f"Concurrent tasks: {args.concurrent}")
    print()

    # Write per-trial configs and build the manifest -----------------------
    manifest_lines = []
    for exp in experiments:
        # Merge base_config + overrides
        cfg = copy.deepcopy(base_config)
        cfg.update(exp["overrides"])
        # Unpack bundled dict-valued overrides (e.g. _arch)
        for k in list(cfg.keys()):
            if isinstance(cfg[k], dict) and k.startswith("_"):
                cfg.update(cfg.pop(k))
        cfg["seed"] = exp["seed"]
        cfg["skip_oracle_plots"] = True  # save time; re-enable for best

        exp_dir = os.path.join(sweep_dir, exp["name"])
        config_path = os.path.join(exp_dir, "config.yaml")
        _write_trial_config(cfg, config_path)

        cmd = (
            f"python slurm/slurm_experiment.py"
            f" --config {config_path}"
            f" --output-dir {exp_dir}"
            f" --seed {exp['seed']}"
            f" --run-id {exp['name']}"
        )
        manifest_lines.append(cmd)

    manifest_path = os.path.join(sweep_dir, "manifest.txt")
    with open(manifest_path, "w") as f:
        f.write("\n".join(manifest_lines) + "\n")
    print(f"Wrote manifest ({len(manifest_lines)} lines) to {manifest_path}")

    # Submit array job(s) --------------------------------------------------
    job_ids = submit_array_job(
        slurm_template=args.slurm_template,
        manifest_path=manifest_path,
        n_tasks=len(manifest_lines),
        tasks_per_job=args.tasks_per_job,
        job_name=sweep_name,
        max_array_size=args.max_array_size,
        concurrent=args.concurrent,
        dependency=args.dependency,
        dry_run=args.dry_run,
    )

    # Save job IDs for monitoring
    if job_ids:
        ids_path = os.path.join(sweep_dir, "slurm_job_ids.txt")
        with open(ids_path, "w") as f:
            f.write("\n".join(job_ids) + "\n")
        print(f"\n{len(job_ids)} array job(s) submitted. IDs saved to {ids_path}")
        print(f"Monitor with:  squeue -u $USER")
        print(f"Aggregate with:  python slurm_sweep.py --sweep-config {args.sweep_config} --aggregate {sweep_dir}")
    elif not args.dry_run:
        print("No jobs were submitted.")


if __name__ == "__main__":
    main()

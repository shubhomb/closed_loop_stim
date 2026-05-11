#!/usr/bin/env python3
"""SLURM-based hyperparameter sweep orchestrator for vis_stim on NOTS.

Mirrors ``patterns_5k/slurm/slurm_sweep.py``: expands a search space into
per-trial YAMLs, builds a manifest of ``slurm_experiment.py`` calls, and
submits a SLURM job array (with optional task-packing).

Workflow
--------
1. ``python slurm/slurm_sweep.py --sweep-config <yaml>``
   -> Expands search space x seeds, writes per-trial configs and a
      manifest, then submits one or more SLURM arrays.
2. Each array task runs ``slurm_experiment.py`` independently.
3. ``python slurm/slurm_sweep.py --sweep-config <yaml> --aggregate <dir>``
   -> Collects ``summary_metrics.json`` from every trial and writes
      ``sweep_results.csv`` + ``sweep_summary.csv``.
"""

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Search-space helpers (inlined; no torch/numpy required on the login node)
# ---------------------------------------------------------------------------

def _flatten_grid(search_space: dict) -> list:
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
                    pts = [lo * ((hi / lo) ** (i / (n_pts - 1)))
                           for i in range(n_pts)] if lo > 0 and n_pts > 1 else [lo]
                    value_lists.append([round(p, 8) for p in pts])
                elif stype == "float_uniform":
                    pts = [lo + (hi - lo) * i / (n_pts - 1)
                           for i in range(n_pts)] if n_pts > 1 else [lo]
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
    parts = [f"trial_{trial_idx:04d}"]
    flat = {}
    for k, v in overrides.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    for key in ("lr", "dropout", "weight_decay",
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


def _write_trial_config(cfg: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def _config_fingerprint(overrides: dict) -> str:
    canonical = json.dumps(overrides, sort_keys=True, default=str)
    return hashlib.md5(canonical.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Job submission
# ---------------------------------------------------------------------------

def submit_array_job(slurm_template, manifest_path, n_tasks,
                     tasks_per_job=1, job_name="sweep",
                     max_array_size=500, concurrent=50,
                     dependency: Optional[str] = None,
                     dry_run: bool = False) -> list:
    n_array_tasks = -(-n_tasks // tasks_per_job)
    job_ids, start, chunk_idx = [], 0, 0

    while start < n_array_tasks:
        end = min(start + max_array_size, n_array_tasks) - 1
        array_spec = f"{start}-{end}%{concurrent}"
        chunk_name = (f"{job_name}_chunk{chunk_idx}"
                      if end < n_array_tasks - 1 or start > 0 else job_name)

        sbatch_cmd = ["sbatch", f"--array={array_spec}",
                      f"--job-name={chunk_name}"]
        if dependency:
            sbatch_cmd.append(f"--dependency={dependency}")

        scratch = os.environ.get("SHARED_SCRATCH", "")
        user = os.environ.get("USER", "unknown")
        if scratch:
            log_dir = os.path.join(scratch, user, "vis_stim_logs")
            os.makedirs(log_dir, exist_ok=True)
            sbatch_cmd.append(f"--output={log_dir}/vis_stim_%A_%a.out")
            sbatch_cmd.append(f"--error={log_dir}/vis_stim_%A_%a.err")

        sbatch_cmd += [slurm_template, manifest_path, str(tasks_per_job)]

        n_chunk_tasks = end - start + 1
        n_chunk_trials = min(n_chunk_tasks * tasks_per_job,
                              n_tasks - start * tasks_per_job)

        if dry_run:
            print(f"  [dry-run] {' '.join(sbatch_cmd)}")
            print(f"            ({n_chunk_tasks} array tasks x {tasks_per_job} "
                  f"trials/task = {n_chunk_trials} trials)")
        else:
            r = subprocess.run(sbatch_cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  sbatch FAILED: {r.stderr.strip()}", file=sys.stderr)
            else:
                jid = r.stdout.strip().split()[-1]
                job_ids.append(jid)
                print(f"  Submitted array job {jid}: {chunk_name}  "
                      f"({n_chunk_tasks} array tasks x {tasks_per_job} trials/task)")

        start = end + 1
        chunk_idx += 1

    return job_ids


# ---------------------------------------------------------------------------
# Sweep expansion
# ---------------------------------------------------------------------------

def expand_sweep(sweep_cfg: dict, n_seeds: int, start_seed: int):
    strategy = sweep_cfg.get("strategy", "grid")
    search_space = sweep_cfg.get("search_space", {})
    base_seed = sweep_cfg.get("seed", 42)
    rng = random.Random(base_seed)

    if strategy == "grid":
        override_list = _flatten_grid(search_space)
    elif strategy == "random":
        n_trials = sweep_cfg.get("n_trials", 20)
        override_list = [_sample_random(search_space, rng)
                         for _ in range(n_trials)]
    else:
        raise ValueError(f"Unsupported strategy: {strategy}. "
                         "Use 'grid' or 'random'.")

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
    import pandas as pd
    rows = []
    for entry in sorted(os.listdir(sweep_dir)):
        metrics_path = os.path.join(sweep_dir, entry, "summary_metrics.json")
        if not os.path.isfile(metrics_path):
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        trial_cfg_path = os.path.join(sweep_dir, entry, "config.yaml")
        if os.path.isfile(trial_cfg_path):
            with open(trial_cfg_path) as f:
                tcfg = yaml.safe_load(f)
            tcfg_core = {k: v for k, v in tcfg.items() if k != "seed"}
            m["config_id"] = _config_fingerprint(tcfg_core)
        m["experiment"] = entry
        rows.append(m)

    if not rows:
        print("No results found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    csv_path = os.path.join(sweep_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} trial results to {csv_path}")

    metric_cols = [c for c in df.columns
                   if df[c].dtype in ("float64", "float32", "int64")]
    if "config_id" in df.columns and metric_cols:
        grouped = df.groupby("config_id")[metric_cols]
        summary = grouped.agg(["mean", "sem", "count"])
        summary.columns = ["_".join(c) for c in summary.columns]
        summary_path = os.path.join(sweep_dir, "sweep_summary.csv")
        summary.to_csv(summary_path)
        print(f"Wrote grouped summary to {summary_path}")

        main_metric = sweep_cfg.get("metric", "test_corr")
        mean_col = f"{main_metric}_mean"
        sem_col = f"{main_metric}_sem"
        if mean_col in summary.columns:
            ascending = sweep_cfg.get("direction", "maximize") == "minimize"
            leaderboard = summary.sort_values(mean_col, ascending=ascending)
            print(f"\n{'='*70}\nLEADERBOARD (by {main_metric})\n{'='*70}")
            for cid, row in leaderboard.head(15).iterrows():
                n = int(row.get(f"{main_metric}_count", 0))
                print(f"  {cid}: {mean_col}={row[mean_col]:.6f} "
                      f"+- {row[sem_col]:.6f}  (n={n})")
            print(f"{'='*70}\n")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SLURM-based hyperparameter sweep for vis_stim on NOTS")
    parser.add_argument("--sweep-config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--slurm-template", type=str,
                        default="slurm/nots_gpu.slurm")
    parser.add_argument("--aggregate", type=str, default=None, metavar="DIR")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dependency", type=str, default=None)
    parser.add_argument("--max-array-size", type=int, default=500)
    parser.add_argument("--concurrent", type=int, default=50)
    parser.add_argument("--tasks-per-job", type=int, default=10)
    args = parser.parse_args()

    with open(args.sweep_config) as f:
        sweep_cfg = yaml.safe_load(f)

    if args.aggregate:
        aggregate_results(args.aggregate, sweep_cfg)
        return

    sweep_name = sweep_cfg.get("sweep_name", "vis_stim_sweep")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    scratch = os.environ.get("SHARED_SCRATCH", "")
    user = os.environ.get("USER", "unknown")
    if scratch:
        default_results = os.path.join(scratch, user, "vis_stim_results")
    else:
        default_results = "results"
    sweep_dir = args.output_dir or os.path.join(default_results,
                                                f"{sweep_name}_{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)

    import shutil
    shutil.copy(args.sweep_config, os.path.join(sweep_dir, "sweep_config.yaml"))

    base_config = sweep_cfg.get("base_config", {})
    experiments = expand_sweep(sweep_cfg, args.seeds, args.start_seed)

    n_configs = len(experiments) // args.seeds
    n_array_tasks = -(-len(experiments) // args.tasks_per_job)
    print(f"Sweep: {sweep_name}")
    print(f"Strategy: {sweep_cfg.get('strategy', 'grid')}")
    print(f"Configs: {n_configs} x {args.seeds} seeds = "
          f"{len(experiments)} trials")
    print(f"Tasks per job: {args.tasks_per_job} -> {n_array_tasks} array tasks")
    print(f"Output: {sweep_dir}")
    print(f"SLURM template: {args.slurm_template}")
    print(f"Concurrent tasks: {args.concurrent}\n")

    manifest_lines = []
    for exp in experiments:
        cfg = copy.deepcopy(base_config)
        cfg.update(exp["overrides"])
        for k in list(cfg.keys()):
            if isinstance(cfg[k], dict) and k.startswith("_"):
                cfg.update(cfg.pop(k))
        cfg["seed"] = exp["seed"]

        exp_dir = os.path.join(sweep_dir, exp["name"])
        config_path = os.path.join(exp_dir, "config.yaml")
        _write_trial_config(cfg, config_path)

        cmd = (f"python slurm/slurm_experiment.py"
               f" --config {config_path}"
               f" --output-dir {exp_dir}"
               f" --seed {exp['seed']}"
               f" --run-id {exp['name']}")
        manifest_lines.append(cmd)

    manifest_path = os.path.join(sweep_dir, "manifest.txt")
    with open(manifest_path, "w") as f:
        f.write("\n".join(manifest_lines) + "\n")
    print(f"Wrote manifest ({len(manifest_lines)} lines) to {manifest_path}")

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

    if job_ids:
        ids_path = os.path.join(sweep_dir, "slurm_job_ids.txt")
        with open(ids_path, "w") as f:
            f.write("\n".join(job_ids) + "\n")
        print(f"\n{len(job_ids)} array job(s) submitted. IDs -> {ids_path}")
        print(f"Monitor:    squeue -u $USER")
        print(f"Aggregate:  python slurm/slurm_sweep.py "
              f"--sweep-config {args.sweep_config} --aggregate {sweep_dir}")
    elif not args.dry_run:
        print("No jobs were submitted.")


if __name__ == "__main__":
    main()

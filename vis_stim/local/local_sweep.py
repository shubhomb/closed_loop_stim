#!/usr/bin/env python3
"""
Local multi-GPU hyperparameter sweep orchestrator for vis_stim.

A local-workstation analog of ``slurm/slurm_sweep.py``. Reuses the same
sweep YAML format and the same per-trial entry point
(``slurm.slurm_experiment.slurm_experiment``) so trials run identically
on the cluster and locally.

Parallelism model
-----------------
One worker process per GPU. Each worker pins ``CUDA_VISIBLE_DEVICES``
before ``torch`` is imported, so the worker sees exactly one GPU
(as ``cuda:0``) and cannot accidentally allocate on the other.

Output layout matches the SLURM sweeps so ``slurm/aggregate_sweeps.sh``
works on local results unchanged.

Strategy: grid only (matches the cluster sweeps in use).

Usage
-----
    # Two GPUs, all configs in the YAML
    python local/local_sweep.py --sweep-config local/local_sweep_test.yaml

    # Override datadir if the YAML points to a cluster path
    python local/local_sweep.py --sweep-config slurm/slurm_sweep_simple_cnn.yaml \\
        --datadir data/vis_stim_data/

    # Pin to a single GPU
    python local/local_sweep.py --sweep-config local/local_sweep_test.yaml --gpus 0

    # Dry-run (print trials, do not execute)
    python local/local_sweep.py --sweep-config local/local_sweep_test.yaml --dry-run

    # Resume after Ctrl-C — skips trials whose summary_metrics.json exists
    python local/local_sweep.py --sweep-config local/local_sweep_test.yaml \\
        --resume results/local_test_2026-05-11_..._

Aggregate identically to slurm:
    bash slurm/aggregate_sweeps.sh results/local_test_<timestamp>
"""

import argparse
import copy
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import yaml

# Allow `from slurm.slurm_sweep import ...` regardless of cwd
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from slurm.slurm_sweep import (  # noqa: E402
    _flatten_grid,
    _make_experiment_name,
    _config_fingerprint,
    _write_trial_config,
    aggregate_results,
)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _run_one_trial(args: tuple) -> dict:
    """Worker entry point. Runs in a fresh process.

    Pins ``CUDA_VISIBLE_DEVICES`` *before* importing torch so the worker
    only ever sees a single GPU. Calls the same ``slurm_experiment``
    function that the cluster invokes per array task.
    """
    gpu_id, exp_dir, config_path, seed, run_id = args

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Match the slurm template's thread caps so a local 2-trial run does
    # not oversubscribe CPU cores.
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

    # Defer the heavy imports until after CUDA_VISIBLE_DEVICES is set
    sys.path.insert(0, _PROJECT_ROOT)
    from slurm.slurm_experiment import slurm_experiment  # noqa: WPS433

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["seed"] = seed

    t0 = time.time()
    try:
        summary = slurm_experiment(cfg, exp_dir)
        summary["status"] = "completed"
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        summary = {"status": "failed", "error": str(e)}

    elapsed = time.time() - t0
    summary["wall_time_s"] = round(elapsed, 1)
    summary["gpu_id"] = gpu_id
    summary["hostname"] = os.uname().nodename
    summary["run_id"] = run_id

    metrics_path = os.path.join(exp_dir, "summary_metrics.json")
    try:
        if os.path.isfile(metrics_path):
            with open(metrics_path, "r") as f:
                existing = json.load(f)
            existing.update({k: summary[k] for k in
                             ("wall_time_s", "gpu_id", "hostname", "run_id", "status")
                             if k in summary})
            with open(metrics_path, "w") as f:
                json.dump(existing, f, indent=2, default=str)
        else:
            os.makedirs(exp_dir, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
    except Exception:  # noqa: BLE001
        pass

    return {"run_id": run_id, "elapsed": elapsed, "gpu_id": gpu_id,
            "status": summary.get("status", "unknown"), "exp_dir": exp_dir}


# ---------------------------------------------------------------------------
# Trial expansion (grid only)
# ---------------------------------------------------------------------------

def _expand_trials(sweep_cfg: dict, n_seeds: int, start_seed: int) -> list[dict]:
    """Return a list of trial dicts (overrides + seed + name)."""
    if sweep_cfg.get("strategy", "grid") != "grid":
        raise ValueError("local_sweep.py only supports grid search.")

    search_space = sweep_cfg.get("search_space", {})
    override_list = _flatten_grid(search_space)

    trials = []
    for idx, overrides in enumerate(override_list):
        cfp = _config_fingerprint(overrides)
        for so in range(n_seeds):
            seed = start_seed + so
            name = _make_experiment_name(idx, overrides) + f"_seed{seed}"
            trials.append({
                "overrides": overrides, "seed": seed,
                "name": name, "config_id": cfp, "trial_idx": idx,
            })
    return trials


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Local multi-GPU grid sweep (mirrors slurm/slurm_sweep.py).")
    parser.add_argument("--sweep-config", type=str, required=True,
                        help="Path to sweep YAML (same format as slurm sweeps).")
    parser.add_argument("--gpus", type=str, default="0,1",
                        help="Comma-separated GPU IDs to use (default: 0,1).")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Seeds per config.")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--datadir", type=str, default=None,
                        help="Override base_config.datadir (handy when YAML "
                             "points to a cluster path).")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, metavar="DIR")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate", type=str, default=None, metavar="DIR",
                        help="Skip running; aggregate from DIR (delegates to "
                             "slurm.slurm_sweep.aggregate_results).")
    args = parser.parse_args()

    with open(args.sweep_config, "r") as f:
        sweep_cfg = yaml.safe_load(f)

    if args.aggregate:
        aggregate_results(args.aggregate, sweep_cfg)
        return

    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpu_ids:
        raise SystemExit("--gpus must include at least one GPU id")

    sweep_name = sweep_cfg.get("sweep_name", "sweep")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.resume:
        sweep_dir = args.resume
        if not os.path.isdir(sweep_dir):
            raise SystemExit(f"--resume dir not found: {sweep_dir}")
    elif args.output_dir:
        sweep_dir = args.output_dir
    else:
        sweep_dir = os.path.join("results", f"local_{sweep_name}_{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)

    # Save a copy of the sweep config (idempotent on resume)
    cfg_copy = os.path.join(sweep_dir, "sweep_config.yaml")
    if not os.path.exists(cfg_copy):
        shutil.copy(args.sweep_config, cfg_copy)

    # Logging
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(sweep_dir, "local_sweep.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("local_sweep")

    base_config = sweep_cfg.get("base_config", {})
    if args.datadir:
        base_config = {**base_config, "datadir": args.datadir}

    trials = _expand_trials(sweep_cfg, args.seeds, args.start_seed)
    n_configs = len(trials) // max(args.seeds, 1)

    log.info(f"Sweep: {sweep_name}  (local, grid)")
    log.info(f"Configs: {n_configs}  x  {args.seeds} seeds  =  {len(trials)} trials")
    log.info(f"GPUs: {gpu_ids}")
    log.info(f"Output: {sweep_dir}")

    # Write per-trial configs; collect work items
    work_items = []
    skipped = 0
    for trial in trials:
        cfg = copy.deepcopy(base_config)
        cfg.update(trial["overrides"])
        for k in list(cfg.keys()):
            if isinstance(cfg[k], dict) and k.startswith("_"):
                cfg.update(cfg.pop(k))
        cfg["seed"] = trial["seed"]

        exp_dir = os.path.join(sweep_dir, trial["name"])
        config_path = os.path.join(exp_dir, "config.yaml")

        metrics_path = os.path.join(exp_dir, "summary_metrics.json")
        if args.resume and os.path.isfile(metrics_path):
            skipped += 1
            continue

        _write_trial_config(cfg, config_path)
        work_items.append({
            "exp_dir": exp_dir,
            "config_path": config_path,
            "seed": trial["seed"],
            "run_id": trial["name"],
        })

    if skipped:
        log.info(f"Resume: skipping {skipped} already-finished trials")
    log.info(f"To run: {len(work_items)} trials")

    if args.dry_run:
        for w in work_items[:10]:
            log.info(f"  [dry-run] {w['run_id']} -> {w['exp_dir']}")
        if len(work_items) > 10:
            log.info(f"  ... and {len(work_items) - 10} more")
        return

    if not work_items:
        log.info("Nothing to do.")
        return

    # Round-robin assignment of GPUs
    tasks = []
    for i, w in enumerate(work_items):
        gpu = gpu_ids[i % len(gpu_ids)]
        tasks.append((gpu, w["exp_dir"], w["config_path"], w["seed"], w["run_id"]))

    t_start = time.time()
    n_done = 0
    n_fail = 0
    # max_tasks_per_child=1 forces a fresh process per trial. Without it,
    # ProcessPoolExecutor reuses workers, and once torch is imported in a
    # worker with CUDA_VISIBLE_DEVICES=X, that worker is permanently
    # pinned to GPU X — a second trial routed to it with a different
    # gpu_id silently lands on the wrong device.
    with ProcessPoolExecutor(
        max_workers=len(gpu_ids), max_tasks_per_child=1,
    ) as pool:
        futures = {pool.submit(_run_one_trial, t): t for t in tasks}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                res = fut.result()
                status = res["status"]
                n_done += 1
                if status != "completed":
                    n_fail += 1
                elapsed = res["elapsed"]
                log.info(
                    f"[{n_done}/{len(tasks)}] gpu{res['gpu_id']}  "
                    f"{res['run_id']}  {status}  ({elapsed:.1f}s)"
                )
            except Exception as e:  # noqa: BLE001
                n_done += 1
                n_fail += 1
                log.error(f"Trial crashed: {t[4]}: {e}")

    total = time.time() - t_start
    log.info("=" * 60)
    log.info(f"Done: {n_done - n_fail} ok, {n_fail} failed, wall time {total:.1f}s "
             f"({total / 60:.1f} min) on GPUs {gpu_ids}")
    log.info(f"Aggregate with: bash slurm/aggregate_sweeps.sh {sweep_dir}")


if __name__ == "__main__":
    main()

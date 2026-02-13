"""
Hyperparameter sweep orchestrator.

Reads a sweep_config.yaml, generates experiment configs via grid / random /
Optuna search, and runs each experiment through run_experiment.py.

All results land under  results/<sweep_name>_<timestamp>/  with:
  - sweep_config.yaml          (copy of the sweep spec)
  - sweep_results.csv          (one row per trial with all metrics)
  - <experiment_name>/         (per-experiment artifacts)
      ├── config.yaml
      ├── experiment.log
      ├── best_stim_spike_model.pt
      ├── summary_metrics.json
      ├── training_history.json
      ├── *.png  (figures)
      └── oracle_trials_by_pattern/

Usage:
    python run_sweep.py                                  # default sweep_config.yaml
    python run_sweep.py --sweep-config my_sweep.yaml
    python run_sweep.py --dry-run                        # preview experiments only
    python run_sweep.py --strategy optuna --n-trials 30  # override strategy
"""

import argparse
import copy
import itertools
import json
import logging
import os
import random
import shutil
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_grid(search_space: dict) -> list[dict]:
    """Expand a grid search space into a list of flat config dicts."""
    keys, value_lists = [], []
    for k, spec in search_space.items():
        keys.append(k)
        if isinstance(spec, dict):
            value_lists.append(spec["values"])
        elif isinstance(spec, list):
            value_lists.append(spec)
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
            elif stype in ("int_uniform",):
                cfg[k] = rng.randint(spec["low"], spec["high"])
            elif stype in ("float_uniform",):
                cfg[k] = rng.uniform(spec["low"], spec["high"])
            elif stype in ("log_uniform", "float_log_uniform"):
                log_val = rng.uniform(np.log(spec["low"]), np.log(spec["high"]))
                cfg[k] = float(np.exp(log_val))
            else:
                cfg[k] = rng.choice(spec["values"])
        else:
            cfg[k] = spec
    return cfg


def _make_experiment_name(trial_idx: int, overrides: dict) -> str:
    """Create a readable experiment sub-folder name."""
    parts = [f"trial_{trial_idx:04d}"]
    # Include a few key params for readability
    for key in ("model_type", "learning_rate", "dropout", "init_state", "history"):
        if key in overrides:
            val = overrides[key]
            if isinstance(val, float):
                parts.append(f"{key}={val:.1e}" if val < 0.01 else f"{key}={val}")
            elif isinstance(val, list):
                parts.append(f"{key}={'x'.join(str(v) for v in val)}")
            else:
                parts.append(f"{key}={val}")
    return "_".join(parts)


def _build_optuna_search_space(search_space: dict):
    """Return a function that Optuna calls to suggest params."""
    def suggest(trial):
        cfg = {}
        for k, spec in search_space.items():
            if isinstance(spec, list):
                cfg[k] = trial.suggest_categorical(k, spec)
            elif isinstance(spec, dict):
                stype = spec.get("type", "categorical")
                if stype == "categorical":
                    # Optuna needs hashable choices — convert lists to tuples
                    choices = spec["values"]
                    has_lists = any(isinstance(v, list) for v in choices)
                    if has_lists:
                        str_choices = [json.dumps(v) for v in choices]
                        picked = trial.suggest_categorical(k, str_choices)
                        cfg[k] = json.loads(picked)
                    else:
                        cfg[k] = trial.suggest_categorical(k, choices)
                elif stype == "int_uniform":
                    cfg[k] = trial.suggest_int(k, spec["low"], spec["high"])
                elif stype == "float_uniform":
                    cfg[k] = trial.suggest_float(k, spec["low"], spec["high"])
                elif stype in ("log_uniform", "float_log_uniform"):
                    cfg[k] = trial.suggest_float(k, spec["low"], spec["high"], log=True)
                else:
                    cfg[k] = trial.suggest_categorical(k, spec["values"])
            else:
                cfg[k] = spec
        return cfg
    return suggest


# ---------------------------------------------------------------------------
# Grid / Random sweep runner
# ---------------------------------------------------------------------------

def run_grid_or_random_sweep(sweep_cfg: dict, sweep_dir: str, dry_run: bool = False):
    """Run grid or random sweep without Optuna dependency."""
    from run_experiment import run_experiment

    strategy = sweep_cfg.get("strategy", "grid")
    base_config = sweep_cfg.get("base_config", {})
    search_space = sweep_cfg.get("search_space", {})
    seed = sweep_cfg.get("seed", 42)
    rng = random.Random(seed)
    np.random.seed(seed)

    if strategy == "grid":
        override_list = _flatten_grid(search_space)
    else:  # random
        n_trials = sweep_cfg.get("n_trials", 20)
        override_list = [_sample_random(search_space, rng) for _ in range(n_trials)]

    logger = logging.getLogger("sweep")
    logger.info(f"Strategy: {strategy} — {len(override_list)} experiments to run")

    results = []
    for idx, overrides in enumerate(override_list):
        exp_name = _make_experiment_name(idx, overrides)
        exp_dir = os.path.join(sweep_dir, exp_name)

        # Merge base + overrides
        cfg = copy.deepcopy(base_config)
        cfg.update(overrides)

        logger.info(f"\n{'='*70}")
        logger.info(f"[{idx+1}/{len(override_list)}] {exp_name}")
        logger.info(f"  Overrides: {json.dumps(overrides, default=str)}")
        logger.info(f"  Output: {exp_dir}")

        if dry_run:
            logger.info("  (dry run — skipping)")
            results.append({"experiment": exp_name, "status": "dry_run", **overrides})
            continue

        try:
            summary = run_experiment(cfg, exp_dir)
            summary["experiment"] = exp_name
            summary["status"] = "completed"
            summary.update(overrides)
            results.append(summary)
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            traceback.print_exc()
            results.append({"experiment": exp_name, "status": "failed", "error": str(e), **overrides})

    return results


# ---------------------------------------------------------------------------
# Optuna sweep runner
# ---------------------------------------------------------------------------

def run_optuna_sweep(sweep_cfg: dict, sweep_dir: str, dry_run: bool = False):
    """Run Bayesian optimization sweep using Optuna."""
    try:
        import optuna
    except ImportError:
        print("Optuna not installed. Install with: pip install optuna")
        print("Falling back to random search.")
        sweep_cfg = copy.deepcopy(sweep_cfg)
        sweep_cfg["strategy"] = "random"
        return run_grid_or_random_sweep(sweep_cfg, sweep_dir, dry_run)

    from run_experiment import run_experiment

    base_config = sweep_cfg.get("base_config", {})
    search_space = sweep_cfg.get("search_space", {})
    n_trials = sweep_cfg.get("n_trials", 20)
    metric = sweep_cfg.get("metric", "test_corr")
    direction = sweep_cfg.get("direction", "maximize")
    seed = sweep_cfg.get("seed", 42)

    suggest_fn = _build_optuna_search_space(search_space)
    logger = logging.getLogger("sweep")

    results = []
    trial_counter = [0]  # mutable counter for closure

    def objective(trial):
        overrides = suggest_fn(trial)
        idx = trial_counter[0]
        trial_counter[0] += 1

        exp_name = _make_experiment_name(idx, overrides)
        exp_dir = os.path.join(sweep_dir, exp_name)

        cfg = copy.deepcopy(base_config)
        cfg.update(overrides)

        logger.info(f"\n{'='*70}")
        logger.info(f"[Optuna trial {trial.number}] {exp_name}")
        logger.info(f"  Params: {json.dumps(overrides, default=str)}")

        if dry_run:
            results.append({"experiment": exp_name, "status": "dry_run", **overrides})
            return 0.0

        try:
            summary = run_experiment(cfg, exp_dir)
            summary["experiment"] = exp_name
            summary["status"] = "completed"
            summary.update(overrides)
            results.append(summary)

            value = summary.get(metric, float("nan"))
            if np.isnan(value):
                raise optuna.TrialPruned(f"Metric '{metric}' is NaN")
            return value
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            traceback.print_exc()
            results.append({"experiment": exp_name, "status": "failed", "error": str(e), **overrides})
            raise optuna.TrialPruned(str(e))

    # Create Optuna study
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=sweep_cfg.get("sweep_name", "sweep"),
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Save Optuna study summary
    try:
        study_df = study.trials_dataframe()
        study_df.to_csv(os.path.join(sweep_dir, "optuna_trials.csv"), index=False)
    except Exception:
        pass

    best = study.best_trial
    logger.info(f"\nBest trial: {best.number} — {metric}={best.value:.6f}")
    logger.info(f"  Params: {best.params}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep")
    parser.add_argument("--sweep-config", type=str, default="sweep_config.yaml",
                        help="Path to sweep config YAML")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview experiments without running them")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Override sweep strategy (grid/random/optuna)")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Override number of trials (random/optuna)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override the sweep output directory")
    args = parser.parse_args()

    # Load sweep config
    with open(args.sweep_config, "r") as f:
        sweep_cfg = yaml.safe_load(f)

    if args.strategy:
        sweep_cfg["strategy"] = args.strategy
    if args.n_trials:
        sweep_cfg["n_trials"] = args.n_trials

    # Create sweep output directory
    sweep_name = sweep_cfg.get("sweep_name", "sweep")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output_dir:
        sweep_dir = args.output_dir
    else:
        sweep_dir = os.path.join("results", f"{sweep_name}_{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)

    # Setup logging
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(sweep_dir, "sweep.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("sweep")

    # Save a copy of the sweep config
    shutil.copy(args.sweep_config, os.path.join(sweep_dir, "sweep_config.yaml"))

    logger.info(f"Sweep: {sweep_name}")
    logger.info(f"Strategy: {sweep_cfg.get('strategy', 'grid')}")
    logger.info(f"Output: {sweep_dir}")
    logger.info(f"Metric: {sweep_cfg.get('metric', 'test_corr')} ({sweep_cfg.get('direction', 'maximize')})")

    # Dispatch
    strategy = sweep_cfg.get("strategy", "grid")
    if strategy in ("grid", "random"):
        results = run_grid_or_random_sweep(sweep_cfg, sweep_dir, dry_run=args.dry_run)
    elif strategy == "optuna":
        results = run_optuna_sweep(sweep_cfg, sweep_dir, dry_run=args.dry_run)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Save consolidated results
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(sweep_dir, "sweep_results.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"\nSaved {len(results)} results to {csv_path}")

        # Print leaderboard
        completed = df[df["status"] == "completed"]
        if not completed.empty:
            metric = sweep_cfg.get("metric", "test_corr")
            if metric in completed.columns:
                ascending = sweep_cfg.get("direction", "maximize") == "minimize"
                leaderboard = completed.sort_values(metric, ascending=ascending)
                logger.info(f"\n{'='*70}")
                logger.info("LEADERBOARD")
                logger.info(f"{'='*70}")
                for i, row in leaderboard.head(10).iterrows():
                    logger.info(f"  {row['experiment']}: {metric}={row[metric]:.6f}")
                logger.info(f"{'='*70}")

                # Save best config
                best_row = leaderboard.iloc[0]
                best_exp_dir = os.path.join(sweep_dir, best_row["experiment"])
                best_config_src = os.path.join(best_exp_dir, "config.yaml")
                if os.path.exists(best_config_src):
                    shutil.copy(best_config_src, os.path.join(sweep_dir, "best_config.yaml"))
                    logger.info(f"Best config saved to {os.path.join(sweep_dir, 'best_config.yaml')}")

    logger.info(f"\nSweep complete! All results under: {sweep_dir}")


if __name__ == "__main__":
    main()

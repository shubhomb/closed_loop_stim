# vis_stim Hyperparameter Sweeps on NOTS

Mirror of `patterns_5k/slurm/` adapted to train `SimpleCausalSpikeCNN`
on the visual-stim dataset. Each trial does data load -> train -> eval
and writes `summary_metrics.json`. Greedy stim generation is **not**
run during the sweep -- pick the winning config and run
`run_greedy.py --model-dir <best_run>` separately.

## Files

| File | Purpose |
|---|---|
| `../run_experiment.py` | One-trial entry point: trains a `SimpleCausalSpikeCNN` and writes `summary_metrics.json`, `best.pth`, `history.json`, `config.yaml`. |
| `slurm_experiment.py` | Thin SLURM wrapper around `run_experiment.run_experiment`. |
| `slurm_sweep.py` | Orchestrator: expands the search space, writes per-trial YAMLs and a manifest, submits a SLURM job array. Also aggregates results. |
| `slurm_sweep_simple_cnn.yaml` | First-cut search space (LR x weight_decay x dropout x architecture). |
| `nots_gpu.slurm` / `nots_cpu.slurm` | NOTS array-job templates. Reads manifest line `[TASK_ID*N .. TASK_ID*N+N-1]`. |
| `launch_all_sweeps.sh` | Convenience wrapper that submits every sweep YAML in this folder. |
| `aggregate_sweeps.sh` | Walks one or more sweep dirs and produces `sweep_results.csv` + `sweep_summary.csv` + a leaderboard. |
| `setup_nots_env.sh` | Verifies the shared `patterns5k_3_11` conda env and adds `plotly` if missing. |
| `sync_to_cluster.sh` | rsync code (vis_stim + patterns_5k) and data to NOTS. |

## Quick start

On your laptop:
```bash
# 1) Push code+data to NOTS
bash slurm/sync_to_cluster.sh
```

On NOTS (login node):
```bash
# 2) Set up env (only first time -- shared with patterns_5k)
srun --partition=commons --account=commons --cpus-per-task=2 --mem=8G --time=00:30:00 --pty bash
bash ../patterns_5k/slurm/setup_nots_env.sh   # only if not already done
bash slurm/setup_nots_env.sh                   # verifies + adds plotly
exit

# 3) Edit slurm/slurm_sweep_simple_cnn.yaml so datadir points at
#    /scratch/$USER/vis_stim_data

# 4) Dry-run
cd ~/vis_stim
python slurm/slurm_sweep.py \
    --sweep-config slurm/slurm_sweep_simple_cnn.yaml \
    --slurm-template slurm/nots_gpu.slurm \
    --seeds 3 --tasks-per-job 10 --dry-run

# 5) Submit for real
bash slurm/launch_all_sweeps.sh

# 6) Monitor
squeue -u $USER

# 7) When jobs finish, aggregate
bash slurm/aggregate_sweeps.sh \
    $SHARED_SCRATCH/$USER/vis_stim_results/vis_stim_simple_cnn_*
```

## Output structure

```
$SHARED_SCRATCH/$USER/vis_stim_results/vis_stim_simple_cnn_<timestamp>/
  sweep_config.yaml             # copy of the spec
  manifest.txt                  # one slurm_experiment cmd per line
  slurm_job_ids.txt             # submitted array job IDs
  sweep_results.csv             # one row per trial (after aggregation)
  sweep_summary.csv             # mean +/- SEM grouped by config_id
  trial_0000_lr=0.001_dropout=0.3_..._seed0/
    config.yaml
    best.pth, last.pth
    history.json
    training_curves.png
    summary_metrics.json        # best_val_loss, val_corr, test_loss, test_corr, ...
  trial_0001_.../
  ...
```

## Choosing the best run

```bash
# After aggregation, the leaderboard prints to stdout.
# To grab the best model dir programmatically:
python - <<'PY'
import pandas as pd, glob, os
sweep = sorted(glob.glob(os.path.expandvars(
    '$SHARED_SCRATCH/$USER/vis_stim_results/vis_stim_simple_cnn_*')))[-1]
df = pd.read_csv(os.path.join(sweep, 'sweep_results.csv'))
best = df.loc[df['test_corr'].idxmax()]
print(os.path.join(sweep, best['experiment']))
PY
```

Then re-run greedy on that model:
```bash
python run_greedy.py --model-dir <that_path>
```

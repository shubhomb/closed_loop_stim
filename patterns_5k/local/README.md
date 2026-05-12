# Local multi-GPU sweep

A local-workstation analog of `slurm/slurm_sweep.py`. Same sweep YAML
format, same per-trial entry point, same output layout — so the existing
aggregation script works on local results unchanged.

## Quick start

```bash
cd patterns_5k

# Practice run: 4 trials, 1 fold each, ~5 epochs (smoke test both GPUs)
python local/local_sweep.py --sweep-config local/local_sweep_test.yaml

# Aggregate when it finishes (mirrors the slurm workflow)
bash slurm/aggregate_sweeps.sh results/local_local_test_<timestamp>
```

## Running a real sweep locally

The cluster YAMLs hard-code `/scratch/sb272/...` for `datadir`. Override
it with `--datadir`:

```bash
python local/local_sweep.py \
    --sweep-config slurm/slurm_sweep_base_cnn.yaml \
    --datadir data/oracle_ICMS_150/ \
    --gpus 0,1
```

## Flags

- `--sweep-config` — sweep YAML (same format as `slurm/slurm_sweep_*.yaml`).
- `--gpus 0,1` — comma-separated GPU IDs to use (default: `0,1`).
- `--seeds N` — seeds per config when no `k_cross` is set (default 5).
- `--k-cross K` — overrides yaml `k_cross` and `--seeds`.
- `--datadir PATH` — override `base_config.datadir` (cluster YAMLs use cluster paths).
- `--resume DIR` — skip trials whose `summary_metrics.json` already exists.
- `--dry-run` — print the trial plan, don't execute.

## Parallelism model

One worker process per GPU. Each worker sets `CUDA_VISIBLE_DEVICES` to a
single ID **before** importing torch, so the worker only ever sees one
device (as `cuda:0`). Trials are distributed round-robin across workers.

Grid search only. Optuna isn't supported — use it on the cluster or in
`run_sweep.py` if you need adaptive search.

## Watching it run

```bash
watch -n 1 nvidia-smi          # both GPUs should be busy
tail -f results/local_*/local_sweep.log
```

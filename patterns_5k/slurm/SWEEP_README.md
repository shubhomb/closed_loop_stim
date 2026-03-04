# Hyperparameter Sweep System

Automated hyperparameter search for neural stimulation → spike prediction experiments.

## Quick Start

```bash
# Single experiment (uses config.yaml)
python run_experiment.py --config config.yaml

# Single experiment with overrides
python run_experiment.py --config config.yaml --overrides '{"learning_rate": 0.01, "model_type": "mlp"}'

# Preview all experiments in a sweep (no training)
python run_sweep.py --dry-run

# Run full sweep
python run_sweep.py

# Use Optuna bayesian optimization (50 trials)
python run_sweep.py --strategy optuna --n-trials 50

# Random search (30 trials)
python run_sweep.py --strategy random --n-trials 30
```

## Files

| File | Purpose |
|---|---|
| `run_experiment.py` | Standalone script that runs one full experiment (data → train → evaluate → figures). Extracted from `5k_dataset_icms150_simple.ipynb`. |
| `run_sweep.py` | Orchestrator that reads `sweep_config.yaml`, generates experiment configs, and runs them sequentially. |
| `sweep_config.yaml` | Master config defining the search space, base config, and strategy. |

## Output Structure

Each sweep creates a time-stamped directory under `results/`:

```
results/<sweep_name>_<timestamp>/
├── sweep_config.yaml          # Copy of the sweep spec used
├── sweep.log                  # Master log for the sweep
├── sweep_results.csv          # One row per trial with all metrics + hyperparams
├── best_config.yaml           # Config of the best experiment (auto-copied)
├── optuna_trials.csv          # (Optuna only) Full Optuna trial dataframe
│
├── trial_0000_model_type=cnn_lr=0.001_.../
│   ├── config.yaml            # Full config for this experiment
│   ├── experiment.log         # Detailed training log
│   ├── best_stim_spike_model.pt
│   ├── summary_metrics.json   # test_corr, best_val_corr, LOO metrics, etc.
│   ├── training_history.json  # Per-epoch loss/corr/lr
│   ├── training_history.png
│   ├── test_prediction_comparison.png
│   ├── pattern_selectivity_analysis.png
│   ├── model_vs_LOO_single_sample.png
│   └── oracle_trials_by_pattern/
│       ├── pattern_4001.png
│       ├── pattern_4002.png
│       └── ...
│
├── trial_0001_.../
│   └── ...
└── ...
```

## Search Strategies

### Grid (`strategy: grid`)
Exhaustive search over all combinations. Edit `search_space` in `sweep_config.yaml` to define value lists. Distribution specs (e.g. `log_uniform`) are auto-discretized into 3 grid points.

### Random (`strategy: random`)
Random sampling from distributions. Set `n_trials` to control budget.

### Optuna (`strategy: optuna`)
Bayesian optimization using TPE sampler. Requires `pip install optuna`. Automatically falls back to random search if Optuna is not installed.

## Configuring the Search Space

Edit `sweep_config.yaml`. The `search_space` section supports:

```yaml
search_space:
  # Categorical: try each value
  model_type:
    type: categorical
    values: ["cnn", "mlp"]

  # Log-uniform: sample LR in log space 
  learning_rate:
    type: log_uniform
    low: 0.0001
    high: 0.01

  # List values (shorthand for categorical)
  dropout:
    type: categorical
    values: [0.1, 0.2, 0.3]

  # Nested lists work too (e.g. layer configs)
  conv_channels:
    type: categorical
    values:
      - [64]
      - [128]
      - [64, 32]
```

All parameters from `config.yaml` can be searched. Common ones to explore:
- **Architecture**: `model_type`, `conv_channels`, `kernel_sizes`, `fc_dims`, `hidden_dims`
- **Training**: `learning_rate`, `dropout`, `weight_decay`, `batch_size`
- **Dataset**: `init_state`, `history`, `input_bin_size_ms`, `n_input_bins`

## Dependencies

The sweep system uses the same dependencies as the notebook. Optionally:
```bash
pip install optuna  # For Bayesian optimization
```

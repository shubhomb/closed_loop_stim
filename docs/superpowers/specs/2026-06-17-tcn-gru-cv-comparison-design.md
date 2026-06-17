# TCN vs GRU 5-fold CV Comparison

**Date:** 2026-06-17
**Notebook:** `patterns_5k/rnn_pattern_encoder.ipynb`

## Goal

Compare four neural spike-prediction models under 5-fold cross-validation,
using identical metrics, and produce two comparative bar charts (full test
correlation, global fraction variance explained).

## Models (4)

| Name         | Architecture            | history | Config |
|--------------|-------------------------|---------|--------|
| `GRU_hist`   | `SequenceGRU`           | on      | hidden=128, num_layers=2, fc_dims=[512] |
| `GRU_nohist` | `SequenceGRU`           | off     | same |
| `TCN_hist`   | `SimpleCausalSpikeCNN`  | on      | kernel_sizes=[3,3], 2 conv layers, fc_dims=[512] |
| `TCN_nohist` | `SimpleCausalSpikeCNN`  | off     | same |

- TCN is the existing causal-conv CNN (`SimpleCausalSpikeCNN`), 2 conv layers,
  kernel size 3 each, causal (left-only) padding.
- `history` on = `BinnedStimSpikeDataset(history=H)` concatenates lagged
  spike-count channels (`+n_neurons` input channels). Teacher-forced.

## Parameter matching

- **Fixed for all four models:** `fc_dims=[512]`.
- GRU reference config fixed (hidden=128, num_layers=2).
- TCN `conv_channels` (the two conv-layer widths) are tuned so the TCN's
  **no-history** total parameter count is within ~5% of the GRU's no-history
  count. The FC head is NOT changed for matching — only conv channels.
- history/no-history variants of the same architecture differ slightly
  (history adds `n_neurons` input channels). We match the no-history variants;
  all four actual param counts are printed before training.

## Cross-validation scheme

- Train pool = non-oracle trials (fixed). Test = oracle trials (fixed).
- For each fold `f` in 0..4:
  - Resample a fresh validation set from the non-oracle pool with
    `random_state = SEED + f` (~15% of non-oracle), remaining non-oracle = train.
  - Train a fresh model with early stopping on val_corr.
  - Evaluate on the **fixed oracle test set**.
- 4 models x 5 folds = 20 training runs.
- `CV_EPOCHS` / `CV_PATIENCE` exposed at top of CV section (default to HP values)
  so the run can be dialled down.

## Metrics (identical for all models)

Computed on the fixed oracle test set after training:

1. **Full test correlation** — `get_per_neuron_temporal_corr(model, test_loader)`,
   mean over neurons (same as existing `all_test_corr`).
2. **Global fraction variance explained** —
   `fraction_variance_explained(y_true, y_pred, global_variance=True)`, mean over
   neurons. Predictions/targets via `collect_model_preds_and_targets`.

Both from `patterns_5k/metrics.py` (no changes needed).

## Outputs

- Two grouped Plotly bar charts (x = 4 models):
  - Chart 1: full test correlation. Bar = mean over 5 folds, error bar = std err (std/sqrt(5)).
  - Chart 2: global FVE. Same aggregation.
- Saved to `RUN_DIR`. Results dict (`{model: {metric: [per-fold values]}}`) kept
  in-notebook and optionally dumped to JSON.

## Implementation

- New cells appended to `rnn_pattern_encoder.ipynb` after the existing
  single-model flow.
- `run_fold(arch, history, fold)` helper reuses existing `make_dataset`,
  `get_model`, `train_epoch`/`validate`, and the two metric functions.
- No changes to `models.py` or `metrics.py`.

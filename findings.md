# Findings & Decisions

## Current Task: `rnn_pattern_encoder.ipynb` Model Comparison

### Requirements
- Modify `patterns_5k/rnn_pattern_encoder.ipynb`.
- Use `patterns_5k/model_playground.ipynb` as the reference.
- Compare TCN with/without history and GRU with/without history.
- Evaluate R^2 and correlation as a function of coarsening factor, including 10 ms vs smoothed/coarsened 100 ms behavior.
- Add a control MLP with no dynamics that learns instantaneous response to stimulation pulses.

### Research Findings
- `rnn_pattern_encoder.ipynb` already contains a late section titled "5-fold CV comparison: TCN ± history vs GRU ± history" with model specs for `GRU_nohist`, `GRU_hist`, `TCN_nohist`, and `TCN_hist`.
- Existing repo utilities include temporal coarsening helpers:
  - `patterns_5k/utils.py`: `coarsen_2d(arr, factor, method='mean')`.
  - `patterns_5k/metrics.py`: `_coarsen`, `get_per_neuron_temporal_corr(model_tuple, loader, coarse_factor=0)`, `collect_model_preds_and_targets(..., coarse_factor=0)`, and LOO equivalents.
- Existing model utilities include `SequenceGRU`, `SimpleCausalSpikeCNN`, history-aware dataset support, and autoregressive prediction helpers for history models.
- Need inspect `model_playground.ipynb` more narrowly because a broad text search hit huge embedded output.
- `model_playground.ipynb` evaluates temporal correlation and FVE at a user-selected `COARSE_BIN_MS` by deriving `factor = COARSE_BIN_MS // fine_bin_ms`, then calling `get_per_neuron_temporal_corr`, `collect_model_preds_and_targets`, `collect_loo_preds_and_targets`, and `fraction_variance_explained`.
- The current `rnn_pattern_encoder.ipynb` CV section ends with bar plots for a single uncoarsened `test_corr` and `global_fve`. It does not sweep coarsening factors and does not include an MLP control.
- Best path: extend the existing CV section with an additional `MLP_control` model spec and return per-coarsening correlation/R^2 metrics from `run_fold`, then plot metrics vs coarsening factor.
- Implemented in `rnn_pattern_encoder.ipynb`:
  - Updated the CV markdown to describe five models and the coarsening sweep.
  - Added notebook-local `InstantaneousStimMLP`, applied independently to each time bin.
  - Added `MODEL_SPECS` entry `("MLP_control", "mlp", False)`.
  - Added `CV_COARSEN_FACTORS = [0, 2, 3, 5, 6, 10]`, mapping to 10/20/30/50/60/100 ms for 10 ms output bins.
  - Added `evaluate_model_by_coarsening`, which runs the trained model once, coarsens predictions/targets in memory, and computes mean temporal correlation plus global R²/FVE.
  - Replaced final plots with native 10 ms bars and metric-vs-coarsening line plots for correlation and R².

### Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Use global FVE as R² | `metrics.fraction_variance_explained(..., global_variance=True)` already implements a global test-set R²/FVE measure. |
| Keep MLP local to notebook | It is an experiment control, not yet a reusable repo model family. |

### Issues Encountered
| Issue | Resolution |
|-------|------------|
| An f-string architecture-summary line was split while writing notebook JSON. | Reconstructed the affected source line and revalidated with `ast.parse`. |

---

## Requirements
- Populate `vis_stim/data/icms_150_6_2_26/dynamics_analysis.ipynb`.
- Add the same `StimCreator` and same trained TCN model used for greedy pattern generation.
- Source information should come from `vis_stim/data/icms_150_6_2_26/greedy_pattern_gen_6_1_26.ipynb`.
- Required sweep directory: `results/local_local_arch_full_2026-06-03_01-24-25`.
- Use the corresponding `SWEEP_CONFIG` from the source notebook.

## Research Findings
- Source notebook path: `vis_stim/data/icms_150_6_2_26/greedy_pattern_gen_6_1_26.ipynb`.
- Target notebook path: `vis_stim/data/icms_150_6_2_26/dynamics_analysis.ipynb`.
- Source notebook cell 10 defines `SWEEP_DIR = "results/local_local_arch_full_2026-06-03_01-24-25"` and `SWEEP_CONFIG = os.path.join(VIS_STIM_DIR, 'local', 'local_sweep_arch_full.yaml')`.
- Source notebook cell 10 rebuilds `best_model` from the best sweep trial's `config.yaml` and `best.pth`, then aliases `model = best_model` and updates `CFG`.
- Source notebook cell 14 defines `StimCreator`; cell 15 instantiates it as `StimCreator(model, device, channel_to_index, n_bins=VIS_ORIENT_TIME // 10, proj=proj)` for each orientation.
- Target notebook already has model-loading and `StimCreator` sections, so the next step is to compare them against the source notebook and patch only any stale/missing content.
- Target notebook has the same 27 source cells as the source notebook; no target patch was needed.
- The sweep best trial selected by `test_corr` is `trial_0277_lr=3.0e-03_weight_decay=1.0e-05_kernel_sizes=3x3_conv_channels=512x512_fc_dims=512_seed1`.
- The selected trial has `best.pth` and `config.yaml`; key config values include `input_bin_ms=10`, `output_bin_ms=10`, `max_time_ms=660`, `conv_channels=[512, 512]`, `kernel_sizes=[3, 3]`, `fc_dims=[512]`, `encoding_mode=current`, `lr=0.003`, `weight_decay=1.0e-05`, `seed=1`.
- Follow-up edit: `dynamics_analysis.ipynb` now includes an explicit markdown note before the model/sweep loading block and a more specific greedy-pattern `StimCreator` markdown header. Runtime code was left unchanged.
- New follow-up requirement: plot change/loss-to-target distance across greedy optimization steps using the intermediate distances computed inside `StimCreator.create_stim`.
- Implemented distance tracking in `StimCreator.create_stim`:
  - `last_distance_history`: baseline plus every accepted greedy update.
  - `last_distance_step_history`: baseline plus the distance after each 60 ms outer step.
  - `last_distance_update_steps`: the 60 ms step index associated with each accepted update.
- Added plotting cell after greedy stim generation. It uses `distance_step_histories` to plot per-orientation gray traces, mean +/- SEM distance to target, and mean +/- SEM improvement from baseline.
- Added a long arbitrary PC-space reconstruction section. It builds a 390-bin target sequence: 0 deg, gray, 90 deg, gray, 180 deg, gray, 270 deg, final 400 ms gray. This is 65 one-step greedy chunks. Gray is PC=(0, 0). The solver targets PC1/PC2 only via `proj=pca.components_.T`.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Extract source code cells programmatically | Notebook output blobs are large; parsing code cells avoids noisy saved Plotly/HTML output. |
| Keep `create_stim` return values unchanged | Existing downstream cells unpack two values; storing history on the object avoids breaking them. |
| Use 60 ms outer-step history for the main figure | This maps directly to the step loop in `create_stim` and gives equal-length traces across orientations. |
| Add final 400 ms gray epoch | The 0/gray/90/gray/180/gray/270 sequence is 3500 ms; adding 400 ms gray makes 3900 ms, divisible by 60 ms. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Shell Python did not have `pandas`; notebook verification needing `sweep_results.csv` could not use pandas. | Used Python's standard-library `csv` module instead. |

## Resources
- `vis_stim/data/icms_150_6_2_26/greedy_pattern_gen_6_1_26.ipynb`
- `vis_stim/data/icms_150_6_2_26/dynamics_analysis.ipynb`

## Visual/Browser Findings
- None.

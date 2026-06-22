# Findings & Decisions

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

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Extract source code cells programmatically | Notebook output blobs are large; parsing code cells avoids noisy saved Plotly/HTML output. |
| Keep `create_stim` return values unchanged | Existing downstream cells unpack two values; storing history on the object avoids breaking them. |
| Use 60 ms outer-step history for the main figure | This maps directly to the step loop in `create_stim` and gives equal-length traces across orientations. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Shell Python did not have `pandas`; notebook verification needing `sweep_results.csv` could not use pandas. | Used Python's standard-library `csv` module instead. |

## Resources
- `vis_stim/data/icms_150_6_2_26/greedy_pattern_gen_6_1_26.ipynb`
- `vis_stim/data/icms_150_6_2_26/dynamics_analysis.ipynb`

## Visual/Browser Findings
- None.

# Cross-Session Transfer Training — Design

**Date:** 2026-05-31
**Notebook:** `vis_stim/cross_session_training.ipynb`

## Goal

Quantify how many stimulation patterns are needed to finetune a stim→spike
model to a new recording session. A shared CNN backbone is reused across
sessions; per-session input adapters and projection heads handle the differing
stim-channel and neuron counts. The headline measurement is a data-efficiency
curve: held-out test accuracy (Poisson NLL + correlation) of the target session
as a function of the number of target patterns used for finetuning, compared
against a from-scratch baseline.

## Datasets (three distinct recording sessions)

| Key        | Path                                          | Stim ch | Neurons | Oracle structure |
|------------|-----------------------------------------------|---------|---------|------------------|
| `vis_stim` | `vis_stim/data/`                              | 40      | 72      | 100 patterns × 10 reps |
| `icms150`  | `patterns_5k/data/oracle_ICMS_150/`           | 42      | 71      | 50 patterns × 20 reps |
| `icms148`  | `patterns_5k/data/original_ICMS_148/`         | 44      | 47      | **none** (all 5000 unique) |

- `vis_stim` uses `Combined_Pattern_Registrations.pkl` + `All_Shank_Spk_Vecs.npy`.
  Its spike-vector field is `unit_index` (renamed to `neuron_id`).
- `icms150` / `icms148` use `pattern_registrations.pkl` / `Pattern_Registrations.pkl`
  + `SpkVecs.npy`.

**Hard assertion:** `icms148` has no repeated (oracle) patterns, so it can never
serve as the held-out evaluation target. Code asserts `target_key != 'icms148'`
wherever an evaluation target is selected.

## Architecture: `CrossSessionSpikeCNN`

Decomposes the existing `SimpleCausalSpikeCNN` into three parts:

1. **Per-session input adapter** — `Conv1d(n_stim_channels_session, COMMON_CH,
   kernel_size=1)`. Maps each session's stim channels to a shared `COMMON_CH`
   width. Kernel size 1 → no temporal reduction, so `use_init_state` valid-conv
   semantics are preserved. Stored in a `ModuleDict` keyed by session.
2. **Shared backbone** — the `conv_stack` from `SimpleCausalSpikeCNN`, operating
   on `COMMON_CH` input channels. Shared across all sessions.
3. **Per-session projection head** — the `fc` stack mapping backbone features →
   that session's `n_neurons`. Stored in a `ModuleDict` keyed by session.

`forward(x, session_key)` routes `adapter[session_key] → backbone →
head[session_key]`.

Defined in the notebook initially; promotable to `patterns_5k/models.py` later.

## Data pipeline

`build_session_data(session_key, cfg)` per session:

1. Loads pattern registrations + spike vectors via existing utils
   (`read_pattern_json`, `preprocess_pattern_stimulations_df`,
   `trial_breakout_spikes_and_patterns`), handling the per-session file names
   and the spike-column rename.
2. Builds `channel_to_index`, `spiking_neuron_to_index`; returns
   `n_stim_channels`, `n_neurons`.
3. Returns train / held-out `BinnedStimSpikeDataset`s using the shared CFG knobs
   (`current` encoding, 10 ms bins, `use_init_state`, etc.).

Splits per session:

- **Oracle patterns** → held-out evaluation set (repeated patterns).
- **Non-oracle patterns** → training pool.
- `N_TRAIN_EXAMPLES` caps how many non-oracle patterns the source trains on
  (`None` = all).
- `EVAL_USE_SINGLE_TRIALS` (default `False`): when `True`, the held-out eval set
  also folds in single-trial (non-oracle) held-out patterns alongside oracles.

## Experiment config (top of notebook)

```python
CFG_XS = dict(
    SOURCE = 'icms150',          # pretraining session (chosen per run)
    TARGET = 'vis_stim',         # finetuning target (asserted != 'icms148')
    COMMON_CH = 40,              # shared backbone input width (try over-complete e.g. 64 later)
    N_TRAIN_EXAMPLES = None,     # cap on source non-oracle training patterns
    EVAL_USE_SINGLE_TRIALS = False,
    FINETUNE_N_GRID = [10, 25, 50, 100, 250, 500],
    PROTOCOLS = ['frozen_backbone', 'full_finetune', 'from_scratch'],
    SEEDS = [0, 1, 2],
    # plus existing arch/training knobs (conv_channels, kernel_sizes, lr, epochs)
)
```

## Notebook flow

1. **Setup & config** — imports, device, `CFG_XS`, session path registry.
2. **Per-session data loaders** — `build_session_data` for all three; print
   shapes; assert `icms148` never set as eval target.
3. **Model definition** — `CrossSessionSpikeCNN`.
4. **Pretrain on SOURCE** — full training of `adapter[source]` + backbone +
   `head[source]` on source non-oracle data (capped by `N_TRAIN_EXAMPLES`);
   save checkpoint; report source held-out (oracle) NLL + correlation.
5. **Headline eval entry point** — load pretrained model; prediction on the
   TARGET held-out oracle set is the sweep's evaluation.
6. **Finetuning data-efficiency sweep** — `run_finetune_sweep(source_model,
   target_key, cfg)`: for each `N` in `FINETUNE_N_GRID`, each protocol, each
   seed:
   - `frozen_backbone`: pretrained backbone frozen, fresh target adapter+head,
     train on N target patterns.
   - `full_finetune`: pretrained backbone trainable, fresh target adapter+head,
     train on N target patterns.
   - `from_scratch`: random backbone+adapter+head, train on N target patterns.

   Evaluate each on the target held-out set → (Poisson NLL, correlation).
   Collect into a tidy DataFrame; save to `results/<run>/xs_sweep.csv`.
7. **Plot** — two panels (NLL vs N, correlation vs N), one line per protocol,
   error bands over seeds, horizontal source-on-source ceiling line in each.
   Saved to `results/<run>/`.

## Metric

Both metrics reported, averaged over the held-out evaluation set, via the
existing `validate(...)` (returns `(loss, corr)`) and `compute_correlation`:

- **Test Poisson NLL loss** (the training criterion, `PoissonNLLLoss`).
- **Per-bin Pearson correlation** of predicted vs actual spike counts.

`EVAL_USE_SINGLE_TRIALS` optionally extends the eval set with single-trial
held-out patterns.

## Reuse / isolation

Reused unchanged: `read_pattern_json`, `preprocess_pattern_stimulations_df`,
`trial_breakout_spikes_and_patterns`, `BinnedStimSpikeDataset`, `train_epoch`,
`validate`, `compute_correlation`.

New (notebook-local, promotable): `build_session_data`,
`CrossSessionSpikeCNN`, `run_finetune_sweep`, plotting.

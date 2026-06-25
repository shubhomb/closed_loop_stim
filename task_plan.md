# Task Plan: RNN Pattern Encoder Model Comparison

## Goal
Modify `patterns_5k/rnn_pattern_encoder.ipynb` using patterns from `patterns_5k/model_playground.ipynb` so it compares:
- TCN without history
- TCN with history
- GRU without history
- GRU with history
- control MLP with no dynamics / instantaneous stimulation response

The comparison should report R^2 and correlation as a function of temporal coarsening factor, e.g. smoothing/binning from 10 ms up toward 100 ms.

## Current Phase
Complete

## Phases

### Phase 1: Source Discovery
- [x] Inspect `rnn_pattern_encoder.ipynb` current structure.
- [x] Inspect `model_playground.ipynb` for history/no-history and coarsening logic.
- [x] Inspect supporting model/dataset utilities in `models.py`, `utils.py`, and `metrics.py`.
- **Status:** complete

### Phase 2: Design Notebook Changes
- [x] Decide where to insert shared helpers, MLP control, coarsening evaluation, and comparison plots.
- [x] Keep edits compatible with existing notebook variables and repo APIs.
- **Status:** complete

### Phase 3: Implementation
- [x] Patch `rnn_pattern_encoder.ipynb` with comparison helpers and experiment cells.
- [x] Add/adjust any supporting code only if the notebook cannot define it locally.
- **Status:** complete

### Phase 4: Verification
- [x] Validate notebook JSON.
- [x] Parse edited code cells with `ast` where possible.
- [x] Run lightweight smoke checks that do not require full training.
- **Status:** complete

## Key Questions
1. How does `model_playground.ipynb` define/evaluate coarsening factor?
2. How are history channels represented for TCN/GRU in the existing data pipeline?
3. Should the MLP control predict each time bin independently from instantaneous stimulation channels, and ignore temporal context entirely?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Keep the comparison in the notebook unless shared utilities are already present | The user asked to modify `rnn_pattern_encoder.ipynb`; notebook-local experiment helpers avoid broad API churn. |
| Extend the existing CV section instead of adding a parallel workflow | The notebook already had TCN/GRU ± history CV; extending it minimizes duplicated training code. |
| Implement the MLP control as a per-bin shared MLP over stimulation channels | This enforces "no dynamics": no recurrence, convolution, history channels, or temporal context. |
| Collect native predictions once per fold and coarsen in memory | Avoids redundant oracle forward passes for every coarsening factor. |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Notebook cell 23 had an unterminated f-string after structured source replacement | 1 | Repaired the split print line by reconstructing the affected source lines. |
| First two repair scripts hit Python string-quoting syntax errors while trying to emit `{'='*60}` literally | 1-2 | Switched to a double-quoted replacement string and line-index reconstruction. |

---

# Archived Plan: Populate Dynamics Analysis With Greedy Pattern Model Components

## Goal
Add the same `StimCreator` setup and trained TCN model loading used by `greedy_pattern_gen_6_1_26.ipynb` into `dynamics_analysis.ipynb`.

## Current Phase
Complete

## Phases

### Phase 1: Source Discovery
- [x] Locate source and target notebooks.
- [x] Extract `SWEEP_DIR`, `SWEEP_CONFIG`, model loading, and `StimCreator` construction from the source notebook.
- [x] Identify target notebook structure and insertion point.
- **Status:** complete

### Phase 2: Implementation
- [x] Confirm `dynamics_analysis.ipynb` has cells that reproduce the same model and stim creator.
- [x] Preserve existing notebook content.
- **Status:** complete

### Phase 3: Verification
- [x] Parse the target notebook as valid JSON.
- [x] Check the inserted code references the requested sweep and same config.
- [x] Summarize changes for the user.
- **Status:** complete

### Phase 4: Loss-Trajectory Figure
- [x] Track per-optimization-step distances inside `StimCreator.create_stim`.
- [x] Collect distance histories while generating `est_stims`.
- [x] Add a figure cell showing distance to target as a function of optimization step.
- [x] Validate the notebook JSON after edits.
- **Status:** complete

### Phase 5: Long Arbitrary PC-Space Target
- [x] Add a cell that builds a long PC1/PC2 target sequence: 0, gray, 90, gray, 180, gray, 270, final gray.
- [x] Solve independent 60 ms one-step stims for the PC-space target chunks.
- [x] Concatenate the one-step stims and compare model-predicted PC1/PC2 to the target.
- [x] Validate notebook JSON and new code syntax.
- **Status:** complete

## Key Questions
1. How does the source notebook define `SWEEP_CONFIG` and load the trained TCN?
2. How does the source notebook instantiate `StimCreator`?
3. Where should the new cells go in `dynamics_analysis.ipynb`?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Use notebook JSON patching | Keeps the `.ipynb` valid and avoids manual cell-output churn. |
| Leave `dynamics_analysis.ipynb` unchanged | It already contains source-identical cells for the requested model and `StimCreator`. |
| Store distance history on `StimCreator` | Preserves the existing `(created_stim, final_dist)` return signature while exposing the optimization trajectory. |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Shell Python lacked `pandas` for `sweep_results.csv` inspection | 1 | Used standard-library `csv` parsing instead. |

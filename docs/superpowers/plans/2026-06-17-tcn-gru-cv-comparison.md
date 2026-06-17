# TCN vs GRU 5-fold CV Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Append cells to `patterns_5k/rnn_pattern_encoder.ipynb` that train 4 models (TCN±history, GRU±history) under 5-fold CV and produce two Plotly comparison bar charts (full test correlation, global FVE).

**Architecture:** Reuse the notebook's existing pipeline (`make_dataset`, `get_model`, `train_epoch`/`validate`) and metrics (`get_per_neuron_temporal_corr`, `fraction_variance_explained`, `collect_model_preds_and_targets`). A `run_fold(arch, history, fold)` helper trains one fresh model on a per-fold-resampled val split and evaluates on the fixed oracle test set. Results aggregated into a dict → two Plotly bar figures with std-err bars. No changes to `models.py` or `metrics.py`.

**Tech Stack:** PyTorch, scikit-learn (`train_test_split`), Plotly, numpy.

## Global Constraints

- `fc_dims=[512]` for ALL four models (GRU and TCN). Verbatim, not changed for param-matching.
- TCN: `SimpleCausalSpikeCNN`, `kernel_sizes=[3, 3]` (2 conv layers, causal padding, `use_init_state=False`).
- GRU: `SequenceGRU`, `hidden_size=128`, `num_layers=2`.
- Param-match rule: tune TCN `conv_channels` so TCN **no-history** total params are within ±5% of GRU **no-history** total params. Print all four actual counts before training.
- CV: train pool = non-oracle trials (fixed); test = oracle trials (fixed). Per fold `f` in 0..4, resample val from non-oracle with `random_state = SEED + f` (~15%); remaining non-oracle = train. 4 models × 5 folds = 20 runs.
- Metrics (identical for all models, on fixed oracle test set): (1) full test correlation = mean over neurons of `get_per_neuron_temporal_corr`; (2) global FVE = mean over neurons of `fraction_variance_explained(..., global_variance=True)`.
- Bars = mean over 5 folds; error bars = standard error (std/√5).
- All new cells appended AFTER the existing single-model flow (after cell 20). Do not modify existing cells 0–20.
- `CV_EPOCHS` / `CV_PATIENCE` exposed at top of the CV section, defaulting to `HP["num_epochs"]` / `HP["patience"]`.
- History lag: `HP["history"]` is currently `0`. The "history-on" models must use a real lag, so the CV section defines `CV_HISTORY = HP["history"] if HP["history"] > 0 else 1` (1-bin = 10 ms teacher-forced lag) and uses `CV_HISTORY` for history-on datasets and `+n_neurons` input channels. History-off uses `history=0`.

---

### Task 1: CV config + param-matching cell

Adds a markdown header cell and a config/param-matching code cell. Determines TCN `conv_channels` that match the GRU no-history param count, and prints all four counts.

**Files:**
- Modify: `patterns_5k/rnn_pattern_encoder.ipynb` (append cells)

**Interfaces:**
- Consumes (from existing notebook globals): `get_model`, `HP`, `NUM_STIM_LEVELS`, `n_stim_channels`, `n_neurons`, `device`, `SEED`.
- Produces: globals `CV_EPOCHS`, `CV_PATIENCE`, `FC_DIMS` (=[512]), `GRU_KW` (dict), `TCN_KW` (dict), `MODEL_SPECS` (list of `(name, arch, history)` tuples), `count_params(model)->int`. `TCN_KW["conv_channels"]` is set so TCN no-hist params are within ±5% of GRU no-hist.

- [ ] **Step 1: Append markdown cell**

Append a markdown cell:
```markdown
## 5-fold CV comparison: TCN ± history vs GRU ± history

Trains 4 models (TCN/GRU × history on/off) under 5-fold CV. Train pool is the
non-oracle trials; the validation split is resampled per fold; every model is
evaluated on the fixed oracle test set. fc_dims=[512] for all models; TCN conv
channels are tuned to match the GRU no-history parameter count (±5%). Bars show
mean ± standard error over the 5 folds.
```

- [ ] **Step 2: Append config + param-match code cell**

```python
# ---- CV comparison config ----
CV_EPOCHS = HP["num_epochs"]
CV_PATIENCE = HP["patience"]
N_FOLDS = 5
FC_DIMS = [512]
CV_HISTORY = HP["history"] if HP["history"] > 0 else 1   # 10ms teacher-forced lag for history-on models

def count_params(m):
    return sum(p.numel() for p in m.parameters())

# Fixed GRU config (reference for param matching)
def build_gru(history):
    in_ch = n_stim_channels + (n_neurons if history else 0)
    return get_model(
        "gru",
        n_stim_channels=in_ch, n_neurons=n_neurons,
        n_input_bins=HP["n_input_bins"], n_output_bins=HP["n_output_bins"],
        hidden_size=128, num_layers=2, embedding_dim=0,
        fc_dims=FC_DIMS, dropout=HP["dropout"],
        num_stim_levels=NUM_STIM_LEVELS, bidirectional=False,
        init_state=False, n_initial_state_bins=0,
    )

def build_tcn(history, conv_channels):
    in_ch = n_stim_channels + (n_neurons if history else 0)
    return get_model(
        "cnn",
        n_stim_channels=in_ch, n_neurons=n_neurons,
        n_input_bins=HP["n_input_bins"], n_output_bins=HP["n_output_bins"],
        embedding_dim=0, conv_channels=conv_channels, kernel_sizes=[3, 3],
        fc_dims=FC_DIMS, dropout=HP["dropout"], num_stim_levels=NUM_STIM_LEVELS,
        use_batch_norm=True, use_init_state=False,
    )

# Target: GRU no-history param count
gru_target = count_params(build_gru(history=False))

# Search a single conv width w (both layers = w) so TCN no-hist params ~ gru_target
best_w, best_diff = None, float("inf")
for w in range(8, 513, 2):
    p = count_params(build_tcn(history=False, conv_channels=[w, w]))
    diff = abs(p - gru_target)
    if diff < best_diff:
        best_diff, best_w = diff, w
TCN_CONV = [best_w, best_w]

GRU_KW = dict(builder=build_gru)
TCN_KW = dict(builder=lambda history: build_tcn(history, TCN_CONV), conv_channels=TCN_CONV)

MODEL_SPECS = [
    ("GRU_nohist", "gru", False),
    ("GRU_hist",   "gru", True),
    ("TCN_nohist", "cnn", False),
    ("TCN_hist",   "cnn", True),
]

# Report actual counts for all four
print(f"GRU no-hist target params: {gru_target:,}")
print(f"TCN conv channels chosen: {TCN_CONV} (within {100*best_diff/gru_target:.1f}% of target)")
for name, arch, hist in MODEL_SPECS:
    m = build_gru(hist) if arch == "gru" else build_tcn(hist, TCN_CONV)
    print(f"  {name:12s} {count_params(m):>10,} params")
    del m
assert best_diff / gru_target <= 0.05, f"TCN no-hist not within 5% of GRU ({best_diff/gru_target:.1%})"
```

- [ ] **Step 3: Run the cell, verify it passes**

Execute the appended cell (in IDE or `jupyter nbconvert --to notebook --execute --inplace`). Expected: prints GRU target, chosen TCN conv channels with "within X%", and a 4-row param table; the final assert passes (TCN no-hist within 5%). If the assert fails, widen the search range — but [8,512] easily spans the GRU's count, so it should pass.

- [ ] **Step 4: Commit**

```bash
git add patterns_5k/rnn_pattern_encoder.ipynb docs/superpowers/plans/2026-06-17-tcn-gru-cv-comparison.md
git commit -m "Add CV config + TCN/GRU param-matching cell to rnn_pattern_encoder

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `make_cv_loaders` + `run_fold` helper cell

Adds a code cell defining a per-fold dataset/loader builder (history-aware, val resampled per fold) and a `run_fold` function that trains one model and returns the two test metrics.

**Files:**
- Modify: `patterns_5k/rnn_pattern_encoder.ipynb` (append cell)

**Interfaces:**
- Consumes: `unique_trials_info`, `pattern_df`, `spike_responses`, `channel_to_index`, `timing_to_pattern`, `BinnedStimSpikeDataset`, `DataLoader`, `train_epoch`, `validate`, `nn`, `torch`, `device`, `HP`, `SEED` (existing globals); `build_gru`, `build_tcn`, `TCN_CONV`, `CV_EPOCHS`, `CV_PATIENCE`, `CV_HISTORY` (Task 1); `get_per_neuron_temporal_corr`, `collect_model_preds_and_targets`, `fraction_variance_explained` (from `metrics`).
- Produces: `make_cv_dataset(trial_indices, history)`, `run_fold(arch, history, fold) -> dict(test_corr=float, global_fve=float)`.

- [ ] **Step 1: Append the helper cell**

```python
from metrics import (
    get_per_neuron_temporal_corr,
    collect_model_preds_and_targets,
    fraction_variance_explained,
)
from sklearn.model_selection import train_test_split as _tts
import numpy as np

# Fixed pools
_oracle_idx = unique_trials_info[unique_trials_info["is_oracle"]]["pattern_timing_index"].tolist()
_nonoracle_idx = unique_trials_info[~unique_trials_info["is_oracle"]]["pattern_timing_index"].tolist()

def make_cv_dataset(trial_indices, history):
    return BinnedStimSpikeDataset(
        pattern_df, spike_responses, channel_to_index, timing_to_pattern,
        trial_indices=trial_indices,
        input_bin_size_ms=HP["input_bin_size_ms"], output_bin_size_ms=HP["output_bin_size_ms"],
        n_input_bins=HP["n_input_bins"], n_output_bins=HP["n_output_bins"],
        max_time_ms=HP["max_time_ms"], output_offset=HP["output_offset"],
        encoding_mode=HP["encoding_mode"], history=(CV_HISTORY if history else 0),
        init_state=False, n_initial_state_bins=0, logger=logger,
    )

def run_fold(arch, history, fold):
    torch.manual_seed(SEED + fold)
    np.random.seed(SEED + fold)
    # Per-fold resampled val split from the non-oracle pool
    tr_idx, va_idx = _tts(_nonoracle_idx, test_size=0.15, random_state=SEED + fold)

    tr_ds = make_cv_dataset(tr_idx, history)
    va_ds = make_cv_dataset(va_idx, history)
    te_ds = make_cv_dataset(_oracle_idx, history)
    tr_loader = DataLoader(tr_ds, batch_size=HP["batch_size"], shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=HP["batch_size"], shuffle=False)
    te_loader = DataLoader(te_ds, batch_size=HP["batch_size"], shuffle=False)

    model = (build_gru(history) if arch == "gru" else build_tcn(history, TCN_CONV)).to(device)

    if HP["criterion_fn"] == "poisson":
        criterion = nn.PoissonNLLLoss(log_input=True, reduction="none", full=True)
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([HP["weight_loss"]], device=device), reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=HP["learning_rate"], weight_decay=HP["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_corr, best_state, patience_ctr = -float("inf"), None, 0
    for _ in range(CV_EPOCHS):
        train_epoch(model, tr_loader, criterion, optimizer, device,
                    sum_loss=HP["sum_loss"], grad_clip=False,
                    weight_loss=HP["weight_loss"], use_init_state=False)
        val_loss, val_corr = validate(model, va_loader, criterion, device,
                    sum_loss=HP["sum_loss"], weight_loss=HP["weight_loss"], use_init_state=False)
        scheduler.step(val_loss)
        if val_corr > best_corr:
            best_corr, patience_ctr = val_corr, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= CV_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Metrics on fixed oracle test set
    model_tuple = (model, None, device)
    neuron_corrs = get_per_neuron_temporal_corr(model_tuple, te_loader)
    test_corr = float(np.mean(neuron_corrs))
    y_pred, y_true = collect_model_preds_and_targets(model_tuple, te_loader)
    _, global_fve = fraction_variance_explained(y_true, y_pred, global_variance=True)
    return dict(test_corr=test_corr, global_fve=float(global_fve))
```

- [ ] **Step 2: Smoke-test the helper with CV_EPOCHS temporarily = 1**

In a scratch cell (do NOT commit the scratch cell), run:
```python
_saved = CV_EPOCHS
CV_EPOCHS = 1
print(run_fold("gru", False, 0))
print(run_fold("cnn", True, 0))
CV_EPOCHS = _saved
```
Expected: two dicts printed, each with finite `test_corr` and `global_fve` floats (no NaN, no shape error). This confirms `run_fold` returns the right shape for both architectures and both history settings. Delete the scratch cell after.

- [ ] **Step 3: Verify metric-function return shapes match usage**

Confirm in the smoke output that `fraction_variance_explained` second return is a scalar (it returns `(neuron_fve, mean_fve)` per `metrics.py:218`). Confirm `get_per_neuron_temporal_corr` returns a `(n_neurons,)` array so `np.mean` is a scalar. If `collect_model_preds_and_targets` ordering is `(all_pred, all_true)`, the call assigns `y_pred, y_true` correctly (it returns pred first per `metrics.py:87`). Expected: no assertion needed — just confirm finite floats in Step 2 output.

- [ ] **Step 4: Commit**

```bash
git add patterns_5k/rnn_pattern_encoder.ipynb
git commit -m "Add run_fold CV helper to rnn_pattern_encoder

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: CV run loop cell

Adds the cell that loops over the 4 model specs × 5 folds, collecting metrics into `cv_results`.

**Files:**
- Modify: `patterns_5k/rnn_pattern_encoder.ipynb` (append cell)

**Interfaces:**
- Consumes: `MODEL_SPECS`, `N_FOLDS`, `run_fold` (Tasks 1–2), `tqdm`.
- Produces: `cv_results` = `{model_name: {"test_corr": [5 floats], "global_fve": [5 floats]}}`.

- [ ] **Step 1: Append the run-loop cell**

```python
from tqdm import tqdm

cv_results = {name: {"test_corr": [], "global_fve": []} for name, _, _ in MODEL_SPECS}

for name, arch, history in MODEL_SPECS:
    for fold in tqdm(range(N_FOLDS), desc=name):
        out = run_fold(arch, history, fold)
        cv_results[name]["test_corr"].append(out["test_corr"])
        cv_results[name]["global_fve"].append(out["global_fve"])
    tc = np.array(cv_results[name]["test_corr"])
    fv = np.array(cv_results[name]["global_fve"])
    logger.info(f"{name}: test_corr {tc.mean():.4f}±{tc.std(ddof=1)/np.sqrt(N_FOLDS):.4f} | "
                f"global_fve {fv.mean():.4f}±{fv.std(ddof=1)/np.sqrt(N_FOLDS):.4f}")

cv_results
```

- [ ] **Step 2: Run the full loop (or reduced CV_EPOCHS first)**

Execute the cell. For a fast first pass, set `CV_EPOCHS` low (e.g. 5) in the Task 1 cell and re-run from there, then bump back to full. Expected: 20 training runs complete; `cv_results` has 4 keys, each with two lists of 5 finite floats. Verify with:
```python
assert set(cv_results) == {n for n,_,_ in MODEL_SPECS}
for v in cv_results.values():
    assert len(v["test_corr"]) == 5 and len(v["global_fve"]) == 5
    assert all(np.isfinite(v["test_corr"])) and all(np.isfinite(v["global_fve"]))
print("cv_results shape OK")
```
(Put this assert in the same cell or a scratch cell; if scratch, delete before commit.)

- [ ] **Step 3: Commit**

```bash
git add patterns_5k/rnn_pattern_encoder.ipynb
git commit -m "Add 4-model x 5-fold CV run loop

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Plotly comparison bar charts cell

Adds the cell that renders two Plotly bar charts (test correlation, global FVE) with std-err error bars, saves them to `RUN_DIR`, and dumps `cv_results` to JSON.

**Files:**
- Modify: `patterns_5k/rnn_pattern_encoder.ipynb` (append cell)

**Interfaces:**
- Consumes: `cv_results` (Task 3), `MODEL_SPECS`, `N_FOLDS`, `RUN_DIR`, `os`, `json`, `np`.
- Produces: two figures saved as `cv_test_corr.png`/`.html` and `cv_global_fve.png`/`.html` in `RUN_DIR`; `cv_results.json` in `RUN_DIR`.

- [ ] **Step 1: Append the plotting cell**

```python
import plotly.graph_objects as go
import json, os

names = [n for n, _, _ in MODEL_SPECS]
colors = {"GRU_nohist": "#1f77b4", "GRU_hist": "#2ca02c",
          "TCN_nohist": "#ff7f0e", "TCN_hist": "#d62728"}

def _bar_fig(metric, title, ytitle):
    means = [np.mean(cv_results[n][metric]) for n in names]
    sems = [np.std(cv_results[n][metric], ddof=1) / np.sqrt(N_FOLDS) for n in names]
    fig = go.Figure(go.Bar(
        x=names, y=means,
        error_y=dict(type="data", array=sems, visible=True),
        marker_color=[colors[n] for n in names],
        text=[f"{m:.3f}" for m in means], textposition="outside",
    ))
    fig.update_layout(title=title, yaxis_title=ytitle, xaxis_title="model",
                      template="plotly_white", width=600, height=450,
                      showlegend=False)
    return fig

fig_corr = _bar_fig("test_corr",
                    "Full test correlation (oracle set, mean ± SEM over 5 folds)",
                    "per-neuron temporal corr (mean over neurons)")
fig_fve = _bar_fig("global_fve",
                   "Global fraction variance explained (oracle set, mean ± SEM over 5 folds)",
                   "global FVE (mean over neurons)")

fig_corr.write_image(os.path.join(RUN_DIR, "cv_test_corr.png"))
fig_corr.write_html(os.path.join(RUN_DIR, "cv_test_corr.html"))
fig_fve.write_image(os.path.join(RUN_DIR, "cv_global_fve.png"))
fig_fve.write_html(os.path.join(RUN_DIR, "cv_global_fve.html"))

with open(os.path.join(RUN_DIR, "cv_results.json"), "w") as f:
    json.dump(cv_results, f, indent=2)

fig_corr.show()
fig_fve.show()
```

- [ ] **Step 2: Run the cell, verify output**

Execute. Expected: two bar charts render inline with 4 bars each and visible error bars; files appear in `RUN_DIR`. Verify with:
```python
for fn in ["cv_test_corr.png", "cv_global_fve.png", "cv_results.json"]:
    assert os.path.exists(os.path.join(RUN_DIR, fn)), fn
print("artifacts written to", RUN_DIR)
```
If `write_image` errors with a kaleido message, run `pip install -U kaleido` (Plotly's static-export engine); the `.html` writes don't need it. If kaleido cannot be installed, drop the two `write_image` lines and keep `.html` + `.show()`.

- [ ] **Step 3: Commit**

```bash
git add patterns_5k/rnn_pattern_encoder.ipynb
git commit -m "Add Plotly CV comparison bar charts + JSON dump

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- 4 models (TCN±hist, GRU±hist) → Task 1 `MODEL_SPECS`, Task 3 loop. ✓
- fc_dims=[512] all models → Task 1 `FC_DIMS`, both builders. ✓
- TCN kernel [3,3], 2 layers → Task 1 `build_tcn`. ✓
- Param-match TCN no-hist to GRU no-hist within 5% via conv channels → Task 1 search + assert. ✓
- Print 4 actual param counts → Task 1 Step 2. ✓
- CV: fixed non-oracle train, per-fold resampled val (`SEED+f`), fixed oracle test → Task 2 `run_fold`. ✓
- Identical metrics (full test corr, global FVE) → Task 2 metric calls. ✓
- Mean ± std err bars → Task 4 `_bar_fig`. ✓
- Plotly charts → Task 4. ✓
- CV_EPOCHS/CV_PATIENCE exposed → Task 1. ✓

**Placeholder scan:** No TBD/TODO; all code blocks complete.

**Type consistency:** `run_fold` returns `dict(test_corr, global_fve)` (Task 2) consumed by `cv_results[...][...]` keys `"test_corr"`/`"global_fve"` (Tasks 3–4). `build_gru(history)`/`build_tcn(history, conv)` signatures consistent across Tasks 1–2. `fraction_variance_explained` returns `(neuron_fve, mean_fve)` — Task 2 unpacks `_, global_fve`. `collect_model_preds_and_targets` returns `(all_pred, all_true)` — Task 2 unpacks `y_pred, y_true`. ✓

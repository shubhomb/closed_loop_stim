"""
Standalone experiment runner extracted from 5k_dataset_icms150_simple.ipynb.

Usage:
    python run_experiment.py --config config.yaml
    python run_experiment.py --config config.yaml --output-dir results/my_experiment
    python run_experiment.py --config config.yaml --overrides '{"learning_rate": 0.01, "model_type": "mlp"}'

All model artifacts, logs, and figures are saved under the experiment output directory.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import (
    SimpleCausalSpikeCNN,
    HistoryCacheCausalCNN,
    train_epoch,
    validate,
)
from utils import (
    DELAY_MODE_TO_INDEX,
    NO_STIM_INDEX,
    NUM_STIM_LEVELS,
    BinnedStimSpikeDataset,
    make_spikes_responses_df,
    preprocess_pattern_stimulations_df,
    read_pattern_json,
    trial_breakout_spikes_and_patterns,
    load_model_from_run_dir,
    load_raw_data
)
from viz import (
    analyze_pattern_responses_by_pattern_name,
    plot_oracle_trials_by_pattern,
    plot_pattern_selectivity,
    plot_psth_per_neuron,
    plot_spike_bin_distribution,
    plot_test_prediction_comparison,
)

from metrics import compute_correlation, fraction_variance_explained


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logger(log_file: str) -> logging.Logger:
    """Create a fresh logger that writes to both file and stdout."""
    # Remove any stale root handlers
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def make_device(logger: logging.Logger) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        logger.info("Using Apple Silicon MPS acceleration")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

# Keys that affect dataset construction — if all match between runs the
# datasets can be shared.
_DATASET_CONFIG_KEYS = frozenset([
    "datadir", "problematic_neurons", "seed", "split_mode",
    "input_bin_size_ms", "output_bin_size_ms", "n_input_bins", "n_output_bins",
    "output_offset", "max_time_ms", "encoding_mode", "init_state",
    "n_initial_state_bins", "history", "batch_size",
])

def run_experiment(cfg: dict, run_dir: str, preloaded_data: dict = None) -> dict:
    """
    Execute a single experiment end-to-end.

    Parameters
    ----------
    cfg : dict
        Full experiment configuration (as loaded from YAML + any overrides).
    run_dir : str
        Directory to write all outputs into.
    preloaded_data : dict, optional
        Output of ``load_raw_data(cfg)``.  When provided the expensive I/O
        step is skipped and the pre-loaded data is reused.

    Returns
    -------
    dict
        Summary metrics (test_loss, test_corr, best_val_corr, etc.)
    """

    os.makedirs(run_dir, exist_ok=True)

    # Save the config used for this run
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    logger = setup_logger(os.path.join(run_dir, "experiment.log"))
    logger.info("=" * 60)
    logger.info(f"Experiment run directory: {run_dir}")
    logger.info("=" * 60)

    # ---- unpack config ----
    datadir = cfg["datadir"]
    problematic_neurons = cfg.get("problematic_neurons", [])
    SEED = cfg.get("seed", 42)
    SPLIT_MODE = cfg.get("split_mode", "oracle")
    INPUT_BIN_SIZE_MS = cfg["input_bin_size_ms"]
    OUTPUT_BIN_SIZE_MS = cfg["output_bin_size_ms"]
    N_INPUT_BINS = cfg["n_input_bins"]
    N_OUTPUT_BINS = cfg["n_output_bins"]
    OUTPUT_OFFSET = cfg.get("output_offset", 0)
    MAX_TIME_MS = cfg.get("max_time_ms", 600)
    BATCH_SIZE = cfg.get("batch_size", 16)
    ENCODING_MODE = cfg.get("encoding_mode", "current")
    INIT_STATE = cfg.get("init_state", False)
    N_INITIAL_STATE_BINS = cfg.get("n_initial_state_bins", 1)
    HISTORY = cfg.get("history", 0)
    SKIP_ORACLE_PLOTS = cfg.get("skip_oracle_plots", False)

    # For CNN with valid convolution + init_state: auto-compute n_initial_state_bins
    # Only the valid-conv reduction; history lookback is handled in __getitem__
    _model_type = cfg.get("model_type", "cnn")
    if _model_type == "cache_cnn":
        # Cache model manages its own spike history via the cache — force
        # history=0 (stim-only dataset) and init_state=False (causal padding).
        HISTORY = 0
        INIT_STATE = False
        N_INITIAL_STATE_BINS = 0
    elif INIT_STATE and _model_type in ("cnn",):
        _kernel_sizes = cfg.get("kernel_sizes", [60])
        N_INITIAL_STATE_BINS = sum(k - 1 for k in _kernel_sizes)

    # USE_INIT_STATE: whether train/validate expect a 3-tuple (x, y, init)
    # from the DataLoader.  For CNN, init_state is prepended in the time dim
    # by the dataset and history channels — the dataset returns 2-tuples.
    # For RNN (without history), init_state is a separate tensor (3-tuple).
    if INIT_STATE and _model_type == "rnn" and (HISTORY is None or HISTORY < 1):
        USE_INIT_STATE = True
    else:
        USE_INIT_STATE = False

    # ================================================================
    # 1. Load data (reuse preloaded_data if provided)
    # ================================================================
    if preloaded_data is not None:
        logger.info("Using preloaded raw data (skipping I/O)")
        pattern_df = preloaded_data["pattern_df"]
        spike_responses = preloaded_data["spike_responses"]
        pattern_stims = preloaded_data["pattern_stims"]
        pattern_polarities = preloaded_data["pattern_polarities"]
        channel_to_index = preloaded_data["channel_to_index"]
        timing_to_pattern = preloaded_data["timing_to_pattern"]
        unique_trials = preloaded_data["unique_trials"]
        spiking_neurons = preloaded_data["spiking_neurons"]
    else:
        raw = load_raw_data(cfg, logger=logger)
        pattern_df = raw["pattern_df"]
        spike_responses = raw["spike_responses"]
        pattern_stims = raw["pattern_stims"]
        pattern_polarities = raw["pattern_polarities"]
        channel_to_index = raw["channel_to_index"]
        timing_to_pattern = raw["timing_to_pattern"]
        unique_trials = raw["unique_trials"]
        spiking_neurons = raw["spiking_neurons"]

    logger.info(f"Stimulation channels: {len(channel_to_index)}, Spiking neurons: {len(spiking_neurons)}")
    logger.info(f"Unique patterns: {len(pattern_stims)}, Total trials: {len(spike_responses)}")

    # ================================================================
    # 2. Train / val / test splits
    # ================================================================
    all_timing_indices = list(spike_responses.keys())
    unique_trials_info = pattern_df[["pattern_timing_index", "pattern_name", "is_oracle"]].drop_duplicates()

    if SPLIT_MODE == "oracle":
        oracle_timing = unique_trials_info[unique_trials_info["is_oracle"]]["pattern_timing_index"].tolist()
        sample_timing = unique_trials_info[~unique_trials_info["is_oracle"]]["pattern_timing_index"].tolist()
        test_indices = oracle_timing
        train_indices, val_indices = train_test_split(sample_timing, test_size=0.15, random_state=SEED)
    else:
        train_val, test_indices = train_test_split(all_timing_indices, test_size=0.15, random_state=SEED)
        train_indices, val_indices = train_test_split(train_val, test_size=0.176, random_state=SEED)

    logger.info(f"Split ({SPLIT_MODE}): train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    dataset_kwargs = dict(
        pattern_df=pattern_df,
        spike_responses=spike_responses,
        channel_to_index=channel_to_index,
        timing_to_pattern=timing_to_pattern,
        input_bin_size_ms=INPUT_BIN_SIZE_MS,
        output_bin_size_ms=OUTPUT_BIN_SIZE_MS,
        n_input_bins=N_INPUT_BINS,
        n_output_bins=N_OUTPUT_BINS,
        max_time_ms=MAX_TIME_MS,
        output_offset=OUTPUT_OFFSET,
        encoding_mode=ENCODING_MODE,
        init_state=INIT_STATE,
        n_initial_state_bins=N_INITIAL_STATE_BINS,
        history=HISTORY,
        logger=logger,
    )

    train_dataset = BinnedStimSpikeDataset(trial_indices=train_indices, **dataset_kwargs)
    val_dataset = BinnedStimSpikeDataset(trial_indices=val_indices, **dataset_kwargs)
    test_dataset = BinnedStimSpikeDataset(trial_indices=test_indices, **dataset_kwargs)

    NUM_WORKERS = cfg.get("num_workers", 4)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0)

    n_stim_channels = train_dataset.n_channels
    n_neurons = train_dataset.n_neurons

    if ENCODING_MODE == "current" and HISTORY is not None and HISTORY > 0:
        n_model_input_channels = n_stim_channels + n_neurons
    else:
        n_model_input_channels = n_stim_channels

    logger.info(f"Samples — train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
    logger.info(f"Input channels: {n_model_input_channels}, Neurons: {n_neurons}")

    device = make_device(logger)

    # ================================================================
    # 3. Build model
    # ================================================================
    MODEL_TYPE = cfg.get("model_type", "cnn")
    EMBEDDING_DIM = cfg.get("embedding_dim", 0)
    DROPOUT = cfg.get("dropout", 0.2)
    HIDDEN_DIMS = cfg.get("hidden_dims", [128, 64])
    CONV_CHANNELS = cfg.get("conv_channels", [128])
    KERNEL_SIZES = cfg.get("kernel_sizes", [60])
    FC_DIMS = cfg.get("fc_dims", [256])
    INIT_BIAS = cfg.get("init_bias", None)
    POOLING_CNN = cfg.get("pooling", "none")
    LINEAR = cfg.get("linear", True)

    if MODEL_TYPE == "cnn":
        model = SimpleCausalSpikeCNN(
            n_stim_channels=n_model_input_channels,
            n_neurons=n_neurons,
            n_input_bins=N_INPUT_BINS,
            n_output_bins=N_OUTPUT_BINS,
            embedding_dim=EMBEDDING_DIM,
            conv_channels=CONV_CHANNELS,
            kernel_sizes=KERNEL_SIZES,
            fc_dims=FC_DIMS,
            dropout=DROPOUT,
            num_stim_levels=NUM_STIM_LEVELS,
            use_init_state=INIT_STATE,
        ).to(device)
    elif MODEL_TYPE == "cache_cnn":
        CACHE_SIZE = cfg.get("cache_size", 5)
        CACHE_EMBED_DIM = cfg.get("cache_embed_dim", 8)
        USE_BATCH_NORM = cfg.get("use_batch_norm", True)
        model = HistoryCacheCausalCNN(
            n_stim_channels=n_stim_channels,
            n_neurons=n_neurons,
            n_input_bins=N_INPUT_BINS,
            n_output_bins=N_OUTPUT_BINS,
            conv_channels=CONV_CHANNELS,
            kernel_sizes=KERNEL_SIZES,
            fc_dims=FC_DIMS,
            dropout=DROPOUT,
            cache_size=CACHE_SIZE,
            cache_embed_dim=CACHE_EMBED_DIM,
            use_batch_norm=USE_BATCH_NORM,
            use_init_state=False,  # always causal padding
        ).to(device)
    elif MODEL_TYPE == "rnn":
        LATENT_DIM = cfg.get("latent_dim", 128)
        NUM_GRU_LAYERS = cfg.get("num_gru_layers", 1)
        model = SpikeRNN(
            n_stim_channels=n_model_input_channels,
            n_neurons=n_neurons,
            n_input_bins=N_INPUT_BINS,
            n_output_bins=N_OUTPUT_BINS,
            embedding_dim=EMBEDDING_DIM,
            latent_dim=LATENT_DIM,
            n_initial_state_bins=N_INITIAL_STATE_BINS,
            num_stim_levels=NUM_STIM_LEVELS,
            num_gru_layers=NUM_GRU_LAYERS,
            dropout=DROPOUT,
            fc_dims=FC_DIMS,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {MODEL_TYPE}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {MODEL_TYPE.upper()} — {trainable_params:,} trainable params ({total_params:,} total)")

    # ================================================================
    # 4. Training
    # ================================================================
    NUM_EPOCHS = cfg.get("num_epochs", 50)
    LEARNING_RATE = cfg.get("learning_rate", 0.001)
    WEIGHT_DECAY = float(cfg.get("weight_decay", 0))  # guard against YAML parsing as string
    WEIGHT_LOSS = cfg.get("weight_loss", 1)
    PATIENCE = cfg.get("patience", 10)
    CRITERION_FN = cfg.get("criterion_fn", "poisson")
    SUM_LOSS = cfg.get("sum_loss", False)

    if CRITERION_FN == "poisson":
        criterion = nn.PoissonNLLLoss(log_input=True, reduction="none", full=True)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([WEIGHT_LOSS], device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history = {"train_loss": [], "val_loss": [], "val_corr": [], "lr": []}
    best_val_loss = float("inf")
    best_val_corr = -float("inf")
    patience_counter = 0
    model_save_path = os.path.join(run_dir, "best_stim_spike_model.pt")

    IS_CACHE_MODEL = MODEL_TYPE == "cache_cnn"
    CACHE_TRAIN_MODE = cfg.get("cache_train_mode", "teacher_forcing")

    # AMP setup
    USE_AMP = cfg.get("use_amp", True) and device.type == "cuda"
    scaler = GradScaler('cuda', enabled=USE_AMP)
    if USE_AMP:
        logger.info("Automatic Mixed Precision (AMP) enabled")

    logger.info("Starting training …")
    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            sum_loss=SUM_LOSS, grad_clip=False, weight_loss=WEIGHT_LOSS,
            use_init_state=USE_INIT_STATE,
            is_cache_model=IS_CACHE_MODEL, cache_mode=CACHE_TRAIN_MODE,
            scaler=scaler, use_amp=USE_AMP,
        )
        val_loss, val_corr = validate(
            model, val_loader, criterion, device,
            sum_loss=SUM_LOSS, weight_loss=WEIGHT_LOSS,
            use_init_state=USE_INIT_STATE,
            is_cache_model=IS_CACHE_MODEL, cache_mode=CACHE_TRAIN_MODE,
            use_amp=USE_AMP,
        )

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_corr"].append(val_corr)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        logger.info(
            f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"Corr: {val_corr:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_corr > best_val_corr:
            best_val_corr = val_corr
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"  ↳ Saved best model (val_corr={best_val_corr:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"Training complete — best val corr: {best_val_corr:.6f}")

    with open(os.path.join(run_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ================================================================
    # 5. Test evaluation
    # ================================================================
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    test_loss, test_corr = validate(
        model, test_loader, criterion, device,
        sum_loss=SUM_LOSS, weight_loss=WEIGHT_LOSS,
        use_init_state=USE_INIT_STATE,
        is_cache_model=IS_CACHE_MODEL, cache_mode=CACHE_TRAIN_MODE,
        use_amp=USE_AMP,
    )
    logger.info(f"Test loss: {test_loss:.6f}, Test corr: {test_corr:.6f}")

    # Generate predictions on test set
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            if USE_INIT_STATE:
                bx, by, bi = batch
                bx = bx.to(device, non_blocking=True)
                with autocast('cuda', enabled=USE_AMP):
                    preds = model(bx)
            elif IS_CACHE_MODEL:
                bx, by = batch
                bx, by_dev = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
                with autocast('cuda', enabled=USE_AMP):
                    preds = model(bx, by_dev, mode=CACHE_TRAIN_MODE)
            else:
                bx, by = batch
                bx = bx.to(device, non_blocking=True)
                with autocast('cuda', enabled=USE_AMP):
                    preds = model(bx)
            all_preds.append(preds.cpu())
            all_targets.append(by)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # Convert log-rates to rates for FVE (model outputs log-rates for PoissonNLLLoss)
    all_preds_rates = np.exp(all_preds)

    # FVE — global variance: denominator is per-neuron variance across ALL trials+time
    # (how much of the across-trial/time variance does the model explain, per neuron)
    neuron_fve_global, fve_global_mean = fraction_variance_explained(
        all_targets, all_preds_rates, global_variance=True
    )
    # FVE — local variance: denominator is per-neuron variance WITHIN each trial
    # (how much of the within-trial temporal variance does the model explain, per neuron,
    # averaged over trials)
    neuron_fve_local, fve_local_mean = fraction_variance_explained(
        all_targets, all_preds_rates, global_variance=False
    )
    logger.info(f"FVE (global, per-neuron mean): {fve_global_mean:.6f}  "
                f"| median: {float(np.median(neuron_fve_global)):.6f}")
    logger.info(f"FVE (local/within-trial, per-neuron mean): {fve_local_mean:.6f}  "
                f"| median: {float(np.median(neuron_fve_local)):.6f}")

    # ── Autoregressive (AR) inference ──
    if HISTORY is not None and HISTORY > 0:
        logger.info("Computing autoregressive (AR) test predictions …")
        ar_preds, ar_targets = _compute_ar_test_predictions(
            model, test_dataset, device, n_neurons,
            n_input_bins=N_INPUT_BINS, n_output_bins=N_OUTPUT_BINS,
            output_offset=OUTPUT_OFFSET, history=HISTORY,
            init_state=INIT_STATE, n_initial_state_bins=N_INITIAL_STATE_BINS,
            max_time_ms=MAX_TIME_MS, input_bin_size_ms=INPUT_BIN_SIZE_MS,
            output_bin_size_ms=OUTPUT_BIN_SIZE_MS,
        )
        _, ar_fve_mean = fraction_variance_explained(
            ar_targets, ar_preds, global_variance=True,
        )
        # Per-neuron temporal correlation (concatenated across all samples)
        ar_corrs = []
        for nidx in range(n_neurons):
            true_n = ar_targets[:, nidx, :].ravel()
            pred_n = ar_preds[:, nidx, :].ravel()
            if true_n.std() > 1e-6 and pred_n.std() > 1e-6:
                ar_corrs.append(pearsonr(true_n, pred_n)[0])
            else:
                ar_corrs.append(0.0)
        ar_test_corr = float(np.mean(ar_corrs))
        logger.info(f"AR FVE: {ar_fve_mean:.6f}, AR test corr: {ar_test_corr:.6f}")
    else:
        ar_fve_mean = float('nan')
        ar_test_corr = float('nan')

    # ================================================================
    # 6. Figures
    # ================================================================
    logger.info("Generating figures …")

    # 6a. Test prediction comparison
    try:
        fig, axes, corr_val, pval = plot_test_prediction_comparison( 
            all_preds, all_targets,
            savepath=os.path.join(run_dir, "test_prediction_comparison.png"),
            logger=logger,
        )
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Could not generate test prediction comparison plot: {e}")

    # 6b. Training history
    if history and len(history.get("train_loss", [])) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(history["train_loss"], label="Train Loss")
        axes[0].plot(history["val_loss"], label="Val Loss")
        axes[0].axhline(y=test_loss, color="r", linestyle="--", label="Test Loss")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss curves")
        axes[0].legend(); axes[0].set_yscale("log"); axes[0].grid(True)
        axes[1].plot(history["lr"], color="green")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("LR"); axes[1].set_title("Learning Rate")
        axes[1].set_yscale("log"); axes[1].grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "training_history.png"), dpi=150)
        plt.close(fig)

    # 6c. Pattern selectivity (if oracle test set)
    try:
        pattern_names, pat_true, pat_pred, pat_counts = analyze_pattern_responses_by_pattern_name(
            model, test_dataset, device, use_init_state=USE_INIT_STATE,
        )
        fig, axes, neuron_correlations = plot_pattern_selectivity(
            pattern_names, pat_true, pat_pred,
            savepath=os.path.join(run_dir, "pattern_selectivity_analysis.png"),
            logger=logger,
        )
        plt.close(fig)
        mean_pattern_corr = float(np.mean(neuron_correlations))
        logger.info(f"Pattern selectivity — mean neuron corr: {mean_pattern_corr:.4f}")
    except Exception as e:
        logger.warning(f"Could not generate pattern selectivity plot: {e}")
        mean_pattern_corr = float("nan")

    # 6d. LOO baseline (single-sample level)
    try:
        loo_metrics = _compute_loo_baseline(
            model, test_dataset, spike_responses, device, n_neurons,
            USE_INIT_STATE, run_dir, logger,
        )
    except Exception as e:
        logger.warning(f"LOO baseline failed: {e}")
        loo_metrics = {}

    # 6e. Oracle trial visualizations
    if SKIP_ORACLE_PLOTS:
        logger.info("Skipping oracle trial plots (skip_oracle_plots=True)")
    else:
      try:
        out_base = os.path.join(run_dir, "oracle_trials_by_pattern")
        plot_oracle_trials_by_pattern(
            model=model,
            test_dataset=test_dataset,
            pattern_df=pattern_df,
            spike_responses=spike_responses,
            pattern_polarities=pattern_polarities,
            output_bin_size_ms=OUTPUT_BIN_SIZE_MS,
            n_input_bins=N_INPUT_BINS,
            n_output_bins=N_OUTPUT_BINS,
            output_offset=OUTPUT_OFFSET,
            max_time_ms=1000,
            n_neurons=n_neurons,
            device=device,
            out_base=out_base,
            input_bin_size_ms=INPUT_BIN_SIZE_MS,
            logger=logger,
            pattern_limit=None,
            use_init_state=USE_INIT_STATE,
            n_initial_state_bins=N_INITIAL_STATE_BINS,
            history=HISTORY,
            init_state=INIT_STATE,
        )
      except Exception as e:
        logger.warning(f"Could not generate oracle trial plots: {e}")

      # 6f. Per-neuron PSTH (model vs ground-truth)
      try:
          psth_dir = os.path.join(run_dir, "psth_per_neuron")
          plot_psth_per_neuron(
              model=model,
              test_dataset=test_dataset,
              device=device,
              out_dir=psth_dir,
              output_bin_size_ms=OUTPUT_BIN_SIZE_MS,
              use_init_state=USE_INIT_STATE,
              logger=logger,
          )
      except Exception as e:
          logger.warning(f"Could not generate PSTH figures: {e}")

    # ================================================================
    # 7. Summary metrics
    # ================================================================
    summary = {
        "run_dir": run_dir,
        "model_type": MODEL_TYPE,
        "batch_avg_test_loss": float(test_loss),
        "batch_avg_test_corr": float(test_corr),
        "all_test_fve_global": float(fve_global_mean),   # per-neuron FVE, variance across all trials
        "all_test_fve_local": float(fve_local_mean),     # per-neuron FVE, variance within each trial
        "AR_FVE": float(ar_fve_mean),
        "AR_test_correlation": float(ar_test_corr),
        "best_val_corr": float(best_val_corr),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "epochs_trained": len(history["train_loss"]),
        "mean_pattern_corr": mean_pattern_corr,
        "history": HISTORY,
        "init_state": INIT_STATE,
        **loo_metrics,
    }
    # Add architecture details for CNN
    if MODEL_TYPE == "cnn":
        summary["kernel_sizes"] = KERNEL_SIZES
        summary["conv_channels"] = CONV_CHANNELS
        summary["fc_dims"] = FC_DIMS
    elif MODEL_TYPE == "cache_cnn":
        summary["kernel_sizes"] = KERNEL_SIZES
        summary["conv_channels"] = CONV_CHANNELS
        summary["fc_dims"] = FC_DIMS
        summary["cache_size"] = CACHE_SIZE
        summary["cache_embed_dim"] = CACHE_EMBED_DIM

    with open(os.path.join(run_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Experiment complete. Summary: {json.dumps(summary, indent=2)}")
    return summary


# ---------------------------------------------------------------------------
# LOO baseline helpers
# ---------------------------------------------------------------------------

def _compute_loo_average_rate(
    model, test_dataset, device, n_neurons, use_init_state,
):
    """
    Compute LOO baseline using time-averaged firing rates.

    Each sample's spike counts are averaged across output time bins before
    computing per-neuron Pearson correlations across samples.

    Returns
    -------
    diff, model_neuron_corrs, loo_neuron_corrs : np.ndarray
        Arrays of shape (n_neurons,).
    """

    # Map pattern names → trial timing indices
    pattern_to_timing = defaultdict(list)
    for timing_idx in test_dataset.trial_indices:
        pname = test_dataset.timing_to_pattern[timing_idx]
        pattern_to_timing[pname].append(timing_idx)

    n_samples = len(test_dataset)
    sample_true_rates = np.zeros((n_samples, n_neurons))
    sample_loo_rates = np.zeros((n_samples, n_neurons))
    sample_model_rates = np.zeros((n_samples, n_neurons))

    # -- Collect model predictions (batched) --
    model.eval()
    loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    model_idx = 0
    with torch.no_grad():
        for batch in loader:
            if use_init_state and len(batch) == 3:
                bx, by, bi = batch
                bx = bx.to(device)
                pred = torch.exp(model(bx)).cpu().numpy()
            else:
                bx, by = batch[:2]
                bx = bx.to(device)
                pred = torch.exp(model(bx)).cpu().numpy()
            # pred shape: (Batch, Neurons, Time) → average over time
            pred_rate = pred.mean(axis=2)
            bs = pred_rate.shape[0]
            sample_model_rates[model_idx : model_idx + bs] = pred_rate
            model_idx += bs

    # -- Collect true & LOO rates (sample by sample) --
    for i in range(n_samples):
        timing_idx, t = test_dataset.samples[i]
        pattern_name_i = test_dataset.timing_to_pattern[timing_idx]
        spikes_full = test_dataset.spike_responses_binned[timing_idx]
        out_start = t + test_dataset.output_offset
        y = spikes_full[:, out_start : out_start + test_dataset.n_output_bins]
        y_rate = y.mean(axis=1)

        other_timings = [ti for ti in pattern_to_timing[pattern_name_i] if ti != timing_idx]
        if len(other_timings) == 0:
            loo_rate = np.zeros_like(y_rate)
        else:
            acc = np.zeros_like(y, dtype=np.float64)
            for oti in other_timings:
                other_spikes = test_dataset.spike_responses_binned[oti]
                acc += other_spikes[:, out_start : out_start + test_dataset.n_output_bins]
            loo_rate = (acc / len(other_timings)).mean(axis=1)

        sample_true_rates[i] = y_rate
        sample_loo_rates[i] = loo_rate

    # -- Per-neuron correlations --
    loo_neuron_corrs, model_neuron_corrs = [], []
    for nidx in range(n_neurons):
        true_n = sample_true_rates[:, nidx]
        loo_n = sample_loo_rates[:, nidx]
        model_n = sample_model_rates[:, nidx]
        r_loo = pearsonr(true_n, loo_n)[0] if true_n.std() > 0 and loo_n.std() > 0 else 0.0
        r_model = pearsonr(true_n, model_n)[0] if true_n.std() > 0 and model_n.std() > 0 else 0.0
        loo_neuron_corrs.append(r_loo)
        model_neuron_corrs.append(r_model)

    loo_neuron_corrs = np.array(loo_neuron_corrs)
    model_neuron_corrs = np.array(model_neuron_corrs)
    diff = model_neuron_corrs - loo_neuron_corrs

    return diff, model_neuron_corrs, loo_neuron_corrs


def _compute_loo_temporal(
    model, test_dataset, device, n_neurons, use_init_state,
):
    """
    Compute LOO baseline preserving temporal dynamics.

    Correlations are computed on the concatenated time-series (all trials,
    all time bins) per neuron.

    Returns
    -------
    diff, model_neuron_corrs, loo_neuron_corrs : np.ndarray
        Arrays of shape (n_neurons,).
    """

    # Map pattern names → trial timing indices
    pattern_to_timing = defaultdict(list)
    for timing_idx in test_dataset.trial_indices:
        pname = test_dataset.timing_to_pattern[timing_idx]
        pattern_to_timing[pname].append(timing_idx)

    n_samples = len(test_dataset)
    n_bins = test_dataset.n_output_bins
    total_time_points = n_samples * n_bins

    # Pre-allocate flattened arrays: (Total_Time, Neurons)
    flat_true_rates = np.zeros((total_time_points, n_neurons), dtype=np.float32)
    flat_loo_rates = np.zeros((total_time_points, n_neurons), dtype=np.float32)
    flat_model_rates = np.zeros((total_time_points, n_neurons), dtype=np.float32)

    # -- Collect model predictions (batched) --
    model.eval()
    loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            if use_init_state and len(batch) >= 3:
                bx, _, bi = batch[:3]
                bx, bi = bx.to(device), bi.to(device)
                raw_pred = model(bx, initial_spikes=bi)
            else:
                bx = batch[0].to(device)
                raw_pred = model(bx)

            # Output shape: (Batch, Neurons, Time)
            pred = torch.exp(raw_pred).cpu().numpy()
            # → (Batch, Time, Neurons) → (Batch*Time, Neurons)
            pred = pred.transpose(0, 2, 1)
            pred_flat = pred.reshape(-1, n_neurons)

            end_cursor = cursor + pred_flat.shape[0]
            flat_model_rates[cursor:end_cursor] = pred_flat
            cursor = end_cursor

    # -- Collect true & LOO (sample by sample) --
    cursor = 0
    for i in range(n_samples):
        timing_idx, t = test_dataset.samples[i]
        pattern_name_i = test_dataset.timing_to_pattern[timing_idx]
        spikes_full = test_dataset.spike_responses_binned[timing_idx]
        out_start = t + test_dataset.output_offset

        # Shape: (Neurons, Time)
        y = spikes_full[:, out_start : out_start + n_bins]

        other_timings = [ti for ti in pattern_to_timing[pattern_name_i] if ti != timing_idx]
        if len(other_timings) == 0:
            loo_rate = np.zeros_like(y)
        else:
            acc = np.zeros_like(y, dtype=np.float64)
            for oti in other_timings:
                other_spikes = test_dataset.spike_responses_binned[oti]
                acc += other_spikes[:, out_start : out_start + n_bins]
            loo_rate = acc / len(other_timings)

        end_cursor = cursor + n_bins
        flat_true_rates[cursor:end_cursor] = y.T
        flat_loo_rates[cursor:end_cursor] = loo_rate.T
        cursor = end_cursor

    # -- Per-neuron correlations --
    loo_neuron_corrs, model_neuron_corrs = [], []
    for nidx in range(n_neurons):
        true_n = flat_true_rates[:, nidx]
        loo_n = flat_loo_rates[:, nidx]
        model_n = flat_model_rates[:, nidx]

        r_loo = pearsonr(true_n, loo_n)[0] if true_n.std() > 1e-6 and loo_n.std() > 1e-6 else 0.0
        r_model = pearsonr(true_n, model_n)[0] if true_n.std() > 1e-6 and model_n.std() > 1e-6 else 0.0

        loo_neuron_corrs.append(r_loo)
        model_neuron_corrs.append(r_model)

    loo_neuron_corrs = np.array(loo_neuron_corrs)
    model_neuron_corrs = np.array(model_neuron_corrs)
    diff = model_neuron_corrs - loo_neuron_corrs

    return diff, model_neuron_corrs, loo_neuron_corrs


def _plot_loo_comparison(
    model_neuron_corrs, loo_neuron_corrs, title_suffix, save_path,
):
    """
    Produce the 3-panel Model-vs-LOO figure.

    Parameters
    ----------
    model_neuron_corrs, loo_neuron_corrs : np.ndarray  (n_neurons,)
    title_suffix : str
        Label appended to subplot titles (e.g. "Average Rate", "Temporal").
    save_path : str
        Full path for the saved figure.
    """
    diff = model_neuron_corrs - loo_neuron_corrs

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: overlapping histograms
    axes[0].hist(model_neuron_corrs, bins=30, alpha=0.6, color="coral",
                 edgecolor="black", label="Model")
    axes[0].hist(loo_neuron_corrs, bins=30, alpha=0.6, color="steelblue",
                 edgecolor="black", label="LOO Avg")
    axes[0].axvline(np.mean(model_neuron_corrs), color="red", linestyle="--",
                    label=f"Model mean r={np.mean(model_neuron_corrs):.3f}")
    axes[0].axvline(np.mean(loo_neuron_corrs), color="blue", linestyle="--",
                    label=f"LOO mean r={np.mean(loo_neuron_corrs):.3f}")
    axes[0].set_xlabel("Pearson r"); axes[0].set_ylabel("# Neurons")
    axes[0].set_title(f"Model vs LOO — {title_suffix}")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # Panel 2: scatter
    axes[1].scatter(loo_neuron_corrs, model_neuron_corrs, alpha=0.6, s=40, edgecolor="w")
    lim = [min(loo_neuron_corrs.min(), model_neuron_corrs.min()) - 0.05, 1.05]
    axes[1].plot(lim, lim, "k--", alpha=0.4)
    axes[1].set_xlabel("LOO r"); axes[1].set_ylabel("Model r")
    axes[1].set_title(f"Per-neuron — {title_suffix}")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: difference histogram
    axes[2].hist(diff, bins=30, edgecolor="black", alpha=0.7, color="mediumpurple")
    axes[2].axvline(0, color="black", linestyle="-", alpha=0.5)
    axes[2].axvline(np.mean(diff), color="red", linestyle="--",
                    label=f"Mean Δr = {np.mean(diff):.4f}")
    axes[2].set_xlabel("Δr (Model − LOO)"); axes[2].set_ylabel("# Neurons")
    axes[2].set_title(f"Δr — {title_suffix} ({(diff > 0).sum()}/{len(diff)} model > LOO)")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    return fig


def _compute_ar_test_predictions(
    model, test_dataset, device, n_neurons,
    n_input_bins, n_output_bins, output_offset, history,
    init_state, n_initial_state_bins,
    max_time_ms, input_bin_size_ms, output_bin_size_ms,
):
    """
    Compute autoregressive (AR) test predictions.

    For each trial, processes sliding windows in temporal order, using the
    model's own predicted rates as lagged spike history instead of ground
    truth.  The init state (from the previous trial) remains ground truth.
    Stimulation channels are always ground truth.

    Returns
    -------
    ar_preds : np.ndarray (N, n_neurons, n_output_bins)
        Predicted firing rates (exp'd) from AR inference.
    targets : np.ndarray (N, n_neurons, n_output_bins)
        Ground-truth spike counts (identical to teacher-forced targets).
    """
    total_output_bins = max_time_ms // output_bin_size_ms
    total_input_bins = max_time_ms // input_bin_size_ms

    max_s_in = total_input_bins - n_input_bins
    max_s_out = total_output_bins - output_offset - n_output_bins
    max_t = min(max_s_in, max_s_out) + 1

    N = len(test_dataset)
    ar_preds = np.zeros((N, n_neurons, n_output_bins), dtype=np.float32)
    targets = np.zeros((N, n_neurons, n_output_bins), dtype=np.float32)

    # (timing_idx, t) → dataset sample index
    sample_lookup = {}
    for sample_idx, (timing_idx, t) in enumerate(test_dataset.samples):
        sample_lookup[(timing_idx, t)] = sample_idx

    # Unique trials
    trial_indices = sorted(set(ti for ti, _ in test_dataset.samples))
    n_trials = len(trial_indices)

    # Pre-compute init states (ground truth from previous trial)
    trial_init = {}
    if init_state and n_initial_state_bins > 0:
        for timing_idx in trial_indices:
            prev = timing_idx - 1
            if prev in test_dataset.spike_responses_binned:
                trial_init[timing_idx] = (
                    test_dataset.spike_responses_binned[prev][:, -n_initial_state_bins:]
                )
            else:
                trial_init[timing_idx] = np.zeros(
                    (n_neurons, n_initial_state_bins), dtype=np.float32,
                )

    # Per-trial AR buffers: accumulate predicted rates over time
    ar_sum = {ti: np.zeros((n_neurons, total_output_bins), dtype=np.float32)
              for ti in trial_indices}
    ar_cnt = {ti: np.zeros(total_output_bins, dtype=np.float32)
              for ti in trial_indices}

    model.eval()
    with torch.no_grad():
        for t in range(max_t):
            # -- Build one input per trial at this time position --
            batch_xs = []
            for timing_idx in trial_indices:
                pattern_name = test_dataset.timing_to_pattern[timing_idx]
                stim_src = test_dataset.pattern_stims[pattern_name]

                # Stim channels
                se = t + n_input_bins
                if se <= stim_src.shape[1]:
                    xs = stim_src[:, t:se].copy()
                else:
                    xs = np.zeros((stim_src.shape[0], n_input_bins), dtype=np.float32)
                    av = max(0, stim_src.shape[1] - t)
                    if av > 0:
                        xs[:, :av] = stim_src[:, t:t + av]

                # Valid-conv context padding
                if init_state and n_initial_state_bins > 0:
                    xs = np.pad(xs, ((0, 0), (n_initial_state_bins, 0)),
                                mode='constant')

                # History channels from AR buffer (replaces ground truth)
                if history > 0:
                    _s = ar_sum[timing_idx]
                    _c = ar_cnt[timing_idx]
                    curr_ar = np.zeros((n_neurons, total_output_bins),
                                       dtype=np.float32)
                    msk = _c > 0
                    if msk.any():
                        curr_ar[:, msk] = _s[:, msk] / _c[msk]

                    ar_slice = curr_ar[:, t:t + n_input_bins]
                    if ar_slice.shape[1] < n_input_bins:
                        ar_slice = np.pad(
                            ar_slice,
                            ((0, 0), (0, n_input_bins - ar_slice.shape[1])),
                        )

                    _init = trial_init.get(timing_idx)
                    if init_state and _init is not None:
                        sfl = np.concatenate([_init, ar_slice], axis=1)
                    else:
                        sfl = ar_slice

                    tt = xs.shape[1]
                    yh = np.zeros((n_neurons, tt), dtype=np.float32)
                    if tt > history:
                        src = sfl[:, :tt - history]
                        yh[:, history:history + src.shape[1]] = src
                    xs = np.concatenate([xs.astype(np.float32), yh], axis=0)

                batch_xs.append(xs)

            # -- Forward pass in sub-batches --
            for c0 in range(0, n_trials, 256):
                c1 = min(c0 + 256, n_trials)
                chunk_xs = batch_xs[c0:c1]
                chunk_ids = trial_indices[c0:c1]

                bx = torch.tensor(np.stack(chunk_xs), dtype=torch.float32).to(device)
                pr_batch = model(bx).cpu().numpy()

                for i, timing_idx in enumerate(chunk_ids):
                    pr = pr_batch[i]  # (n_neurons, n_output_bins)

                    # Store prediction if this position is a dataset sample
                    key = (timing_idx, t)
                    if key in sample_lookup:
                        sidx = sample_lookup[key]
                        ar_preds[sidx] = np.exp(pr)
                        trial_spikes = test_dataset.spike_responses_binned[timing_idx]
                        out_start = t + output_offset
                        targets[sidx] = trial_spikes[
                            :, out_start:out_start + n_output_bins
                        ]

                    # Update AR buffer
                    for o in range(n_output_bins):
                        tb = t + output_offset + o
                        if 0 <= tb < total_output_bins:
                            ar_sum[timing_idx][:, tb] += np.exp(pr[:, o])
                            ar_cnt[timing_idx][tb] += 1

    return ar_preds, targets


def _compute_loo_baseline(
    model, test_dataset, spike_responses, device, n_neurons,
    use_init_state, run_dir, logger,
):
    """Compute LOO baseline (average-rate & temporal) and generate both figures."""

    # --- Average-rate correlations ---
    diff_avg, model_avg, loo_avg = _compute_loo_average_rate(
        model, test_dataset, device, n_neurons, use_init_state,
    )
    _plot_loo_comparison(
        model_avg, loo_avg,
        title_suffix="Average Rate",
        save_path=os.path.join(run_dir, "model_vs_LOO_average_rate.png"),
    )
    logger.info(
        f"LOO (avg rate) — Model mean r: {np.mean(model_avg):.4f}, "
        f"LOO mean r: {np.mean(loo_avg):.4f}, "
        f"Neurons model > LOO: {(diff_avg > 0).sum()}/{len(diff_avg)}"
    )

    # --- Temporal correlations ---
    diff_temp, model_temp, loo_temp = _compute_loo_temporal(
        model, test_dataset, device, n_neurons, use_init_state,
    )
    _plot_loo_comparison(
        model_temp, loo_temp,
        title_suffix="Temporal",
        save_path=os.path.join(run_dir, "model_vs_LOO_temporal.png"),
    )
    logger.info(
        f"LOO (temporal) — Model mean r: {np.mean(model_temp):.4f}, "
        f"LOO mean r: {np.mean(loo_temp):.4f}, "
        f"Neurons model > LOO: {(diff_temp > 0).sum()}/{len(diff_temp)}"
    )

    return {
        "loo_avg_mean_corr": float(np.mean(loo_avg)),
        "model_avg_mean_corr": float(np.mean(model_avg)),
        "neurons_model_beats_loo_avg": int((diff_avg > 0).sum()),
        "loo_temporal_mean_corr": float(np.mean(loo_temp)),
        "all_test_corr": float(np.mean(model_temp)),
        "neurons_model_beats_loo_temporal": int((diff_temp > 0).sum()),
    }

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single experiment")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--overrides", type=str, default=None,
                        help='JSON string of config overrides, e.g. \'{"learning_rate": 0.01}\'')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.overrides:
        overrides = json.loads(args.overrides)
        cfg.update(overrides)

    if args.output_dir:
        run_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("binned_%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join("results", timestamp)

    summary = run_experiment(cfg, run_dir)
    print(f"\n✓ Done. Results in {run_dir}")
    print(f"  all_test_corr={summary.get('all_test_corr', float('nan')):.6f}  best_val_corr={summary['best_val_corr']:.6f}")

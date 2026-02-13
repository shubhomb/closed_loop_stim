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
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import (
    SimpleCausalSpikeCNN,
    SpikeRNN,
    StimToSpikeCNN,
    StimToSpikeMLP,
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
)
from viz import (
    analyze_pattern_responses_by_pattern_name,
    plot_oracle_trials_by_pattern,
    plot_pattern_selectivity,
    plot_spike_bin_distribution,
    plot_test_prediction_comparison,
)


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

def run_experiment(cfg: dict, run_dir: str) -> dict:
    """
    Execute a single experiment end-to-end.

    Parameters
    ----------
    cfg : dict
        Full experiment configuration (as loaded from YAML + any overrides).
    run_dir : str
        Directory to write all outputs into.

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

    if ENCODING_MODE == "current" and HISTORY and HISTORY > 0:
        USE_INIT_STATE = False
    else:
        USE_INIT_STATE = INIT_STATE

    # ================================================================
    # 1. Load data
    # ================================================================
    logger.info("Loading data …")
    spikes_df = make_spikes_responses_df(os.path.join(datadir, "spkVecs.npy"))
    spikes_df = spikes_df[~spikes_df["neuron_id"].isin(problematic_neurons)]
    logger.info(f"{spikes_df['neuron_id'].nunique()} unique neurons after dropping problematic neurons")

    pattern_registrations_path = os.path.join(datadir, "pattern_registrations.pkl")
    pattern_df, min_pattern_timestamp = preprocess_pattern_stimulations_df(
        read_pattern_json(pattern_registrations_path), align_to_stim=True
    )

    channel_to_index = {ch: idx for idx, ch in enumerate(sorted(pattern_df["channel"].dropna().unique()))}
    spiking_neurons = spikes_df["neuron_id"].unique()
    spiking_neuron_to_index = {neuron: idx for idx, neuron in enumerate(spiking_neurons)}
    spiking_neurons.sort()

    logger.info(f"Stimulation channels: {len(channel_to_index)}, Spiking neurons: {len(spiking_neurons)}")

    pattern_stims, pattern_polarities, spike_responses, timing_to_pattern, unique_trials = (
        trial_breakout_spikes_and_patterns(
            spikes_df,
            pattern_df,
            channel_to_index,
            spiking_neurons=spiking_neurons,
            spiking_neuron_to_index=spiking_neuron_to_index,
        )
    )
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
        init_state=USE_INIT_STATE,
        n_initial_state_bins=N_INITIAL_STATE_BINS,
        history=HISTORY,
        logger=logger,
    )

    train_dataset = BinnedStimSpikeDataset(trial_indices=train_indices, **dataset_kwargs)
    val_dataset = BinnedStimSpikeDataset(trial_indices=val_indices, **dataset_kwargs)
    test_dataset = BinnedStimSpikeDataset(trial_indices=test_indices, **dataset_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    n_stim_channels = train_dataset.n_channels
    n_neurons = train_dataset.n_neurons

    if ENCODING_MODE == "current" and HISTORY is not None and HISTORY >= 0:
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
    POOLING_CNN = cfg.get("pooling", "flatten")
    LINEAR = cfg.get("linear", True)

    if MODEL_TYPE == "mlp":
        model = StimToSpikeMLP(
            n_stim_channels=n_model_input_channels,
            n_neurons=n_neurons,
            n_input_bins=N_INPUT_BINS,
            n_output_bins=N_OUTPUT_BINS,
            embedding_dim=EMBEDDING_DIM,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT,
            init_bias=INIT_BIAS,
            linear=LINEAR,
            num_stim_levels=NUM_STIM_LEVELS,
        ).to(device)
    elif MODEL_TYPE == "cnn":
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
            pooling=POOLING_CNN,
            num_stim_levels=NUM_STIM_LEVELS,
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
    WEIGHT_DECAY = cfg.get("weight_decay", 0)
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

    logger.info("Starting training …")
    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            sum_loss=SUM_LOSS, grad_clip=False, weight_loss=WEIGHT_LOSS,
            use_init_state=USE_INIT_STATE,
        )
        val_loss, val_corr = validate(
            model, val_loader, criterion, device,
            sum_loss=SUM_LOSS, weight_loss=WEIGHT_LOSS,
            use_init_state=USE_INIT_STATE,
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
    )
    logger.info(f"Test loss: {test_loss:.6f}, Test corr: {test_corr:.6f}")

    # Generate predictions on test set
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            if USE_INIT_STATE:
                bx, by, bi = batch
                bx = bx.to(device)
                preds = model(bx)
            else:
                bx, by = batch
                bx = bx.to(device)
                preds = model(bx)
            all_preds.append(preds.cpu())
            all_targets.append(by)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

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
            pattern_limit=cfg.get("pattern_limit", 50),
            use_init_state=USE_INIT_STATE,
            n_initial_state_bins=N_INITIAL_STATE_BINS,
        )
    except Exception as e:
        logger.warning(f"Could not generate oracle trial plots: {e}")

    # ================================================================
    # 7. Summary metrics
    # ================================================================
    summary = {
        "run_dir": run_dir,
        "model_type": MODEL_TYPE,
        "test_loss": float(test_loss),
        "test_corr": float(test_corr),
        "best_val_corr": float(best_val_corr),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "epochs_trained": len(history["train_loss"]),
        "mean_pattern_corr": mean_pattern_corr,
        **loo_metrics,
    }

    with open(os.path.join(run_dir, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Experiment complete. Summary: {json.dumps(summary, indent=2)}")
    return summary


# ---------------------------------------------------------------------------
# LOO baseline helper
# ---------------------------------------------------------------------------

def _compute_loo_baseline(
    model, test_dataset, spike_responses, device, n_neurons,
    use_init_state, run_dir, logger,
):
    """Compute LOO baseline and model vs LOO comparison at single-sample level."""
    pattern_to_timing = defaultdict(list)
    for timing_idx in test_dataset.trial_indices:
        pname = test_dataset.timing_to_pattern[timing_idx]
        pattern_to_timing[pname].append(timing_idx)

    n_samples = len(test_dataset)
    sample_true_rates = np.zeros((n_samples, n_neurons))
    sample_loo_rates = np.zeros((n_samples, n_neurons))
    sample_model_rates = np.zeros((n_samples, n_neurons))

    model.eval()
    loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
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
            pred_rate = pred.mean(axis=2)
            bs = pred_rate.shape[0]
            sample_model_rates[model_idx : model_idx + bs] = pred_rate
            model_idx += bs

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

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(model_neuron_corrs, bins=30, alpha=0.6, color="coral", edgecolor="black", label="Model")
    axes[0].hist(loo_neuron_corrs, bins=30, alpha=0.6, color="steelblue", edgecolor="black", label="LOO Avg")
    axes[0].axvline(np.mean(model_neuron_corrs), color="red", linestyle="--",
                    label=f"Model mean r={np.mean(model_neuron_corrs):.3f}")
    axes[0].axvline(np.mean(loo_neuron_corrs), color="blue", linestyle="--",
                    label=f"LOO mean r={np.mean(loo_neuron_corrs):.3f}")
    axes[0].set_xlabel("Pearson r"); axes[0].set_ylabel("# Neurons")
    axes[0].set_title("Single-Sample Correlation: Model vs LOO"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].scatter(loo_neuron_corrs, model_neuron_corrs, alpha=0.6, s=40, edgecolor="w")
    lim = [min(loo_neuron_corrs.min(), model_neuron_corrs.min()) - 0.05, 1.05]
    axes[1].plot(lim, lim, "k--", alpha=0.4); axes[1].set_xlabel("LOO r"); axes[1].set_ylabel("Model r")
    axes[1].set_title("Per-neuron: Model vs LOO"); axes[1].grid(True, alpha=0.3)

    axes[2].hist(diff, bins=30, edgecolor="black", alpha=0.7, color="mediumpurple")
    axes[2].axvline(0, color="black", linestyle="-", alpha=0.5)
    axes[2].axvline(np.mean(diff), color="red", linestyle="--", label=f"Mean Δr = {np.mean(diff):.4f}")
    axes[2].set_xlabel("Δr (Model − LOO)"); axes[2].set_ylabel("# Neurons")
    axes[2].set_title(f"Model vs LOO ({(diff > 0).sum()}/{len(diff)} neurons model > LOO)")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "model_vs_LOO_single_sample.png"), dpi=150)
    plt.close(fig)

    logger.info(f"LOO baseline — Model mean r: {np.mean(model_neuron_corrs):.4f}, "
                f"LOO mean r: {np.mean(loo_neuron_corrs):.4f}, "
                f"Neurons model > LOO: {(diff > 0).sum()}/{len(diff)}")

    return {
        "loo_mean_corr": float(np.mean(loo_neuron_corrs)),
        "model_sample_mean_corr": float(np.mean(model_neuron_corrs)),
        "neurons_model_beats_loo": int((diff > 0).sum()),
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
    print(f"  test_corr={summary['test_corr']:.6f}  best_val_corr={summary['best_val_corr']:.6f}")

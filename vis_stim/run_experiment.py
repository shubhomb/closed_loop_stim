"""Single training experiment for vis_stim hyperparameter sweeps.

Trains a SimpleCausalSpikeCNN on (stim -> spike) data and writes
artifacts to ``run_dir``:

    config.yaml             # the merged config used
    best.pth                # checkpoint with lowest val_loss
    last.pth                # final-epoch checkpoint
    history.json            # per-epoch train_loss, val_loss, val_corr
    training_curves.png
    summary_metrics.json    # val_loss/val_corr/test_loss/test_corr +
                            # n_train, n_val, n_test, run_id, ...

This is the slurm-friendly counterpart of ``run_greedy.py`` -- it skips
the greedy stim generation (and oracle-trial figures) so each trial is
cheap enough to fan out across a SLURM job array.

Programmatic API::

    from run_experiment import run_experiment
    summary = run_experiment(cfg, run_dir)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "patterns_5k") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "patterns_5k"))

from utils import (  # noqa: E402
    BinnedStimSpikeDataset,
    preprocess_pattern_stimulations_df,
    read_pattern_json,
    trial_breakout_spikes_and_patterns,
)
from models import SimpleCausalSpikeCNN, train_epoch, validate  # noqa: E402

# Re-use existing helpers from run_greedy so behaviour stays in sync.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_greedy import (  # noqa: E402
    load_data,
    build_orientation_responses,
    make_loaders,
    plot_training_curves,
)


def pick_device(spec: Optional[str] = None) -> torch.device:
    if spec:
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _train_loop(model, train_loader, val_loader, run_dir: Path, cfg: dict,
                device, criterion, patience: Optional[int] = None):
    """Training loop with optional early stopping on val_loss."""
    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                             weight_decay=cfg["weight_decay"])
    history = {"train_loss": [], "val_loss": [], "val_corr": []}
    best_val = float("inf")
    best_corr = float("nan")
    best_epoch = -1
    best_path = run_dir / "best.pth"
    last_path = run_dir / "last.pth"

    epochs_since_improve = 0
    for epoch in range(1, cfg["n_epochs"] + 1):
        tl = train_epoch(model, train_loader, criterion, optim, device,
                         grad_clip=True, max_norm=1.0)
        vl, vc = validate(model, val_loader, criterion, device)
        history["train_loss"].append(float(tl))
        history["val_loss"].append(float(vl))
        history["val_corr"].append(float(vc))
        print(f"epoch {epoch:>3}/{cfg['n_epochs']}  "
              f"train={tl:.4f}  val={vl:.4f}  val_corr={vc:.4f}")
        sys.stdout.flush()

        torch.save(model.state_dict(), last_path)
        if vl < best_val:
            best_val = float(vl)
            best_corr = float(vc)
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if patience is not None and epochs_since_improve >= patience:
                print(f"early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    return history, best_val, best_corr, best_epoch


def run_experiment(cfg: dict, run_dir, device_spec: Optional[str] = None,
                   preloaded_data=None, prebuilt_datasets=None) -> dict:
    """Train a SimpleCausalSpikeCNN and write artifacts to ``run_dir``.

    Returns a flat ``summary`` dict suitable for JSON serialisation.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(device_spec)
    print(f"[run_experiment] device={device}  run_dir={run_dir}")

    seed = int(cfg.get("seed", 42))
    np.random.seed(seed); torch.manual_seed(seed)

    # ---- Data ----
    if preloaded_data is None:
        spikes_df, pattern_df, visual_trial_df, visual_spikes = load_data(cfg["datadir"])
        sum_responses, orientations, neuron_ids, neuron_to_idx = build_orientation_responses(
            visual_trial_df, visual_spikes, cfg)
        channel_to_index = {ch: i for i, ch in
                            enumerate(sorted(pattern_df["channel"].dropna().unique()))}
        spiking_neurons = np.sort(spikes_df.neuron_id.unique())
        spiking_neuron_to_index = {n: i for i, n in enumerate(spiking_neurons)}
        n_stim_channels = len(channel_to_index)

        pattern_stims, pattern_polarities, spike_responses, timing_to_pattern, _ = (
            trial_breakout_spikes_and_patterns(
                spikes_df, pattern_df, channel_to_index,
                spiking_neurons=spiking_neurons,
                spiking_neuron_to_index=spiking_neuron_to_index,
                stim_time_ms=cfg["max_time_ms"],
            )
        )
        info = pattern_df[["pattern_timing_index", "pattern_name", "is_oracle"]].drop_duplicates()
        train_indices = info[~info.is_oracle].pattern_timing_index.tolist()
        test_indices = info[info.is_oracle].pattern_timing_index.tolist()
    else:
        (pattern_df, spike_responses, channel_to_index, timing_to_pattern,
         spiking_neurons, n_stim_channels, train_indices, test_indices) = preloaded_data

    print(f"[run_experiment] stim_ch={n_stim_channels}  "
          f"neurons={len(spiking_neurons)}  "
          f"train_patterns={len(train_indices)}  test_patterns={len(test_indices)}")

    train_loader, val_loader, test_loader, test_ds, n_tr, n_val = make_loaders(
        pattern_df, spike_responses, channel_to_index, timing_to_pattern,
        train_indices, test_indices, cfg, device,
        prebuilt_datasets=prebuilt_datasets,
    )

    # ---- Model ----
    model = SimpleCausalSpikeCNN(
        n_stim_channels=n_stim_channels,
        n_neurons=len(spiking_neurons),
        n_input_bins=cfg["max_time_ms"] // cfg["input_bin_ms"],
        n_output_bins=cfg["max_time_ms"] // cfg["output_bin_ms"],
        embedding_dim=1,
        conv_channels=cfg["conv_channels"],
        kernel_sizes=cfg["kernel_sizes"],
        fc_dims=cfg["fc_dims"],
        dropout=cfg["dropout"],
        use_batch_norm=cfg.get("use_batch_norm", True),
        use_init_state=False,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[run_experiment] model params: {n_params:,}")

    criterion = nn.PoissonNLLLoss(log_input=True, full=True, reduction="none")

    # ---- Train ----
    t0 = time.time()
    history, best_val, best_corr, best_epoch = _train_loop(
        model, train_loader, val_loader, run_dir, cfg, device, criterion,
        patience=cfg.get("patience"),
    )
    train_secs = time.time() - t0

    # ---- Test on best ckpt ----
    best_path = run_dir / "best.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_corr = validate(model, test_loader, criterion, device)
    print(f"[run_experiment] test_loss={test_loss:.4f}  test_corr={test_corr:.4f}")

    # ---- Persist artefacts ----
    plot_training_curves(history, run_dir / "training_curves.png")

    cfg_out = dict(cfg)
    cfg_out.update(
        n_stim_channels=int(n_stim_channels),
        n_neurons=int(len(spiking_neurons)),
        n_train=int(n_tr), n_val=int(n_val), n_test=int(len(test_ds)),
        n_params=int(n_params),
    )
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg_out, f, default_flow_style=False)
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "best_val_loss": float(best_val),
        "best_val_corr": float(best_corr),
        "best_epoch": int(best_epoch),
        "test_loss": float(test_loss),
        "test_corr": float(test_corr),
        "n_params": int(n_params),
        "n_train": int(n_tr),
        "n_val": int(n_val),
        "n_test": int(len(test_ds)),
        "n_epochs_run": len(history["train_loss"]),
        "train_seconds": round(train_secs, 1),
        "device": str(device),
        "seed": int(seed),
    }
    with open(run_dir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[run_experiment] done in {train_secs:.1f}s  ->  {run_dir}")
    return summary


# ---------------------------------------------------------------------------
# CLI (so it can also be invoked standalone, mirroring slurm_experiment.py)
# ---------------------------------------------------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser(description="Single vis_stim training experiment")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--overrides", default=None,
                   help="JSON string of config overrides")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.overrides:
        cfg.update(json.loads(args.overrides))
        for k in list(cfg.keys()):
            if isinstance(cfg[k], dict) and k.startswith("_"):
                cfg.update(cfg.pop(k))
    if args.seed is not None:
        cfg["seed"] = args.seed

    run_experiment(cfg, args.output_dir, device_spec=args.device)


if __name__ == "__main__":
    main()

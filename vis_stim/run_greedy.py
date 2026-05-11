"""Modular greedy stim generator (port of vis_stim/greedy1.ipynb).

Pipeline:
    1. Load spikes + visual stim + pattern data.
    2. Build PSTH/orientation responses.
    3. Train SimpleCausalSpikeCNN OR load an existing checkpoint.
    4. Run StimCreator (greedy) for every orientation under
       (vis_orient_time, budget_per_step) to make a stim pattern bank.
    5. Save figures + the generated pattern registration pkl.

Layout::

    results/
      <datestamp>_simple_cnn/                # one model
        config.json
        history.json
        best.pth, last.pth
        training_curves.png
        viz/
          oracle_predictions/                # model -> oracle trial figures
          vot<VIS_ORIENT_TIME>_bps<BUDGET>/  # one greedy run
            greedy_config.json
            distances.json
            generated_pattern_registration.pkl
            per_orientation/<orient>.png
            distance_bar.png
            stim_similarity.png
            embedding_3d.html

`--train` / auto-detect: if `--model-dir` exists with a `best.pth`, training is
skipped. Otherwise a fresh `<datestamp>_simple_cnn/` dir is created and the model
trained.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "patterns_5k"))

from utils import (  # noqa: E402
    BinnedStimSpikeDataset,
    preprocess_pattern_stimulations_df,
    read_pattern_json,
    trial_breakout_spikes_and_patterns,
)
from models import SimpleCausalSpikeCNN, train_epoch, validate  # noqa: E402
from viz import plot_single_oracle_trial  # noqa: E402


VIS_STIM_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Greedy stim builder (lifted from greedy1.ipynb).
# ---------------------------------------------------------------------------
class StimCreator:
    BINS_PER_STEP = 6  # 60 ms / 10 ms

    _ENCODING_TABLE = {
        0: ([0, 2, 4], 3),
        -1: ([0, 2, 4], -3),
        1: ([1, 3, 5], 3),
        2: ([0, 1, 2, 3, 4, 5], 3),
    }

    def __init__(self, model, device, channel_to_index, n_bins, proj):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.channel_to_index = channel_to_index
        self.n_stim_channels = len(channel_to_index)
        self.n_bins = n_bins
        self.proj = proj  # (n_neurons, 3) shared embedding axes

    def stim_sequence_from_encoding(self, encoding):
        seq = np.zeros(self.BINS_PER_STEP, dtype=np.float32)
        if encoding in self._ENCODING_TABLE:
            offsets, amp = self._ENCODING_TABLE[encoding]
            for o in offsets:
                seq[o] = amp
        return seq

    def _predict(self, stim, n_bins_truncate=None):
        x = torch.from_numpy(stim).unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            pred = self.model(x)
        if n_bins_truncate is not None:
            pred = pred[:, :, :n_bins_truncate]
        return torch.exp(pred).cpu().numpy()[0]

    def create_stim(self, target_spikes, budget_per_step=2,
                    encodings=(-1, 0, 1, 2), channels=None,
                    require_improvement=True, verbose=False):
        if channels is None:
            channels = list(range(self.n_stim_channels))
        else:
            channels = list(channels)

        n_steps = self.n_bins // self.BINS_PER_STEP
        created = np.zeros((self.n_stim_channels, self.n_bins), dtype=np.float32)
        used = [set() for _ in range(n_steps)]
        proj = self.proj

        def dist(stim):
            pred_sum = self._predict(stim, n_bins_truncate=self.n_bins).sum(axis=1)
            return np.linalg.norm(pred_sum @ proj - target_spikes @ proj)

        curr = dist(created)
        if verbose:
            print(f"baseline dist: {curr:.4f}")
        enc_seqs = [(e, self.stim_sequence_from_encoding(e)) for e in encodings]

        for step in range(n_steps):
            t0, t1 = step * self.BINS_PER_STEP, (step + 1) * self.BINS_PER_STEP
            for _ in range(budget_per_step):
                best = None
                for ch in channels:
                    if ch in used[step]:
                        continue
                    for enc, seq in enc_seqs:
                        cand = created.copy()
                        cand[ch, t0:t1] = seq
                        d = dist(cand)
                        if best is None or d < best[0]:
                            best = (d, ch, enc, seq)
                if best is None:
                    break
                if require_improvement and best[0] >= curr:
                    break
                d, ch, enc, seq = best
                created[ch, t0:t1] = seq
                used[step].add(ch)
                curr = d
                if verbose:
                    print(f"  step {step}: ch={ch}, enc={enc:>2}, dist={d:.4f}")
        return created, curr


# ---------------------------------------------------------------------------
# Data loading + response building
# ---------------------------------------------------------------------------
def load_data(datadir):
    datadir = Path(datadir)
    all_spikes = np.load(datadir / "All_Shank_Spk_Vecs.npy", allow_pickle=False)
    spikes_df = pd.DataFrame(all_spikes)
    spikes_df.rename(columns={"sample_index": "timestamp",
                              "unit_index": "neuron_id"}, inplace=True)
    spikes_df.drop(columns=["segment_index"], inplace=True)

    pattern_df, _ = preprocess_pattern_stimulations_df(
        read_pattern_json(str(datadir / "Combined_Pattern_Registrations.pkl")),
        align_to_stim=True,
    )
    pattern_df = pattern_df.drop_duplicates()

    visual_stim_times = np.load(datadir / "VStim_Onset_TS.npy") * 30
    visual_stim_labels = np.load(datadir / "VStim_Labels.npy")
    visual_trial_df = pd.DataFrame({
        "timestamp": visual_stim_times,
        "orientation": visual_stim_labels,
    })

    min_t, max_t = visual_trial_df.timestamp.min(), visual_trial_df.timestamp.max()
    visual_spikes = spikes_df[
        (spikes_df.timestamp >= min_t) & (spikes_df.timestamp <= max_t)
    ].copy()

    return spikes_df, pattern_df, visual_trial_df, visual_spikes


def build_orientation_responses(visual_trial_df, visual_spikes, cfg):
    samples_per_ms = cfg["sample_rate"] // 1000
    window_len = cfg["stim_ms"] * samples_per_ms
    bin_samples = cfg["bin_ms"] * samples_per_ms
    n_bins = window_len // bin_samples

    orientations = np.sort(visual_trial_df.orientation.unique())
    neuron_ids = np.sort(visual_spikes.neuron_id.unique())
    neuron_to_idx = {n: i for i, n in enumerate(neuron_ids)}
    n_neurons = len(neuron_ids)

    sum_responses = []
    for orient in orientations:
        sr = np.zeros((n_neurons, n_bins))
        trials = visual_trial_df[visual_trial_df.orientation == orient].head(cfg["n_vis_trials"])
        for t_time in trials.timestamp.values:
            ts = visual_spikes[
                (visual_spikes.timestamp >= t_time)
                & (visual_spikes.timestamp < t_time + window_len)
            ]
            if ts.empty:
                continue
            edges = t_time + np.arange(n_bins + 1) * bin_samples
            for nid in ts.neuron_id.unique():
                idx = neuron_to_idx[nid]
                counts, _ = np.histogram(
                    ts[ts.neuron_id == nid].timestamp, bins=edges
                )
                sr[idx] += counts
        sr /= cfg["n_vis_trials"]
        sum_responses.append(sr)
    return (
        np.array(sum_responses),  # (n_orient, n_neurons, n_bins)
        orientations,
        neuron_ids,
        neuron_to_idx,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _float_collate(batch):
    xs = torch.stack([torch.as_tensor(b[0]).float() for b in batch])
    ys = torch.stack([torch.as_tensor(b[1]).float() for b in batch])
    return xs, ys


def make_loaders(pattern_df, spike_responses, channel_to_index, timing_to_pattern,
                 train_indices, test_indices, cfg, device):
    common = dict(
        pattern_df=pattern_df,
        spike_responses=spike_responses,
        channel_to_index=channel_to_index,
        timing_to_pattern=timing_to_pattern,
        encoding_mode=cfg["encoding_mode"],
        input_bin_size_ms=cfg["input_bin_ms"],
        output_bin_size_ms=cfg["output_bin_ms"],
        n_input_bins=cfg["max_time_ms"] // cfg["input_bin_ms"],
        n_output_bins=cfg["max_time_ms"] // cfg["output_bin_ms"],
        max_time_ms=cfg["max_time_ms"],
        output_offset=0,
        init_state=False,
    )
    train_ds = BinnedStimSpikeDataset(trial_indices=train_indices, **common)
    test_ds = BinnedStimSpikeDataset(trial_indices=test_indices, **common)

    n_total = len(train_ds)
    perm = np.random.RandomState(cfg["seed"]).permutation(n_total)
    n_val = max(1, int(n_total * cfg["val_frac"]))
    val_idx, tr_idx = perm[:n_val].tolist(), perm[n_val:].tolist()

    pin = device.type == "cuda"
    train_loader = DataLoader(Subset(train_ds, tr_idx), batch_size=cfg["batch_size"],
                              shuffle=True, collate_fn=_float_collate, pin_memory=pin)
    val_loader = DataLoader(Subset(train_ds, val_idx), batch_size=cfg["batch_size"],
                            shuffle=False, collate_fn=_float_collate, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                             collate_fn=_float_collate, pin_memory=pin)
    return train_loader, val_loader, test_loader, test_ds, len(tr_idx), len(val_idx)


def train_model(model, train_loader, val_loader, run_dir, cfg, device):
    criterion = nn.PoissonNLLLoss(log_input=True, full=True, reduction="none")
    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                             weight_decay=cfg["weight_decay"])
    history = {"train_loss": [], "val_loss": [], "val_corr": []}
    best_val = float("inf")
    best_path = run_dir / "best.pth"
    last_path = run_dir / "last.pth"

    for epoch in range(1, cfg["n_epochs"] + 1):
        tl = train_epoch(model, train_loader, criterion, optim, device,
                         grad_clip=True, max_norm=1.0)
        vl, vc = validate(model, val_loader, criterion, device)
        history["train_loss"].append(float(tl))
        history["val_loss"].append(float(vl))
        history["val_corr"].append(float(vc))
        print(f"epoch {epoch:>3}/{cfg['n_epochs']}  "
              f"train={tl:.4f}  val={vl:.4f}  val_corr={vc:.4f}")
        torch.save(model.state_dict(), last_path)
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), best_path)
    return history, float(best_val), criterion


def plot_training_curves(history, out_path):
    fig, ax = plt.subplots()
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
    ax.set_title("Training curves")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


# ---------------------------------------------------------------------------
# Greedy stim run + figures
# ---------------------------------------------------------------------------
def run_greedy(model, device, channel_to_index, sum_responses, orientations,
               n_neurons, cfg):
    vis_t = cfg["vis_orient_time"]
    n_bins = vis_t // cfg["output_bin_ms"]
    targets = sum_responses[:, :, :n_bins].sum(axis=2)  # (n_orient, n_neurons)

    # Shared (CIS, PC1, PC2) projection from these very targets — same as notebook.
    mean_vec = targets.mean(axis=0, keepdims=True)
    pca = PCA(n_components=2).fit(targets)
    proj, _ = np.linalg.qr(np.array([
        mean_vec[0], pca.components_[0], pca.components_[1]
    ]).T)

    creator = StimCreator(model, device, channel_to_index, n_bins=n_bins, proj=proj)
    est_stims, distances = [], []
    for o_idx, orient in enumerate(orientations):
        stim, d = creator.create_stim(
            target_spikes=targets[o_idx],
            budget_per_step=cfg["budget_per_step"],
            encodings=tuple(cfg["encodings"]),
            require_improvement=cfg["require_improvement"],
            verbose=True,
        )
        est_stims.append(stim)
        distances.append(float(d))
        print(f"orientation {orient}: dist={d:.4f}")
    return targets, np.array(distances), est_stims, proj


def fig_per_orientation(targets, est_stims, distances, orientations, model,
                        device, orient_total, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    mu = orient_total.mean(axis=0); sigma = orient_total.std(axis=0)
    selectivity = (orient_total - mu) / (sigma + 1e-9)
    arrow_off = -22

    for o_idx, deg in enumerate(orientations):
        target = targets[o_idx]
        stim = est_stims[o_idx]
        x = torch.from_numpy(stim).unsqueeze(0).to(device).float()
        with torch.no_grad():
            pred = torch.exp(model(x)).cpu().numpy()[0]
        pred_counts = pred.sum(axis=1)

        f, ax = plt.subplots(1, 3, figsize=(12, 4))
        vmax = max(np.abs(target).max(), np.abs(pred_counts).max(), 1e-6)
        ax[0].imshow(target.reshape(-1, 1), aspect="auto", cmap="Reds",
                     vmin=0, vmax=vmax)
        ax[0].set_title(f"{int(deg)}deg target"); ax[0].set_ylabel("Neuron")
        plt.colorbar(ax[0].images[0], ax=ax[0])
        for nidx in np.argsort(selectivity[o_idx])[::-1][:5]:
            ax[0].annotate(
                f"n{int(nidx)}",
                xy=(0, nidx), xycoords=("axes fraction", "data"),
                xytext=(arrow_off, 0), textcoords="offset points",
                ha="right", va="center", fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.6, shrinkA=0, shrinkB=2),
                annotation_clip=False,
            )

        ax[1].imshow(pred_counts.reshape(-1, 1), aspect="auto", cmap="Reds",
                     vmin=0, vmax=vmax)
        ax[1].set_title(f"{int(deg)}deg pred (d={distances[o_idx]:.3f})")
        plt.colorbar(ax[1].images[0], ax=ax[1])

        amax = max(np.abs(stim).max(), 1)
        ax[2].imshow(stim, aspect="auto", cmap="RdBu_r", vmin=-amax, vmax=amax)
        ax[2].set_title("Greedy stim"); ax[2].set_xlabel("Bin (10 ms)")
        ax[2].set_ylabel("Stim ch")
        plt.colorbar(ax[2].images[0], ax=ax[2], label="uA")

        f.tight_layout()
        f.savefig(out_dir / f"{int(deg):03d}deg.png", dpi=120)
        plt.close(f)


def fig_distance_bar(distances, orientations, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(orientations)), distances)
    ax.hlines(distances.mean(), -0.5, len(orientations) - 0.5,
              colors="red", linestyles="dashed", label="mean")
    ax.set_xticks(range(len(orientations)))
    ax.set_xticklabels([f"{int(d)}" for d in orientations], rotation=90)
    ax.set_ylabel("Final L2"); ax.set_title(
        f"Greedy stim distance (mean={distances.mean():.3f})")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def fig_stim_similarity(est_stims, orientations, out_path):
    flat = np.stack([s.flatten() for s in est_stims], axis=0)
    sim = normalize(flat, norm="l2") @ normalize(flat, norm="l2").T
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(orientations)))
    ax.set_yticks(range(len(orientations)))
    ax.set_xticklabels([f"{int(d)}" for d in orientations], rotation=90)
    ax.set_yticklabels([f"{int(d)}" for d in orientations])
    ax.set_title("Greedy stim cosine similarity")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def fig_embedding_3d(targets, est_stims, model, device, orientations, proj,
                     visual_trial_df, visual_spikes, neuron_to_idx, n_neurons,
                     cfg, out_path):
    import plotly.graph_objects as go

    coords = targets @ proj
    greedy_pred = np.zeros((len(orientations), n_neurons), dtype=np.float32)
    for i, stim in enumerate(est_stims):
        x = torch.from_numpy(stim).unsqueeze(0).to(device).float()
        with torch.no_grad():
            rate = torch.exp(model(x))
        greedy_pred[i] = rate[0].sum(dim=1).cpu().numpy()
    greedy_coords = greedy_pred @ proj

    vis_window = (cfg["vis_orient_time"] * cfg["sample_rate"]) // 1000
    trial_coords, trial_doubled = [], []
    for orient in orientations:
        times = visual_trial_df[
            visual_trial_df.orientation == orient
        ].timestamp.values[: cfg["n_vis_trials"]]
        for t in times:
            ts = visual_spikes[
                (visual_spikes.timestamp >= t)
                & (visual_spikes.timestamp < t + vis_window)
            ]
            x = np.zeros(n_neurons)
            if not ts.empty:
                for nid, c in ts.groupby("neuron_id").size().items():
                    if nid in neuron_to_idx:
                        x[neuron_to_idx[nid]] = c
            trial_coords.append(x @ proj)
            trial_doubled.append((int(orient) * 2) % 360)
    trial_coords = np.array(trial_coords)
    trial_doubled = np.array(trial_doubled)
    doubled = (orientations * 2) % 360

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=trial_coords[:, 0], y=trial_coords[:, 1], z=trial_coords[:, 2],
        mode="markers",
        marker=dict(size=3, color=trial_doubled, colorscale="HSV",
                    cmin=0, cmax=360, opacity=0.1),
        name="trials"))
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode="markers+text",
        text=[f"{int(o)}" for o in orientations],
        marker=dict(size=8, color=doubled, colorscale="HSV", cmin=0, cmax=360),
        name="observed avg"))
    fig.add_trace(go.Scatter3d(
        x=greedy_coords[:, 0], y=greedy_coords[:, 1], z=greedy_coords[:, 2],
        mode="markers",
        marker=dict(size=8, color=doubled, colorscale="HSV", cmin=0, cmax=360,
                    symbol="diamond"),
        name="greedy CNN pred"))
    for i in range(len(orientations)):
        fig.add_trace(go.Scatter3d(
            x=[coords[i, 0], greedy_coords[i, 0]],
            y=[coords[i, 1], greedy_coords[i, 1]],
            z=[coords[i, 2], greedy_coords[i, 2]],
            mode="lines", line=dict(color="gray", width=2, dash="dot"),
            showlegend=False))
    fig.update_layout(
        title=f"Embedding (vot={cfg['vis_orient_time']} ms, bps={cfg['budget_per_step']})",
        scene=dict(xaxis_title="CIS", yaxis_title="PC1", zaxis_title="PC2"),
        width=900, height=700,
    )
    fig.write_html(str(out_path))


# ---------------------------------------------------------------------------
# Pattern registration export
# ---------------------------------------------------------------------------
def export_pattern_registration(est_stims, channel_to_index, k_trials,
                                shuffle_seed, out_path,
                                n_shanks=4, max_per_shank=4):
    """Build pattern registrations from est_stims.

    Hardware constraint: channels are partitioned into ``n_shanks`` equal-sized
    shanks by sorted channel ID, and at most ``max_per_shank`` channels per
    shank may fire in any single step. Offending steps are flagged via stdout
    and lowest-ranked (highest channel-index) offenders are dropped to make
    the step legal.
    """
    bps = StimCreator.BINS_PER_STEP
    index_to_channel = {idx: ch for ch, idx in channel_to_index.items()}
    n_steps = est_stims[0].shape[1] // bps

    sorted_channels = sorted(channel_to_index.keys())
    if len(sorted_channels) % n_shanks != 0:
        raise ValueError(f"cannot split {len(sorted_channels)} channels into "
                         f"{n_shanks} equal shanks")
    shank_size = len(sorted_channels) // n_shanks
    channel_to_shank = {ch: i // shank_size for i, ch in enumerate(sorted_channels)}

    def slice_to_dm(sl):
        if not np.any(sl):
            return None
        for dm, (offsets, amp) in StimCreator._ENCODING_TABLE.items():
            target = np.zeros(bps, dtype=np.float32)
            for o in offsets:
                target[o] = amp
            if np.array_equal(sl, target):
                return dm
        return None

    def enforce_shank_limit(cds, ori_idx, step_idx):
        """At most ``max_per_shank`` channels firing simultaneously per shank.

        Two channels conflict only if their delay_mode bin-sets intersect
        (e.g. dm=0 and dm=1 are interleaved -> never coincident, ok to share
        a shank; dm=2 fires every bin so conflicts with everything).

        Walks channels in ascending order; drops a channel if adding it
        would push any (shank, bin) count above max_per_shank.
        """
        # bin-bitmask per delay_mode (6 bins per step)
        dm_bins = {dm: {o for o in offsets}
                   for dm, (offsets, _amp) in StimCreator._ENCODING_TABLE.items()}
        # per-shank, per-bin live count
        live = {}  # shank -> {bin_idx: count}
        kept, dropped_log = [], []
        for cd in cds:
            shank = channel_to_shank[cd["channel"]]
            bins_used = dm_bins[cd["delay_mode"]]
            shank_counts = live.setdefault(shank, {})
            if any(shank_counts.get(b, 0) + 1 > max_per_shank for b in bins_used):
                dropped_log.append((shank, cd["channel"], cd["delay_mode"]))
                continue
            for b in bins_used:
                shank_counts[b] = shank_counts.get(b, 0) + 1
            kept.append(cd)
        if dropped_log:
            by_shank_drop = {}
            for shank, ch, dm in dropped_log:
                by_shank_drop.setdefault(shank, []).append((ch, dm))
            for shank, items in by_shank_drop.items():
                print(f"[shank-violation] orient {ori_idx} step {step_idx} "
                      f"shank {shank}: dropping {len(items)} channel(s) "
                      f"that exceed {max_per_shank} simultaneous: "
                      f"{[(ch, dm) for ch, dm in items]}")
        return kept

    unique_patterns = []
    for ori_idx, stim in enumerate(est_stims):
        steps_lists = []
        for s in range(n_steps):
            cds = []
            for ch in range(stim.shape[0]):
                sl = stim[ch, s * bps:(s + 1) * bps]
                dm = slice_to_dm(sl)
                if dm is not None:
                    cds.append({"channel": int(index_to_channel[ch]),
                                "delay_mode": int(dm)})
                elif np.any(sl):
                    print(f"[warn] orient {ori_idx} ch {ch} step {s}: unrecognized {sl}")
            cds = enforce_shank_limit(cds, ori_idx, s)
            steps_lists.append(cds)
        unique_patterns.append({
            "ori_idx": ori_idx,
            "pattern_name": f"50{ori_idx}",
            "steps_lists": steps_lists,
        })

    rng = np.random.default_rng(shuffle_seed)
    slots = [(p["ori_idx"], p["pattern_name"]) for p in unique_patterns
             for _ in range(k_trials)]
    order = rng.permutation(len(slots))

    generated = []
    for new_pos, slot_idx in enumerate(order):
        ori_idx, pname = slots[slot_idx]
        src = next(p for p in unique_patterns if p["ori_idx"] == ori_idx)
        steps = [
            {"index": i, "channel_delays": copy.deepcopy(cd), "start_timestamp": 0}
            for i, cd in enumerate(src["steps_lists"])
        ]
        generated.append({
            "pattern_name": pname,
            "pattern_lambda": 0.0,
            "pattern_flag_start_timestamp": 0,
            "pattern_timing_index": new_pos + 1,
            "steps": steps,
        })

    with open(out_path, "wb") as f:
        pickle.dump(generated, f)
    print(f"wrote {len(generated)} trials ({len(unique_patterns)} patterns x {k_trials}) -> {out_path}")
    return generated


# ---------------------------------------------------------------------------
# Oracle predictions viz (model-level, not greedy-level)
# ---------------------------------------------------------------------------
def viz_oracle_predictions(model, criterion, device, pattern_df, spike_responses,
                           pattern_polarities, test_loader, cfg, out_dir, n_show=4):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_viz = dict(
        input_bin_size_ms=cfg["input_bin_ms"],
        output_bin_size_ms=cfg["output_bin_ms"],
        n_input_bins=cfg["max_time_ms"] // cfg["input_bin_ms"],
        n_output_bins=cfg["max_time_ms"] // cfg["output_bin_ms"],
        max_time_ms=cfg["max_time_ms"],
        output_offset=0,
        history=0,
        init_state=False,
        encoding_mode=cfg["encoding_mode"],
    )
    raw_data = dict(pattern_df=pattern_df, spike_responses=spike_responses,
                    pattern_polarities=pattern_polarities)
    model_tuple = (model, criterion, device)

    oracle = pattern_df[pattern_df.is_oracle][["pattern_name", "pattern_timing_index"]].drop_duplicates()
    counts = oracle.groupby("pattern_name").size()
    names = list(counts[counts >= 2].index[:n_show]) or list(counts.index[:n_show])
    for pname in names:
        try:
            fig = plot_single_oracle_trial(
                pattern_name=pname, trial_idx=0, model_tuple=model_tuple,
                cfg=cfg_viz, raw_data=raw_data, test_loader=test_loader,
                coarse_factor=1,
            )
            fig.savefig(out_dir / f"{pname}.png", dpi=120)
            plt.close(fig)
        except Exception as e:
            print(f"[warn] oracle viz {pname}: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path,
                   default=VIS_STIM_DIR / "configs" / "default.yaml")
    p.add_argument("--results-root", type=Path,
                   default=VIS_STIM_DIR / "results")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="Existing run dir. If best.pth present, training is skipped.")
    p.add_argument("--datadir", type=Path, default=None)
    p.add_argument("--vis-orient-time", type=int, default=None)
    p.add_argument("--budget-per-step", type=int, default=None)
    p.add_argument("--n-epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None,
                   help="cuda|mps|cpu (default: auto)")
    return p.parse_args()


def pick_device(spec=None):
    if spec:
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    # CLI overrides
    if args.datadir is not None: cfg["datadir"] = str(args.datadir)
    if args.vis_orient_time is not None: cfg["vis_orient_time"] = args.vis_orient_time
    if args.budget_per_step is not None: cfg["budget_per_step"] = args.budget_per_step
    if args.n_epochs is not None: cfg["n_epochs"] = args.n_epochs
    if args.seed is not None: cfg["seed"] = args.seed

    device = pick_device(args.device)
    print("device:", device)

    np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    # ---- 1. Data ----
    spikes_df, pattern_df, visual_trial_df, visual_spikes = load_data(cfg["datadir"])
    sum_responses, orientations, neuron_ids, neuron_to_idx = build_orientation_responses(
        visual_trial_df, visual_spikes, cfg)
    n_neurons = len(neuron_ids)
    orient_total = sum_responses[:, :, : cfg["vis_orient_time"] // cfg["bin_ms"]].sum(axis=2)

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
    print(f"stim ch={n_stim_channels}, neurons={n_neurons}, "
          f"unique patterns={len(pattern_stims)}, trials={len(spike_responses)}")

    info = pattern_df[["pattern_timing_index", "pattern_name", "is_oracle"]].drop_duplicates()
    train_indices = info[~info.is_oracle].pattern_timing_index.tolist()
    test_indices = info[info.is_oracle].pattern_timing_index.tolist()

    train_loader, val_loader, test_loader, test_ds, n_tr, n_val = make_loaders(
        pattern_df, spike_responses, channel_to_index, timing_to_pattern,
        train_indices, test_indices, cfg, device,
    )

    # ---- 2. Model dir / checkpoint ----
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
        use_batch_norm=True,
        use_init_state=False,
    ).to(device)
    criterion = nn.PoissonNLLLoss(log_input=True, full=True, reduction="none")

    if args.model_dir and (args.model_dir / "best.pth").exists():
        run_dir = args.model_dir
        print(f"loading existing model from {run_dir}/best.pth (skipping training)")
        model.load_state_dict(torch.load(run_dir / "best.pth", map_location=device))
        history = None
    else:
        run_name = time.strftime("%Y%m%d_%H%M%S") + "_simple_cnn"
        run_dir = (args.model_dir if args.model_dir
                   else args.results_root / run_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"training model into {run_dir}")
        history, best_val, criterion = train_model(
            model, train_loader, val_loader, run_dir, cfg, device)
        plot_training_curves(history, run_dir / "training_curves.png")
        save_cfg = dict(cfg)
        save_cfg.update(n_stim_channels=n_stim_channels, n_neurons=n_neurons,
                        n_train=n_tr, n_val=n_val, n_test=len(test_ds),
                        best_val_loss=best_val)
        (run_dir / "config.json").write_text(json.dumps(save_cfg, indent=2))
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        model.load_state_dict(torch.load(run_dir / "best.pth", map_location=device))

        viz_oracle_predictions(
            model, criterion, device, pattern_df, spike_responses,
            pattern_polarities, test_loader, cfg,
            out_dir=run_dir / "viz" / "oracle_predictions",
        )

    # ---- 3. Greedy run dir ----
    greedy_dir = (run_dir / "viz"
                  / f"vot{cfg['vis_orient_time']}_bps{cfg['budget_per_step']}")
    greedy_dir.mkdir(parents=True, exist_ok=True)
    (greedy_dir / "greedy_config.json").write_text(json.dumps({
        "vis_orient_time": cfg["vis_orient_time"],
        "budget_per_step": cfg["budget_per_step"],
        "encodings": cfg["encodings"],
        "require_improvement": cfg["require_improvement"],
        "k_trials_per_pattern": cfg["k_trials_per_pattern"],
        "shuffle_seed": cfg["shuffle_seed"],
    }, indent=2))

    targets, distances, est_stims, proj = run_greedy(
        model, device, channel_to_index, sum_responses, orientations,
        n_neurons, cfg)
    (greedy_dir / "distances.json").write_text(json.dumps({
        "orientations": orientations.tolist(),
        "distances": distances.tolist(),
        "mean_distance": float(distances.mean()),
    }, indent=2))

    fig_per_orientation(
        targets, est_stims, distances, orientations, model, device,
        orient_total, greedy_dir / "per_orientation",
    )
    fig_distance_bar(distances, orientations, greedy_dir / "distance_bar.png")
    fig_stim_similarity(est_stims, orientations,
                        greedy_dir / "stim_similarity.png")
    fig_embedding_3d(
        targets, est_stims, model, device, orientations, proj,
        visual_trial_df, visual_spikes, neuron_to_idx, n_neurons, cfg,
        greedy_dir / "embedding_3d.html",
    )

    export_pattern_registration(
        est_stims, channel_to_index,
        k_trials=cfg["k_trials_per_pattern"],
        shuffle_seed=cfg["shuffle_seed"],
        out_path=greedy_dir / "generated_pattern_registration.pkl",
    )

    print(f"\nDone. Model: {run_dir}\nGreedy: {greedy_dir}")


if __name__ == "__main__":
    main()

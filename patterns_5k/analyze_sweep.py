"""
Cross-experiment analysis and visualization for hyperparameter sweeps.

Loads sweep_results.csv (or multiple sweep directories) and creates
comparative plots across trials.

Usage:
    # From Python / notebook:
    from analyze_sweep import SweepAnalyzer
    sa = SweepAnalyzer("results/cnn_history_sweep_2026-02-13_12-47-14")
    sa.plot_metric_vs_param("history", "test_corr")
    sa.plot_all()

    # CLI:
    python analyze_sweep.py results/cnn_history_sweep_*
    python analyze_sweep.py results/cnn_kernel_sweep_* --params kernel_sizes --metrics test_corr neurons_model_beats_loo
"""

import argparse
import glob
import json
import logging
import os
from typing import List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# ── Metric display names ────────────────────────────────────────────
METRIC_LABELS = {
    "test_corr": "Test Correlation",
    "best_val_corr": "Best Val Correlation",
    "test_loss": "Test Loss",
    "mean_pattern_corr": "Mean Pattern Selectivity (r)",
    # New (split) LOO keys
    "model_avg_mean_corr": "Model Avg-Rate Mean Corr",
    "loo_avg_mean_corr": "LOO Avg-Rate Mean Corr",
    "neurons_model_beats_loo_avg": "# Neurons Model > LOO (Avg Rate)",
    "model_temporal_mean_corr": "Model Temporal Mean Corr",
    "loo_temporal_mean_corr": "LOO Temporal Mean Corr",
    "neurons_model_beats_loo_temporal": "# Neurons Model > LOO (Temporal)",
    # Legacy keys (older runs)
    "model_sample_mean_corr": "Model Sample-Level Mean Corr",
    "loo_mean_corr": "LOO Baseline Mean Corr",
    "neurons_model_beats_loo": "# Neurons Model > LOO",
    "epochs_trained": "Epochs Trained",
    "trainable_params": "Trainable Parameters",
    "total_params": "Total Parameters",
}


class SweepAnalyzer:
    """Load and visualize results from one or more sweep directories."""

    def __init__(self, sweep_dirs: Union[str, List[str]]):
        """
        Parameters
        ----------
        sweep_dirs : str or list of str
            Path(s) to sweep output directory(ies). Each must contain
            a ``sweep_results.csv``.
        """
        if isinstance(sweep_dirs, str):
            sweep_dirs = [sweep_dirs]
        self.sweep_dirs = sweep_dirs
        self.df = self._load(sweep_dirs)
        self._detect_params()

    # ── Loading ──────────────────────────────────────────────────────

    @staticmethod
    def _load(dirs: List[str]) -> pd.DataFrame:
        frames = []
        for d in dirs:
            csv = os.path.join(d, "sweep_results.csv")
            if not os.path.exists(csv):
                print(f"Warning: {csv} not found, skipping")
                continue
            tmp = pd.read_csv(csv)
            tmp["sweep_dir"] = d
            # Read sweep name from sweep_config.yaml if available
            scfg = os.path.join(d, "sweep_config.yaml")
            if os.path.exists(scfg):
                import yaml
                with open(scfg) as f:
                    sweep_name = yaml.safe_load(f).get("sweep_name", os.path.basename(d))
                tmp["sweep_name"] = sweep_name
            else:
                tmp["sweep_name"] = os.path.basename(d)
            frames.append(tmp)
        if not frames:
            raise FileNotFoundError("No sweep_results.csv found in any of the provided directories")
        df = pd.concat(frames, ignore_index=True)
        # Keep only completed experiments
        df = df[df["status"] == "completed"].copy()
        # Fix stale run_dir paths: resolve via sweep_dir + experiment name
        df["run_dir"] = df.apply(
            lambda r: SweepAnalyzer._resolve_run_dir(r["run_dir"], r.get("sweep_dir", ""), r.get("experiment", "")),
            axis=1,
        )
        return df

    @staticmethod
    def _resolve_run_dir(run_dir: str, sweep_dir: str, experiment: str) -> str:
        """Return a valid run directory path.

        The ``run_dir`` baked into sweep_results.csv may be stale if the
        sweep directory was renamed after creation.  Fall back to
        ``<sweep_dir>/<experiment>`` when the original path doesn't exist.
        """
        if os.path.isdir(run_dir):
            return run_dir
        candidate = os.path.join(sweep_dir, experiment)
        if os.path.isdir(candidate):
            return candidate
        # last resort: return the original (caller must handle missing dir)
        return run_dir

    def _detect_params(self):
        """Detect which columns are swept parameters vs fixed metrics."""
        # Standard metric columns (always present from summary_metrics.json)
        metric_cols = {
            "run_dir", "model_type", "test_loss", "test_corr", "best_val_corr",
            "total_params", "trainable_params", "epochs_trained",
            "mean_pattern_corr",
            # New LOO keys
            "model_avg_mean_corr", "loo_avg_mean_corr", "neurons_model_beats_loo_avg",
            "model_temporal_mean_corr", "loo_temporal_mean_corr", "neurons_model_beats_loo_temporal",
            # Legacy LOO keys
            "loo_mean_corr", "model_sample_mean_corr", "neurons_model_beats_loo",
            "experiment", "status",
            "sweep_dir", "sweep_name",
        }
        self.param_cols = [c for c in self.df.columns if c not in metric_cols]
        self.metric_cols = [c for c in self.df.columns if c in metric_cols and c not in
                           {"run_dir", "experiment", "status", "sweep_dir", "sweep_name"}]

    # ── Per-experiment training history ──────────────────────────────

    def load_training_histories(self) -> dict:
        """Load training_history.json for each experiment row."""
        histories = {}
        for _, row in self.df.iterrows():
            hpath = os.path.join(row["run_dir"], "training_history.json")
            if os.path.exists(hpath):
                with open(hpath) as f:
                    histories[row["experiment"]] = json.load(f)
        return histories

    # ── Single-param plots ──────────────────────────────────────────

    @staticmethod
    def _parse_param_column(series: pd.Series) -> pd.Series:
        """Normalise a parameter column: parse JSON lists, unwrap singletons."""
        def _parse(v):
            if isinstance(v, str) and v.startswith("["):
                try:
                    v = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    pass
            if isinstance(v, list) and len(v) == 1:
                return v[0]
            # Convert remaining lists to tuples so they're hashable for groupby
            if isinstance(v, list):
                return tuple(v)
            return v
        return series.apply(_parse)

    def plot_metric_vs_param(
        self,
        param: str,
        metric: str,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        show_loo_baseline: bool = True,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot a single metric as a function of one swept parameter,
        averaged over all other conditions.  Error bars show ± 1 SEM.

        Parameters
        ----------
        param : str   Column name of the swept parameter (e.g. "history").
        metric : str  Column name of the metric (e.g. "test_corr").
        show_loo_baseline : bool  If True, draw a horizontal line for LOO baseline
                                   (only when metric is a correlation metric).
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        sub = self.df.dropna(subset=[param, metric]).copy()
        sub["_x"] = self._parse_param_column(sub[param])

        # Group by parsed parameter value → mean ± SEM
        grouped = sub.groupby("_x", sort=True)[metric]
        means = grouped.mean()
        sems = grouped.sem().fillna(0)
        counts = grouped.count()

        # Determine if x is numeric
        try:
            x_numeric = pd.to_numeric(means.index)
            is_numeric = True
        except (ValueError, TypeError):
            x_numeric = np.arange(len(means))
            is_numeric = False

        ax.errorbar(
            x_numeric, means.values, yerr=sems.values,
            fmt="o-", linewidth=2, markersize=8, capsize=4, **kwargs,
        )
        # Annotate counts
        for xi, m, n in zip(x_numeric, means.values, counts.values):
            ax.annotate(f"n={n}", (xi, m), textcoords="offset points",
                        xytext=(0, 10), fontsize=7, ha="center", alpha=0.6)

        # LOO baseline
        _loo_key = "loo_avg_mean_corr" if "loo_avg_mean_corr" in sub.columns else "loo_mean_corr"
        if show_loo_baseline and _loo_key in sub.columns and "corr" in metric.lower():
            loo_val = sub[_loo_key].iloc[0]
            ax.axhline(y=loo_val, color="gray", linestyle="--", alpha=0.7,
                       label=f"LOO baseline ({loo_val:.3f})")
            ax.legend()

        if not is_numeric:
            ax.set_xticks(x_numeric)
            ax.set_xticklabels([str(v) for v in means.index], rotation=45, ha="right")

        ylabel = METRIC_LABELS.get(metric, metric)
        ax.set_xlabel(param.replace("_", " ").title())
        ax.set_ylabel(ylabel)
        ax.set_title(title or f"{ylabel} vs {param}")
        ax.grid(True, alpha=0.3)
        return ax

    # ── Multi-metric dashboard ──────────────────────────────────────

    def plot_dashboard(
        self,
        param: str,
        metrics: Optional[List[str]] = None,
        savepath: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a multi-panel dashboard showing several metrics vs one parameter.
        """
        if metrics is None:
            metrics = [
                "test_corr", "best_val_corr", "test_loss",
                "mean_pattern_corr",
                "neurons_model_beats_loo_avg", "neurons_model_beats_loo_temporal",
                "model_avg_mean_corr", "model_temporal_mean_corr",
                # Legacy fallbacks
                "neurons_model_beats_loo", "model_sample_mean_corr",
            ]
        # Only keep metrics that exist
        metrics = [m for m in metrics if m in self.df.columns]

        n = len(metrics)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
        axes = np.atleast_1d(axes).flatten()

        for i, m in enumerate(metrics):
            self.plot_metric_vs_param(param, m, ax=axes[i])
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        sweep_name = self.df["sweep_name"].iloc[0] if "sweep_name" in self.df.columns else ""
        fig.suptitle(f"Sweep Dashboard: {sweep_name} — varying {param}", fontsize=14, y=1.02)
        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
            print(f"Saved dashboard to {savepath}")
        return fig

    # ── Training curves comparison ──────────────────────────────────

    def plot_training_curves(
        self,
        param: str,
        curve: str = "val_corr",
        savepath: Optional[str] = None,
    ) -> plt.Figure:
        """
        Overlay mean training curves grouped by *param* value.
        Shaded region shows ± 1 SEM across experiments that share a
        parameter value.
        """
        histories = self.load_training_histories()
        if not histories:
            print("No training histories found")
            return None

        sub = self.df.dropna(subset=[param]).copy()
        sub["_x"] = self._parse_param_column(sub[param])

        # Collect curves per group
        group_curves: dict = {}  # param_value -> list[np.array]
        for _, row in sub.iterrows():
            exp = row["experiment"]
            if exp not in histories:
                continue
            vals = histories[exp].get(curve, [])
            if not vals:
                continue
            key = row["_x"]
            group_curves.setdefault(key, []).append(np.array(vals, dtype=float))

        if not group_curves:
            print("No training histories found")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_keys = sorted(group_curves.keys(), key=lambda k: (str(type(k)), k))
        cmap = plt.cm.viridis
        n_groups = len(sorted_keys)

        for idx, key in enumerate(sorted_keys):
            arrays = group_curves[key]
            # Pad to equal length (some may have early-stopped)
            max_len = max(len(a) for a in arrays)
            padded = np.full((len(arrays), max_len), np.nan)
            for j, a in enumerate(arrays):
                padded[j, :len(a)] = a

            mean_curve = np.nanmean(padded, axis=0)
            sem_curve = np.nanstd(padded, axis=0) / np.sqrt(np.sum(~np.isnan(padded), axis=0).clip(1))
            epochs = np.arange(max_len)

            color = cmap(idx / max(1, n_groups - 1))
            label = f"{param}={key} (n={len(arrays)})"
            ax.plot(epochs, mean_curve, color=color, linewidth=2, label=label)
            ax.fill_between(epochs, mean_curve - sem_curve, mean_curve + sem_curve,
                            color=color, alpha=0.15)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(curve.replace("_", " ").title())
        ax.set_title(f"Training Curves: {curve} — grouped by {param}")
        ax.legend(fontsize=8, loc="best", ncol=2)
        ax.grid(True, alpha=0.3)

        if savepath:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
            print(f"Saved training curves to {savepath}")
        return fig

    # ── Bar chart comparison ────────────────────────────────────────

    def plot_comparison_bars(
        self,
        param: str,
        metrics: Optional[List[str]] = None,
        savepath: Optional[str] = None,
    ) -> plt.Figure:
        """
        Side-by-side bar chart comparing multiple metrics across param values.
        Bars show mean ± SEM grouped by *param*.  Raw values are shown; each
        metric gets its own y-axis panel for readability.
        """
        if metrics is None:
            metrics = ["test_corr", "mean_pattern_corr", "neurons_model_beats_loo_avg",
                      "neurons_model_beats_loo_temporal", "neurons_model_beats_loo"]
        metrics = [m for m in metrics if m in self.df.columns]
        if not metrics:
            print("No matching metrics found")
            return None

        sub = self.df.dropna(subset=[param]).copy()
        sub["_x"] = self._parse_param_column(sub[param])

        grouped = sub.groupby("_x", sort=True)
        x_labels = [str(k) for k in grouped.groups.keys()]
        n_groups = len(x_labels)
        x = np.arange(n_groups)
        width = 0.8 / len(metrics)

        fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.5), 6))

        for i, m in enumerate(metrics):
            means = grouped[m].mean().values.astype(float)
            sems = grouped[m].sem().fillna(0).values.astype(float)

            # Normalize across groups for display
            vmin, vmax = means.min(), means.max()
            if vmax > vmin:
                norm_means = (means - vmin) / (vmax - vmin)
                norm_sems = sems / (vmax - vmin)
            else:
                norm_means = np.ones_like(means) * 0.5
                norm_sems = np.zeros_like(sems)

            bars = ax.bar(
                x + i * width - 0.4 + width / 2, norm_means, width,
                yerr=norm_sems, capsize=3,
                label=METRIC_LABELS.get(m, m), alpha=0.8,
            )
            # Annotate with actual mean values
            for bar, v in zip(bars, means):
                fmt = f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.03,
                    fmt, ha="center", va="bottom", fontsize=7, rotation=45,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_xlabel(param.replace("_", " ").title())
        ax.set_ylabel("Normalized Value (mean ± SEM)")
        ax.set_title(f"Metric Comparison Across {param}")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2, axis="y")

        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
            print(f"Saved comparison bars to {savepath}")
        return fig

    # ── Multi-sweep overlay ─────────────────────────────────────────

    def plot_multi_sweep_comparison(
        self,
        metric: str = "test_corr",
        savepath: Optional[str] = None,
    ) -> plt.Figure:
        """
        When multiple sweeps are loaded, compare them on a shared axis.
        Each sweep becomes one line/group, x-axis is experiment index.
        """
        if self.df["sweep_name"].nunique() < 2:
            print("Need ≥ 2 sweeps for multi-sweep comparison")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        for name, group in self.df.groupby("sweep_name"):
            group = group.sort_values("experiment")
            ax.plot(range(len(group)), group[metric].values, "o-", label=name, markersize=6)

        ax.set_xlabel("Experiment Index")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(f"{metric} Across Sweeps")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
        return fig

    # ── Convenience: plot everything ────────────────────────────────

    def plot_all(self, savedir: Optional[str] = None):
        """
        Generate all available plots and save to savedir (defaults to
        first sweep directory).
        """
        if savedir is None:
            savedir = os.path.join(self.sweep_dirs[0], "analysis")
        os.makedirs(savedir, exist_ok=True)

        for param in self.param_cols:
            if param in self.df.columns and self.df[param].nunique() > 1:
                print(f"\n── Analyzing parameter: {param} ──")
                self.plot_dashboard(
                    param, savepath=os.path.join(savedir, f"dashboard_{param}.png")
                )
                plt.close("all")
                self.plot_training_curves(
                    param, curve="val_corr",
                    savepath=os.path.join(savedir, f"training_curves_val_corr_{param}.png"),
                )
                plt.close("all")
                self.plot_training_curves(
                    param, curve="train_loss",
                    savepath=os.path.join(savedir, f"training_curves_train_loss_{param}.png"),
                )
                plt.close("all")
                self.plot_comparison_bars(
                    param, savepath=os.path.join(savedir, f"comparison_bars_{param}.png")
                )
                plt.close("all")

        if self.df["sweep_name"].nunique() >= 2:
            self.plot_multi_sweep_comparison(
                savepath=os.path.join(savedir, "multi_sweep_comparison.png")
            )
            plt.close("all")

        # Generate LOO figures for the best model
        try:
            self.generate_best_loo_figures(savedir=savedir)
        except Exception as e:
            print(f"Warning: could not generate LOO figures for best model: {e}")

        # Save a summary table
        summary = self.df.drop(columns=["run_dir", "sweep_dir"], errors="ignore")
        summary.to_csv(os.path.join(savedir, "summary_table.csv"), index=False)
        print(f"\nAll analysis saved to {savedir}")

    # ── LOO figures for best model ────────────────────────────────

    def generate_best_loo_figures(
        self,
        metric: str = "test_corr",
        savedir: Optional[str] = None,
    ):
        """Load the best model and generate LOO comparison figures.

        Parameters
        ----------
        metric : str
            Column used to rank experiments (highest = best).
        savedir : str, optional
            Where to save the figures.  Defaults to ``<sweep_dir>/analysis``.
        """
        from run_experiment import (
            _compute_loo_average_rate,
            _compute_loo_temporal,
            _plot_loo_comparison,
            load_model_from_run_dir,
            load_raw_data,
        )
        from sklearn.model_selection import train_test_split
        from utils import BinnedStimSpikeDataset

        if savedir is None:
            savedir = os.path.join(self.sweep_dirs[0], "analysis")
        os.makedirs(savedir, exist_ok=True)

        # --- Identify the best experiment ---
        sub = self.df.dropna(subset=[metric])
        if sub.empty:
            print(f"No completed experiments with metric '{metric}'")
            return
        best_row = sub.loc[sub[metric].idxmax()]
        best_run_dir = best_row["run_dir"]
        best_exp = best_row["experiment"]
        print(f"Best model: {best_exp}  ({metric}={best_row[metric]:.6f})")
        print(f"  run_dir: {best_run_dir}")

        if not os.path.isdir(best_run_dir):
            raise FileNotFoundError(
                f"Best experiment directory not found: {best_run_dir}"
            )

        # --- Load config ---
        cfg_path = os.path.join(best_run_dir, "config.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        # --- Load raw data ---
        print("Loading raw data (this may take a minute) …")
        raw = load_raw_data(cfg, logger=logging.getLogger("analyze_sweep"))
        n_stim_channels = raw["n_stim_channels"]
        n_neurons = raw["n_neurons"]

        # --- Load model ---
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        model, cfg, device = load_model_from_run_dir(
            best_run_dir, n_stim_channels, n_neurons, device=device,
        )
        print(f"  Loaded model on {device}")

        # --- Build test dataset ---
        SEED = cfg.get("seed", 42)
        SPLIT_MODE = cfg.get("split_mode", "oracle")
        INIT_STATE = cfg.get("init_state", False)
        HISTORY = cfg.get("history", 0)
        _model_type = cfg.get("model_type", "cnn")
        N_INITIAL_STATE_BINS = cfg.get("n_initial_state_bins", 1)
        if INIT_STATE and _model_type == "cnn":
            _kernel_sizes = cfg.get("kernel_sizes", [60])
            N_INITIAL_STATE_BINS = sum(k - 1 for k in _kernel_sizes)

        pattern_df = raw["pattern_df"]
        spike_responses = raw["spike_responses"]
        unique_trials_info = pattern_df[
            ["pattern_timing_index", "pattern_name", "is_oracle"]
        ].drop_duplicates()

        if SPLIT_MODE == "oracle":
            test_indices = unique_trials_info[
                unique_trials_info["is_oracle"]
            ]["pattern_timing_index"].tolist()
        else:
            all_timing_indices = list(spike_responses.keys())
            _, test_indices = train_test_split(
                all_timing_indices, test_size=0.15, random_state=SEED,
            )

        test_dataset = BinnedStimSpikeDataset(
            trial_indices=test_indices,
            pattern_df=pattern_df,
            spike_responses=spike_responses,
            channel_to_index=raw["channel_to_index"],
            timing_to_pattern=raw["timing_to_pattern"],
            input_bin_size_ms=cfg["input_bin_size_ms"],
            output_bin_size_ms=cfg["output_bin_size_ms"],
            n_input_bins=cfg["n_input_bins"],
            n_output_bins=cfg["n_output_bins"],
            max_time_ms=cfg.get("max_time_ms", 600),
            output_offset=cfg.get("output_offset", 0),
            encoding_mode=cfg.get("encoding_mode", "current"),
            init_state=INIT_STATE,
            n_initial_state_bins=N_INITIAL_STATE_BINS,
            history=HISTORY,
        )

        # USE_INIT_STATE flag (3-tuple from dataloader)
        if INIT_STATE and _model_type == "rnn" and (HISTORY is None or HISTORY < 1):
            use_init_state = True
        else:
            use_init_state = False

        # --- Compute & plot ---
        print("Computing LOO (average rate) …")
        diff_avg, model_avg, loo_avg = _compute_loo_average_rate(
            model, test_dataset, device, n_neurons, use_init_state,
        )
        _plot_loo_comparison(
            model_avg, loo_avg,
            title_suffix=f"Average Rate — {best_exp}",
            save_path=os.path.join(savedir, "best_model_vs_LOO_average_rate.png"),
        )
        plt.close("all")

        print("Computing LOO (temporal) …")
        diff_temp, model_temp, loo_temp = _compute_loo_temporal(
            model, test_dataset, device, n_neurons, use_init_state,
        )
        _plot_loo_comparison(
            model_temp, loo_temp,
            title_suffix=f"Temporal — {best_exp}",
            save_path=os.path.join(savedir, "best_model_vs_LOO_temporal.png"),
        )
        plt.close("all")

        print(
            f"LOO (avg rate): model r={np.mean(model_avg):.4f}, "
            f"LOO r={np.mean(loo_avg):.4f}, "
            f"model > LOO: {(diff_avg > 0).sum()}/{len(diff_avg)}"
        )
        print(
            f"LOO (temporal): model r={np.mean(model_temp):.4f}, "
            f"LOO r={np.mean(loo_temp):.4f}, "
            f"model > LOO: {(diff_temp > 0).sum()}/{len(diff_temp)}"
        )
        print(f"LOO figures saved to {savedir}")

    # ── Repr ────────────────────────────────────────────────────────

    def __repr__(self):
        return (
            f"SweepAnalyzer({len(self.df)} experiments, "
            f"params={self.param_cols}, "
            f"sweeps={self.df['sweep_name'].unique().tolist()})"
        )


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter sweep results")
    parser.add_argument("sweep_dirs", nargs="+", help="Sweep output directories")
    parser.add_argument("--params", nargs="*", default=None,
                        help="Parameters to analyze (auto-detected if omitted)")
    parser.add_argument("--metrics", nargs="*", default=None,
                        help="Metrics to plot")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots")
    args = parser.parse_args()

    sa = SweepAnalyzer(args.sweep_dirs)
    print(sa)

    if args.params:
        sa.param_cols = args.params

    sa.plot_all(savedir=args.output_dir)


if __name__ == "__main__":
    main()

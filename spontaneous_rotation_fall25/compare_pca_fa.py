#!/usr/bin/env python3
"""
compare_pca_fa.py  –  Leave-one-out MSE comparison of PCA, Factor Analysis,
                       and NMF for reconstructing neuron fluorescence traces.

Reference: Yu et al. (2009), Figure 5.
Dataset:   Neuron_Act_Info.mat  (332 neurons, ~30 min @ ~30 Hz)

Usage
-----
# Run all methods locally with varimax FA rotation (default)
python compare_pca_fa.py

# Use promax rotation for FA (requires factor_analyzer package)
python compare_pca_fa.py --fa-rotation promax

# Run only NMF
python compare_pca_fa.py --method NMF

# Run on NOTS via the companion SLURM script
sbatch slurm/run_pca_fa.slurm "python compare_pca_fa.py --method FactorAnalysis --fa-rotation promax"
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster use
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import FactorAnalysis, NMF, PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class FactorAnalyzerWrapper:
    """
    Thin sklearn-compatible wrapper around factor_analyzer.FactorAnalyzer.
    Used when an oblique rotation (e.g. promax) is requested, since sklearn's
    FactorAnalysis only supports orthogonal rotations (varimax, quartimax).

    Exposes fit_transform / transform / components_ to match sklearn's interface.
    """

    def __init__(self, n_components: int, rotation: str = "promax", max_iter: int = 1000):
        try:
            from factor_analyzer import FactorAnalyzer  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'factor_analyzer' package is required for oblique rotations. "
                "Install it with:  pip install factor_analyzer"
            )
        self.n_components = n_components
        self.rotation = rotation
        self.max_iter = max_iter
        self._fa = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        from factor_analyzer import FactorAnalyzer
        self._fa = FactorAnalyzer(
            n_factors=self.n_components,
            rotation=self.rotation,
            max_iter=self.max_iter,
        )
        self._fa.fit(X)
        return self._fa.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._fa.transform(X)

    @property
    def loadings_(self) -> np.ndarray:
        """Shape: (n_features, n_factors) — same convention as factor_analyzer."""
        return self._fa.loadings_

    @property
    def components_(self) -> np.ndarray:
        """Shape: (n_factors, n_features) — sklearn convention."""
        return self._fa.loadings_.T


class NMFWrapper:
    """
    sklearn-compatible NMF that handles data with negative values by
    shifting each feature to be non-negative before fitting.  The shift
    is computed from the training set and reused in transform().

    This allows NMF to run after --standardize without crashing, though
    the interpretation is that we're factorising (X - min_train) ≥ 0.
    """

    def __init__(self, n_components: int, max_iter: int = 1000, random_state: int = 42):
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        self._nmf = NMF(n_components=n_components, max_iter=max_iter, random_state=random_state)
        self._col_min: np.ndarray | None = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self._col_min = X.min(axis=0)
        return self._nmf.fit_transform(X - self._col_min)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_shifted = np.clip(X - self._col_min, 0, None)
        return self._nmf.transform(X_shifted)

    @property
    def components_(self) -> np.ndarray:
        return self._nmf.components_


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model(method_name: str, n_components: int, fa_rotation: str,
               fa_max_iter: int, fa_tol: float, seed: int = 42):
    """Return an unfitted sklearn-compatible model for the given method."""
    if method_name == "PCA":
        return PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    elif method_name == "FactorAnalysis":
        if fa_rotation == "promax":
            return FactorAnalyzerWrapper(n_components, rotation="promax", max_iter=fa_max_iter)
        else:
            rot = None if fa_rotation == "none" else fa_rotation
            return FactorAnalysis(
                n_components=n_components,
                svd_method="randomized",
                rotation=rot,
                max_iter=fa_max_iter,
                tol=fa_tol,
            )
    elif method_name == "NMF":
        return NMFWrapper(n_components=n_components, max_iter=fa_max_iter, random_state=seed)
    else:
        raise ValueError(f"Unknown method: {method_name!r}")


def method_label(method_name: str, fa_rotation: str) -> str:
    """Human-readable display name used in plot titles and legend entries."""
    if method_name == "FactorAnalysis":
        rot = fa_rotation if fa_rotation != "none" else "no rotation"
        return f"FA ({rot})"
    return method_name


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(mat_path: str):
    """Load Neuron_Act_Info.mat and return (neuron_activity, neuron_range)."""
    data = loadmat(mat_path)["Neuron_Act_Info"][0][0]
    roi_mask, neuron_activity, max_value = data
    neuron_range = int(np.amin(roi_mask)), int(np.amax(roi_mask))
    print(f"ROI mask shape:        {roi_mask.shape}")
    print(f"Neuron activity shape: {neuron_activity.shape}  (neurons × time)")
    print(f"Neuron ID range:       {neuron_range}")
    return neuron_activity, neuron_range


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_method(
    method_name: str,
    test_n_pcs: list,
    decomposition_train: np.ndarray,
    decomposition_test: np.ndarray,
    selected_neurons: list,
    neuron_range: tuple,
    n_neurons_per_group: int,
    n_subsets: int,
    fa_rotation: str = "varimax",
    fa_max_iter: int = 1000,
    fa_tol: float = 1e-2,
    seed: int = 42,
    verbose: bool = True,
):
    """
    Evaluate a method over a grid of component counts.

    For each n_components:
      - fit on train timepoints, excluding held-out neurons
      - regress latents → held-out neuron activity (train set)
      - predict on test timepoints, measure squared error

    Returns (mean_errors, std_errors) each of shape (len(test_n_pcs),).
    """
    plot_errors = []
    plot_error_std = []
    label = method_label(method_name, fa_rotation)

    for n_components in test_n_pcs:
        model = make_model(method_name, n_components, fa_rotation, fa_max_iter, fa_tol, seed)
        line = LinearRegression()
        errors = []

        for sample_neuron_set in selected_neurons:
            training_neurons = np.setdiff1d(np.arange(neuron_range[1]), sample_neuron_set)
            assert len(training_neurons) == neuron_range[1] - n_neurons_per_group

            train_z = model.fit_transform(decomposition_train[:, training_neurons])
            line.fit(train_z, decomposition_train[:, sample_neuron_set])

            test_z = model.transform(decomposition_test[:, training_neurons])
            pred = line.predict(test_z)
            err = np.sum(
                np.square(pred - decomposition_test[:, sample_neuron_set]), axis=0
            )
            assert err.shape[0] == n_neurons_per_group
            errors.append(np.mean(err))

        assert len(errors) == n_subsets
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        if verbose:
            print(
                f"  {label:22s}  n_components={n_components:4d}"
                f"  mean_err={mean_err:.5f}  std={std_err:.5f}"
            )
        plot_errors.append(mean_err)
        plot_error_std.append(std_err)

    return np.array(plot_errors), np.array(plot_error_std)


# ---------------------------------------------------------------------------
# Variance explained
# ---------------------------------------------------------------------------

def compute_variance_explained(
    method_name: str,
    test_n_pcs: list,
    X_train: np.ndarray,
    fa_rotation: str = "varimax",
    fa_max_iter: int = 1000,
    fa_tol: float = 1e-2,
    seed: int = 42,
) -> np.ndarray:
    """
    Fit each method on the full training set (all neurons) and return the
    cumulative proportion of variance explained at each n_components.

    PCA          : sum(explained_variance_ratio_)
    FactorAnalysis : loadings-based —
                     fa_loadings = model.components_.T  (n_features × n_factors)
                     var_exp_k   = Σ_features  loadings[:, k]²
                     cum_prop    = Σ_k  var_exp_k / Σ_neurons var(X_train)
    NMF          : reconstruction R²  =  1 − ||X − WH||² / ||X − mean(X)||²
                   (NMF has no canonical ordering of components, so this is
                    the total R² for the k-component model at each grid point)
    """
    total_var = np.sum(X_train.var(axis=0))
    cum_var_exp = []

    for n_components in test_n_pcs:
        model = make_model(method_name, n_components, fa_rotation, fa_max_iter, fa_tol, seed)

        if method_name == "PCA":
            model.fit(X_train)
            cum_var_exp.append(float(np.sum(model.explained_variance_ratio_)))

        elif method_name == "FactorAnalysis":
            model.fit_transform(X_train)          # FactorAnalyzerWrapper needs fit_transform
            fa_loadings = model.components_.T      # (n_features, n_factors)
            var_exp = np.sum(fa_loadings ** 2, axis=0)
            cum_var_exp.append(float(np.sum(var_exp) / total_var))

        elif method_name == "NMF":
            W = model.fit_transform(X_train)       # NMFWrapper shifts internally
            H = model.components_
            X_shifted = X_train - model._col_min   # use same shift for residual
            reconstruction = W @ H
            ss_res = np.sum((X_shifted - reconstruction) ** 2)
            ss_tot = np.sum((X_shifted - X_shifted.mean(axis=0)) ** 2)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            cum_var_exp.append(r2)

    return np.array(cum_var_exp)


def plot_variance_explained(
    method_names: list,
    labels: dict,
    var_results: dict,
    test_n_pcs: list,
    output_dir: str,
):
    """Overlay cumulative variance explained curves for all methods."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in method_names:
        ax.plot(test_n_pcs, var_results[name] * 100, marker="o", label=labels[name])

    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title("Variance explained vs number of components")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, "variance_explained.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {fig_path}")
    plt.close()
    return fig_path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    method_names: list,
    labels: dict,
    results: dict,
    test_n_pcs: list,
    n_subsets: int,
    n_neurons_per_group: int,
    output_dir: str,
):
    """One subplot per method: mean ± std MSE vs n_components."""
    fig, axes = plt.subplots(1, len(method_names), figsize=(6 * len(method_names), 6), sharey=True)
    if len(method_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, method_names):
        errs, stds = results[name]
        ax.plot(test_n_pcs, errs, marker="o", label="train ratio 0.8")
        ax.fill_between(test_n_pcs, errs - stds, errs + stds, alpha=0.3)
        ax.set_xlabel("Number of components")
        ax.set_ylabel("Avg MSE on test time-series over holdout neurons")
        ax.set_title(labels[name])
        ax.grid(True)

    plt.legend()
    plt.suptitle(f"MSE: {n_subsets} groups × {n_neurons_per_group} neurons")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "mse_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {fig_path}")
    plt.close()
    return fig_path


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="PCA / FactorAnalysis / NMF leave-one-out MSE comparison"
    )

    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument(
        "--data",
        default=os.path.join(here, "Neuron_Act_Info.mat"),
        help="Path to Neuron_Act_Info.mat (default: same directory as script)",
    )
    p.add_argument(
        "--method",
        choices=["PCA", "FactorAnalysis", "NMF", "all"],
        default="all",
        help="Which method(s) to evaluate (default: all)",
    )
    p.add_argument(
        "--fa-rotation",
        choices=["none", "varimax", "quartimax", "promax"],
        default="varimax",
        help=(
            "Rotation applied to FA loadings.  'varimax' and 'quartimax' use "
            "sklearn; 'promax' uses factor_analyzer (oblique).  (default: varimax)"
        ),
    )
    p.add_argument(
        "--n-components",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10, 30, 50, 60, 80, 100, 200],
        help="Component counts to sweep (default: 1 3 5 10 30 50 60 80 100 200)",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of timepoints used for training (default: 0.8)",
    )
    p.add_argument(
        "--n-subsets",
        type=int,
        default=10,
        help="Number of held-out neuron subsets (default: 10)",
    )
    p.add_argument(
        "--n-neurons-per-group",
        type=int,
        default=30,
        help="Neurons per held-out subset (default: 30)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join(here, "results"),
        help="Directory to save output figures (default: ./results/)",
    )
    p.add_argument(
        "--fa-max-iter",
        type=int,
        default=1000,
        help="Max iterations for FactorAnalysis / NMF (default: 1000)",
    )
    p.add_argument(
        "--fa-tol",
        type=float,
        default=1e-2,
        help="Convergence tolerance for FactorAnalysis (default: 1e-2)",
    )
    p.add_argument(
        "--standardize",
        action="store_true",
        default=False,
        help=(
            "Mean-center and unit-variance scale all neurons before fitting "
            "(StandardScaler, fit on train only).  Note: NMF will shift the "
            "scaled data back to non-negative range internally."
        ),
    )
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    # --- Load data ---
    neuron_activity, neuron_range = load_data(args.data)

    # --- Prepare dataset (transpose to time × neurons) ---
    dataset = neuron_activity.T  # shape: (T, N)

    np.random.seed(args.seed)
    T = dataset.shape[0]
    train_idx = np.random.choice(T, size=int(args.train_ratio * T), replace=False)
    test_idx = np.setdiff1d(np.arange(T), train_idx)
    train_data = dataset[train_idx, :]
    test_data = dataset[test_idx, :]
    print(f"Train shape: {train_data.shape}  Test shape: {test_data.shape}")

    # --- Optional standardization (fit scaler on train only) ---
    if args.standardize:
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)
        print("Standardized: mean-centered and unit-variance scaled.")

    # --- Sample held-out neuron subsets ---
    selected_neurons = []
    for i in range(args.n_subsets):
        np.random.seed(i)
        selected_neurons.append(
            np.random.choice(neuron_range[1], size=args.n_neurons_per_group, replace=False)
        )
    print(f"Sampled {args.n_subsets} subsets of {args.n_neurons_per_group} neurons each.")

    # --- Determine which methods to run ---
    if args.method == "all":
        method_names = ["PCA", "FactorAnalysis", "NMF"]
    else:
        method_names = [args.method]

    labels = {name: method_label(name, args.fa_rotation) for name in method_names}

    # --- Run evaluation ---
    results = {}
    var_results = {}

    for name in method_names:
        print(f"\n{'='*70}")
        print(f"Method: {labels[name]}")
        print(f"{'='*70}")
        t0 = time.time()

        errs, stds = test_method(
            method_name=name,
            test_n_pcs=args.n_components,
            decomposition_train=train_data,
            decomposition_test=test_data,
            selected_neurons=selected_neurons,
            neuron_range=neuron_range,
            n_neurons_per_group=args.n_neurons_per_group,
            n_subsets=args.n_subsets,
            fa_rotation=args.fa_rotation,
            fa_max_iter=args.fa_max_iter,
            fa_tol=args.fa_tol,
            seed=args.seed,
            verbose=args.verbose,
        )
        print(f"Done in {time.time() - t0:.1f}s")
        results[name] = (errs, stds)

        print(f"  Computing variance explained...")
        var_results[name] = compute_variance_explained(
            method_name=name,
            test_n_pcs=args.n_components,
            X_train=train_data,
            fa_rotation=args.fa_rotation,
            fa_max_iter=args.fa_max_iter,
            fa_tol=args.fa_tol,
            seed=args.seed,
        )

    # --- Save arrays ---
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(
        os.path.join(args.output_dir, "results.npz"),
        n_components=np.array(args.n_components),
        **{f"{name}_errors": results[name][0] for name in method_names},
        **{f"{name}_stds": results[name][1] for name in method_names},
        **{f"{name}_cum_var_exp": var_results[name] for name in method_names},
    )

    # --- Plots ---
    plot_results(
        method_names=method_names,
        labels=labels,
        results=results,
        test_n_pcs=args.n_components,
        n_subsets=args.n_subsets,
        n_neurons_per_group=args.n_neurons_per_group,
        output_dir=args.output_dir,
    )
    plot_variance_explained(
        method_names=method_names,
        labels=labels,
        var_results=var_results,
        test_n_pcs=args.n_components,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

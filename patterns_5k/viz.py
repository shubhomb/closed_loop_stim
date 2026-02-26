from matplotlib import colors
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import os 
import logging
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from io import BytesIO
import pandas as pd
from PIL import Image
from utils import bin_spike_response
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from metrics import _coarsen



# Colormap for polarity: index mapping is vmin=-1, vmax=1 so
# -1 -> blue (negative), 0 -> white (no stim), +1 -> red (positive).
polarity_cmap = ListedColormap(['blue', 'white', 'red'])


def plot_spike_bin_distribution(dataset, mode='counts', max_count=None, figsize=(16, 5), savepath=None, logger=None):
    """Plot the distribution of spike counts across all target bins.
    
    Args:
        dataset: BinnedStimSpikeDataset instance
        mode: 'counts' to show full spike count distribution, 
              'binary' to show only zero vs non-zero bins
        max_count: Maximum spike count to show (bins above this are grouped). 
                   If None, uses the max observed count. Only used in 'counts' mode.
        figsize: Figure size tuple
        savepath: If provided, save the figure to this path
        logger: Optional logger instance
    
    Returns:
        fig, axes: matplotlib figure and axes objects
    """
    if mode not in ['counts', 'binary']:
        raise ValueError(f"mode must be 'counts' or 'binary', got '{mode}'")
    
    # Collect all spike counts from targets
    all_counts = []
    for timing_idx in dataset.trial_indices:
        spikes = dataset.spike_responses_binned[timing_idx] # will be binned at dataset output_bin_ms
        for t in range(dataset.n_output_bins + 1):
            out_start = t + dataset.output_offset
            y = spikes[:, out_start : out_start + dataset.n_output_bins]
            all_counts.extend(y.flatten().tolist())
    
    all_counts = np.array(all_counts)
    
    # Compute statistics
    sparsity = dataset.compute_sparsity()
    sample_sparsity = dataset.compute_sample_sparsity()
    mean_count = all_counts.mean()
    std_count = all_counts.std()
    max_observed = int(all_counts.max())
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # =====================
    # Column 1: Spike count distribution (binary or counts mode)
    # =====================
    if mode == 'binary':
        # Binary mode: show only zero vs non-zero
        zero_count = sparsity['zero_bins']
        nonzero_count = sparsity['nonzero_bins']
        
        axes[0].bar(['Zero (0)', 'Non-zero (≥1)'], [zero_count, nonzero_count], 
                    color=['steelblue', 'coral'], edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Spike Bin Category')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Binary Spike Distribution\n(bin_size={dataset.output_bin_size}ms, n_output_bins={dataset.n_output_bins})')
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add statistics text box
        stats_text = (f'Total bins: {sparsity["total_bins"]:,}\n'
                      f'Zero: {zero_count:,} ({sparsity["sparsity_pct"]:.1f}%)\n'
                      f'Non-zero: {nonzero_count:,} ({100 - sparsity["sparsity_pct"]:.1f}%)')
        axes[0].text(0.95, 0.95, stats_text, transform=axes[0].transAxes, fontsize=9,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    else:  # mode == 'counts'
        if max_count is None:
            max_count = max_observed  # Show all counts
        
        # Create histogram bins for all counts
        bins = np.arange(0, max_count + 2) - 0.5  # Centers at 0, 1, 2, ...
        
        # Left plot: histogram of spike counts
        counts_clipped = np.clip(all_counts, 0, max_count)
        axes[0].hist(counts_clipped, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Spike Count per Bin')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Distribution of Spike Counts\n(bin_size={dataset.output_bin_size}ms, n_output_bins={dataset.n_output_bins})')
        axes[0].set_xticks(range(0, max_count + 1))
        if max_count < max_observed:
            axes[0].set_xticklabels([str(i) for i in range(max_count)] + [f'{max_count}+'])
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = (f'Total bins: {sparsity["total_bins"]:,}\n'
                      f'Zero bins: {sparsity["zero_bins"]:,} ({sparsity["sparsity_pct"]:.1f}%)\n'
                      f'Mean: {mean_count:.3f}\n'
                      f'Std: {std_count:.3f}\n'
                      f'Max: {max_observed}')
        axes[0].text(0.95, 0.95, stats_text, transform=axes[0].transAxes, fontsize=9,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # =====================
    # Column 2: Per-sample sparsity (fraction of neurons with zero spikes)
    # =====================
    per_sample_zero_frac = sample_sparsity['per_sample_zero_frac']
    
    # Histogram of per-sample zero fractions
    axes[1].hist(per_sample_zero_frac * 100, bins=50, edgecolor='black', alpha=0.7, color='mediumpurple')
    axes[1].set_xlabel('% Neurons with Zero Spikes')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title(f'Per-Sample Sparsity (across {dataset.n_neurons} neurons)')
    axes[1].grid(True, alpha=0.3)
    
    # Add statistics text box
    sample_stats_text = (f'Total samples: {sample_sparsity["total_samples"]:,}\n'
                         f'Mean: {sample_sparsity["mean_zero_frac"]*100:.1f}%\n'
                         f'Std: {sample_sparsity["std_zero_frac"]*100:.1f}%\n'
                         f'Min: {per_sample_zero_frac.min()*100:.1f}%\n'
                         f'Max: {per_sample_zero_frac.max()*100:.1f}%')
    axes[1].text(0.05, 0.95, sample_stats_text, transform=axes[1].transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # =====================
    # Column 3: Per-neuron average spike rate with SEM error bars
    # =====================
    # Collect spike rates per neuron per trial for SEM calculation
    neuron_rates_per_trial = []  # List of (n_neurons,) arrays
    for timing_idx in dataset.trial_indices:
        spikes = dataset.spike_responses_binned[timing_idx]  # (n_neurons, total_bins)
        neuron_rates_per_trial.append(spikes.mean(axis=1))  # average across time bins
    
    neuron_rates_per_trial = np.array(neuron_rates_per_trial)  # (n_trials, n_neurons)
    
    # Compute mean and SEM across trials
    neuron_means = neuron_rates_per_trial.mean(axis=0)  # (n_neurons,)
    neuron_sem = neuron_rates_per_trial.std(axis=0) / np.sqrt(len(dataset.trial_indices))  # SEM
    
    x = np.arange(len(neuron_means))
    axes[2].bar(x, neuron_means, yerr=neuron_sem, alpha=0.7, color='coral', 
                capsize=2, error_kw={'elinewidth': 1, 'capthick': 1})
    axes[2].set_xlabel('Neuron Index')
    axes[2].set_ylabel(f'Mean Spikes per {dataset.output_bin_size}ms Bin')
    axes[2].set_title(f'Average Spike Rate per Neuron (± SEM, n={len(dataset.trial_indices)} trials)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        if logger:
            logger.info(f"Saved spike distribution plot ({mode} mode) to {savepath}")
    
    return fig, axes


def plot_test_prediction_comparison(all_preds, all_targets, savepath=None, logger=None):
    """
    Plot test prediction comparison: scatter plot and bar chart.
    
    Args:
        all_preds: (n_samples, n_neurons, n_output_bins) - log predictions from model
        all_targets: (n_samples, n_neurons, n_output_bins) - target spike counts
        savepath: If provided, save the figure to this path
        logger: Optional logger instance
    
    Returns:
        fig, axes: matplotlib figure and axes objects
        corr: Pearson correlation coefficient
        pval: p-value for correlation
    """
    # Convert log predictions to actual rates (for Poisson)
    pred_rates = np.exp(all_preds)
    target_counts = all_targets
    
    # Average across all samples and output bins
    avg_pred_per_neuron = pred_rates.mean(axis=(0, 2))  # (n_neurons,)
    avg_target_per_neuron = target_counts.mean(axis=(0, 2))  # (n_neurons,)
    
    # Compute standard error of the mean (SEM) for predictions
    sem_pred_per_neuron = pred_rates.std(axis=(0, 2)) / np.sqrt(pred_rates.shape[0])
    sem_target_per_neuron = target_counts.std(axis=(0, 2)) / np.sqrt(target_counts.shape[0])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot: predicted vs actual per neuron
    axes[0].scatter(avg_target_per_neuron, avg_pred_per_neuron, alpha=0.6)
    max_val = max(avg_target_per_neuron.max(), avg_pred_per_neuron.max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', label='Target')
    axes[0].set_xlabel('Actual Average Spike Count')
    axes[0].set_ylabel('Predicted Average Spike Count')
    axes[0].set_title('Average Prediction vs Actual per Neuron (Test Set)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Bar chart comparison for first 20 neurons
    n_show = min(20, len(avg_target_per_neuron))
    x = np.arange(n_show)
    width = 0.35
    axes[1].bar(x - width/2, avg_target_per_neuron[:n_show], width, yerr=sem_target_per_neuron[:n_show], 
                label='Actual', alpha=0.8, capsize=3)
    axes[1].bar(x + width/2, avg_pred_per_neuron[:n_show], width, yerr=sem_pred_per_neuron[:n_show], 
                label='Predicted', alpha=0.8, capsize=3)
    axes[1].set_xlabel('Neuron Index')
    axes[1].set_ylabel('Average Spike Count per Bin')
    axes[1].set_title(f'First {n_show} Neurons: Actual vs Predicted Means')
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath, dpi=150)
    
    # Compute correlation
    corr, pval = pearsonr(avg_target_per_neuron, avg_pred_per_neuron)
    if logger:
        logger.info(f"Correlation between predicted and actual per-neuron averages: r={corr:.4f}, p={pval:.2e}")
    
    return fig, axes, corr, pval


def analyze_pattern_responses_by_pattern_name(model, dataset, device, use_init_state=False):
    """
    Analyze model predictions grouped by actual pattern identity (pattern_name).
    
    Args:
        model: PyTorch model
        dataset: BinnedStimSpikeDataset instance
        device: torch device
        use_init_state: If True, pass initial_spikes to model forward
    
    Returns:
        pattern_names: list of unique pattern names
        pattern_true_means: (n_patterns, n_neurons) - avg actual spike rate per pattern
        pattern_pred_means: (n_patterns, n_neurons) - avg predicted rate per pattern  
        pattern_counts: (n_patterns,) - number of samples per pattern
    """
    model.eval()
    
    # Get unique pattern names from the dataset's trial indices
    pattern_names_list = []
    for timing_idx in dataset.trial_indices:
        pattern_name = dataset.timing_to_pattern[timing_idx]
        if pattern_name not in pattern_names_list:
            pattern_names_list.append(pattern_name)
    
    pattern_names_list = sorted(pattern_names_list)
    pattern_to_idx = {p: i for i, p in enumerate(pattern_names_list)}
    n_patterns = len(pattern_names_list)
    n_neurons = dataset.n_neurons
    
    print(f"Found {n_patterns} unique oracle patterns in test set")
    
    # Initialize accumulators
    pattern_true_sums = np.zeros((n_patterns, n_neurons))
    pattern_pred_sums = np.zeros((n_patterns, n_neurons))
    pattern_counts = np.zeros(n_patterns)
    
    # Process all samples
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    sample_idx = 0
    
    with torch.no_grad():
        for batch in loader:
            if use_init_state and len(batch) == 3:
                batch_x, batch_y, batch_init = batch
                batch_x, batch_init = batch_x.to(device), batch_init.to(device)
                # Get predictions (convert log rates to rates)
                if hasattr(model, 'forward') and 'initial_spikes' in model.forward.__code__.co_varnames:
                    preds = torch.exp(model(batch_x, initial_spikes=batch_init)).cpu().numpy()
                else:
                    preds = torch.exp(model(batch_x)).cpu().numpy()
            else:
                batch_x, batch_y = batch[:2]
                batch_x = batch_x.to(device)
                # Get predictions (convert log rates to rates)
                preds = torch.exp(model(batch_x)).cpu().numpy()  # (batch, n_neurons, n_output_bins)
            targets = batch_y.numpy()  # (batch, n_neurons, n_output_bins)
            
            # Average across output bins
            preds_rate = preds.mean(axis=2)  # (batch, n_neurons)
            targets_rate = targets.mean(axis=2)  # (batch, n_neurons)
            
            # Assign each sample to its pattern
            for i in range(batch_x.shape[0]):
                timing_idx, t = dataset.samples[sample_idx]
                pattern_name = dataset.timing_to_pattern[timing_idx]
                pattern_idx = pattern_to_idx[pattern_name]
                
                pattern_true_sums[pattern_idx] += targets_rate[i]
                pattern_pred_sums[pattern_idx] += preds_rate[i]
                pattern_counts[pattern_idx] += 1
                
                sample_idx += 1
    
    # Compute means
    pattern_counts_safe = np.maximum(pattern_counts, 1)[:, None]
    pattern_true_means = pattern_true_sums / pattern_counts_safe
    pattern_pred_means = pattern_pred_sums / pattern_counts_safe
    
    # Print summary
    print(f"Samples per pattern: min={pattern_counts.min():.0f}, max={pattern_counts.max():.0f}, mean={pattern_counts.mean():.1f}")
    
    return pattern_names_list, pattern_true_means, pattern_pred_means, pattern_counts


def plot_pattern_selectivity(pattern_names, pat_true, pat_pred, savepath=None, logger=None):
    """
    Plot pattern selectivity analysis: top patterns, scatter, and correlation distribution.
    
    Args:
        pattern_names: list of pattern names
        pat_true: (n_patterns, n_neurons) - actual responses
        pat_pred: (n_patterns, n_neurons) - predicted responses
        savepath: If provided, save the figure to this path
        logger: Optional logger instance
    
    Returns:
        fig, axes: matplotlib figure and axes objects
        neuron_correlations: array of per-neuron correlations
    """
    # 1. Find the most "interesting" neuron (highest variance in true responses across patterns)
    neuron_variances = np.var(pat_true, axis=0)
    best_neuron_idx = np.argmax(neuron_variances)
    
    # 2. Get data for this neuron
    n_true = pat_true[:, best_neuron_idx]  # (n_patterns,)
    n_pred = pat_pred[:, best_neuron_idx]  # (n_patterns,)
    
    # 3. Sort patterns by TRUE response magnitude
    sorted_indices = np.argsort(n_true)[::-1]  # Descending order
    top_k = min(20, len(pattern_names))  # Show top 20 patterns
    
    top_indices = sorted_indices[:top_k]
    top_true = n_true[top_indices]
    top_pred = n_pred[top_indices]
    top_names = [pattern_names[i] for i in top_indices]
    
    # 4. Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # --- Plot 1: The "Ranking" Plot ---
    indices = np.arange(top_k)
    width = 0.35
    
    axes[0].bar(indices - width/2, top_true, width, label='Actual Response', color='steelblue', alpha=0.8)
    axes[0].bar(indices + width/2, top_pred, width, label='Predicted Response', color='coral', alpha=0.8)
    axes[0].set_xticks(indices)
    axes[0].set_xticklabels([str(n) for n in top_names], rotation=45, ha='right')
    axes[0].set_title(f"Neuron {best_neuron_idx}: Top {top_k} Patterns by Actual Response")
    axes[0].set_xlabel("Pattern Name")
    axes[0].set_ylabel("Average Spike Rate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # --- Plot 2: Scatter of ALL Patterns for this Neuron ---
    correlation, pval = pearsonr(n_true, n_pred)
    
    axes[1].scatter(n_true, n_pred, alpha=0.7, c='purple', edgecolor='w', s=60)
    max_val = max(n_true.max(), n_pred.max())
    axes[1].plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y=x')
    axes[1].set_title(f"Pattern Selectivity (Neuron {best_neuron_idx})\nPearson r = {correlation:.4f}, p = {pval:.2e}")
    axes[1].set_xlabel("Actual Response to Pattern")
    axes[1].set_ylabel("Predicted Response to Pattern")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # --- Plot 3: Correlation across ALL neurons ---
    neuron_correlations = []
    for neuron_idx in range(pat_true.shape[1]):
        if pat_true[:, neuron_idx].std() > 0 and pat_pred[:, neuron_idx].std() > 0:
            r, _ = pearsonr(pat_true[:, neuron_idx], pat_pred[:, neuron_idx])
            neuron_correlations.append(r)
        else:
            neuron_correlations.append(0)
    
    neuron_correlations = np.array(neuron_correlations)
    
    axes[2].hist(neuron_correlations, bins=30, edgecolor='black', alpha=0.7, color='teal')
    axes[2].axvline(np.mean(neuron_correlations), color='red', linestyle='--', 
                    label=f'Mean r = {np.mean(neuron_correlations):.3f}')
    axes[2].set_title(f"Pattern Selectivity Correlation per Neuron\n(across {len(pattern_names)} patterns)")
    axes[2].set_xlabel("Pearson r (Actual vs Predicted)")
    axes[2].set_ylabel("Number of Neurons")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath, dpi=150)
    
    if logger:
        logger.info(f"Pattern selectivity summary:")
        logger.info(f"  Mean correlation across neurons: {np.mean(neuron_correlations):.4f}")
        logger.info(f"  Median correlation: {np.median(neuron_correlations):.4f}")
        logger.info(f"  Neurons with r > 0.5: {(neuron_correlations > 0.5).sum()} / {len(neuron_correlations)}")
        logger.info(f"  Neurons with r > 0.3: {(neuron_correlations > 0.3).sum()} / {len(neuron_correlations)}")
    
    return fig, axes, neuron_correlations

def compute_avg_spikes_across_trials(timing_indices, spike_responses_dict, bin_size_ms, max_time_ms):
    """Compute average number of spikes across trials for each neuron in each bin."""
    n_trials = len(timing_indices)
    if n_trials == 0:
        raise ValueError("No timing indices provided for averaging spikes.")
    spike_sum = None

    for i, timing_idx in enumerate(timing_indices):
        binned = bin_spike_response(spike_responses_dict[timing_idx], bin_size_ms, max_time_ms, remainder="append")
        if spike_sum is None:
            spike_sum = np.zeros_like(binned, dtype=np.float32)
        spike_sum += binned

    return spike_sum / n_trials


def plot_oracle_trials_by_pattern(model, test_dataset, pattern_df, spike_responses, pattern_polarities,
                                   output_bin_size_ms, n_input_bins, n_output_bins, output_offset, max_time_ms,
                                   n_neurons, device, out_base, input_bin_size_ms=None, logger=None, pattern_limit=None,
                                   use_init_state=False, n_initial_state_bins=None, history=0, init_state=False):
    """
    Generate per-pattern visualizations for oracle trials.
    
    Args:
        model: PyTorch model
        test_dataset: BinnedStimSpikeDataset for test set
        pattern_df: DataFrame with pattern info
        spike_responses: dict mapping timing_idx -> spike response arrays (in ms, 1ms resolution)
        pattern_polarities: dict mapping pattern_name -> polarity arrays
        bin_size: OUTPUT temporal resolution in ms (for spike visualization)
        n_input_bins, n_output_bins, output_offset: dataset config
        max_time_ms: maximum time to consider in ms
        n_neurons: number of neurons
        device: torch device
        out_base: output directory
        input_bin_size_ms: INPUT temporal resolution in ms (for stim). If None, uses test_dataset.input_bin_size
        logger: Optional logger instance
        use_init_state: Whether to use initial state for RNN models
        n_initial_state_bins: Number of initial state bins (for plotting). If None, gets from test_dataset.
        history: Number of output bins to lag and concatenate as spike history input (for Pillow models). Default 0 (no history).
    """
    os.makedirs(out_base, exist_ok=True)
    
    # Get input bin size from dataset if not provided
    if input_bin_size_ms is None:
        input_bin_size_ms = test_dataset.input_bin_size
    
    # Get n_initial_state_bins from dataset if not provided
    if n_initial_state_bins is None:
        n_initial_state_bins = getattr(test_dataset, 'n_initial_state_bins', 1)
    
    # Calculate the time offset for initial state bins (negative time before 0)
    init_state_time_ms = n_initial_state_bins * output_bin_size_ms if use_init_state else 0
    
    # spike_responses are in 1ms resolution, so use bin_size and max_time_ms directly
    
    # Get unique pattern info
    unique_trials = pattern_df[['pattern_timing_index', 'pattern_name', 'is_oracle']].drop_duplicates()
    oracle_patterns = unique_trials[unique_trials['is_oracle']]['pattern_name'].unique()
    
    if logger:
        logger.info(f"Found {len(oracle_patterns)} oracle patterns")
        logger.info(f"Using input_bin_size={input_bin_size_ms}ms, output_bin_size={output_bin_size_ms}ms, n_input_bins={n_input_bins}, n_output_bins={n_output_bins}, output_offset={output_offset}")
    
    
    
    # Calculate number of bins at each resolution
    total_input_bins = max_time_ms // input_bin_size_ms   # e.g., 600 // 10 = 60
    total_output_bins = max_time_ms // output_bin_size_ms          # e.g., 600 // 60 = 10
    
    # Valid starting positions (using input resolution for indexing stim)
    max_start_for_output = total_output_bins - output_offset - n_output_bins
    max_start_for_input = total_input_bins - n_input_bins
    max_valid_start = min(max_start_for_output, max_start_for_input)
    
    if logger:
        logger.info(f"Total input bins: {total_input_bins}, Total output bins: {total_output_bins}")
        logger.info(f"Valid input positions: 0 to {max_valid_start}")
    
    # Process each oracle pattern
    for p_idx, pattern_name in enumerate(oracle_patterns):        # Go through each oracle pattern
        if pattern_limit is not None and p_idx >= pattern_limit:
            if logger:
                logger.info(f"Pattern limit of {pattern_limit} reached, stopping.")
            break   
        pattern_dir = os.path.join(out_base, f"pattern_{pattern_name}")
        os.makedirs(pattern_dir, exist_ok=True)
        
        # Get all timing indices for this pattern
        timing_list = unique_trials[unique_trials['pattern_name'] == pattern_name]['pattern_timing_index'].tolist()
        timing_list = sorted(timing_list)
        
        # Get the binned stim pattern for this pattern from TEST dataset
        stim_binned = test_dataset.pattern_stims[pattern_name]  # (n_channels, total_bins)
        
        # Get polarity data for this pattern - extend to max_time_ms
        polarity_600 = pattern_polarities[pattern_name]  # (n_channels, 600)
        polarity_base = np.zeros((polarity_600.shape[0], max_time_ms))
        polarity_base[:, :min(600, max_time_ms)] = polarity_600[:, :min(600, max_time_ms)]

        # Make the polarity plot last for 10 bins (added w shifted versions)
        polarity_plot = np.zeros_like(polarity_base)
        for shift in range(10):
            if shift == 0:
                polarity_plot += polarity_base
            else:
                polarity_plot[:, shift:] += polarity_base[:, :-shift]
        
        for trial_idx in range(len(timing_list)):
            timing_idx = timing_list[trial_idx]
            
            # Actual spike response - binned (spike_responses are in 1ms resolution)
            actual_response_binned = bin_spike_response(spike_responses[timing_idx], output_bin_size_ms, max_time_ms, remainder="append")
            
            # Average spikes across other trials
            other_trial_indices = [t for t in timing_list if t != timing_idx]
            avg_other_trials = compute_avg_spikes_across_trials(
                other_trial_indices, spike_responses, output_bin_size_ms, max_time_ms
            )
            
            # For RNN models with init_state, get actual spikes for the init_state period
            # This will be used to prepend to the actual and average plots
            actual_init_state_for_plot = None
            avg_init_state_for_plot = None
            
            # Build model predictions bin-by-bin
            # pred_array is at OUTPUT resolution (e.g., 60ms bins)
            pred_array = np.full((n_neurons, total_output_bins), 0, dtype=np.float32)
            
            model.eval()
            with torch.no_grad():
                # Handle init_state for RNN models with autoregressive extension
                if use_init_state and hasattr(model, 'forward') and 'initial_spikes' in model.forward.__code__.co_varnames:
                    # Get init_state from test_dataset if available
                    n_init_bins = getattr(test_dataset, 'n_initial_state_bins', 1)
                    
                    if hasattr(test_dataset, 'init_state') and test_dataset.init_state:
                        # Get the previous trial's last bins as init_state
                        prev_idx = timing_idx - 1
                        if prev_idx in test_dataset.spike_responses_binned:
                            init_state = test_dataset.spike_responses_binned[prev_idx][:, -n_init_bins:]
                        else:
                            init_state = np.zeros((n_neurons, n_init_bins), dtype=np.float32)
                    else:
                        init_state = np.zeros((n_neurons, n_init_bins), dtype=np.float32)
                    
                    # Store the initial init_state for plotting (before autoregressive updates)
                    init_state_for_plot = init_state.copy()
                    
                    # Get actual spike init_state for this trial (from previous trial's last bins)
                    actual_init_state_for_plot = init_state.copy()
                    
                    # Compute average init_state across other trials
                    avg_init_state = np.zeros((n_neurons, n_init_bins), dtype=np.float32)
                    valid_other_count = 0
                    for other_idx in other_trial_indices:
                        other_prev_idx = other_idx - 1
                        if other_prev_idx in test_dataset.spike_responses_binned:
                            avg_init_state += test_dataset.spike_responses_binned[other_prev_idx][:, -n_init_bins:]
                            valid_other_count += 1
                    if valid_other_count > 0:
                        avg_init_state /= valid_other_count
                    avg_init_state_for_plot = avg_init_state
                    
                    # Autoregressive prediction: extend timeseries using model's own output
                    # Calculate how many windows we need to cover max_time_ms
                    bins_per_output = n_input_bins // n_output_bins  # e.g., 60 / 10 = 6
                    current_init_state = init_state.copy()
                    
                    # Slide through the timeseries, making predictions and using output as next init_state
                    current_output_start = 0
                    window_idx = 0
                    
                    while current_output_start < total_output_bins:
                        # Calculate input window start in input resolution
                        input_start = window_idx * n_input_bins
                        input_end = input_start + n_input_bins
                        
                        # Check if we have enough input data (use actual stim shape, not total_input_bins)
                        actual_stim_bins = stim_binned.shape[1]
                        if input_start >= actual_stim_bins:
                            # No more stim data available, use zeros
                            x = np.zeros((stim_binned.shape[0], n_input_bins), dtype=np.float32)
                        elif input_end > actual_stim_bins:
                            # Partial stim data available, pad with zeros
                            x = np.zeros((stim_binned.shape[0], n_input_bins), dtype=np.float32)
                            available = actual_stim_bins - input_start
                            x[:, :available] = stim_binned[:, input_start:actual_stim_bins]
                        else:
                            x = stim_binned[:, input_start:input_end]
                        batch_x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)                        
                        batch_init = torch.tensor(current_init_state, dtype=torch.float32).unsqueeze(0).to(device)
                        
                        preds = model(batch_x, initial_spikes=batch_init).cpu().numpy()  # (1, n_neurons, n_output_bins)
                        preds_rates = np.exp(preds[0])  # Convert log to rates
                        
                        # Place predictions in the output array
                        for o in range(n_output_bins):
                            target_bin = current_output_start + o
                            if target_bin < total_output_bins:
                                pred_array[:, target_bin] = preds_rates[:, o]
                        
                        # Use the last n_init_bins of predicted spike rates as next init_state
                        if n_init_bins <= n_output_bins:
                            current_init_state = preds_rates[:, -n_init_bins:]
                        else:
                            # If n_init_bins > n_output_bins, we need to keep some history
                            new_init = np.zeros((n_neurons, n_init_bins), dtype=np.float32)
                            new_init[:, :-n_output_bins] = current_init_state[:, n_output_bins:]
                            new_init[:, -n_output_bins:] = preds_rates
                            current_init_state = new_init
                        
                        current_output_start += n_output_bins
                        window_idx += 1
                    
                    # pred_array already contains rates (not log rates)
                    pred_rates = pred_array
                else:
                    # Non-RNN models: no separate init_state tensor
                    init_state_for_plot = None
                    
                    if history > 0:
                        # Pillow/history model: slide through the trial autoregressively,
                        # concatenating lagged spike history to stim input at each step.
                        # Use actual spikes (teacher-forced) as the history source.
                        actual_spikes_binned = bin_spike_response(
                            spike_responses[timing_idx], output_bin_size_ms, max_time_ms, remainder="append"
                        )  # (n_neurons, total_output_bins)
                        
                        # Accumulate predictions and counts for averaging overlapping windows
                        pred_sum = np.zeros((n_neurons, total_output_bins), dtype=np.float32)
                        pred_counts = np.zeros(total_output_bins, dtype=np.float32)
                        
                        # Build init_state spike data once for this trial (if needed)
                        _init_spikes_for_viz = None
                        if init_state and n_initial_state_bins and n_initial_state_bins > 0:
                            prev_idx = timing_idx - 1
                            if prev_idx in test_dataset.spike_responses_binned:
                                _init_spikes_for_viz = test_dataset.spike_responses_binned[prev_idx][:, -n_initial_state_bins:]
                            else:
                                _init_spikes_for_viz = np.zeros((n_neurons, n_initial_state_bins), dtype=np.float32)
                        
                        for t in range(max_valid_start + 1):
                            # Stim input: (n_channels, n_input_bins)
                            stim_end = t + n_input_bins
                            actual_stim_bins = stim_binned.shape[1]
                            if stim_end <= actual_stim_bins:
                                x_stim = stim_binned[:, t:stim_end].copy()
                            else:
                                x_stim = np.zeros((stim_binned.shape[0], n_input_bins), dtype=np.float32)
                                available = max(0, actual_stim_bins - t)
                                if available > 0:
                                    x_stim[:, :available] = stim_binned[:, t:t + available]
                            
                            # Spike history: shift actual spikes by `history` bins
                            # When init_state is active, prepend previous trial spike data
                            # so the valid conv has enough context.
                            if init_state and _init_spikes_for_viz is not None:
                                x_stim = np.pad(x_stim, ((0, 0), (n_initial_state_bins, 0)),
                                                mode='constant', constant_values=0)
                                spikes_for_lag = np.concatenate([
                                    _init_spikes_for_viz,
                                    actual_spikes_binned[:, t:t + n_input_bins]
                                ], axis=1)
                            else:
                                spikes_for_lag = actual_spikes_binned[:, t:t + n_input_bins]
                            
                            total_time = x_stim.shape[1]
                            y_history = np.zeros((n_neurons, total_time), dtype=np.float32)
                            if total_time > history:
                                src = spikes_for_lag[:, :total_time - history]
                                y_history[:, history:history + src.shape[1]] = src
                            
                            # Concatenate stim + spike history along channel dim
                            x_combined = np.concatenate([x_stim.astype(np.float32), y_history], axis=0)
                            batch_x = torch.tensor(x_combined, dtype=torch.float32).unsqueeze(0).to(device)
                            preds = model(batch_x).cpu().numpy()
                            
                            for o in range(n_output_bins):
                                target_bin = t + output_offset + o
                                if target_bin < total_output_bins:
                                    pred_sum[:, target_bin] += preds[0, :, o]
                                    pred_counts[target_bin] += 1
                        
                        # Average predictions for bins covered by multiple windows
                        mask = pred_counts > 0
                        pred_array[:, mask] = pred_sum[:, mask] / pred_counts[mask]
                    else:
                        # Simple non-history model: single forward pass
                        x = stim_binned[:, :n_input_bins]  # (n_channels, n_input_bins)
                        if init_state and n_initial_state_bins and n_initial_state_bins > 0:
                            x = np.pad(x, ((0, 0), (n_initial_state_bins, 0)),
                                       mode='constant', constant_values=0)
                        batch_x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
                        preds = model(batch_x).cpu().numpy()  # (1, n_neurons, n_output_bins)
                        
                        # preds[0] has shape (n_neurons, n_output_bins)
                        # Place predictions at the appropriate output bins
                        for o in range(n_output_bins):
                            target_bin = output_offset + o
                            if target_bin < total_output_bins:
                                pred_array[:, target_bin] = preds[0, :, o]
                    
                    # Convert log predictions to rates
                    pred_rates = np.exp(pred_array)
            
            # Dynamic vmax calculation – based on actual spike data only,
            # so model predictions don't inflate the scale and wash out the data.
            # Predictions that exceed vmax will simply saturate to black.
            local_max = max(actual_response_binned.max(), avg_other_trials.max(), pred_rates.max())
            spike_vmax = max(1.0, local_max) + 0.2  # continuous 0 (white) → max+0.2 (black)
            spike_vmin = 0
            norm = colors.Normalize(vmin=0, vmax=spike_vmax)
            
            # Continuous Greys colormap from white to black
            spike_cmap = plt.cm.Greys
            
            # Brighter polarity colormap (Blue, White, Red)
            bright_polarity_cmap = ListedColormap(["#1E8FFF", '#FFFFFF', "#D02525"])
            
            # Calculate x-axis range: negative for init_state, positive for predictions
            x_min = -init_state_time_ms if use_init_state else 0
            x_max = max_time_ms
            
            # For RNN with init_state: prepend init_state to all arrays for visualization
            if use_init_state and init_state_for_plot is not None:
                # init_state_for_plot: (n_neurons, n_init_bins), pred_rates: (n_neurons, total_output_bins)
                pred_rates_extended = np.concatenate([init_state_for_plot, pred_rates], axis=1)
                # Extend actual spikes with init_state data
                actual_response_extended = np.concatenate([actual_init_state_for_plot, actual_response_binned], axis=1)
                # Extend average spikes with init_state data
                avg_other_trials_extended = np.concatenate([avg_init_state_for_plot, avg_other_trials], axis=1)
            else:
                pred_rates_extended = pred_rates
                actual_response_extended = actual_response_binned
                avg_other_trials_extended = avg_other_trials
            
            # Create figure with 4 panels
            fig, axes = plt.subplots(4, 1, figsize=(4, 7), constrained_layout=True)
            
            # Panel 0: Stimulation pattern colored by polarity
            im0 = axes[0].imshow(polarity_plot, aspect='auto', cmap=bright_polarity_cmap, vmin=-1, vmax=1, 
                                  interpolation='nearest', extent=[0, max_time_ms, polarity_plot.shape[0], 0])
            axes[0].set_title(f"Pattern {pattern_name} - trial {trial_idx} - Stimulation")
            axes[0].set_ylabel('Channel')
            axes[0].axvline(x=600, color='gray', linestyle='--', linewidth=1, alpha=0.9)
            axes[0].axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            axes[0].set_xlim(x_min, x_max)
            
            # Visibility tweaks for polarity
            cbar0 = fig.colorbar(im0, ax=axes[0], orientation='vertical', shrink=0.8, label='Polarity', ticks=[-1, 0, 1])
            cbar0.ax.set_yticklabels(['-1', '0', '+1'])
            
            # Panel 1: Actual spike counts (with init_state prepended for RNN)
            im1 = axes[1].imshow(actual_response_extended, aspect='auto', cmap=spike_cmap, interpolation='nearest', 
                                  vmin=spike_vmin, vmax=spike_vmax,
                                  extent=[x_min, max_time_ms, actual_response_extended.shape[0], 0])
            axes[1].axvline(x=600, color='red', linestyle='--', linewidth=1, alpha=0.7)
            axes[1].axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            axes[1].set_title(f'Actual spikes ({output_bin_size_ms}ms bins)')
            axes[1].set_ylabel('Neuron')
            axes[1].set_xlim(x_min, x_max)
            
            # Panel 2: Model prediction (with init_state prepended for RNN)
            im2 = axes[2].imshow(pred_rates_extended, aspect='auto', cmap=spike_cmap, interpolation='nearest', 
                                  vmin=spike_vmin, vmax=spike_vmax,
                                  extent=[x_min, max_time_ms, pred_rates_extended.shape[0], 0])
            axes[2].axvline(x=600, color='red', linestyle='--', linewidth=1, alpha=0.7)
            axes[2].axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            title_suffix = " (w/ init state)" if use_init_state and init_state_for_plot is not None else ""
            axes[2].set_title(f'Model prediction (rate, {output_bin_size_ms}ms bins){title_suffix}')
            axes[2].set_ylabel('Neuron')
            axes[2].set_xlim(x_min, x_max)
            
            # Panel 3: Average spikes across other oracle trials (with init_state prepended for RNN)
            im3 = axes[3].imshow(avg_other_trials_extended, aspect='auto', cmap=spike_cmap, interpolation='nearest', 
                                  vmin=spike_vmin, vmax=spike_vmax,
                                  extent=[x_min, max_time_ms, avg_other_trials_extended.shape[0], 0])
            axes[3].axvline(x=600, color='red', linestyle='--', linewidth=1, alpha=0.7)
            axes[3].axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            n_other = len(other_trial_indices)
            axes[3].set_title(f'Avg spikes over {n_other} other oracle trials ({output_bin_size_ms}ms bins)')
            axes[3].set_ylabel('Neuron')
            axes[3].set_xlabel('Time (ms)')
            axes[3].set_xlim(x_min, x_max)
            
            # Shared colorbar for Panels 1, 2, 3
            # Use im3 as mappable, assuming all share same vmin/vmax
            cbar_shared = fig.colorbar(im3, ax=axes[1:], orientation='vertical', shrink=0.8, label='Spike count', norm=norm)
            cbar_shared.set_ticks(np.arange(spike_vmin, spike_vmax + 0.2))
            
            # Save figure as compressed JPEG using PIL
            buf = BytesIO()
            plt.savefig(buf, dpi=100, format='png', bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            fig_path = os.path.join(pattern_dir, f"trial_{trial_idx:02d}_timing_{timing_idx}.jpg")
            img.save(fig_path, format='JPEG', quality=70, optimize=True)
            buf.close()
            plt.close(fig)
            
            if trial_idx == 0 and logger:
                logger.info(f"Saved pattern {pattern_name} trial {trial_idx} figure to {fig_path}")
        
        if logger:
            logger.info(f"Completed pattern {pattern_name} ({p_idx+1}/{len(oracle_patterns)})")
    
    if logger:
        logger.info(f"Saved all oracle pattern figures to {out_base}")


def plot_oracle_pattern_average_responses(unique_trials, spike_responses, spiking_neurons, savepath,plot_duration=60000, bin_size=1800):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Make a plot showing average for each oracle pattern across its trials
    # Optionally bin the time axis by bin_size_frames (e.g., 1800 frames = 60 ms bins at 30kHz)
    if bin_size < 1:
        raise ValueError("bin_size_frames must be >= 1")

    oracle_patterns = unique_trials[unique_trials['is_oracle']]['pattern_name'].unique()
    
    for pattern_name in oracle_patterns:  
        print ("\n===============================================\nOracle trial: ", pattern_name)
        trials = unique_trials[unique_trials['pattern_name'] == pattern_name]
        n_neurons = len(spiking_neurons)

        n_bins = (plot_duration + bin_size - 1) // bin_size
        print (f"Plotting average spike response for pattern {pattern_name} over {len(trials)} trials with {n_bins} bins of size {bin_size}  over {plot_duration} ")
        avg_spike_response = np.zeros((n_neurons, n_bins))

        for _, trial_info in trials.iterrows():
            timing_idx = trial_info['pattern_timing_index']
            resp = spike_responses[timing_idx]
            # Bin the spike response
            binned = bin_spike_response(resp, bin_size, max_time=plot_duration, remainder="append")
            avg_spike_response += binned
        avg_spike_response /= len(trials)
        
        # Plot the average spike response as a heatmap
        plt.figure(figsize=(10, 6)) 

        # Extent spans the full plot_duration_frames, even when binned
        plt.imshow(
            avg_spike_response,
            aspect='auto',
            cmap='Greys',
            interpolation='nearest',
            extent=[0, plot_duration, avg_spike_response.shape[0], 0],
        )
        plt.colorbar(label='Average Spike Count per Bin')
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index')
        plt.title(f'Average Spike Response for Oracle Pattern: {pattern_name}')
        plt.tight_layout()
        plt.savefig(f'{savepath}/average_spike_response_oracle_pattern_{pattern_name}.png')
        plt.close()


def plot_stimulation_polarity_timeseries(pattern_stims, pattern_polarities, pattern_name, savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    stim = pattern_stims[pattern_name]
    polarity = pattern_polarities[pattern_name]
    print(f"Pattern {pattern_name}: stim shape {stim.shape}, polarity shape {polarity.shape}")
    
    # Plot the stimulation pattern colored by p olarity
    plt.figure(figsize=(10, 6)) 
    im = plt.imshow(polarity, aspect='auto', cmap=polarity_cmap, vmin=-1, vmax=1, interpolation='nearest')
    cbar = plt.colorbar(im, label='Pulse Polarity', ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['Negative (-1)', 'No Stim', 'Positive (+1)'])
    plt.xlabel('Time (ms)')
    plt.ylabel('Channel Index')
    plt.title(f'Stimulation Pattern: {pattern_name} (Colored by Polarity)')
    plt.tight_layout()
    plt.savefig(f'{savepath}/stim_pattern_polarity_{pattern_name}.png')
    plt.close()



def plot_stimulation_polarity_frame_res(pattern_name, trial, pattern_df, savepath):
    '''
    Plot the stimulation pattern at frame resolution (n_electrodes x 61000).
    Each step is colored by delay_mode and spans from step_start_timestamp to
    the next step's step_start_timestamp (or +1800 frames for the last step).
    The time origin is min(pattern_flag_start_timestamp, earliest step_start_timestamp).
    
    :param pattern_name: integer pattern name (e.g. 4033)
    :param trial: trial number (uses 'trial' column if present, else 'pattern_timing_index')
    :param pattern_df: full pattern DataFrame (all patterns) — used to build the
                       global electrode list so the y-axis is consistent across plots
    :param savepath: base directory; a subdirectory per pattern_name is created
    '''
    import matplotlib.patches as mpatches

    # ------------------------------------------------------------------
    # 1. Build a GLOBAL electrode list from the full DataFrame (before
    #    subsetting by pattern/trial) so the y-axis is identical for all
    #    patterns and trials.
    # ------------------------------------------------------------------
    all_channels = sorted(pattern_df["channel"].dropna().unique())
    ch_enc = {ch: idx for idx, ch in enumerate(all_channels)}
    n_electrodes = len(all_channels)
    # Labels for the y-axis (actual electrode names)
    channel_labels = [str(int(ch)) for ch in all_channels]

    # ------------------------------------------------------------------
    # 2. Subset to the requested pattern_name + trial
    # ------------------------------------------------------------------
    trial_col = 'trial' if 'trial' in pattern_df.columns else 'pattern_timing_index'
    trial_df = pattern_df[(pattern_df['pattern_name'] == pattern_name) &
                          (pattern_df[trial_col] == trial)]

    if trial_df.empty:
        print(f"No data found for pattern {pattern_name}, trial {trial}")
        return

    # ------------------------------------------------------------------
    # 3. Determine time origin
    # ------------------------------------------------------------------
    pat_start = (trial_df['pattern_flag_start_timestamp'].iloc[0]
                 if 'pattern_flag_start_timestamp' in trial_df.columns
                 else float('inf'))
    min_step_start = trial_df['step_start_timestamp'].min()
    start_time = int(min(pat_start, min_step_start))

    duration_frames = 61000

    # ------------------------------------------------------------------
    # 4. Build a lookup: step_index -> (step_start, step_end) in absolute
    #    frames.  step_end is the next step's start; for the last step we
    #    add 1800 frames (~60 ms @ 30 kHz).
    # ------------------------------------------------------------------
    step_starts = (trial_df[['step_index', 'step_start_timestamp']]
                   .drop_duplicates()
                   .sort_values('step_index'))
    step_start_list = step_starts['step_start_timestamp'].astype(int).tolist()
    step_index_list = step_starts['step_index'].astype(int).tolist()

    step_end_map = {}   # step_index -> absolute end frame
    for i, si in enumerate(step_index_list):
        if i + 1 < len(step_index_list):
            step_end_map[si] = step_start_list[i + 1]
        else:
            step_end_map[si] = step_start_list[i] + 1800  # ~60 ms

    # ------------------------------------------------------------------
    # 5. Fill the visualisation matrix
    # ------------------------------------------------------------------
    vis_data = np.zeros((n_electrodes, duration_frames), dtype=int)

    mode_color_map = {
        0:  1,   # Gray
        1:  2,   # Red
        -1: 3,   # Blue
        2:  4,   # Green
    }

    for _, row in trial_df.iterrows():
        if pd.isna(row['channel']) or pd.isna(row.get('delay_mode', np.nan)):
            continue
        try:
            si = int(row['step_index'])
            abs_start = int(row['step_start_timestamp'])
            abs_end = step_end_map.get(si, abs_start + 1800)

            rel_start = abs_start - start_time
            rel_end   = abs_end   - start_time

            # Clip to [0, duration_frames)
            r_s = max(0, rel_start)
            r_e = min(duration_frames, rel_end)
            if r_e <= r_s:
                continue

            ch = int(row['channel'])
            if ch not in ch_enc:
                continue
            ch_idx = ch_enc[ch]

            mode = int(row['delay_mode'])
            val = mode_color_map.get(mode, 1)
            vis_data[ch_idx, r_s:r_e] = val
        except (ValueError, KeyError):
            continue

    # ------------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(20, 8))

    colors = ['white', 'gray', 'red', 'blue', 'green']
    cmap = ListedColormap(colors)

    ax.imshow(vis_data, aspect='auto', cmap=cmap, interpolation='nearest',
              vmin=0, vmax=4, extent=[0, duration_frames, n_electrodes, 0])

    ax.set_title(f"Stimulation Pattern: {pattern_name} | Trial: {trial}\nResolution: 1 frame (@30kHz)")
    ax.set_xlabel("Time (frames from pattern start)")
    ax.set_ylabel("Electrode")

    # Y-axis: show actual electrode names
    ax.set_yticks(np.arange(n_electrodes) + 0.5)
    ax.set_yticklabels(channel_labels, fontsize=6)

    # Legend
    legend_patches = [
        mpatches.Patch(color='gray',  label='delay_mode 0'),
        mpatches.Patch(color='red',   label='delay_mode 1'),
        mpatches.Patch(color='blue',  label='delay_mode -1'),
        mpatches.Patch(color='green', label='delay_mode 2'),
    ]
    ax.legend(handles=legend_patches, loc='upper right')

    # ------------------------------------------------------------------
    # 7. Save  (savepath / pattern_name / trial.png)
    # ------------------------------------------------------------------
    pattern_dir = os.path.join(savepath, str(pattern_name))
    os.makedirs(pattern_dir, exist_ok=True)

    save_file = os.path.join(pattern_dir, f"{trial}.png")
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_stimulation_polarity_frame_res_alpha(pattern_name, pattern_df, savepath, alpha=0.2):
    '''
    Overlay all oracle trials of a given pattern on one figure with transparent
    patches.  Where stimulation aligns across trials the colour saturates;
    misalignment is visible as fringes.

    :param pattern_name: integer pattern name (e.g. 4033)
    :param pattern_df: full pattern DataFrame (all patterns)
    :param savepath: base directory; a subdirectory per pattern_name is created
    :param alpha: transparency per trial per patch (default 0.2)
    '''
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection

    # ------------------------------------------------------------------
    # 1. Global electrode list (consistent y-axis across all calls)
    # ------------------------------------------------------------------
    all_channels = sorted(pattern_df["channel"].dropna().unique())
    ch_enc = {ch: idx for idx, ch in enumerate(all_channels)}
    n_electrodes = len(all_channels)
    channel_labels = [str(int(ch)) for ch in all_channels]

    # ------------------------------------------------------------------
    # 2. Get all trials for this pattern
    # ------------------------------------------------------------------
    trial_col = 'trial' if 'trial' in pattern_df.columns else 'pattern_timing_index'
    pattern_subset = pattern_df[pattern_df['pattern_name'] == pattern_name]

    if pattern_subset.empty:
        print(f"No data found for pattern {pattern_name}")
        return

    all_trials = sorted(pattern_subset[trial_col].unique())
    n_trials = len(all_trials)

    duration_frames = 20000  # ~>600 ms at 30 kHz, chosen to capture the main stimulation period across all trials

    mode_color = {
        0:  'gray',
        1:  'red',
        -1: 'blue',
        2:  'green',
    }

    # ------------------------------------------------------------------
    # 3. Collect rectangles across ALL trials
    # ------------------------------------------------------------------
    rects_by_color = {c: [] for c in mode_color.values()}

    for trial in all_trials:
        trial_df = pattern_subset[pattern_subset[trial_col] == trial]

        # Time origin for this trial
        pat_start = (trial_df['pattern_flag_start_timestamp'].iloc[0]
                     if 'pattern_flag_start_timestamp' in trial_df.columns
                     else float('inf'))
        min_step_start = trial_df['step_start_timestamp'].min()
        start_time = int(min(pat_start, min_step_start))

        # step_index -> end frame lookup for this trial
        step_starts = (trial_df[['step_index', 'step_start_timestamp']]
                       .drop_duplicates()
                       .sort_values('step_index'))
        step_start_list = step_starts['step_start_timestamp'].astype(int).tolist()
        step_index_list = step_starts['step_index'].astype(int).tolist()

        step_end_map = {}
        for i, si in enumerate(step_index_list):
            if i + 1 < len(step_index_list):
                step_end_map[si] = step_start_list[i + 1]
            else:
                step_end_map[si] = step_start_list[i] + 1800

        for _, row in trial_df.iterrows():
            if pd.isna(row['channel']) or pd.isna(row.get('delay_mode', np.nan)):
                continue
            try:
                si = int(row['step_index'])
                abs_start = int(row['step_start_timestamp'])
                abs_end = step_end_map.get(si, abs_start + 1800)

                rel_start = abs_start - start_time
                rel_end   = abs_end   - start_time

                r_s = max(0, rel_start)
                r_e = min(duration_frames, rel_end)
                if r_e <= r_s:
                    continue

                ch = int(row['channel'])
                if ch not in ch_enc:
                    continue
                ch_idx = ch_enc[ch]

                mode = int(row['delay_mode'])
                color = mode_color.get(mode, 'gray')
                rects_by_color[color].append(
                    Rectangle((r_s, ch_idx), r_e - r_s, 1)
                )
            except (ValueError, KeyError):
                continue

    # ------------------------------------------------------------------
    # 4. Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_facecolor('white')

    for color, rects in rects_by_color.items():
        if not rects:
            continue
        pc = PatchCollection(rects, facecolor=color, edgecolor='none', alpha=alpha)
        ax.add_collection(pc)

    ax.set_xlim(0, duration_frames)
    ax.set_ylim(n_electrodes, 0)
    ax.set_aspect('auto')

    ax.set_title(f"Stimulation Pattern: {pattern_name} | All {n_trials} trials overlaid\n"
                 f"Resolution: 1 frame (@30kHz) | alpha={alpha}")
    ax.set_xlabel("Time (frames from pattern start)")
    ax.set_ylabel("Electrode")

    ax.set_yticks(np.arange(n_electrodes) + 0.5)
    ax.set_yticklabels(channel_labels, fontsize=6)

    legend_patches = [
        mpatches.Patch(color='gray',  alpha=alpha, label='delay_mode 0'),
        mpatches.Patch(color='red',   alpha=alpha, label='delay_mode 1'),
        mpatches.Patch(color='blue',  alpha=alpha, label='delay_mode -1'),
        mpatches.Patch(color='green', alpha=alpha, label='delay_mode 2'),
    ]
    ax.legend(handles=legend_patches, loc='upper right')

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------

    save_file = os.path.join(savepath,f"{pattern_name}_all_trials_alpha.png")
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── PSTH: model vs ground-truth per neuron ──────────────────────────

def plot_psth_per_neuron(
    model,
    test_dataset,
    device,
    out_dir,
    output_bin_size_ms=10,
    use_init_state=False,
    logger=None,
):
    """Generate per-neuron PSTH figures comparing model predictions to ground truth.

    For each oracle pattern the ground-truth firing rate is the trial-averaged
    spike count per output bin (i.e. the mean across the ~20 oracle repeats).
    Model predictions are also averaged across the same set of samples.

    One PNG is saved per neuron inside *out_dir*, with one subplot per oracle
    pattern.

    Parameters
    ----------
    model : nn.Module
    test_dataset : BinnedStimSpikeDataset
    device : torch.device
    out_dir : str
        Directory to write per-neuron PNGs into.
    output_bin_size_ms : int
        Bin width in ms (used for x-axis labels).
    use_init_state : bool
    logger : logging.Logger, optional
    """
    os.makedirs(out_dir, exist_ok=True)
    _log = logger.info if logger else (lambda m: None)

    n_neurons = test_dataset.n_neurons
    n_output_bins = test_dataset.n_output_bins

    # ── 1. Collect model predictions (batched) ──
    model.eval()
    loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # pred_all shape: (n_samples, n_neurons, n_output_bins)
    pred_chunks = []
    with torch.no_grad():
        for batch in loader:
            if use_init_state and len(batch) == 3:
                bx, _, bi = batch
                bx = bx.to(device)
                pred = torch.exp(model(bx)).cpu().numpy()
            else:
                bx = batch[0].to(device)
                pred = torch.exp(model(bx)).cpu().numpy()
            pred_chunks.append(pred)
    pred_all = np.concatenate(pred_chunks, axis=0)  # (N_samples, neurons, bins)

    # ── 2. Group samples by pattern name ──
    #   For each pattern collect indices into pred_all / test_dataset.samples
    from collections import defaultdict
    pattern_to_sample_idxs = defaultdict(list)
    for i, (timing_idx, t) in enumerate(test_dataset.samples):
        pname = test_dataset.timing_to_pattern[timing_idx]
        pattern_to_sample_idxs[pname].append(i)

    pattern_names = sorted(pattern_to_sample_idxs.keys())
    n_patterns = len(pattern_names)
    _log(f"PSTH: {n_patterns} patterns, {n_neurons} neurons, {n_output_bins} bins")

    # ── 3. Compute per-pattern averages ──
    # true_avg[p]  -> (n_neurons, n_output_bins)
    # pred_avg[p]  -> (n_neurons, n_output_bins)
    # true_sem[p]  -> (n_neurons, n_output_bins)
    true_avg = {}
    pred_avg = {}
    true_sem = {}
    for pname in pattern_names:
        idxs = pattern_to_sample_idxs[pname]
        # ground-truth spikes from the dataset
        true_stack = np.stack(
            [
                test_dataset.spike_responses_binned[test_dataset.samples[i][0]][
                    :,
                    test_dataset.samples[i][1]
                    + test_dataset.output_offset : test_dataset.samples[i][1]
                    + test_dataset.output_offset
                    + n_output_bins,
                ]
                for i in idxs
            ],
            axis=0,
        )  # (n_trials, n_neurons, n_output_bins)
        pred_stack = pred_all[idxs]  # (n_trials, n_neurons, n_output_bins)

        true_avg[pname] = true_stack.mean(axis=0)
        true_sem[pname] = true_stack.std(axis=0) / np.sqrt(true_stack.shape[0])
        pred_avg[pname] = pred_stack.mean(axis=0)

    # ── 4. Plot one figure per neuron ──
    time_axis = np.arange(n_output_bins) * output_bin_size_ms  # ms

    ncols = min(10, n_patterns)
    nrows = (n_patterns + ncols - 1) // ncols

    for nidx in range(n_neurons):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(2.4 * ncols, 2.0 * nrows),
            sharex=True, sharey=True,
        )
        axes_flat = np.atleast_1d(axes).flatten()

        for pidx, pname in enumerate(pattern_names):
            ax = axes_flat[pidx]
            true_mean_n = true_avg[pname][nidx]
            true_sem_n = true_sem[pname][nidx]
            pred_mean_n = pred_avg[pname][nidx]

            ax.fill_between(
                time_axis,
                true_mean_n - true_sem_n,
                true_mean_n + true_sem_n,
                color="steelblue", alpha=0.25,
            )
            ax.plot(time_axis, true_mean_n, color="steelblue", linewidth=1, label="True")
            ax.plot(time_axis, pred_mean_n, color="coral", linewidth=1, label="Model")
            ax.set_title(pname, fontsize=6, pad=2)
            ax.tick_params(labelsize=5)

        # Hide unused axes
        for j in range(n_patterns, len(axes_flat)):
            axes_flat[j].set_visible(False)

        # Shared legend & labels
        axes_flat[0].legend(fontsize=5, loc="upper right")
        fig.supxlabel("Time (ms)", fontsize=8)
        fig.supylabel("Firing rate (spikes / bin)", fontsize=8)
        fig.suptitle(f"PSTH — Neuron {nidx}", fontsize=10)
        fig.tight_layout(rect=[0.02, 0.02, 1, 0.96])
        fig.savefig(os.path.join(out_dir, f"neuron_{nidx:03d}.png"), dpi=120)
        plt.close(fig)

    _log(f"PSTH figures saved to {out_dir} ({n_neurons} neurons)")



def plot_neuron_pattern_traces(neuron_idx, pattern_name, model_tuple, cfg, raw_data, test_loader,
                                coarse_factor=1, coarse_method='mean', save_dir=None):
    """
    Two-panel figure for one neuron × one oracle pattern.

    Top: mean ± SEM spike response across trials + mean model prediction.
    Bottom: all individual trial traces (vertically offset) + model prediction
            traces overlaid.

    Args:
        neuron_idx: int, neuron index (0-based)
        pattern_name: pattern identifier
        model_tuple: (model, cfg_unused, device)
        cfg: model config dict
        raw_data: output of load_raw_data
        test_loader: DataLoader (only .dataset used)
        coarse_factor: temporal coarsening factor (1 = no coarsening)
        coarse_method: 'mean' or 'sum'
        save_dir: if not None, save figure there and close it

    Returns:
        fig (or None if saved and closed)
    """
    model, _, device = model_tuple
    test_dataset = test_loader.dataset
    pattern_df = raw_data["pattern_df"]
    spike_responses = raw_data["spike_responses"]

    output_bin_size_ms = cfg['output_bin_size_ms']
    max_time_ms = cfg['max_time_ms']
    n_input_bins = cfg['n_input_bins']
    n_output_bins = cfg['n_output_bins']
    output_offset = cfg.get('output_offset', 0)
    history = cfg.get('history', 0)
    init_state_flag = cfg.get('init_state', False)
    n_neurons_total = test_dataset.n_neurons
    n_initial_state_bins = getattr(test_dataset, 'n_initial_state_bins', 0)

    total_output_bins = max_time_ms // output_bin_size_ms

    def _bin_1d(arr, factor, method):
        if factor <= 1:
            return arr
        n = len(arr)
        coarse = n // factor
        reshaped = arr[:coarse * factor].reshape(coarse, factor)
        return reshaped.sum(axis=1) if method == 'sum' else reshaped.mean(axis=1)

    def _sliding_window_predict_single(mdl, dev, stim_src, spikes_src, spike_binned_dict,
                                        timing):
        """Sliding-window teacher-forced prediction → (n_neurons, total_output_bins)."""
        tot_out = max_time_ms // output_bin_size_ms
        tot_in = max_time_ms // cfg['input_bin_size_ms']
        max_s = min(tot_out - output_offset - n_output_bins, tot_in - n_input_bins)

        p_sum = np.zeros((n_neurons_total, tot_out), dtype=np.float32)
        p_cnt = np.zeros(tot_out, dtype=np.float32)

        _init = None
        if init_state_flag and n_initial_state_bins > 0:
            prev = timing - 1
            if prev in spike_binned_dict:
                _init = spike_binned_dict[prev][:, -n_initial_state_bins:]
            else:
                _init = np.zeros((n_neurons_total, n_initial_state_bins), dtype=np.float32)

        mdl.eval()
        with torch.no_grad():
            for t in range(max_s + 1):
                se = t + n_input_bins
                actual_bins = stim_src.shape[1]
                if se <= actual_bins:
                    xs = stim_src[:, t:se].copy()
                else:
                    xs = np.zeros((stim_src.shape[0], n_input_bins), dtype=np.float32)
                    av = max(0, actual_bins - t)
                    if av > 0:
                        xs[:, :av] = stim_src[:, t:t+av]

                if init_state_flag and _init is not None:
                    xs = np.pad(xs, ((0, 0), (n_initial_state_bins, 0)), mode='constant')

                if history > 0:
                    if init_state_flag and _init is not None:
                        sfl = np.concatenate([_init, spikes_src[:, t:t + n_input_bins]], axis=1)
                    else:
                        sfl = spikes_src[:, t:t + n_input_bins]
                    tt = xs.shape[1]
                    yh = np.zeros((n_neurons_total, tt), dtype=np.float32)
                    if tt > history:
                        src = sfl[:, :tt - history]
                        yh[:, history:history + src.shape[1]] = src
                    xs = np.concatenate([xs.astype(np.float32), yh], axis=0)

                bx = torch.tensor(xs, dtype=torch.float32).unsqueeze(0).to(dev)
                pr = mdl(bx).cpu().numpy()
                for o in range(n_output_bins):
                    tb = t + output_offset + o
                    if tb < tot_out:
                        p_sum[:, tb] += pr[0, :, o]
                        p_cnt[tb] += 1

        msk = p_cnt > 0
        pa = np.zeros_like(p_sum)
        pa[:, msk] = p_sum[:, msk] / p_cnt[msk]
        return np.exp(pa)

    # --- Gather all trials for this pattern ---
    unique_trials = pattern_df[['pattern_timing_index', 'pattern_name', 'is_oracle']].drop_duplicates()
    timing_list = sorted(
        unique_trials[unique_trials['pattern_name'] == pattern_name]['pattern_timing_index'].tolist()
    )
    n_trials = len(timing_list)

    stim_binned = test_dataset.pattern_stims[pattern_name]

    all_actual = []   # list of 1-d arrays (coarse_bins,)
    all_pred   = []

    for timing_idx in timing_list:
        # Actual binned spikes for this neuron
        actual_full = bin_spike_response(spike_responses[timing_idx],
                                         output_bin_size_ms, max_time_ms, remainder="append")
        actual_neuron = actual_full[neuron_idx]  # (total_output_bins,)

        # Model prediction for this trial
        pred_full = _sliding_window_predict_single(
            model, device, stim_binned, actual_full,
            test_dataset.spike_responses_binned, timing_idx)
        pred_neuron = pred_full[neuron_idx]

        all_actual.append(_bin_1d(actual_neuron, coarse_factor, coarse_method))
        all_pred.append(_bin_1d(pred_neuron, coarse_factor, coarse_method))

    all_actual = np.array(all_actual)  # (n_trials, coarse_bins)
    all_pred   = np.array(all_pred)

    n_bins = all_actual.shape[1]
    coarse_bin_ms = output_bin_size_ms * coarse_factor
    time_ms = np.arange(n_bins) * coarse_bin_ms + coarse_bin_ms / 2  # bin centres

    actual_mean = all_actual.mean(axis=0)
    actual_sem  = all_actual.std(axis=0) / np.sqrt(n_trials)
    pred_mean   = all_pred.mean(axis=0)
    pred_sem    = all_pred.std(axis=0) / np.sqrt(n_trials)

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

    # Top: Mean ± SEM
    axes[0].plot(time_ms, actual_mean, color='black', lw=1.5, label='Actual (mean)')
    axes[0].fill_between(time_ms, actual_mean - actual_sem, actual_mean + actual_sem,
                         color='black', alpha=0.15)
    axes[0].plot(time_ms, pred_mean, color='tab:red', lw=1.5, label='Model pred (mean)')
    axes[0].fill_between(time_ms, pred_mean - pred_sem, pred_mean + pred_sem,
                         color='tab:red', alpha=0.15)
    axes[0].set_ylabel('Spike count / rate')
    axes[0].set_title(f'Neuron {neuron_idx} — Pattern {pattern_name}  '
                      f'({n_trials} trials, {coarse_bin_ms}ms bins)')
    axes[0].legend(fontsize=8)
    axes[0].axvline(x=600, color='gray', ls='--', lw=0.8)
    axes[0].set_xlim(time_ms[0] - coarse_bin_ms / 2, time_ms[-1] + coarse_bin_ms / 2)

    # Bottom: individual traces offset
    for k in range(n_trials):
        axes[1].plot(time_ms, all_actual[k], color='black', lw=0.6, alpha=0.7)
        axes[1].plot(time_ms, all_pred[k], color='tab:red', lw=0.6, alpha=0.7)

    axes[1].set_ylabel('Firing rate')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_title('Individual trial traces (black=actual, red=model)')
    axes[1].axvline(x=600, color='gray', ls='--', lw=0.8)
    axes[1].set_xlim(time_ms[0] - coarse_bin_ms / 2, time_ms[-1] + coarse_bin_ms / 2)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f'neuron{neuron_idx:03d}_pattern{pattern_name}.png')
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    plt.show()
    return fig


def generate_all_neuron_pattern_plots(model_tuple, cfg, raw_data, test_loader,
                                       coarse_factor=1, coarse_method='mean',
                                       save_dir='results/neuron_pattern_traces',
                                       neuron_indices=None):
    """
    Generate per-neuron per-pattern trace figures for all oracle patterns.

    Args:
        neuron_indices: list of neuron indices, or None → all neurons.
        Other args: same as plot_neuron_pattern_traces.
    """
    pattern_df = raw_data["pattern_df"]
    unique_trials = pattern_df[['pattern_name', 'is_oracle']].drop_duplicates()
    oracle_patterns = sorted(unique_trials[unique_trials['is_oracle']]['pattern_name'].unique())

    n_neurons_total = test_loader.dataset.n_neurons
    if neuron_indices is None:
        neuron_indices = list(range(n_neurons_total))

    total = len(neuron_indices) * len(oracle_patterns)
    print(f"Generating {total} figures ({len(neuron_indices)} neurons × {len(oracle_patterns)} patterns) → {save_dir}/")

    for count, neuron_idx in enumerate(neuron_indices):
        for pat in oracle_patterns:
            plot_neuron_pattern_traces(
                neuron_idx, pat, model_tuple, cfg, raw_data, test_loader,
                coarse_factor=coarse_factor, coarse_method=coarse_method,
                save_dir=save_dir)
        done = (count + 1) * len(oracle_patterns)
        print(f"  Neuron {neuron_idx:>3d} done  ({done}/{total})")

    print("All figures saved.")


def plot_single_oracle_trial(pattern_name, trial_idx, model_tuple, cfg, raw_data, test_loader,
                              coarse_factor=1, coarse_method='mean',
                              nonhistory_model_tuple=None, nonhistory_cfg=None,
                              nonhistory_test_loader=None,
                              save_path=None):
    """Plot a single oracle trial comparing model predictions.

    Panels (top → bottom):
      1. Stimulation polarity
      2. Actual spikes
      3. History model, teacher-forced (ground-truth history)
      4. History model, autoregressive  *(only if history > 0)*
      5. No-history model prediction    *(only if provided)*
      6. LOO trial average

    Parameters
    ----------
    pattern_name : int/str
    trial_idx : int          – index within this pattern's oracle trials
    model_tuple : tuple      – ``(model, cfg_or_crit, device)``
    cfg : dict               – experiment config
    raw_data : dict          – from ``load_raw_data``
    test_loader : DataLoader
    coarse_factor, coarse_method : int, str – temporal coarsening
    nonhistory_model_tuple, nonhistory_cfg, nonhistory_test_loader : optional
        Second (no-history) model to compare.
    save_path : str or None  – save figure here; if None the path is derived.

    Returns
    -------
    fig
    """
    from matplotlib.colors import PowerNorm
    from models import sliding_window_predict_trial, sliding_window_predict_trial_ar
    from utils import coarsen_2d

    model, _, device = model_tuple
    test_dataset = test_loader.dataset
    pattern_df   = raw_data["pattern_df"]
    spike_responses    = raw_data["spike_responses"]
    pattern_polarities = raw_data["pattern_polarities"]

    output_bin_size_ms = cfg['output_bin_size_ms']
    max_time_ms        = cfg['max_time_ms']
    history            = cfg.get('history', 0)
    n_neurons          = test_dataset.n_neurons

    # --- timing index ---
    unique_trials = pattern_df[['pattern_timing_index', 'pattern_name', 'is_oracle']].drop_duplicates()
    timing_list = sorted(
        unique_trials[unique_trials['pattern_name'] == pattern_name]['pattern_timing_index'].tolist()
    )
    if trial_idx >= len(timing_list):
        raise ValueError(f"Trial {trial_idx} out of range (pattern {pattern_name} has {len(timing_list)} trials)")
    timing_idx = timing_list[trial_idx]

    # --- actual response ---
    actual_response = bin_spike_response(spike_responses[timing_idx],
                                         output_bin_size_ms, max_time_ms, remainder="append")

    # --- LOO average ---
    other_timings = [t for t in timing_list if t != timing_idx]
    loo_avg = compute_avg_spikes_across_trials(other_timings, spike_responses,
                                                output_bin_size_ms, max_time_ms)

    # --- teacher-forced prediction ---
    pred_rates = sliding_window_predict_trial(model_tuple, cfg, test_loader, timing_idx)

    # --- autoregressive prediction (if history > 0) ---
    ar_pred_rates = None
    if history and history > 0:
        ar_pred_rates = sliding_window_predict_trial_ar(model_tuple, cfg, test_loader, timing_idx)

    # --- non-history model prediction ---
    nohist_pred_rates = None
    if nonhistory_model_tuple is not None and nonhistory_cfg is not None:
        nohist_pred_rates = sliding_window_predict_trial(
            nonhistory_model_tuple, nonhistory_cfg,
            nonhistory_test_loader or test_loader, timing_idx)

    # --- coarsen ---
    actual_response = coarsen_2d(actual_response, coarse_factor, coarse_method)
    loo_avg         = coarsen_2d(loo_avg, coarse_factor, coarse_method)
    pred_rates      = coarsen_2d(pred_rates, coarse_factor, coarse_method)
    if ar_pred_rates is not None:
        ar_pred_rates = coarsen_2d(ar_pred_rates, coarse_factor, coarse_method)
    if nohist_pred_rates is not None:
        nohist_pred_rates = coarsen_2d(nohist_pred_rates, coarse_factor, coarse_method)

    coarse_bin_ms     = output_bin_size_ms * coarse_factor
    coarse_total_bins = actual_response.shape[1]
    display_max_ms    = coarse_total_bins * coarse_bin_ms

    # --- polarity heatmap ---
    polarity_600 = pattern_polarities[pattern_name]
    polarity_base = np.zeros((polarity_600.shape[0], max_time_ms))
    polarity_base[:, :min(600, max_time_ms)] = polarity_600[:, :min(600, max_time_ms)]
    polarity_plot = np.zeros_like(polarity_base)
    for shift in range(10):
        polarity_plot[:, shift:] += polarity_base[:, :max_time_ms - shift] if shift > 0 else polarity_base

    # --- figure ---
    spike_vmax = 3.4
    pnorm = PowerNorm(gamma=0.5, vmin=0, vmax=spike_vmax)
    spike_cmap = plt.cm.Greys
    bright_polarity_cmap = ListedColormap(["#1E8FFF", '#FFFFFF', "#D02525"])

    agg_label = coarse_method if coarse_factor > 1 else ""
    bin_label = f"{coarse_bin_ms}ms bins" + (f", {agg_label}" if agg_label else "")

    n_panels = 3  # stim + actual + teacher-forced
    if ar_pred_rates is not None:
        n_panels += 1
    if nohist_pred_rates is not None:
        n_panels += 1
    n_panels += 1  # LOO

    fig, axes = plt.subplots(n_panels, 1, figsize=(6, 2 * n_panels), constrained_layout=True)

    # Panel 0: stimulation
    axes[0].imshow(polarity_plot, aspect='auto', cmap=bright_polarity_cmap, vmin=-1, vmax=1,
                   interpolation='nearest', extent=[0, max_time_ms, polarity_plot.shape[0], 0])
    axes[0].set_title(f"Pattern {pattern_name} – Trial {trial_idx} – Stimulation")
    axes[0].set_ylabel('Channel')
    axes[0].axvline(x=600, color='gray', linestyle='--', linewidth=1)

    # Panel 1: actual spikes
    axes[1].imshow(actual_response, aspect='auto', cmap=spike_cmap, norm=pnorm,
                   interpolation='nearest', extent=[0, display_max_ms, n_neurons, 0])
    axes[1].set_title(f'Actual spikes ({bin_label})')
    axes[1].set_ylabel('Neuron')
    axes[1].axvline(x=600, color='red', linestyle='--', linewidth=1, alpha=0.7)

    # Panel 2: teacher-forced
    hist_label = f" (history={history})" if history and history > 0 else ""
    im2 = axes[2].imshow(pred_rates, aspect='auto', cmap=spike_cmap, norm=pnorm,
                   interpolation='nearest', extent=[0, display_max_ms, n_neurons, 0])
    axes[2].set_title(f'Ground Truth Observed History prediction{hist_label} ({bin_label})')
    axes[2].set_ylabel('Neuron')
    axes[2].axvline(x=600, color='red', linestyle='--', linewidth=1, alpha=0.7)

    panel_idx = 3

    # Panel: AR
    if ar_pred_rates is not None:
        axes[panel_idx].imshow(ar_pred_rates, aspect='auto', cmap=spike_cmap, norm=pnorm,
                       interpolation='nearest', extent=[0, display_max_ms, n_neurons, 0])
        axes[panel_idx].set_title(f'Self Generated (AR) History Prediction (history={history}) ({bin_label})')
        axes[panel_idx].set_ylabel('Neuron')
        axes[panel_idx].axvline(x=600, color='red', linestyle='--', linewidth=1, alpha=0.7)
        panel_idx += 1

    # Panel: non-history model
    if nohist_pred_rates is not None:
        axes[panel_idx].imshow(nohist_pred_rates, aspect='auto', cmap=spike_cmap, norm=pnorm,
                       interpolation='nearest', extent=[0, display_max_ms, n_neurons, 0])
        axes[panel_idx].set_title(f'No History (Stim only) Model Prediction ({bin_label})')
        axes[panel_idx].set_ylabel('Neuron')
        axes[panel_idx].axvline(x=600, color='red', linestyle='--', linewidth=1, alpha=0.7)
        panel_idx += 1

    # Panel: LOO
    axes[panel_idx].imshow(loo_avg, aspect='auto', cmap=spike_cmap, norm=pnorm,
                   interpolation='nearest', extent=[0, display_max_ms, n_neurons, 0])
    axes[panel_idx].set_title(f'LOO avg over {len(other_timings)} other trials ({bin_label})')
    axes[panel_idx].set_ylabel('Neuron')
    axes[panel_idx].set_xlabel('Time (ms)')
    axes[panel_idx].axvline(x=600, color='red', linestyle='--', linewidth=1, alpha=0.7)

    fig.colorbar(im2, ax=axes[1:].tolist(), orientation='vertical', shrink=0.8, label='Spike count')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()
    return fig


def fa_input_output(model_tuple, cfg, raw_data, model_label='',
                    n_trajectories_per_split=20, seed_vis=0,
                    output_coarse_factor=6, n_factors=10):
    """Factor Analysis point-cloud plots (top 3 of n_factors) on model inputs
    and outputs, colored by train/val/test.

    - Inputs: all time bins (no subsampling).
    - Outputs: coarsened spike counts via _coarsen.
    - Only start (o) and end (x) points are plotted per trial.
    - Axes annotated with shared variance (sum of squared loadings / total variance).
    """
    model, _, device = model_tuple
    model.eval()

    # --- Reconstruct train / val / test trial indices ---
    pattern_df = raw_data["pattern_df"]
    unique_trials_info = pattern_df[["pattern_timing_index", "pattern_name", "is_oracle"]].drop_duplicates()
    oracle_timing = unique_trials_info[unique_trials_info["is_oracle"]]["pattern_timing_index"].tolist()
    sample_timing = unique_trials_info[~unique_trials_info["is_oracle"]]["pattern_timing_index"].tolist()
    exp_seed = cfg.get("seed", 42)
    train_indices, val_indices = train_test_split(sample_timing, test_size=0.15, random_state=exp_seed)
    test_indices = oracle_timing

    # Auto-compute n_initial_state_bins
    n_initial_state_bins = cfg.get('n_initial_state_bins', 1)
    if cfg.get('init_state', False) and cfg.get('model_type', 'cnn') == 'cnn':
        _ks = cfg.get('kernel_sizes', [60])
        n_initial_state_bins = sum(k - 1 for k in _ks)

    ds_kwargs = dict(
        pattern_df=raw_data["pattern_df"],
        spike_responses=raw_data["spike_responses"],
        channel_to_index=raw_data["channel_to_index"],
        timing_to_pattern=raw_data["timing_to_pattern"],
        input_bin_size_ms=cfg['input_bin_size_ms'],
        output_bin_size_ms=cfg['output_bin_size_ms'],
        n_input_bins=cfg['n_input_bins'],
        n_output_bins=cfg['n_output_bins'],
        max_time_ms=cfg['max_time_ms'],
        output_offset=cfg.get('output_offset', 0),
        encoding_mode=cfg.get('encoding_mode', 'current'),
        init_state=cfg.get('init_state', False),
        n_initial_state_bins=n_initial_state_bins,
        history=cfg.get('history', 0),
        logger=None,
    )

    # --- Collect per-trial data ---
    splits = {'Train': train_indices, 'Val': val_indices, 'Test': test_indices}
    in_rows, out_rows = [], []
    traj_info = []
    cumulative_in = 0
    cumulative_out = 0

    with torch.no_grad():
        for split_name, indices in splits.items():
            ds = BinnedStimSpikeDataset(trial_indices=indices, **ds_kwargs)
            for i in range(len(ds)):
                x, y = ds[i]
                x_tensor = x.unsqueeze(0).to(device)
                pred = torch.exp(model(x_tensor)).cpu().numpy().squeeze(0)

                x_t = x.numpy().T   # (T_in, C)
                T_in = x_t.shape[0]

                pred_3d = pred[np.newaxis, :, :]
                pred_coarse = _coarsen(pred_3d, factor=output_coarse_factor)
                p_t = pred_coarse.squeeze(0).T  # (T_coarse, neurons)
                T_out = p_t.shape[0]

                in_rows.append(x_t)
                out_rows.append(p_t)
                traj_info.append({
                    'split': split_name,
                    'in_start': cumulative_in,
                    'in_end': cumulative_in + T_in,
                    'out_start': cumulative_out,
                    'out_end': cumulative_out + T_out,
                })
                cumulative_in += T_in
                cumulative_out += T_out

    all_in  = np.concatenate(in_rows, axis=0)
    all_out = np.concatenate(out_rows, axis=0)

    print(f"[{model_label}] FA input matrix: {all_in.shape}  (timepoints x features)")
    print(f"[{model_label}] FA output matrix: {all_out.shape}  (coarsened timepoints x neurons)")

    # --- Factor Analysis ---
    fa_in  = FactorAnalysis(n_components=n_factors, random_state=0).fit(all_in)
    fa_out = FactorAnalysis(n_components=n_factors, random_state=0).fit(all_out)
    fc_in  = fa_in.transform(all_in)
    fc_out = fa_out.transform(all_out)

    def _shared_variance_ratios(fa_obj):
        """Shared variance per factor as fraction of total variance."""
        loadings = fa_obj.components_  # (n_factors, n_features)
        factor_var = (loadings ** 2).sum(axis=1)  # per factor
        total_var = factor_var.sum() + fa_obj.noise_variance_.sum()
        return factor_var / total_var

    sv_in  = _shared_variance_ratios(fa_in)
    sv_out = _shared_variance_ratios(fa_out)
    print(f"[{model_label}] Input shared variance (top 3): "
          f"{sv_in[0]*100:.1f}%, {sv_in[1]*100:.1f}%, {sv_in[2]*100:.1f}%  "
          f"(total shared: {sv_in.sum()*100:.1f}%)")
    print(f"[{model_label}] Output shared variance (top 3): "
          f"{sv_out[0]*100:.1f}%, {sv_out[1]*100:.1f}%, {sv_out[2]*100:.1f}%  "
          f"(total shared: {sv_out.sum()*100:.1f}%)")

    # --- Subsample trials per split ---
    rng = np.random.RandomState(seed_vis)
    colors = {'Train': 'tab:blue', 'Val': 'tab:orange', 'Test': 'tab:green'}

    split_indices = defaultdict(list)
    for idx, info in enumerate(traj_info):
        split_indices[info['split']].append(idx)

    selected = {}
    for s in ['Train', 'Val', 'Test']:
        idxs = split_indices[s]
        if len(idxs) > n_trajectories_per_split:
            chosen = rng.choice(len(idxs), n_trajectories_per_split, replace=False)
            selected[s] = [idxs[c] for c in chosen]
        else:
            selected[s] = idxs

    # --- 3D point-cloud helper (start + end only, no trajectories) ---
    def _plot(fc_data, sv, title_prefix, key_start, key_end):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for split_name in ['Train', 'Val', 'Test']:
            c = colors[split_name]
            starts, ends = [], []
            for idx in selected[split_name]:
                info = traj_info[idx]
                s, e = info[key_start], info[key_end]
                seg = fc_data[s:e]
                if seg.shape[0] == 0:
                    continue
                starts.append(seg[0, :3])
                ends.append(seg[-1, :3])
            if starts:
                starts = np.array(starts)
                ends = np.array(ends)
                ax.scatter(starts[:, 0], starts[:, 1], starts[:, 2],
                           color=c, marker='o', s=25, alpha=0.6,
                           label=f'{split_name} start ({len(starts)})')
                ax.scatter(ends[:, 0], ends[:, 1], ends[:, 2],
                           color=c, marker='x', s=25, alpha=0.6,
                           label=f'{split_name} end')

        ax.set_xlabel(f'F1 ({sv[0]*100:.1f}% shared)')
        ax.set_ylabel(f'F2 ({sv[1]*100:.1f}% shared)')
        ax.set_zlabel(f'F3 ({sv[2]*100:.1f}% shared)')
        title = title_prefix
        if model_label:
            title += f' — {model_label}'
        ax.set_title(title)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
        return fig

    fig1 = _plot(fc_in,  sv_in,  'FA Point Cloud — Model Inputs',  'in_start',  'in_end')
    fig2 = _plot(fc_out, sv_out, 'FA Point Cloud — Model Outputs (coarse)',  'out_start', 'out_end')

    return fig1, fig2, fa_in, fa_out

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import os 
import logging
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from io import BytesIO
from PIL import Image
from utils import bin_spike_response


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
        spikes = dataset.spike_responses_binned[timing_idx]
        max_start = dataset.total_bins - dataset.output_offset - dataset.n_output_bins
        for t in range(max_start + 1):
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
        axes[0].set_title(f'Binary Spike Distribution\n(bin_size={dataset.bin_size}ms, n_output_bins={dataset.n_output_bins})')
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
        axes[0].set_title(f'Distribution of Spike Counts\n(bin_size={dataset.bin_size}ms, n_output_bins={dataset.n_output_bins})')
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
    axes[2].set_ylabel(f'Mean Spikes per {dataset.bin_size}ms Bin')
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
    axes[1].set_title(f'First {n_show} Neurons: Actual vs Predicted')
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


def analyze_pattern_responses_by_pattern_name(model, dataset, device):
    """
    Analyze model predictions grouped by actual pattern identity (pattern_name).
    
    Args:
        model: PyTorch model
        dataset: BinnedStimSpikeDataset instance
        device: torch device
    
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
        for batch_x, batch_y in loader:
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


def plot_oracle_trials_by_pattern(model, test_dataset, pattern_df, spike_responses, pattern_polarities,
                                   bin_size, n_input_bins, n_output_bins, output_offset, max_time_ms,
                                   n_neurons, device, out_base, logger=None, pattern_limit=None):
    """
    Generate per-pattern visualizations for oracle trials.
    
    Args:
        model: PyTorch model
        test_dataset: BinnedStimSpikeDataset for test set
        pattern_df: DataFrame with pattern info
        spike_responses: dict mapping timing_idx -> spike response arrays (in ms, 1ms resolution)
        pattern_polarities: dict mapping pattern_name -> polarity arrays
        bin_size: temporal resolution in ms
        n_input_bins, n_output_bins, output_offset: dataset config
        max_time_ms: maximum time to consider in ms
        n_neurons: number of neurons
        device: torch device
        out_base: output directory
        logger: Optional logger instance
    """
    os.makedirs(out_base, exist_ok=True)
    
    # spike_responses are in 1ms resolution, so use bin_size and max_time_ms directly
    
    # Get unique pattern info
    unique_trials = pattern_df[['pattern_timing_index', 'pattern_name', 'is_oracle']].drop_duplicates()
    oracle_patterns = unique_trials[unique_trials['is_oracle']]['pattern_name'].unique()
    
    if logger:
        logger.info(f"Found {len(oracle_patterns)} oracle patterns")
        logger.info(f"Using bin_size={bin_size}ms, n_input_bins={n_input_bins}, n_output_bins={n_output_bins}, output_offset={output_offset}")
    
    
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
    
    # Calculate number of target bins we can predict
    total_bins = max_time_ms // bin_size
    max_start_for_output = total_bins - output_offset - n_output_bins
    max_start_for_input = total_bins - n_input_bins
    max_valid_start = min(max_start_for_output, max_start_for_input)
    
    if logger:
        logger.info(f"Total bins: {total_bins}, Valid input positions: 0 to {max_valid_start}")
    
    # Process each oracle pattern
    print ("oracle patterns: ", oracle_patterns)
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
        polarity_plot = np.zeros((polarity_600.shape[0], max_time_ms))
        polarity_plot[:, :min(600, max_time_ms)] = polarity_600[:, :min(600, max_time_ms)]
        
        for trial_idx in range(len(timing_list)):
            timing_idx = timing_list[trial_idx]
            
            # Actual spike response - binned (spike_responses are in 1ms resolution)
            actual_response_binned = bin_spike_response(spike_responses[timing_idx], bin_size, max_time_ms, remainder="append")
            
            # Average spikes across other trials
            other_trial_indices = [t for t in timing_list if t != timing_idx]
            avg_other_trials = compute_avg_spikes_across_trials(
                other_trial_indices, spike_responses, bin_size, max_time_ms
            )
            
            # Build model predictions bin-by-bin
            pred_array = np.full((n_neurons, total_bins), 0, dtype=np.float32)
            
            model.eval()
            with torch.no_grad():
                batch_inputs = []
                target_bin_indices = []
                
                for t in range(max_valid_start + 1):
                    x = stim_binned[:, t : t + n_input_bins]
                    batch_inputs.append(x)
                    target_bin_indices.append(t + output_offset)
                
                batch_x = torch.tensor(np.array(batch_inputs), dtype=torch.long).to(device)
                preds = model(batch_x).cpu().numpy()
                
                for i, target_bin in enumerate(target_bin_indices):
                    pred_array[:, target_bin] = preds[i, :, 0]
            
            # Convert log predictions to rates
            pred_rates = np.exp(pred_array)
            
            # Fixed colormap range: 0-7, values >7 saturate to same color
            spike_vmin = 0
            spike_vmax = 5
            spike_cmap = 'Greys'  # 0=white, vmax=black
            
            # Create figure with 4 panels
            fig, axes = plt.subplots(4, 1, figsize=(8, 7), constrained_layout=True)
            
            # Panel 0: Stimulation pattern colored by polarity
            im0 = axes[0].imshow(polarity_plot, aspect='auto', cmap=polarity_cmap, vmin=-1, vmax=1, 
                                  interpolation='nearest', extent=[0, max_time_ms, polarity_plot.shape[0], 0])
            axes[0].set_title(f"Pattern {pattern_name} - trial {trial_idx} - Stimulation")
            axes[0].set_ylabel('Channel')
            axes[0].axvline(x=600, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            cbar0 = fig.colorbar(im0, ax=axes[0], orientation='vertical', shrink=0.8, label='Polarity', ticks=[-1, 0, 1])
            cbar0.ax.set_yticklabels(['-1', '0', '+1'])
            
            # Panel 1: Actual spike counts (full binned response, not just target bins)
            im1 = axes[1].imshow(actual_response_binned, aspect='auto', cmap=spike_cmap, interpolation='nearest', 
                                  vmin=spike_vmin, vmax=spike_vmax,
                                  extent=[0, max_time_ms, actual_response_binned.shape[0], 0])
            axes[1].axvline(x=600, color='red', linestyle='--', linewidth=1, alpha=0.7)
            axes[1].set_title(f'Actual spikes ({bin_size}ms bins)')
            axes[1].set_ylabel('Neuron')
            cbar1 = fig.colorbar(im1, ax=axes[1], orientation='vertical', shrink=0.8, label='Spike count')
            cbar1.set_ticks([spike_vmin, spike_vmax])
            cbar1.set_ticklabels(['0', f'{spike_vmax}+'])
            
            # Panel 2: Model prediction (rates) at target bins 
            im2 = axes[2].imshow(pred_rates, aspect='auto', cmap=spike_cmap, interpolation='nearest', 
                                  vmin=spike_vmin, vmax=spike_vmax,
                                  extent=[0, max_time_ms, pred_rates.shape[0], 0])
            axes[2].axvline(x=600, color='red', linestyle='--', linewidth=1, alpha=0.7)
            axes[2].set_title(f'Model prediction (rate, {bin_size}ms bins)')
            axes[2].set_ylabel('Neuron')
            cbar2 = fig.colorbar(im2, ax=axes[2], orientation='vertical', shrink=0.8, label='Predicted rate')
            cbar2.set_ticks([spike_vmin, spike_vmax])
            cbar2.set_ticklabels(['0', f'{spike_vmax}+'])
            
            # Panel 3: Average spikes across other oracle trials
            im3 = axes[3].imshow(avg_other_trials, aspect='auto', cmap=spike_cmap, interpolation='nearest', 
                                  vmin=spike_vmin, vmax=spike_vmax,
                                  extent=[0, max_time_ms, avg_other_trials.shape[0], 0])
            axes[3].axvline(x=600, color='red', linestyle='--', linewidth=1, alpha=0.7)
            n_other = len(other_trial_indices)
            axes[3].set_title(f'Avg spikes over {n_other} other oracle trials ({bin_size}ms bins)')
            axes[3].set_ylabel('Neuron')
            axes[3].set_xlabel('Time (ms)')
            cbar3 = fig.colorbar(im3, ax=axes[3], orientation='vertical', shrink=0.8, label='Spike count')
            cbar3.set_ticks([spike_vmin, spike_vmax])
            cbar3.set_ticklabels(['0', f'{spike_vmax}+'])
            
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


def plot_oracle_pattern_average_responses(unique_trials, spike_responses, spiking_neurons, savepath,
                                          plot_duration=60000, bin_size=1800):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Make a plot showing average for each oracle pattern across its trials
    # Optionally bin the time axis by bin_size_frames (e.g., 1800 frames = 60 ms bins at 30kHz)
    if bin_size < 1:
        raise ValueError("bin_size_frames must be >= 1")

    oracle_patterns = unique_trials[unique_trials['is_oracle']]['pattern_name'].unique()
    for pattern_name in oracle_patterns:  
        trials = unique_trials[unique_trials['pattern_name'] == pattern_name]
        n_neurons = len(spiking_neurons)

        n_bins = (plot_duration + bin_size - 1) // bin_size
        avg_spike_response = np.zeros((n_neurons, n_bins))

        for _, trial_info in trials.iterrows():
            timing_idx = trial_info['pattern_timing_index']
            resp = spike_responses[timing_idx]
            binned = bin_spike_response(resp, bin_size, max_time=plot_duration, remainder="append")
            avg_spike_response += binned
        avg_spike_response /= len(trials)
        
        # Plot the average spike response as a heatmap
        plt.figure(figsize=(10, 6)) 

        # Extent spans the full plot_duration_frames, even when binned
        plt.imshow(
            avg_spike_response,
            aspect='auto',
            cmap='viridis',
            interpolation='nearest',
            extent=[0, plot_duration, avg_spike_response.shape[0], 0],
        )
        plt.colorbar(label='Average Spike Count per Bin')
        plt.xlabel('Time (frames)')
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



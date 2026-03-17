import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from collections import defaultdict
import numpy as np


def compute_correlation(pred_log_rates, target_spikes, dim=1):
    """
    Computes Pearson correlation coefficient.
    
    Args:
        pred_log_rates: (batch, n_neurons, ...) or (batch, ...)
        target_spikes: (batch, n_neurons, ...) or (batch, ...)
        dim: The dimension representing neurons/features. 
             Usually, inputs are (batch, n_neurons, time). 
             We want correlation over (batch, time) for EACH neuron.
    
    Returns:
        Mean correlation across the specified dimension (e.g., mean across neurons).
    """
    # 1. Convert log rates to rates
    pred_rates = torch.exp(pred_log_rates).detach().cpu().numpy()
    targets = target_spikes.detach().cpu().numpy()
    
    # 2. Reshape to (n_neurons, everything_else)
    # Assuming input is (Batch, Neurons, Time) or (Batch, Neurons)    
    n_neurons = pred_rates.shape[dim]
    
    # Transpose to put neurons first: (Neurons, Batch, Time)
    pred_rates = pred_rates.transpose(dim, 0, 2) if pred_rates.ndim == 3 else pred_rates.transpose(dim, 0)
    targets = targets.transpose(dim, 0, 2) if targets.ndim == 3 else targets.transpose(dim, 0)
    
    # Flatten the remaining dimensions for each neuron
    pred_flat = pred_rates.reshape(n_neurons, -1)
    targets_flat = targets.reshape(n_neurons, -1)
    
    corrs = []
    for i in range(n_neurons):
        p = pred_flat[i]
        t = targets_flat[i]
        
        # Handle constant input (std=0) to avoid NaN
        if np.std(p) < 1e-6 or np.std(t) < 1e-6:
            corrs.append(1.0)
        else:
            # pearsonr returns (correlation, p-value), we want index 0
            corrs.append(np.corrcoef(p, t)[0, 1])
            
    # Return the average correlation across all neurons
    return np.mean(corrs)


# --- Helper: run model on its test loader, return per-neuron temporal correlation ---
def get_per_neuron_temporal_corr(model_tuple, loader, coarse_factor=0):
    """
    Concatenate all (sample × time-bin) predictions and targets,
    then compute Pearson r per neuron across the full time-series.
    
    Returns array of shape (n_neurons,).
    """
    model, _, device = model_tuple
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            bx, by = batch[0].to(device), batch[1]
            pred = torch.exp(model(bx)).cpu().numpy()   # (B, neurons, bins)
            all_pred.append(pred)
            all_true.append(by.numpy())
    all_pred = np.concatenate(all_pred, axis=0)  # (N, neurons, bins)
    all_true = np.concatenate(all_true, axis=0)
    if coarse_factor > 0:
        all_pred = _coarsen(all_pred, factor=coarse_factor)
        all_true = _coarsen(all_true, factor=coarse_factor)
    n_neurons = all_pred.shape[1]

    corrs = np.zeros(n_neurons)
    for n in range(n_neurons):
        p = all_pred[:, n, :].ravel()
        t = all_true[:, n, :].ravel()
        if p.std() > 1e-8 and t.std() > 1e-8:
            corrs[n] = pearsonr(p, t)[0]
    return corrs


def collect_model_preds_and_targets(model_tuple, loader, coarse_factor=0):
    """Return (all_pred, all_true) as arrays of shape (N, neurons, bins)."""
    model, _, device = model_tuple
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            bx, by = batch[0].to(device), batch[1]
            pred = torch.exp(model(bx)).cpu().numpy()
            all_pred.append(pred)
            all_true.append(by.numpy())
    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    if coarse_factor > 0:
        all_pred = _coarsen(all_pred, factor=coarse_factor)
        all_true = _coarsen(all_true, factor=coarse_factor)
    return all_pred, all_true

def collect_loo_preds_and_targets(test_dataset, coarse_factor=0, loo_flag=True):
    """Return (loo_pred, true) as arrays of shape (N, neurons, bins)."""
    pattern_to_timings = defaultdict(list)
    for timing_idx in test_dataset.trial_indices:
        pname = test_dataset.timing_to_pattern[timing_idx]
        pattern_to_timings[pname].append(timing_idx)

    n_neurons = test_dataset.n_neurons
    n_fine_bins = test_dataset.n_output_bins
    n_out_bins = n_fine_bins // coarse_factor if coarse_factor > 0 else n_fine_bins
    n_samples = len(test_dataset)

    all_true = np.zeros((n_samples, n_neurons, n_out_bins), dtype=np.float32)
    all_loo  = np.zeros((n_samples, n_neurons, n_out_bins), dtype=np.float32)

    for i in range(n_samples):
        timing_idx, t = test_dataset.samples[i]
        pname = test_dataset.timing_to_pattern[timing_idx]
        spikes_full = test_dataset.spike_responses_binned[timing_idx]
        out_start = t + test_dataset.output_offset
        y = spikes_full[:, out_start : out_start + n_fine_bins]

        other_timings = [ti for ti in pattern_to_timings[pname] if ti != timing_idx]
        if not loo_flag: # add original trial to get a full sample mean
            other_timings.append(timing_idx)

        if len(other_timings) == 0:
            loo = np.zeros_like(y)
        else:
            acc = np.zeros_like(y, dtype=np.float64)
            for oti in other_timings:
                other = test_dataset.spike_responses_binned[oti]
                acc += other[:, out_start : out_start + n_fine_bins]
            loo = acc / len(other_timings)

        if coarse_factor > 0:
            trim = n_out_bins * coarse_factor
            y   = y[:, :trim].reshape(n_neurons, n_out_bins, coarse_factor).mean(axis=2)
            loo = loo[:, :trim].reshape(n_neurons, n_out_bins, coarse_factor).mean(axis=2)

        all_true[i] = y
        all_loo[i]  = loo

    return all_loo, all_true

# --- Helper: LOO temporal correlation ---
def get_loo_temporal_corr(test_dataset, coarse_factor=0):
    """
    For each sample, the LOO 'prediction' is the average of all other
    trials of the same pattern (same time offset).  Pearson r is computed
    per neuron across the concatenated time-series of all samples.
    """
    # Map pattern → list of timing indices
    pattern_to_timings = defaultdict(list)
    for timing_idx in test_dataset.trial_indices:
        pname = test_dataset.timing_to_pattern[timing_idx]
        pattern_to_timings[pname].append(timing_idx)

    n_neurons = test_dataset.n_neurons
    n_fine_bins = test_dataset.n_output_bins          # always read full fine resolution
    n_out_bins = n_fine_bins // coarse_factor if coarse_factor > 0 else n_fine_bins
    n_samples = len(test_dataset)

    flat_true = np.zeros((n_samples * n_out_bins, n_neurons), dtype=np.float32)
    flat_loo  = np.zeros((n_samples * n_out_bins, n_neurons), dtype=np.float32)

    cursor = 0
    for i in range(n_samples):
        timing_idx, t = test_dataset.samples[i]
        pname = test_dataset.timing_to_pattern[timing_idx]
        spikes_full = test_dataset.spike_responses_binned[timing_idx]
        out_start = t + test_dataset.output_offset
        y = spikes_full[:, out_start : out_start + n_fine_bins]  # (neurons, fine_bins)

        other_timings = [ti for ti in pattern_to_timings[pname] if ti != timing_idx]
        if len(other_timings) == 0:
            loo = np.zeros_like(y)
        else:
            acc = np.zeros_like(y, dtype=np.float64)
            for oti in other_timings:
                other = test_dataset.spike_responses_binned[oti]
                acc += other[:, out_start : out_start + n_fine_bins]
            loo = acc / len(other_timings)

        # Coarsen if needed: (neurons, fine_bins) → (neurons, coarse_bins)
        if coarse_factor > 0:
            trim = n_out_bins * coarse_factor
            y   = y[:, :trim].reshape(n_neurons, n_out_bins, coarse_factor).mean(axis=2)
            loo = loo[:, :trim].reshape(n_neurons, n_out_bins, coarse_factor).mean(axis=2)

        end = cursor + n_out_bins
        flat_true[cursor:end] = y.T
        flat_loo[cursor:end]  = loo.T
        cursor = end

    corrs = np.zeros(n_neurons)
    for n in range(n_neurons):
        p = flat_loo[:, n]
        t = flat_true[:, n]
        if p.std() > 1e-8 and t.std() > 1e-8:
            corrs[n] = pearsonr(p, t)[0]
    return corrs

def _coarsen(arr_3d, factor=0):
    """Average (N, neurons, fine_bins) → (N, neurons, coarse_bins) by reshaping."""
    if factor == 0:
        print ("No coarsening applied (factor=0), returning original array.")
        return arr_3d
    N, neurons, fine = arr_3d.shape
    coarse = fine // factor
    trimmed = arr_3d[:, :, :coarse * factor]  # drop remainder bins if any
    return trimmed.reshape(N, neurons, coarse, factor).mean(axis=3)

def fraction_variance_explained(y_true, y_pred, global_variance=False):
    """Fraction of variance explained (FVE / R²) per neuron.

    Parameters
    ----------
    y_true, y_pred : ndarray, shape (N_samples, neurons, time)
    global_variance : bool
        If True, denominator uses each neuron's mean firing rate across
        **all** samples and time bins in y_true (i.e. the test set mean).
        If False, denominator uses each sample's own temporal mean per
        neuron, and FVE is averaged over samples.

    Returns
    -------
    neuron_fve : ndarray (neurons,)
    mean_fve   : float
    """
    assert y_pred.shape == y_true.shape
    batch_size, num_neurons, _ = y_true.shape

    # 1. Determine the Mean to subtract
    if global_variance:
        # Mean across ALL patterns (batch) and time
        # Shape becomes (1, neurons, 1) for broadcasting
        mu = y_true.mean(axis=(0, 2), keepdims=True) 
        numerator = np.sum((y_pred - y_true)**2, axis=(0, 2))  # sum over batch and time
        denominator = np.sum((y_true - mu)**2, axis=(0, 2))  # sum over batch and time
    else:
        # Mean per pattern and per neuron
        # Shape becomes (batch, neurons, 1)
        mu = y_true.mean(axis=2, keepdims=True)
        numerator = np.sum((y_pred - y_true)**2, axis=2)
        denominator = np.sum((y_true - mu)**2, axis=2)


    # 3. Handle division by zero (constant firing rates)
    # If denominator is tiny, the variance is 0. 
    # Usually we mask these out or set R2 to 0 (or 1 depending on philosophy, here using your mask approach)
    valid_mask = denominator > 1e-10
    
    # Initialize grid of R2 values
    if global_variance:
        r_squared_batch_neuron = np.zeros((num_neurons,))  # only one R2 per neuron
    else:   
        r_squared_batch_neuron = np.zeros((batch_size, num_neurons))
    
    # Perform division only where valid
    # Note: We must index both arrays with the mask to maintain shape alignment
    r_squared_batch_neuron[valid_mask] = 1 - (numerator[valid_mask] / denominator[valid_mask])

    # 4. Aggregate results
    # Average across the batch dimension to get one R2 value per neuron
    if global_variance:
        # r_squared_batch_neuron is already (neurons,) since we computed global variance
        neuron_r2 = r_squared_batch_neuron
    else:
        neuron_r2 = np.mean(r_squared_batch_neuron, axis=0)
    
    return neuron_r2, np.mean(neuron_r2)



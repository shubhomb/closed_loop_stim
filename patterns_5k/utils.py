import numpy as np
import pandas as pd
import logging  
import pickle
import os 

import torch
from torch.utils.data import Dataset

# Mapping from delay_mode to integer index for categorical encoding
# Index 4 = no stimulation (default)
DELAY_MODE_TO_INDEX = {0: 0, 1: 1, -1: 2, 2: 3}  # delay_mode -> category index
NO_STIM_INDEX = 4  # Category for no stimulation
NUM_STIM_LEVELS = 5  # Total categories: 4 delay modes + 1 no-stim


def make_spikes_responses_df(spkVecs_file):
    """
    Load spike vectors from a pickle file and convert to a pandas DataFrame.
    
    Args:
        spkVecs_file (str): Path to the pickle file containing spike vectors.
    """
    spikes_df = pd.DataFrame(np.load(spkVecs_file)) # size = (3396476,))
    spikes_df.columns = ['timestamp', 'neuron_id', 'segment_index']
    spikes_df.drop(columns=['segment_index'], inplace=True)
    spikes_df['timestamp'] = spikes_df['timestamp'].astype(int)
    spikes_df['neuron_id'] = spikes_df['neuron_id'].astype(int) # noncontinuous neuron ids from with numbers corresponding to shank location
    spikes_df.sort_values(by=['timestamp', 'neuron_id'], inplace=True)
    return spikes_df

def read_pattern_json(pattern_registrations_pkl_path):
    data = pickle.load(open(pattern_registrations_pkl_path, "rb"))
    # 1. Flatten the Pattern -> Steps level
    # We use record_path to reach the 'steps' and meta to keep parent info
    df = pd.json_normalize(
        data, 
        record_path=['steps'], 
        meta=[
            'pattern_name', # given name of the pattern
            'pattern_lambda', # given lambda parameter that generated the pattern 
            'pattern_flag_start_timestamp', # given starting timpestamp of pattern
            'pattern_timing_index' # given order of pattern from 1-5,000 (first to last)
        ],
        record_prefix='step_'
    )
    # 2. Flatten the 'step_channel_delays' list into individual rows
    # This creates a row for every delay entry. Steps with [] will become NaN.
    df = df.explode('step_channel_delays').reset_index(drop=True)

    # 3. Convert the dictionaries in 'step_channel_delays' into separate columns
    delays_df = pd.json_normalize(df['step_channel_delays'])
    
    # 4. Store pattern length
    df['pattern_idx_length'] = df.groupby('pattern_timing_index')['step_index'].transform('max') + 1

    # 5. Combine and cleanup
    final_df = pd.concat([df.drop(columns=['step_channel_delays']), delays_df], axis=1)
    
    # 6. Convert timestamps to integers and subtract original timestamp from step timestamps
    final_df['step_start_timestamp'] = final_df['step_start_timestamp'].astype(int)
    final_df['pattern_flag_start_timestamp'] = final_df['pattern_flag_start_timestamp'].astype(int)
    
    # 7. Add is_oracle column (True for pattern_name 4001-4050)
    final_df['is_oracle'] = (final_df['pattern_name'] >= 4001) & (final_df['pattern_name'] <= 4050)
    
    # 8. Add trial column
    # For non-oracle patterns: trial = 1 (single trial each)
    # For oracle patterns: trial = 1-10 (10 repeats of each pattern)
    # We assign trial numbers based on the order of appearance (pattern_timing_index)
    final_df['trial'] = 1  # Default for non-oracle
    
    # For oracle patterns, assign trial number based on order of occurrence
    oracle_mask = final_df['is_oracle']
    if oracle_mask.any():
        # Get unique (pattern_name, pattern_timing_index) pairs for oracle patterns
        oracle_occurrences = final_df.loc[oracle_mask, ['pattern_name', 'pattern_timing_index']].drop_duplicates()
        oracle_occurrences = oracle_occurrences.sort_values('pattern_timing_index')
        
        # Assign trial number (1-10) for each pattern_name based on order of appearance
        oracle_occurrences['trial'] = oracle_occurrences.groupby('pattern_name').cumcount() + 1
        
        # Merge back to get trial numbers for oracle patterns
        trial_map = oracle_occurrences.set_index('pattern_timing_index')['trial'].to_dict()
        final_df.loc[oracle_mask, 'trial'] = final_df.loc[oracle_mask, 'pattern_timing_index'].map(trial_map)

    return final_df

def preprocess_pattern_stimulations_df(pattern_df):
    """
    Preprocess the pattern stimulations DataFrame by adding useful columns.
    
    Args:
        pattern_stimulations_df (pd.DataFrame): DataFrame containing pattern stimulations.
    """
    # Add pattern duration column
    # define pattern end as pattern's start time - 1
    # build unique patterns and set end = next pattern's start - 1
    patterns = (pattern_df.groupby(['pattern_timing_index', 'pattern_name'])['pattern_flag_start_timestamp']
                .first().reset_index())
    patterns = patterns.sort_values('pattern_timing_index').reset_index(drop=True)


    # end = next start - 1
    patterns['pattern_end_timestamp'] = patterns['pattern_flag_start_timestamp'].shift(-1) - 1
    patterns.loc[patterns.index[-1], 'pattern_end_timestamp'] = patterns.loc[patterns.index[-1], 'pattern_flag_start_timestamp'] + 1999  # assume 2000 ms for last pattern

    # ensure integer timestamps
    patterns['pattern_end_timestamp'] = patterns['pattern_end_timestamp'].astype(int)
    patterns['pattern_flag_start_timestamp'] = patterns['pattern_flag_start_timestamp'].astype(int)

    # merge end times back into the full step-level dataframe
    pattern_df = pattern_df.merge(
        patterns[['pattern_timing_index', 'pattern_end_timestamp']],
        on='pattern_timing_index',
        how='left'
    )

    # how many times each pattern appears in separate timestamps
    pattern_counts = pattern_df[['pattern_name', 'pattern_timing_index']].drop_duplicates().groupby('pattern_name').size()
    min_pattern_timestamp = pattern_df['pattern_flag_start_timestamp'].min()

    # Center times to 0 for easier intepretation
    pattern_df['pattern_flag_start_timestamp'] -= min_pattern_timestamp
    pattern_df['pattern_end_timestamp'] -= min_pattern_timestamp
    pattern_df['pattern_duration'] = pattern_df['pattern_end_timestamp'] - pattern_df['pattern_flag_start_timestamp']
    pattern_df['step_start_timestamp'] -= min_pattern_timestamp
    return pattern_df, min_pattern_timestamp




def trial_breakout_spikes_and_patterns(spikes_df, pattern_df, channel_to_index, spiking_neurons, stim_time_ms=600, post_stim_ms=1400, step_time_ms=60, spiking_neuron_to_index=None):
    """
    For each trial (pattern presentation), extract spike responses within a specified time window.
    
    Args:
        spikes_df (pd.DataFrame): DataFrame containing spike data with 'timestamp' and 'neuron_id'.
        pattern_df (pd.DataFrame): DataFrame containing pattern stimulations with 'pattern_flag_start_timestamp'.
    """
    # Inputs: 600 ms (10 60ms steps) x 44 stimulating channels with 4 delay modes.
    # Key by pattern_timing_index to preserve all 10 trials of oracle patterns
    pattern_stims = {}  # keyed by pattern_name (stimulus is same for all trials of same pattern)
    pattern_polarities = {}  # keyed by pattern_name, tracks polarity (-1 or 1) for each pulse
    spike_responses = {}  # keyed by pattern_timing_index (unique per trial)
    timing_to_pattern = {}  # map timing_index -> pattern_name for lookups

    # Get unique trials (pattern_timing_index, pattern_name pairs)
    unique_trials = pattern_df[['pattern_timing_index', 'pattern_name', 'pattern_flag_start_timestamp', 'pattern_end_timestamp', 'is_oracle', 'trial']].drop_duplicates()
    for _, trial_info in unique_trials.iterrows():
        timing_idx = trial_info['pattern_timing_index']
        pattern_name = trial_info['pattern_name']
        pattern_start_time = trial_info['pattern_flag_start_timestamp']
        pattern_end_time = trial_info['pattern_end_timestamp']
        
        timing_to_pattern[timing_idx] = pattern_name
        n_channels = len(channel_to_index)
        # Build stim pattern (same for all trials of same pattern, only compute once)
        if pattern_name not in pattern_stims: 
            logging.info(f"Building stim pattern for pattern_name {pattern_name}")
            pattern_subset = pattern_df[pattern_df['pattern_name'] == pattern_name].drop_duplicates(subset=['step_index', 'channel', 'delay_mode'])
            stim = np.zeros((n_channels, stim_time_ms))  # 44 channels, 600 ms duration
            polarity = np.zeros((n_channels, stim_time_ms))  # Track polarity: -1 for delay_mode -1, +1 otherwise
            for idx, row in pattern_subset.iterrows():
                step_index = row['step_index']
                stim_ms = step_time_ms * step_index  # each step is 60 ms
                if pd.isna(row['channel']):
                    continue  # No stimulation for this step
                channel_index = channel_to_index[int(row['channel'])]
                delay_mode = int(row['delay_mode'])
                # Polarity is -1 for delay_mode -1, +1 otherwise
                pulse_polarity = -1 if delay_mode == -1 else 1
                
                if delay_mode == 0:  # 3 pulses, 50 Hz, starting at stim_ms
                    for pulse in range(3):
                        pulse_time = stim_ms + pulse * 20
                        if pulse_time < 600:
                            stim[channel_index, pulse_time] = 3
                            polarity[channel_index, pulse_time] = pulse_polarity
                elif delay_mode == 1:  # 3 pulses, 50 Hz, starting at 10ms after stim_ms
                    for pulse in range(3):
                        pulse_time = stim_ms + 10 + pulse * 20
                        if pulse_time < 600:
                            stim[channel_index, pulse_time] = 3
                            polarity[channel_index, pulse_time] = pulse_polarity
                elif delay_mode == -1:  # same as 0 but with reverse phase structure
                    for pulse in range(3): 
                        pulse_time = stim_ms + pulse * 20
                        if pulse_time < 600:
                            stim[channel_index, pulse_time] = 3
                            polarity[channel_index, pulse_time] = pulse_polarity
                elif delay_mode == 2:  # 6 pulses, 100 Hz, starting at stim_ms
                    for pulse in range(6):
                        pulse_time = stim_ms + pulse * 10
                        if pulse_time < 600:
                            stim[channel_index, pulse_time] = 3
                            polarity[channel_index, pulse_time] = pulse_polarity
            pattern_stims[pattern_name] = stim
            pattern_polarities[pattern_name] = polarity
        
        # Build spike responses for this specific trial (keyed by timing_index)
        spikes_during_pattern = spikes_df[(spikes_df['timestamp'] >= pattern_start_time) & (spikes_df['timestamp'] < pattern_end_time)]
        spike_responses_pattern = np.zeros((len(spiking_neurons), stim_time_ms + post_stim_ms))  # e.g., 2000 ms window
        for idx, row in spikes_during_pattern.iterrows():
            neuron_id = row['neuron_id']
            neuron_index = spiking_neuron_to_index[neuron_id]
            spike_time = row['timestamp'] - pattern_start_time # this is in frames, centered on the pattern_start_time
            if 0 <= spike_time < 30000 * 2:
                ms_idx = (spike_time) // 30 
                spike_responses_pattern[neuron_index, ms_idx] += 1
        spike_responses[timing_idx] = spike_responses_pattern
    return pattern_stims, pattern_polarities, spike_responses, timing_to_pattern, unique_trials


import numpy as np

def bin_spike_response(spike_resp, bin_size, max_time=None, remainder='drop'):
    """
    Bin spike responses to specified bin size.
    
    Parameters:
    -----------
    spike_resp : np.ndarray
        (n_neurons, time_ms) array of 1ms resolution spike data.
    bin_size : int
        The size of the bin to sum over.
    max_time : int, optional
        Cutoff time for the input data.
    remainder : str, optional
        'drop' (default) - Discard the last chunk of data if it is smaller than bin_size.
        'append' - Pad the data with zeros to complete the last bin and include it.
    """
    
    # 1. Slice to max_time if provided
    if max_time is not None:
        spike_resp = spike_resp[:, :max_time]
    else:
        max_time = spike_resp.shape[1]

    # 2. Handle 'append' logic
    # If we need to keep the remainder, we pad the array with zeros until it fits the bin_size
    if remainder == 'append':
        leftover = max_time % bin_size
        if leftover > 0:
            pad_width = bin_size - leftover
            # Pad axis 1 (time) with zeros at the end
            # ((0,0), (0, pad_width)) means: nothing on axis 0 (neurons), 0 before and pad_width after on axis 1
            spike_resp = np.pad(spike_resp, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            
            # Update max_time to match the new padded length
            max_time = spike_resp.shape[1]

    # 3. Standard Binning Logic
    n_neurons_local = spike_resp.shape[0]
    n_bins = max_time // bin_size
    
    # The slice [:n_bins * bin_size] handles the 'drop' logic automatically 
    # (by excluding the end) if we didn't pad above.
    reshaped = spike_resp[:, :n_bins * bin_size].reshape(n_neurons_local, n_bins, bin_size)
    binned = reshaped.sum(axis=2).astype(np.float32)
    
    return binned

class BinnedStimSpikeDataset(Dataset):
    """
    Dataset that creates samples at the individual time bin level.
    
    Each sample consists of:
    - Input (X): Stimulation pattern over `n_input_bins` consecutive bins
      Shape: (n_channels, n_input_bins) with categorical indices [0-4]
    - Output (Y): Spike counts over `n_output_bins` consecutive bins
      Shape: (n_neurons, n_output_bins)
    
    Samples are generated for each valid starting position in each trial.
    For a trial of length T bins, with n_input_bins=1 and n_output_bins=2,
    we generate samples at positions t=0, 1, ..., T-2 (output extends 2 bins from t).
    
    Args:
        pattern_df: DataFrame with step-level stimulation information
        spike_responses: dict mapping pattern_timing_index -> (n_neurons, 2000) numpy array
        channel_to_index: dict mapping channel name -> index
        timing_to_pattern: dict mapping pattern_timing_index -> pattern_name
        trial_indices: list of pattern_timing_index values to include
        bin_size_ms: temporal resolution in milliseconds (e.g., 10, 20)
        n_input_bins: number of consecutive stim bins to use as input (default 1)
        n_output_bins: number of consecutive spike bins to predict (default 1)
        max_time_ms: maximum time to consider (default 2000)
        output_offset: how many bins after input start to begin output (default 0 = same window)
        logger: optional logger instance for logging messages
    """
    def __init__(self, pattern_df, spike_responses, channel_to_index, timing_to_pattern,
                 trial_indices=None, bin_size_ms=10, n_input_bins=1, n_output_bins=1,
                 max_time_ms=2000, output_offset=0, logger=None):
        
        if trial_indices is None:
            trial_indices = list(spike_responses.keys())
        
        assert max_time_ms % bin_size_ms == 0, f"max_time_ms ({max_time_ms}) must be divisible by bin_size_ms ({bin_size_ms})"
        
        self.trial_indices = trial_indices
        self.timing_to_pattern = timing_to_pattern
        self.n_channels = len(channel_to_index)
        self.bin_size = bin_size_ms
        self.n_input_bins = n_input_bins
        self.n_output_bins = n_output_bins
        self.max_time_ms = max_time_ms
        self.output_offset = output_offset
        self.total_bins = max_time_ms // bin_size_ms
        self._logger = logger
        
        # Pre-compute all binned stimulation patterns (one per unique pattern)
        # Shape: (n_channels, total_bins) with categorical indices
        self.pattern_stims = {}
        unique_patterns = pattern_df['pattern_name'].unique()
        for pattern_name in unique_patterns:
            pattern_subset = pattern_df[pattern_df['pattern_name'] == pattern_name].drop_duplicates(
                subset=['step_index', 'channel', 'delay_mode'])
            self.pattern_stims[pattern_name] = self._encode_pattern_binned(
                pattern_subset, channel_to_index, bin_size_ms, max_time_ms)
        
        # Pre-compute all binned spike responses
        # Shape: (n_neurons, total_bins) with spike counts
        self.spike_responses_binned = {}
        for timing_idx in trial_indices:
            raw_spikes = spike_responses[timing_idx][:, :max_time_ms]
            self.spike_responses_binned[timing_idx] = self._bin_spikes(raw_spikes, bin_size_ms)
        
        # Calculate valid starting positions for each trial
        # Output ends at: t + output_offset + n_output_bins
        # So max valid t is: total_bins - output_offset - n_output_bins
        # Also need: t + n_input_bins <= total_bins
        max_start_for_output = self.total_bins - output_offset - n_output_bins
        max_start_for_input = self.total_bins - n_input_bins
        max_valid_start = min(max_start_for_output, max_start_for_input)
        
        # Generate (trial_idx, time_bin) pairs for all samples
        self.samples = []
        for timing_idx in trial_indices:
            for t in range(max_valid_start + 1):
                self.samples.append((timing_idx, t))
        
        # Get neuron count from first trial
        sample_key = trial_indices[0]
        self.n_neurons = spike_responses[sample_key].shape[0]
        
        self._log(f"BinnedStimSpikeDataset initialized:")
        self._log(f"  bin_size_ms={bin_size_ms}, n_input_bins={n_input_bins}, n_output_bins={n_output_bins}")
        self._log(f"  output_offset={output_offset}, total_bins per trial={self.total_bins}")
        self._log(f"  Valid start positions per trial: 0 to {max_valid_start}")
        self._log(f"  Total samples: {len(self.samples)} ({len(trial_indices)} trials × {max_valid_start + 1} positions)")
    
    def _log(self, message):
        """Log a message using the provided logger or print."""
        if self._logger:
            self._logger.info(message)
        else:
            logging.info(message)
    
    def _encode_pattern_binned(self, pattern_subset, channel_to_index, bin_size_ms, max_time_ms):
        """Encode a pattern into binned categorical format.
        
        Returns:
            stim: (n_channels, total_bins) with categorical indices [0-4]
        """
        n_channels = len(channel_to_index)
        total_bins = max_time_ms // bin_size_ms
        stim = np.full((n_channels, total_bins), NO_STIM_INDEX, dtype=np.int64)
        assert bin_size_ms <= 60 # if bin size is more than 60, then we can have multiple stimulations per bin which would need to be encoded differently
        for _, row in pattern_subset.iterrows():
            if pd.isna(row['channel']):
                continue
            step_index = int(row['step_index'])
            if step_index >= 10:  # Only first 10 steps (600ms of stimulation)
                continue
            
            channel_index = channel_to_index[int(row['channel'])]
            delay_mode = int(row['delay_mode'])
            category_idx = DELAY_MODE_TO_INDEX[delay_mode]
            
            # Each 60ms step contains stimulation
            # Mark all bins within this 60ms window with the delay mode category
            step_start_ms = step_index * 60
            step_end_ms = step_start_ms + 60
            
            start_bin = step_start_ms // bin_size_ms
            end_bin = min(step_end_ms // bin_size_ms, total_bins)
            
            for b in range(start_bin, end_bin):
                stim[channel_index, b] = category_idx
        
        return stim
    
    def _bin_spikes(self, spike_resp, bin_size_ms):
        """Bin spike responses into counts per bin.
        
        Args:
            spike_resp: (n_neurons, time_ms) binary spike array
            bin_size_ms: bin width in ms
        Returns:
            binned: (n_neurons, n_bins) spike counts
        """
        # Use the shared bin_spike_response function
        max_time = spike_resp.shape[1]
        return bin_spike_response(spike_resp, bin_size_ms, max_time)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        timing_idx, t = self.samples[idx]
        pattern_name = self.timing_to_pattern[timing_idx]
        
        # Extract input: stim bins [t, t + n_input_bins)
        stim_full = self.pattern_stims[pattern_name]
        x = stim_full[:, t : t + self.n_input_bins]  # (n_channels, n_input_bins)
        
        # Extract output: spike bins [t + output_offset, t + output_offset + n_output_bins)
        spikes_full = self.spike_responses_binned[timing_idx]
        out_start = t + self.output_offset
        y = spikes_full[:, out_start : out_start + self.n_output_bins]  # (n_neurons, n_output_bins)
        
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.float32)
    
    def compute_sparsity(self):
        """Compute sparsity statistics for the dataset.
        
        Returns:
            dict with sparsity metrics:
                - total_bins: total number of spike bins across all samples
                - zero_bins: number of bins with 0 spikes
                - nonzero_bins: number of bins with >= 1 spike
                - sparsity_pct: percentage of bins that are zero
        """
        total_bins = 0
        zero_bins = 0
        
        for timing_idx in self.trial_indices:
            spikes = self.spike_responses_binned[timing_idx]  # (n_neurons, total_bins)
            # Count across valid output positions
            max_start = self.total_bins - self.output_offset - self.n_output_bins
            for t in range(max_start + 1):
                out_start = t + self.output_offset
                y = spikes[:, out_start : out_start + self.n_output_bins]
                total_bins += y.size
                zero_bins += (y == 0).sum()
        
        nonzero_bins = total_bins - zero_bins
        sparsity_pct = 100 * zero_bins / total_bins if total_bins > 0 else 0
        
        return {
            'total_bins': int(total_bins),
            'zero_bins': int(zero_bins),
            'nonzero_bins': int(nonzero_bins),
            'sparsity_pct': float(sparsity_pct)
        }
    
    def compute_sample_sparsity(self):
        """Compute per-sample sparsity (fraction of neurons with zero spikes per sample).
        
        Returns:
            dict with sample-level sparsity metrics:
                - total_samples: total number of samples
                - per_sample_zero_frac: array of shape (n_samples,) with fraction of neurons 
                                         that have zero spikes in each sample
                - mean_zero_frac: mean fraction of zero neurons across samples
                - std_zero_frac: std of fraction of zero neurons across samples
        """
        per_sample_zero_frac = []
        
        for timing_idx in self.trial_indices:
            spikes = self.spike_responses_binned[timing_idx]  # (n_neurons, total_bins)
            max_start = self.total_bins - self.output_offset - self.n_output_bins
            for t in range(max_start + 1):
                out_start = t + self.output_offset
                y = spikes[:, out_start : out_start + self.n_output_bins]  # (n_neurons, n_output_bins)
                # Sum across output bins to get total spikes per neuron in this sample
                neuron_totals = y.sum(axis=1)  # (n_neurons,)
                zero_frac = (neuron_totals == 0).mean()
                per_sample_zero_frac.append(zero_frac)
        
        per_sample_zero_frac = np.array(per_sample_zero_frac)
        
        return {
            'total_samples': len(per_sample_zero_frac),
            'per_sample_zero_frac': per_sample_zero_frac,
            'mean_zero_frac': float(per_sample_zero_frac.mean()),
            'std_zero_frac': float(per_sample_zero_frac.std())
        }


def compute_correlation(pred_log_rates, target_spikes):
    """
    Computes Pearson correlation between predicted rates and actual spike counts.
    Input shapes: (batch, n_neurons, n_bins)
    """
    pred_rates = torch.exp(pred_log_rates).detach().cpu().numpy().flatten()
    targets = target_spikes.detach().cpu().numpy().flatten()
    
    if np.std(pred_rates) == 0 or np.std(targets) == 0:
        return 0.0
        
    return np.corrcoef(pred_rates, targets)[0, 1]
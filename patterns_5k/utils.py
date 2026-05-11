import numpy as np
import pandas as pd
import logging  
import pickle
import os 
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from models import SimpleCausalSpikeCNN

# Mapping from delay_mode to integer index for categorical encoding
# Index 4 = no stimulation (default)
DELAY_MODE_TO_INDEX = {0: 0, 1: 1, -1: 2, 2: 3}  # delay_mode -> category index
NO_STIM_INDEX = 4  # Category for no stimulation
NUM_STIM_LEVELS = 5  # Total categories: 4 delay modes + 1 no-stim


'''Utility functions for loading data, defining datasets, and other common tasks.'''

def load_raw_data(cfg: dict, logger=None) -> dict:
    """Load and preprocess the raw spike / pattern data used in an experimental config. 

    This is the expensive I/O step (~45 s).  The result can be reused
    across experiments that share the same *datadir* and
    *problematic_neurons*.

    Returns a dict with:
        pattern_df, spike_responses, pattern_stims, pattern_polarities,
        channel_to_index, timing_to_pattern, unique_trials,
        spiking_neurons, n_stim_channels, n_neurons
    """
    _log = logger.info if logger else print

    datadir = cfg["datadir"]
    problematic_neurons = cfg.get("problematic_neurons", [])

    # Fallback: if configured datadir doesn't exist locally, try to find
    # the data under the local 'data/' directory by matching the last
    # path component (e.g. "oracle_ICMS_150").
    spkvecs_path = os.path.join(datadir, "SpkVecs.npy")
    if not os.path.isfile(spkvecs_path):
        basename = os.path.basename(os.path.normpath(datadir))
        local_candidate = os.path.join("data", basename)
        if os.path.isfile(os.path.join(local_candidate, "SpkVecs.npy")):
            _log(f"Datadir '{datadir}' not found; falling back to '{local_candidate}/'")
            datadir = local_candidate
        else:
            raise FileNotFoundError(
                f"SpkVecs.npy not found at '{spkvecs_path}' or fallback '{local_candidate}/SpkVecs.npy'")

    _log("Loading data …")
    spikes_df = make_spikes_responses_df(os.path.join(datadir, "SpkVecs.npy"))
    spikes_df = spikes_df[~spikes_df["neuron_id"].isin(problematic_neurons)]
    _log(f"{spikes_df['neuron_id'].nunique()} unique neurons after dropping problematic neurons")

    pattern_registrations_path = os.path.join(datadir, "pattern_registrations.pkl")
    pattern_df, min_pattern_timestamp = preprocess_pattern_stimulations_df(
        read_pattern_json(pattern_registrations_path), align_to_stim=True
    )

    channel_to_index = {ch: idx for idx, ch in enumerate(sorted(pattern_df["channel"].dropna().unique()))}
    spiking_neurons = spikes_df["neuron_id"].unique()
    spiking_neuron_to_index = {neuron: idx for idx, neuron in enumerate(spiking_neurons)}
    spiking_neurons.sort()

    _log(f"Stimulation channels: {len(channel_to_index)}, Spiking neurons: {len(spiking_neurons)}")

    pattern_stims, pattern_polarities, spike_responses, timing_to_pattern, unique_trials = (
        trial_breakout_spikes_and_patterns(
            spikes_df,
            pattern_df,
            channel_to_index,
            spiking_neurons=spiking_neurons,
            spiking_neuron_to_index=spiking_neuron_to_index,
        )
    )
    _log(f"Unique patterns: {len(pattern_stims)}, Total trials: {len(spike_responses)}")

    return {
        "pattern_df": pattern_df,
        "spike_responses": spike_responses,
        "pattern_stims": pattern_stims,
        "pattern_polarities": pattern_polarities,
        "channel_to_index": channel_to_index,
        "timing_to_pattern": timing_to_pattern,
        "unique_trials": unique_trials,
        "spiking_neurons": spiking_neurons,
        "n_stim_channels": len(channel_to_index),
        "n_neurons": len(spiking_neurons),
    }



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
    # After explode, steps with empty [] become NaN — drop those rows first
    df = df.dropna(subset=['step_channel_delays']).reset_index(drop=True)
    delays_df = pd.json_normalize(df['step_channel_delays'])
    
    # 4. Store pattern length
    df['pattern_idx_length'] = df.groupby('pattern_timing_index')['step_index'].transform('max') + 1

    # 5. Combine and cleanup
    final_df = pd.concat([df.drop(columns=['step_channel_delays']), delays_df], axis=1)
    
    # 6. Convert timestamps to integers and subtract original timestamp from step timestamps
    final_df['step_start_timestamp'] = final_df['step_start_timestamp'].astype(int)
    final_df['pattern_flag_start_timestamp'] = final_df['pattern_flag_start_timestamp'].astype(int)
    
    # 7. Add is_oracle column: True for any pattern_name that appears more than once
    repeat_counts = (
        final_df[['pattern_name', 'pattern_timing_index']]
        .drop_duplicates()
        .groupby('pattern_name')['pattern_timing_index']
        .count()
    )
    oracle_pattern_names = set(repeat_counts[repeat_counts > 1].index)
    final_df['is_oracle'] = final_df['pattern_name'].isin(oracle_pattern_names)

    # 8. Add trial column: cumulative occurrence index per pattern_name
    final_df['trial'] = 1
    oracle_mask = final_df['is_oracle']
    if oracle_mask.any():
        oracle_occurrences = final_df.loc[oracle_mask, ['pattern_name', 'pattern_timing_index']].drop_duplicates()
        oracle_occurrences = oracle_occurrences.sort_values('pattern_timing_index')
        oracle_occurrences['trial'] = oracle_occurrences.groupby('pattern_name').cumcount() + 1
        trial_map = oracle_occurrences.set_index('pattern_timing_index')['trial'].to_dict()
        final_df.loc[oracle_mask, 'trial'] = final_df.loc[oracle_mask, 'pattern_timing_index'].map(trial_map)

    return final_df

def preprocess_pattern_stimulations_df(pattern_df, align_to_stim=False, center_to_0=False):
    """
    Preprocess the pattern stimulations DataFrame by adding useful columns.
    
    Args:
        pattern_stimulations_df (pd.DataFrame): DataFrame containing pattern stimulations.
        align_to_stim(bool): whether each pattern start should be defined by the original pattern flag start or by the first stimulation onset
        center_to_0 (bool): whether to subtract min_pattern_idx
    """
    # Add pattern duration column
    # define pattern end as pattern's start time - 1
    # build unique patterns and set end = next pattern's start - 1
    patterns = (pattern_df.groupby(['pattern_timing_index', 'pattern_name'])['pattern_flag_start_timestamp']
                .first().reset_index())
    patterns = patterns.sort_values('pattern_timing_index').reset_index(drop=True)

    if align_to_stim:
        # Use the earliest step_start_timestamp per trial as the aligned flag.
        # step_index==0 can be missing when its step_channel_delays was empty
        # (dropped upstream), and even when present it isn't guaranteed to be
        # the numerically smallest timestamp — grouping by min on the full
        # stim-bearing rows is the robust definition of "first stim".
        first_step_timestamps = (
            pattern_df
            .groupby('pattern_timing_index')['step_start_timestamp']
            .min()
            .reset_index()
            .rename(columns={'step_start_timestamp': 'first_stim_timestamp'})
        )
        # Merge and replace pattern_flag_start_timestamp with the first stimulation timestamp.
        # Fall back to original flag timestamp when step_index=0 was dropped
        # (e.g. because that step had empty channel_delays).
        patterns = patterns.merge(first_step_timestamps, on='pattern_timing_index', how='left')
        patterns['first_stim_timestamp'] = patterns['first_stim_timestamp'].fillna(
            patterns['pattern_flag_start_timestamp'])
        patterns['pattern_flag_start_timestamp'] = patterns['first_stim_timestamp']
        patterns = patterns.drop(columns=['first_stim_timestamp'])

    # end = next start - 1
    patterns['pattern_end_timestamp'] = patterns['pattern_flag_start_timestamp'].shift(-1) - 1
    # For last pattern, assume 2000ms duration. Timestamps are in frames at 30 kHz, so 2000ms = 60000 frames
    patterns.loc[patterns.index[-1], 'pattern_end_timestamp'] = patterns.loc[patterns.index[-1], 'pattern_flag_start_timestamp'] + 60000 - 1  # 2000 ms = 60000 frames

    # ensure integer timestamps
    patterns['pattern_end_timestamp'] = patterns['pattern_end_timestamp'].astype(int)
    patterns['pattern_flag_start_timestamp'] = patterns['pattern_flag_start_timestamp'].astype(int)

    # merge end times (and updated start times if align_to_stim) back into the full step-level dataframe
    if align_to_stim:
        # Also update pattern_flag_start_timestamp in the original dataframe
        pattern_df = pattern_df.drop(columns=['pattern_flag_start_timestamp'], errors='ignore')
        pattern_df = pattern_df.merge(
            patterns[['pattern_timing_index', 'pattern_end_timestamp', 'pattern_flag_start_timestamp']],
            on='pattern_timing_index',
            how='left'
        )
    else:
        pattern_df = pattern_df.merge(
            patterns[['pattern_timing_index', 'pattern_end_timestamp']],
            on='pattern_timing_index',
            how='left'
        )

    # how many times each pattern appears in separate timestamps
    pattern_counts = pattern_df[['pattern_name', 'pattern_timing_index']].drop_duplicates().groupby('pattern_name').size()
    min_pattern_timestamp = pattern_df['pattern_flag_start_timestamp'].min()

    # Center times to 0 for easier intepretation
    if center_to_0: 
        pattern_df['pattern_flag_start_timestamp'] -= min_pattern_timestamp
        pattern_df['pattern_end_timestamp'] -= min_pattern_timestamp
        pattern_df['step_start_timestamp'] -= min_pattern_timestamp
    pattern_df['pattern_duration'] = pattern_df['pattern_end_timestamp'] - pattern_df['pattern_flag_start_timestamp']
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
            pattern_subset = pattern_df[pattern_df['pattern_name'] == pattern_name].drop_duplicates(subset=['step_index', 'channel', 'delay_mode'])
            stim = np.zeros((n_channels, stim_time_ms))  # 44 channels, 600 ms duration
            polarity = np.zeros((n_channels, stim_time_ms))  # Track polarity: -1 for delay_mode -1, +1 otherwise
            # Use the first occurrence's pattern_flag_start_timestamp as reference for this pattern
            pattern_start_ref = pattern_df[pattern_df['pattern_name'] == pattern_name]['pattern_flag_start_timestamp'].iloc[0]
            for idx, row in pattern_subset.iterrows():
                # Convert actual step timestamp to ms relative to pattern start
                stim_ms = int((row['step_start_timestamp'] - pattern_start_ref) / 30)
                if pd.isna(row['channel']):
                    continue  # No stimulation for this step
                channel_index = channel_to_index[int(row['channel'])]
                delay_mode = int(row['delay_mode'])
                # Polarity is -1 for delay_mode -1, +1 otherwise
                pulse_polarity = -1 if delay_mode == -1 else 1 # This is the correct definition of polarity based on the original code's use of delay_mode -1 as the only negative case
                
                if delay_mode == 0:  # 3 pulses, 50 Hz, starting at stim_ms
                    for pulse in range(3):
                        pulse_time = stim_ms + pulse * 20
                        if 0 <= pulse_time < stim_time_ms:
                            stim[channel_index, pulse_time] = 3
                            polarity[channel_index, pulse_time] = pulse_polarity
                elif delay_mode == 1:  # 3 pulses, 50 Hz, starting at 10ms after stim_ms
                    for pulse in range(3):
                        pulse_time = stim_ms + 10 + pulse * 20
                        if 0 <= pulse_time < stim_time_ms:
                            stim[channel_index, pulse_time] = 3
                            polarity[channel_index, pulse_time] = pulse_polarity
                elif delay_mode == -1:  # same as 0 but with reverse phase structure
                    for pulse in range(3):
                        pulse_time = stim_ms + pulse * 20
                        if 0 <= pulse_time < stim_time_ms:
                            stim[channel_index, pulse_time] = 3
                            polarity[channel_index, pulse_time] = pulse_polarity
                elif delay_mode == 2:  # 6 pulses, 100 Hz, starting at stim_ms
                    for pulse in range(6):
                        pulse_time = stim_ms + pulse * 10
                        if 0 <= pulse_time < stim_time_ms:
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
            spike_time = row['timestamp'] - pattern_start_time  # frames, relative to pattern start
            ms_idx = spike_time // 30
            if 0 <= ms_idx < spike_responses_pattern.shape[1]:
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


def encode_pattern_nominal_current_10ms(pattern_subset, channel_to_index, max_time_ms):
    """Encode a pattern in 'current' mode at 10ms bins using NOMINAL step starts
    (step_index * 60ms) instead of real timestamps. Pulses land at clean offsets
    relative to step starts: dm 0/-1 -> {0,2,4}, dm 1 -> {1,3,5}, dm 2 -> {0..5}.

    Returns: (n_channels, max_time_ms // 10) int8 array with values in {-3, 0, +3}.
    """
    n_channels = len(channel_to_index)
    bin_ms = 10
    total_bins = max_time_ms // bin_ms
    stim = np.zeros((n_channels, total_bins), dtype=np.int8)
    for _, row in pattern_subset.iterrows():
        if pd.isna(row['channel']):
            continue
        step_index = int(row['step_index'])
        if step_index * 60 >= max_time_ms:
            continue
        ch = channel_to_index[int(row['channel'])]
        delay_mode = int(row['delay_mode'])
        step_start_ms = step_index * 60
        if delay_mode == 0:
            offsets, amp = (0, 20, 40), 3
        elif delay_mode == 1:
            offsets, amp = (10, 30, 50), 3
        elif delay_mode == -1:
            offsets, amp = (0, 20, 40), -3
        elif delay_mode == 2:
            offsets, amp = (0, 10, 20, 30, 40, 50), 3
        else:
            continue
        for dt in offsets:
            b = (step_start_ms + dt) // bin_ms
            if 0 <= b < total_bins:
                stim[ch, b] = amp
    return stim


def build_pattern_stims_nominal(pattern_df, channel_to_index, max_time_ms):
    """Build a {pattern_name -> (n_channels, n_bins) int8} dict using nominal step
    starts so every 60ms step contains a clean pulse triplet at known offsets."""
    out = {}
    for pname in pattern_df['pattern_name'].unique():
        subset = pattern_df[pattern_df['pattern_name'] == pname].drop_duplicates(
            subset=['step_index', 'channel', 'delay_mode'])
        out[pname] = encode_pattern_nominal_current_10ms(subset, channel_to_index, max_time_ms)
    return out


def coarsen_2d(arr, factor, method='mean'):
    """Coarsen a 2-D array ``(rows, fine_bins)`` → ``(rows, coarse_bins)``.

    Parameters
    ----------
    arr : np.ndarray, shape ``(R, T)``
    factor : int  – coarsening factor (≤1 returns *arr* unchanged).
    method : str  – ``'mean'`` or ``'sum'``.
    """
    if factor <= 1:
        return arr
    R, T = arr.shape
    coarse = T // factor
    reshaped = arr[:, :coarse * factor].reshape(R, coarse, factor)
    return reshaped.sum(axis=2) if method == 'sum' else reshaped.mean(axis=2)


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
        encoding_mode: "categorical" (default) for categorical encoding of stim patterns
        trial_indices: list of pattern_timing_index values to include
        input_bin_size_ms: temporal resolution for inputs in milliseconds (e.g., 10)
        output_bin_size_ms: temporal resolution for outputs in milliseconds (e.g., 60)
        n_input_bins: number of consecutive stim bins to use as input (default 1)
        n_output_bins: number of consecutive spike bins to predict (default 1)
        max_time_ms: maximum time to consider (default 2000)
        output_offset: how many bins after input start to begin output (default 0 = same window)
        logger: optional logger instance for logging messages
    """
    def __init__(self, pattern_df, spike_responses, channel_to_index, timing_to_pattern, trial_indices=None, encoding_mode="categorical",
                 input_bin_size_ms=10, output_bin_size_ms=60, n_input_bins=1, n_output_bins=1,
                 max_time_ms=2000, output_offset=0, init_state=False, n_initial_state_bins=1, history=None, logger=None):
        
        
        if trial_indices is None:
            trial_indices = list(spike_responses.keys())
                
        self.trial_indices = trial_indices
        self.timing_to_pattern = timing_to_pattern
        self.encoding_mode = encoding_mode
        self.n_channels = len(channel_to_index)
        self.input_bin_size = input_bin_size_ms
        self.output_bin_size = output_bin_size_ms
        self.n_input_bins = n_input_bins
        self.n_output_bins = n_output_bins
        self.max_time_ms = max_time_ms
        self.output_offset = output_offset
        self.total_bins_input = max_time_ms // input_bin_size_ms
        self.total_bins_output = max_time_ms // output_bin_size_ms
        self._logger = logger
        self.init_state = init_state
        self.n_initial_state_bins = n_initial_state_bins
        self.history = history if history is not None else None
        # Pre-compute all binned stimulation patterns (one per unique pattern)
        # Shape: (n_channels, total_bins) with categorical indices
        self.pattern_stims = {}
        self.responses = {}
        unique_patterns = pattern_df['pattern_name'].unique()

        # Inputs are binned at input_bin_size_ms
        for pattern_name in unique_patterns:
            pattern_subset = pattern_df[pattern_df['pattern_name'] == pattern_name].drop_duplicates(
                subset=['step_index', 'channel', 'delay_mode'])
            self.pattern_stims[pattern_name] = self._encode_pattern_binned(
                pattern_subset, channel_to_index, input_bin_size_ms, max_time_ms)
            
        # Pre-compute all binned spike response with output bin size
        # Shape: (n_neurons, total_bins) with spike counts
        self.spike_responses_binned = {}
        non_specified = 0 
        for timing_idx in trial_indices:
            raw_spikes = spike_responses[timing_idx][:, :max_time_ms]
            self.spike_responses_binned[timing_idx] = self._bin_spikes(raw_spikes, output_bin_size_ms)
            if self.init_state and timing_idx - 1 not in trial_indices: # need response spikes for part of a previous trial
                if timing_idx == 0:
                    # repeat first bin as initial state if gl first trial
                    self.spike_responses_binned[timing_idx - 1] = self._bin_spikes(spike_responses[timing_idx], output_bin_size_ms)
                else:
                    self.spike_responses_binned[timing_idx - 1] = self._bin_spikes(spike_responses[timing_idx - 1], output_bin_size_ms)
                non_specified += 1
        print (f"{non_specified} trials were added in spike responses for initial state that were not in the specified trial_indices")
        
        # Calculate valid starting positions for each trial
        # Output ends at: t + output_offset + n_output_bins
        # So max valid t is: total_bins - output_offset - n_output_bins
        # Also need: t + n_input_bins <= total_bins
        max_start_for_output = self.total_bins_output - output_offset - n_output_bins
        max_start_for_input = self.total_bins_input - n_input_bins
        max_valid_start = min(max_start_for_output, max_start_for_input)
        
        # Generate (trial_idx, time_bin) pairs for all samples
        self.samples = []
        for timing_idx in trial_indices:
            for t in range(max_valid_start + 1): # because you can have a sample starting at multiple valid time bins within the same trial
                self.samples.append((timing_idx, t))
        
        # Get neuron count from first trial
        sample_key = trial_indices[0]
        self.n_neurons = spike_responses[sample_key].shape[0]
        
        self._log(f"BinnedStimSpikeDataset initialized:")
        self._log(f"  input_bin_size_ms={input_bin_size_ms}, output_bin_size_ms={output_bin_size_ms}")
        self._log(f"  n_input_bins={n_input_bins}, n_output_bins={n_output_bins}")
        self._log(f"  output_offset={output_offset}, total_bins_input={self.total_bins_input}, total_bins_output={self.total_bins_output}")
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
        assert bin_size_ms <= 60 # if bin size is more than 60, then we can have multiple stimulations per bin which would need to be encoded differently
        # Use the real step-start time (from step_start_timestamp relative to
        # pattern_flag_start_timestamp, converted from 30 kHz frames to ms)
        # instead of the nominal step_index * 60 ms. Pulses within the step are
        # still placed at their nominal intra-step offsets.
        def _step_start_ms(row):
            return (row['step_start_timestamp'] - row['pattern_flag_start_timestamp']) / 30.0

        if self.encoding_mode == "categorical":
            stim = np.full((n_channels, total_bins), NO_STIM_INDEX, dtype=np.int64)
            for _, row in pattern_subset.iterrows():
                if pd.isna(row['channel']):
                    continue
                step_index = int(row['step_index'])
                if step_index >= 10:  # Only first 10 steps (600ms of stimulation)
                    continue

                channel_index = channel_to_index[int(row['channel'])]
                delay_mode = int(row['delay_mode'])
                category_idx = DELAY_MODE_TO_INDEX[delay_mode]

                step_start_ms = _step_start_ms(row)
                step_end_ms   = step_start_ms + 60

                start_bin = int(step_start_ms // bin_size_ms)
                end_bin   = min(int(step_end_ms // bin_size_ms), total_bins)

                for b in range(max(0, start_bin), end_bin):
                    stim[channel_index, b] = category_idx
        elif self.encoding_mode == "current":
            stim = np.full((n_channels, total_bins), 0, dtype=np.int64)
            assert bin_size_ms == 10, "Current encoding only supported for 10ms bins"
            for _, row in pattern_subset.iterrows():
                if pd.isna(row['channel']):
                    continue

                channel_index = channel_to_index[int(row['channel'])]
                delay_mode = int(row['delay_mode'])

                step_start_ms = _step_start_ms(row)

                # Pulses within the step are assumed to fire at fixed offsets
                # from the real step start (nominal intra-step timing).
                if delay_mode == 0:               # 3 pulses @ 50 Hz, anodic
                    offsets, amp = (0, 20, 40), 3
                elif delay_mode == 1:             # 3 pulses @ 50 Hz, cathodic, +10 ms
                    offsets, amp = (10, 30, 50), -3
                elif delay_mode == -1:            # reverse phase of mode 0
                    offsets, amp = (0, 20, 40), -3
                elif delay_mode == 2:             # 6 pulses @ 100 Hz
                    offsets, amp = (0, 10, 20, 30, 40, 50), 3
                else:
                    continue

                for dt in offsets:
                    b = int((step_start_ms + dt) // bin_size_ms)
                    if 0 <= b < total_bins:
                        stim[channel_index, b] = amp
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
        stim_full = self.pattern_stims[self.timing_to_pattern[timing_idx]]  # (n_channels, total_bins_input)
        trial_spikes = self.spike_responses_binned[timing_idx]  # (n_neurons, total_bins_output)

        # Determine how many previous-trial spike bins to prepend.
        # For valid conv context we need n_initial_state_bins = sum(k-1).
        # For history lookback we need an additional self.history bins so
        # that even the first output bin has real (non-zero) history data.
        history_val = self.history if (self.history is not None and self.history > 0) else 0

        if self.init_state:
            # Stim: pad with n_initial_state_bins zeros (valid-conv context only)
            stim_full = np.pad(stim_full, ((0, 0), (self.n_initial_state_bins, 0)),
                               mode='constant', constant_values=0)

            # Spikes: prepend extra bins for history lookback on top of conv context
            spike_prepend = self.n_initial_state_bins + history_val
            prev_trial = self.spike_responses_binned[timing_idx - 1]
            available = prev_trial.shape[1]
            if spike_prepend <= available:
                previous_trial_spikes = prev_trial[:, -spike_prepend:]
            else:
                previous_trial_spikes = np.zeros((self.n_neurons, spike_prepend), dtype=np.float32)
                previous_trial_spikes[:, -available:] = prev_trial
            spikes_full = np.concatenate([previous_trial_spikes, trial_spikes], axis=1)

            x_width = self.n_initial_state_bins + self.n_input_bins
            
            # Offset between spike and stim indexing
            # stim_full[k] aligns with spikes_full[k + history_val]
            spike_offset = history_val
        else:
            spikes_full = trial_spikes
            x_width = self.n_input_bins
            spike_offset = 0

        x = stim_full[:, t : t + x_width].copy()  # (n_channels, x_width)
        out_start = t + self.output_offset + self.n_initial_state_bins + spike_offset if self.init_state else t + self.output_offset
        y = spikes_full[:, out_start : out_start + self.n_output_bins]  # (n_neurons, n_output_bins)

        if self.history is not None and self.history > 0:
            # Lagged spike history aligned with x's time positions.
            # With spike_offset, lag_start lands at valid indices in spikes_full
            # even at t=0 when init_state provides extra previous-trial context.
            n_time = x_width
            lag_start = t + spike_offset - self.history
            y_history = np.zeros((self.n_neurons, n_time), dtype=np.float32)
            if lag_start >= 0:
                y_history[:] = spikes_full[:, lag_start : lag_start + n_time]
            else:
                valid_from = -lag_start
                y_history[:, valid_from:] = spikes_full[:, 0 : n_time - valid_from]

            return (
                torch.cat([torch.tensor(x, dtype=torch.float32),
                           torch.tensor(y_history, dtype=torch.float32)], dim=0),
                torch.tensor(y, dtype=torch.float32)
            )
        else:
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        

 
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
            spikes = self.spike_responses_binned[timing_idx]  # (n_neurons, total_bins_output)
            # Count across valid output positions
            max_start = self.total_bins_output - self.output_offset - self.n_output_bins
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
            spikes = self.spike_responses_binned[timing_idx]  # (n_neurons, total_bins_output)
            max_start = self.total_bins_output - self.output_offset - self.n_output_bins
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

    def visualize_sample(self, idx, cfg=None, ax=None):
        """Visualize the input (X) and output (Y) produced by __getitem__ for a given index.

        Shows two vertically-aligned heatmaps:
          - Top: stimulation input channels over input bins
          - Bottom: spike output neurons over output bins
        If the model uses spike history, the input panel shows stim channels
        in the upper rows and history channels in the lower rows, separated
        by a horizontal line.

        When *cfg* is provided, overlays the convolutional receptive field
        (RF) for a few representative output bins so you can verify that
        the convolution is causal and that predictions align with the
        lagged input.

        Args:
            idx: sample index (0 to len-1)
            cfg: optional model config dict (must contain 'kernel_sizes';
                 'init_state' is read to choose valid-conv vs causal-pad mode).
            ax: optional pair of matplotlib Axes ``(ax_input, ax_output)``.
                If *None* a new figure is created.

        Returns:
            fig: the matplotlib Figure (or *None* when *ax* was provided)
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        timing_idx, t = self.samples[idx]
        pattern_name = self.timing_to_pattern[timing_idx]
        sample = self[idx]          # returns (x, y) or (x, y, init)
        x = sample[0].numpy()      # (channels, time_x)
        y = sample[1].numpy()      # (neurons, n_output_bins)

        has_history = self.history is not None and self.history > 0
        n_stim_channels = self.n_channels
        n_time_x = x.shape[1]

        # Time axes in ms
        x_bin_ms = self.input_bin_size
        y_bin_ms = self.output_bin_size
        history_lag = self.history if has_history else 0

        # Stim time extent
        stim_time_start = t * x_bin_ms
        if self.init_state:
            stim_time_start -= self.n_initial_state_bins * x_bin_ms
        stim_time_end = stim_time_start + n_time_x * x_bin_ms

        # History time extent — each column is shifted left by history bins
        hist_time_start = stim_time_start - history_lag * x_bin_ms
        hist_time_end = stim_time_end - history_lag * x_bin_ms

        # Backwards-compat aliases used by RF overlay code
        x_time_start = stim_time_start
        x_time_end = stim_time_end

        out_start_bin = t + self.output_offset
        y_time_start = out_start_bin * y_bin_ms
        y_time_end = y_time_start + self.n_output_bins * y_bin_ms

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)
        else:
            fig = None

        ax_in, ax_out = ax[0], ax[1]

        # --- Input panel ---
        if has_history:
            # Show stim and history channels with separate time extents
            # so each is displayed at its true temporal position.
            stim_part = x[:n_stim_channels, :]
            hist_part = x[n_stim_channels:, :]

            ax_in.imshow(stim_part, aspect='auto', interpolation='nearest',
                         cmap='binary', vmin=0,
                         extent=[stim_time_start, stim_time_end,
                                 n_stim_channels, 0])
            ax_in.imshow(hist_part, aspect='auto', interpolation='nearest',
                         cmap='binary', vmin=0,
                         extent=[hist_time_start, hist_time_end,
                                 x.shape[0], n_stim_channels])

            # Reset axis limits so both imshow calls are visible
            combined_xlo = min(stim_time_start, hist_time_start)
            combined_xhi = max(stim_time_end, hist_time_end)
            ax_in.set_xlim(combined_xlo, combined_xhi)
            ax_in.set_ylim(x.shape[0], 0)

            ax_in.axhline(y=n_stim_channels, color='red', ls='--', lw=1,
                          label='stim | history')
            ax_in.legend(loc='lower right', fontsize=7)
        else:
            ax_in.imshow(x, aspect='auto', interpolation='nearest',
                         cmap='binary', vmin=0,
                         extent=[stim_time_start, stim_time_end, x.shape[0], 0])
            
        ax_in.set_ylabel('Channel' if not has_history else 'Stim / History ch')
        ax_in.set_title(
            f'Input  —  sample {idx}  (timing_idx={timing_idx}, t={t}, '
            f'pattern={pattern_name})')

        # --- Output panel ---
        # Swapped to 'binary' and added vmin=0
        ax_out.imshow(y, aspect='auto', interpolation='nearest',
                      cmap='binary', vmin=0,
                      extent=[y_time_start, y_time_end, y.shape[0], 0])
                      
        ax_out.set_ylabel('Neuron')
        ax_out.set_xlabel('Time (ms)')
        ax_out.set_title(
            f'Output  —  {self.n_output_bins} bins × {y_bin_ms}ms  '
            f'(offset={self.output_offset})')
        # --- Receptive-field overlay (when cfg is provided) ---
        if cfg is not None:
            from matplotlib.patches import ConnectionPatch
            kernel_sizes = cfg.get('kernel_sizes', [3])
            use_init_state = cfg.get('init_state', False)
            history_lag = self.history if has_history else 0
            RF = 1 + sum(k - 1 for k in kernel_sizes)  # receptive field in bins
            n_out = self.n_output_bins

            # Pick a few output bins to highlight (up to 3)
            if n_out <= 3:


                highlight_bins = list(range(n_out))
            else:
                highlight_bins = np.linspace(0, n_out - 1, 3, dtype=int).tolist()

            cmap_hl = plt.cm.tab10

            for i, o in enumerate(highlight_bins):
                color = cmap_hl(i / max(len(highlight_bins) - 1, 1))

                if use_init_state:
                    rf_lo = o
                    rf_hi = o + RF
                else:
                    rf_lo = o - RF + 1
                    rf_hi = o + 1

                # Convert bin indices to ms in the input panel's coordinate frame
                rf_lo_ms = x_time_start + rf_lo * x_bin_ms
                rf_hi_ms = x_time_start + rf_hi * x_bin_ms

                # Output vertical line position — at bin end to align with RF right edge
                out_end_ms = y_time_start + (o + 1) * y_bin_ms

                if has_history and history_lag > 0:
                    # --- Stim channels: RF at input column positions ---
                    rect_stim = Rectangle(
                        (rf_lo_ms, 0), rf_hi_ms - rf_lo_ms, n_stim_channels,
                        linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.15)
                    ax_in.add_patch(rect_stim)

                    # --- History channels: displaced left by history * bin_ms ---
                    hist_offset_ms = history_lag * x_bin_ms
                    hist_rf_lo_ms = rf_lo_ms - hist_offset_ms
                    hist_rf_hi_ms = rf_hi_ms - hist_offset_ms
                    rect_hist = Rectangle(
                        (hist_rf_lo_ms, n_stim_channels),
                        hist_rf_hi_ms - hist_rf_lo_ms,
                        x.shape[0] - n_stim_channels,
                        linewidth=1.5, edgecolor=color, facecolor=color,
                        alpha=0.15, linestyle='--')
                    ax_in.add_patch(rect_hist)

                else:
                    # No history — single rectangle spanning all channels
                    rect = Rectangle(
                        (rf_lo_ms, 0), rf_hi_ms - rf_lo_ms, x.shape[0],
                        linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.12)
                    ax_in.add_patch(rect)

                # Matching vertical line on output panel at this output bin
                ax_out.axvline(out_end_ms, color=color, ls='--', lw=1.2, alpha=0.6)

                # Connecting line: right edge of stim RF (bottom of input) → output line (top of output)
                con_stim = ConnectionPatch(
                    xyA=(rf_hi_ms, x.shape[0]), coordsA=ax_in.transData,
                    xyB=(out_end_ms, 0), coordsB=ax_out.transData,
                    color=color, lw=1, alpha=0.5, ls=':')
                fig.add_artist(con_stim)

                # Connecting line: right edge of history RF → output line
                if has_history and history_lag > 0:
                    con_hist = ConnectionPatch(
                        xyA=(hist_rf_hi_ms, x.shape[0]), coordsA=ax_in.transData,
                        xyB=(out_end_ms, 0), coordsB=ax_out.transData,
                        color=color, lw=1, alpha=0.5, ls='--')
                    fig.add_artist(con_hist)

                # Small tick label at top of input panel
                ax_in.annotate(
                    f'o={o}', xy=(rf_hi_ms, 0), fontsize=6, color=color,
                    ha='right', va='bottom', rotation=90)

            # Align x-axes — extend left to include history data extent
            xlim_lo = min(x_time_start, y_time_start)
            if has_history and history_lag > 0:
                xlim_lo = min(xlim_lo, hist_time_start)
            xlim_hi = max(x_time_end, y_time_end)
            ax_in.set_xlim(xlim_lo, xlim_hi)
            ax_out.set_xlim(xlim_lo, xlim_hi)

            mode_str = 'valid (init_state)' if use_init_state else 'causal pad'
            hist_str = ''
            if has_history:
                hist_str = (f', history={history_lag} '
                            f'(lag={history_lag * x_bin_ms}ms)')
            ax_in.set_title(
                f'Input  —  sample {idx}  (timing_idx={timing_idx}, t={t}, '
                f'pattern={pattern_name})\n'
                f'RF={RF} bins ({RF * x_bin_ms}ms), '
                f'kernels={kernel_sizes}, mode={mode_str}{hist_str}')

        if created_fig:
            plt.show()
        return fig


def open_model_and_data(results_dir, n_stim_channels=42, n_neurons=63):
    model, _, _ = load_model_from_run_dir(results_dir, n_stim_channels=n_stim_channels, n_neurons=n_neurons)  
    with open(os.path.join(results_dir, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
        output_bin_size_ms = cfg['output_bin_size_ms']
        max_time_ms = cfg['max_time_ms']

        raw_data = load_raw_data(cfg, logger=None) # pattern df, spike_responses, etc. 

        # Auto-compute n_initial_state_bins for CNN + init_state
        # Only valid-conv reduction; history lookback is handled in __getitem__
        n_initial_state_bins = cfg.get('n_initial_state_bins', 1)
        if cfg.get('init_state', False) and cfg.get('model_type', 'cnn') == 'cnn':
            _ks = cfg.get('kernel_sizes', [60])
            n_initial_state_bins = sum(k - 1 for k in _ks)

        history_dataset_kwargs = dict(
            pattern_df=raw_data["pattern_df"],
            spike_responses=raw_data["spike_responses"],
            channel_to_index=raw_data["channel_to_index"],
            timing_to_pattern=raw_data["timing_to_pattern"],
            input_bin_size_ms=cfg['input_bin_size_ms'],
            output_bin_size_ms=cfg['output_bin_size_ms'],
            n_input_bins=cfg['n_input_bins'],
            n_output_bins=cfg['n_output_bins'],
            max_time_ms=cfg['max_time_ms'],
            output_offset=cfg['output_offset'],
            encoding_mode=cfg['encoding_mode'],
            init_state=cfg['init_state'],
            n_initial_state_bins=n_initial_state_bins,
            history=cfg['history'],
            logger=None,
        )


    unique_trials_info = raw_data["pattern_df"][["pattern_timing_index", "pattern_name", "is_oracle"]].drop_duplicates()
    oracle_timing = unique_trials_info[unique_trials_info["is_oracle"]]["pattern_timing_index"].tolist()
    test_indices = oracle_timing
    test_dataset = BinnedStimSpikeDataset(trial_indices=test_indices, **history_dataset_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    return model, cfg, raw_data, test_loader


def get_oracle_trial_response_and_loo_average(raw_data, pattern_name, output_bin_size_ms=10, max_time_ms=600):
    spike_responses = raw_data["spike_responses"]
    pattern_df = raw_data["pattern_df"]
    timing_list = sorted(pattern_df[pattern_df['pattern_name'] == pattern_name]['pattern_timing_index'].tolist())
    for trial_idx in range(len(timing_list)):
        timing_idx = timing_list[trial_idx]
        
        # Actual spike response - binned (spike_responses are in 1ms resolution)
        actual_response_binned = bin_spike_response(spike_responses[timing_idx], output_bin_size_ms, max_time_ms, remainder="append")
        
        # Average spikes across other trials
        other_trial_indices = [t for t in timing_list if t != timing_idx]
        avg_other_trials = compute_avg_spikes_across_trials(
            other_trial_indices, spike_responses, output_bin_size_ms, max_time_ms
            )
    return actual_response_binned, avg_other_trials


def load_model_from_run_dir(run_dir, n_stim_channels, n_neurons, device=None):
    """Load a trained model from an experiment run directory.

    Parameters
    ----------
    run_dir : str
        Path to the experiment directory containing ``config.yaml`` and
        ``best_stim_spike_model.pt``.
    n_stim_channels : int
        Number of stimulation channels (from dataset / raw data).
    n_neurons : int
        Number of neurons (from  / raw data).
    device : torch.device, optional
        Device to place the model on.  Defaults to CPU.

    Returns
    -------
    model : nn.Module
        The loaded model in eval mode.
    cfg : dict
        The experiment config.
    device : torch.device
    """
    cfg_path = os.path.join(run_dir, "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if device is None:
        device = torch.device("cpu")

    # --- Derive model input channels ---
    ENCODING_MODE = cfg.get("encoding_mode", "current")
    HISTORY = cfg.get("history", 0)
    INIT_STATE = cfg.get("init_state", False)
    N_INPUT_BINS = cfg["n_input_bins"]
    N_OUTPUT_BINS = cfg["n_output_bins"]
    N_INITIAL_STATE_BINS = cfg.get("n_initial_state_bins", 1)

    if ENCODING_MODE == "current" and HISTORY is not None and HISTORY > 0:
        n_model_input_channels = n_stim_channels + n_neurons
    else:
        n_model_input_channels = n_stim_channels

    MODEL_TYPE = cfg.get("model_type", "cnn")

    # For CNN + init_state: auto-compute n_initial_state_bins
    # Only the valid-conv reduction; history lookback is handled in __getitem__
    if INIT_STATE and MODEL_TYPE == "cnn":
        _kernel_sizes = cfg.get("kernel_sizes", [60])
        N_INITIAL_STATE_BINS = sum(k - 1 for k in _kernel_sizes)

    # --- Build model ---
    EMBEDDING_DIM = cfg.get("embedding_dim", 0)
    DROPOUT = cfg.get("dropout", 0.2)
    CONV_CHANNELS = cfg.get("conv_channels", [128])
    KERNEL_SIZES = cfg.get("kernel_sizes", [60])
    FC_DIMS = cfg.get("fc_dims", [256])

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
        )
    else:
        raise ValueError(f"Unknown model_type: {MODEL_TYPE}")

    # --- Load weights ---
    weights_path = os.path.join(run_dir, "best_stim_spike_model.pt")
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
    model = model.to(device)
    model.eval()

    return model, cfg, device




from numpy.lib.stride_tricks import sliding_window_view
from collections import defaultdict
import numpy as np


from numpy.lib.stride_tricks import sliding_window_view
from collections import defaultdict
import numpy as np

def build_encoder_dataset(spike_responses, timing_to_pattern, pattern_stims, pattern_polarities,
                  oracle_pattern_indices, *,
                  WINDOW_SIZE, SPIKE_LATENCY=0, MAX_TIME_MS,
                  MODE='step', OVERLAPS=False,
                  n_stim_channels, n_amplitudes, n_neurons,
                  oracle_mode='per_trial'):
    """
    Build (X, Y) train/test arrays.

    oracle_mode controls how oracle trials are used in training:
      - 'per_trial':  each oracle trial is its own training row (test = empty).
      - 'averaged':   oracle trials are averaged per pname (test = empty).
                      ** test set will be empty in this mode -- caller's responsibility
                         to know that. (Per the spec, test is not averaged oracles.)
      - 'leftout':    oracle trials are excluded from training entirely; test set is
                      every oracle trial, per-trial.

    Non-oracle trials (timing_idx not in test_indices) always go into training, per-trial.

    Returns dict with: train_X, train_Y, train_timings, test_X, test_Y, test_timings.
    """
    if oracle_mode not in ('per_trial', 'averaged', 'leftout'):
        raise ValueError(f"oracle_mode must be one of 'per_trial', 'averaged', 'leftout'; got {oracle_mode!r}")

    delay_mode_to_lridx = {-1: 0, 0: 1, 1: 2, 2: 3}

    def encode_one(timing_idx):
        spk   = spike_responses[timing_idx]
        pname = timing_to_pattern[timing_idx]
        win   = spk[:, SPIKE_LATENCY:SPIKE_LATENCY + MAX_TIME_MS]
        stim  = pattern_stims[pname]
        pol   = pattern_polarities[pname]
        signed_stim = stim * pol

        if MODE == 'step':
            if WINDOW_SIZE % 60 != 0:
                raise ValueError('For step mode, WINDOW_SIZE must be a multiple of 60ms.')
            stim_one_hot = np.zeros((n_stim_channels * 4, signed_stim.shape[1] // 60))
            for step in range(0, signed_stim.shape[1], 60):
                for ch in range(n_stim_channels):
                    dm = None
                    # dm=2: 100Hz anodic. Synthesizer writes +3 every 10ms, so step and step+10 are both +3.
                    if signed_stim[ch, step] == 3 and signed_stim[ch, step + 10] == 3:
                        dm = 2
                    # dm=0: 50Hz anodic. +3 at step start, step+10 empty (next pulse at step+20).
                    elif signed_stim[ch, step] == 3:
                        dm = 0
                    # dm=1: 50Hz anodic, delayed 10ms. step empty, +3 at step+10.
                    elif signed_stim[ch, step + 10] == 3:
                        dm = 1
                    # dm=-1: 50Hz cathodic. -3 at step start.
                    elif signed_stim[ch, step] == -3:
                        dm = -1
                    if dm is not None:
                        stim_one_hot[ch * 4 + delay_mode_to_lridx[dm], step // 60] = 1
        elif MODE == 'pulse':
            stim_one_hot = np.zeros((n_stim_channels * n_amplitudes, win.shape[1]))
            for ch in range(n_stim_channels):
                for a_idx, amp_val in enumerate([-3, 3]):
                    stim_one_hot[ch * n_amplitudes + a_idx, :] = (signed_stim[ch, :] == amp_val)
        else:
            raise ValueError(f'unknown MODE {MODE!r}')

        if OVERLAPS:
            # Sample at step boundaries so spike-window count == stim-window count.
            spk_view = sliding_window_view(win, window_shape=WINDOW_SIZE, axis=1)[:, ::60, :]
            Y = spk_view.sum(axis=-1).T
            if MODE == 'step':
                stim_view = sliding_window_view(stim_one_hot, window_shape=WINDOW_SIZE // 60, axis=1)
                X = stim_view.sum(axis=-1).T
            else:
                stim_view = sliding_window_view(stim_one_hot, window_shape=WINDOW_SIZE, axis=1)[:, ::60, :]
                X = stim_view.sum(axis=-1).T
        else:
            Y = win.reshape(n_neurons, -1, WINDOW_SIZE).sum(axis=-1).T
            if MODE == 'step':
                X = stim_one_hot.reshape(stim_one_hot.shape[0], -1, WINDOW_SIZE // 60).sum(axis=-1).T
            else:
                X = stim_one_hot.reshape(stim_one_hot.shape[0], -1, WINDOW_SIZE).sum(axis=-1).T
        return X, Y

    oracle_set = set(oracle_pattern_indices)

    # --- Non-oracle trials -> training (per-trial) ---
    nonoracle_X, nonoracle_Y, nonoracle_T = [], [], []
    # --- Oracle trials -> grouped by pname ---
    oracle_X_by_pname = {}                   # pname -> X (identical across trials)
    oracle_Y_by_pname = defaultdict(list)
    oracle_timing_by_pname = defaultdict(list)

    for timing_idx in spike_responses.keys():
        X, Y = encode_one(timing_idx)
        if timing_idx in oracle_set:
            pname = timing_to_pattern[timing_idx]
            oracle_X_by_pname[pname] = X
            oracle_Y_by_pname[pname].append(Y)
            oracle_timing_by_pname[pname].append(timing_idx)
        else:
            nonoracle_X.append(X)
            nonoracle_Y.append(Y)
            nonoracle_T.extend([timing_idx] * Y.shape[0])

    train_X_blocks, train_Y_blocks, train_T = list(nonoracle_X), list(nonoracle_Y), list(nonoracle_T)
    test_X_blocks,  test_Y_blocks,  test_T  = [], [], []

    pnames = list(oracle_X_by_pname.keys())

    if oracle_mode == 'per_trial':
        for p in pnames:
            ys = oracle_Y_by_pname[p]
            n_win = ys[0].shape[0]
            Y_block = np.vstack(ys)
            X_block = np.tile(oracle_X_by_pname[p], (len(ys), 1))
            train_X_blocks.append(X_block)
            train_Y_blocks.append(Y_block)
            for ti in oracle_timing_by_pname[p]:
                train_T.extend([ti] * n_win)              # repeat per window
        # test is empty
    elif oracle_mode == 'averaged':
        for p in pnames:
            ys = oracle_Y_by_pname[p]
            Y_avg = np.mean(np.stack(ys, axis=0), axis=0)   # (n_win, n_neurons)
            n_win = Y_avg.shape[0]
            train_X_blocks.append(oracle_X_by_pname[p])
            train_Y_blocks.append(Y_avg)
            train_T.extend([oracle_timing_by_pname[p][0]] * n_win)
        # test is empty (per-spec: test is not averaged oracles)
    elif oracle_mode == 'leftout':
        for p in pnames:
            ys = oracle_Y_by_pname[p]
            n_win = ys[0].shape[0]
            Y_block = np.vstack(ys)
            X_block = np.tile(oracle_X_by_pname[p], (len(ys), 1))
            test_X_blocks.append(X_block)
            test_Y_blocks.append(Y_block)
            for ti in oracle_timing_by_pname[p]:
                test_T.extend([ti] * n_win)

    def _stack(Xs, Ys, Ts):
        if not Xs:
            return (np.zeros((0, 0)), np.zeros((0, 0)), np.array([]))
        return np.vstack(Xs), np.vstack(Ys), np.array(Ts)

    train_X, train_Y, train_T = _stack(train_X_blocks, train_Y_blocks, train_T)
    test_X,  test_Y,  test_T  = _stack(test_X_blocks,  test_Y_blocks,  test_T)

    return {
        'train_X': train_X, 'train_Y': train_Y, 'train_timings': train_T,
        'test_X':  test_X,  'test_Y':  test_Y,  'test_timings':  test_T,
        'oracle_pnames': pnames,
    }

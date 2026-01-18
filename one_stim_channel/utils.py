import os
import sys
import datetime
import logging
import random
import pandas as pd


import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm, colors
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
import scipy
import scipy.io



def set_seed(seed: int = 42) -> None:
    """Sets the random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If using CUDA (GPU), also set the seed for the current GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # For deterministic behavior on CuDNN backend
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")



def make_snippets(activity, stim_times_sess, length, overlap=True, aligned=False):
    '''
    Create snippets of activity and stimulation data for RNN training.
    Each snippet is of size (length, n_rois) for activity and
    (length, n_electrodes) for stimulation.
    :param activity:
    :param stim_times_sess:
    :param length:
    :param overlap:
    :param aligned:
    :return:
    '''
    initial_conditions = []
    activity_snippets = []
    stim_snippets = []
    if aligned:
        raise NotImplementedError("Not yet implemented!")
    for session_id in range(3):
        activity_from_session = activity[session_id, ...]
        stimulation_from_session = stim_times_sess[session_id, ...]
        # overlapping snippets of size snippet_length
        if overlap:
            num_snippets = activity_from_session.shape[0] - length + 1
            print (num_snippets)
            for i in range(num_snippets):
                if i == 0: # for first timepoint, assume initial condition is identical to first activity frame 
                    initial_conditions.append(activity_from_session[0])
                else:
                    initial_conditions.append(activity_from_session[i-1])
                activity_snippets.append(activity_from_session[i:i + length])
                stim_snippets.append(stimulation_from_session[i:i + length])
        else:
            for i in range(0, activity_from_session.shape[0] - length, length + 1):
                if i == 0:
                    initial_conditions.append(activity_from_session[0])
                else:
                    initial_conditions.append(activity_from_session[i-1])
                activity_snippets.append(activity_from_session[i:i + length])
                stim_snippets.append(stimulation_from_session[i:i + length])

    initial_conditions = np.array(initial_conditions)
    activity_snippets = np.array(activity_snippets)
    stim_snippets = np.array(stim_snippets)
    return initial_conditions, activity_snippets, stim_snippets


def make_snippets_df(trials_df, activity, length, overlap=True, stride=1, n_electrodes=10):
    """
    Create snippets of activity and stimulation data with full metadata.
    
    Stimulation timing is derived directly from trials_df['stim_time'], which should be
    trial_start_time + STIM_DELAY. This avoids relying on a separate stim_times_sess array.
    
    Two modes:
    1. overlap=True (default): Creates overlapping snippets from the entire timeseries.
       Labels each snippet based on the FIRST stimulation within the snippet window.
       Also tracks ALL stimulations in the snippet for holdout filtering.
    2. overlap=False: Only creates snippets aligned to trial start times (one per trial).
    
    Parameters
    ----------
    trials_df : pd.DataFrame
        DataFrame with columns: session, trial, config, electrode, current, start_time, 
        stim_time, is_stim
    activity : ndarray, shape (n_sessions, n_times, n_rois)
        Activity data (dfof) for each session
    length : int
        Snippet length in frames
    overlap : bool
        If True, create overlapping snippets with given stride. If False, only stim-aligned.
    stride : int
        Step size between snippet starts when overlap=True (default=1 for full overlap)
    n_electrodes : int
        Number of electrodes (default=10)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - session, snippet_start: location info
        - first_config, first_electrode, first_current, first_trial: metadata for FIRST stim
        - all_configs, all_electrodes, all_currents, all_trials: lists of ALL stims in snippet
        - has_stim: whether snippet contains any stimulation
        - initial_condition, activity_snippet, stim_snippet: data arrays
        - valid: bool
    """
    # Build a lookup from (session, stim_time) -> trial metadata
    # stim_time = trial_start_time + STIM_DELAY, so this is the actual time when stim occurs
    stim_lookup = {}
    for _, row in trials_df.iterrows():
        if row['is_stim'] or row['config'] == 31:  # Only add actual stim trials
            key = (row['session'], row['stim_time'])
            stim_lookup[key] = {
                'config': row['config'],
                'electrode': row['electrode'],
                'current': row['current'],
                'trial': row['trial'],
                'is_stim': row['is_stim']
            }
    
    def build_stim_snippet(session, snippet_start, length):
        """Build stim_snippet array from trials_df stim_time values."""
        stim_snippet = np.zeros((length, n_electrodes))
        all_configs = []
        all_electrodes = []
        all_currents = []
        all_trials = []
        all_stim_times_in_snippet = []
        
        snippet_end = snippet_start + length
        for t_abs in range(snippet_start, snippet_end):
            key = (session, t_abs)
            if key in stim_lookup:
                info = stim_lookup[key]
                t_rel = t_abs - snippet_start
                electrode = info['electrode']
                current = info['current']
                if electrode is not None:
                    stim_snippet[t_rel, int(electrode)] = current
                else: # config 31
                    stim_snippet[t_rel, :] = 0  # no stimulation

                all_configs.append(info['config'])
                all_electrodes.append(electrode)
                all_currents.append(current)
                all_trials.append(info['trial'])
                all_stim_times_in_snippet.append(t_rel)
        
        return stim_snippet, all_configs, all_electrodes, all_currents, all_trials, all_stim_times_in_snippet
    
    results = []
    if overlap:
        # Create overlapping snippets from entire timeseries
        for session in range(activity.shape[0]):
            n_times = activity[session].shape[0]
            
            for i in range(0, n_times - length + 1, stride):
                # Get initial condition
                if i == 0:
                    initial_cond = activity[session][0, :]
                else:
                    initial_cond = activity[session][i - 1, :]
                
                # Get activity snippet
                activity_snippet = activity[session][i:i + length, :]
                # Build stim snippet from trials_df
                stim_snippet, all_configs, all_electrodes, all_currents, all_trials, all_stim_times = \
                    build_stim_snippet(session, i, length)
                
                # Determine first stim metadata (if any)
                has_stim = len(all_configs) > 0
                if has_stim:
                    # Sort by time to get first stim
                    first_idx = np.argmin(all_stim_times)
                    first_config = all_configs[first_idx]
                    first_electrode = all_electrodes[first_idx]
                    first_current = all_currents[first_idx]
                    first_trial = all_trials[first_idx]
                    first_stim_time = all_stim_times[first_idx]
                else:
                    first_config = np.nan
                    first_electrode = np.nan
                    first_current = 0
                    first_trial = np.nan
                    first_stim_time = np.nan
                
                results.append({
                    'session': session,
                    'snippet_start': i,
                    # First stim metadata (for primary labeling)
                    'first_config': first_config,
                    'first_electrode': first_electrode,
                    'first_current': first_current,
                    'first_trial': first_trial,
                    'first_stim_time': first_stim_time,  # relative to snippet start
                    # All stims in snippet (for holdout filtering)
                    'all_configs': all_configs if all_configs else [],
                    'all_electrodes': all_electrodes if all_electrodes else [],
                    'all_currents': all_currents if all_currents else [],
                    'all_trials': all_trials if all_trials else [],
                    # Flags
                    'has_stim': has_stim,
                    'stim_at_t0': first_stim_time == 0 if has_stim else False,
                    'num_stims': len(all_configs),
                    # Data
                    'initial_condition': initial_cond,
                    'activity_snippet': activity_snippet,
                    'stim_snippet': stim_snippet,
                    'valid': True
                })
    else:
        # Only create snippets aligned to trial start times
        for idx, row in trials_df.iterrows():
            session = row['session']
            start_time = row['start_time']
            stim_time = row['stim_time']
            
            n_times = activity[session].shape[0]
            
            # Check if snippet fits within session
            if start_time + length > n_times:
                results.append({
                    'session': session,
                    'snippet_start': start_time,
                    'first_config': row['config'],
                    'first_electrode': row['electrode'],
                    'first_current': row['current'],
                    'first_trial': row['trial'],
                    'first_stim_time': stim_time - start_time if row['is_stim'] else -1,
                    'all_configs': [row['config']] if row['is_stim'] else [],
                    'all_electrodes': [row['electrode']] if row['is_stim'] else [],
                    'all_currents': [row['current']] if row['is_stim'] else [],
                    'all_trials': [row['trial']],
                    'has_stim': row['is_stim'],
                    'stim_at_t0': (stim_time - start_time) == 0 if row['is_stim'] else False,
                    'num_stims': 1 if row['is_stim'] else 0,
                    'initial_condition': None,
                    'activity_snippet': None,
                    'stim_snippet': None,
                    'valid': False
                })
                continue
            
            # Get initial condition
            if start_time == 0:
                initial_cond = activity[session][0, :]
            else:
                initial_cond = activity[session][start_time - 1, :]
            
            # Get activity snippet
            activity_snippet = activity[session][start_time:start_time + length, :]
            # Build stim snippet from trials_df
            stim_snippet, all_configs, all_electrodes, all_currents, all_trials, all_stim_times = \
                build_stim_snippet(session, start_time, length)
            # The primary stim for this snippet (from this trial's row)
            if row['is_stim']: 
                stim_time_rel = stim_time - start_time
            # Determine first stim info (might be from this trial or another)
            has_stim = len(all_configs) > 0
            if has_stim:
                first_idx = np.argmin(all_stim_times)
                first_stim_time = all_stim_times[first_idx]
            else:
                first_stim_time = -1
            
            results.append({
                'session': session,
                'snippet_start': start_time,
                'first_config': row['config'] if row['is_stim'] else (all_configs[0] if has_stim else 31),
                'first_electrode': row['electrode'] if row['is_stim'] else (all_electrodes[0] if has_stim else np.nan),
                'first_current': row['current'] if row['is_stim'] else (all_currents[0] if has_stim else 0),
                'first_trial': row['trial'],
                'first_stim_time': stim_time_rel if row['is_stim'] else np.nan,
                'all_configs': all_configs,
                'all_electrodes': all_electrodes,
                'all_currents': all_currents,
                'all_trials': all_trials,
                'has_stim': has_stim or row['is_stim'],
                'stim_at_t0': stim_time_rel == 0 if row['is_stim'] else False,
                'num_stims': len(all_configs),
                'initial_condition': initial_cond,
                'activity_snippet': activity_snippet,
                'stim_snippet': stim_snippet,
                'valid': True
            })
    
    return pd.DataFrame(results)


def snippets_df_to_arrays(snippets_df):
    """
    Convert a snippets DataFrame to arrays for model training.
    
    Parameters
    ----------
    snippets_df : pd.DataFrame
        DataFrame from make_snippets_df with valid snippets
        
    Returns
    -------
    initial_conditions : ndarray, shape (n_samples, n_rois)
    activity_snippets : ndarray, shape (n_samples, length, n_rois)
    stim_snippets : ndarray, shape (n_samples, length, n_electrodes)
    """
    valid_df = snippets_df[snippets_df['valid']].copy()
    
    initial_conditions = np.stack(valid_df['initial_condition'].values)
    activity_snippets = np.stack(valid_df['activity_snippet'].values)
    stim_snippets = np.stack(valid_df['stim_snippet'].values)
    
    return initial_conditions, activity_snippets, stim_snippets


def get_next_versioned_directory(base_dir_name='rnn_training'):
    """
    Creates a time-stamped directory, checking for existing versions.
    If 'rnn_training/20251218_211900' exists, it creates 'rnn_training/20251218_211900_2'.
    """
    # 1. Get the base timestamp string
    DATETIME_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2. Start with the base directory path
    base_path = os.path.join(base_dir_name, DATETIME_STAMP)

    # 3. Initialize path and version counter
    current_path = base_path
    # 4. Create the new, unique directory
    os.makedirs(current_path)

    # 6. Return the path of the created directory
    return current_path

# --- Usage Example ---
new_dir_path = get_next_versioned_directory('rnn_training')
print(f"Created new directory: {new_dir_path}")


def partition_trials(
    initial_conds,
    stim_snippets,
    activity_snippets,
    SNIPPET_LENGTH,
    val_size=0.15,
    test_size=0.15,
    method="timeorder",
):
    """
    Partition trials into train / val / test with no temporal overlap.

    Ensures that snippets in different sets cannot overlap in time by
    inserting SNIPPET_LENGTH gaps at boundaries.

    Returns:
        (init_train, stim_train, act_train,
         init_val,   stim_val,   act_val,
         init_test,  stim_test,  act_test)
    """

    # -------------------- Sanity checks --------------------
    assert initial_conds.shape[0] == stim_snippets.shape[0] == activity_snippets.shape[0]
    assert 0 < val_size < 1
    assert 0 < test_size < 1
    assert val_size + test_size < 1
    assert SNIPPET_LENGTH >= 1

    n_trials = activity_snippets.shape[0]

    if method != "timeorder":
        raise ValueError("Only method='timeorder' is supported.")

    # -------------------- Compute split indices --------------------
    n_test = int(round(n_trials * test_size))
    n_val  = int(round(n_trials * val_size))

    test_start = n_trials - n_test
    val_start  = test_start - n_val

    # -------------------- Enforce non-overlap --------------------
    # Remove SNIPPET_LENGTH samples around boundaries
    train_end = val_start - SNIPPET_LENGTH
    val_end   = test_start - SNIPPET_LENGTH

    if train_end <= 0 or val_end <= val_start:
        raise ValueError(
            "Not enough data to enforce non-overlapping splits. "
            "Reduce SNIPPET_LENGTH or val/test sizes."
        )

    # -------------------- Slice data --------------------
    init_train = initial_conds[:train_end]
    stim_train = stim_snippets[:train_end]
    act_train  = activity_snippets[:train_end]

    init_val = initial_conds[val_start:val_end]
    stim_val = stim_snippets[val_start:val_end]
    act_val  = activity_snippets[val_start:val_end]

    init_test = initial_conds[test_start:]
    stim_test = stim_snippets[test_start:]
    act_test  = activity_snippets[test_start:]

    return (
        init_train, stim_train, act_train,
        init_val,   stim_val,   act_val,
        init_test,  stim_test,  act_test,
    )



def collate_to_device(batch, device):
    # First unzip samples
    inputs_and_init, targets = zip(*batch)

    # Now unzip the nested tuple
    inputs, activity_initial = zip(*inputs_and_init)

    inputs = torch.stack(inputs).to(device, non_blocking=True)
    activity_initial = torch.stack(activity_initial).to(device, non_blocking=True)
    targets = torch.stack(targets).to(device, non_blocking=True)

    return (inputs, activity_initial), targets


# ============================
# ROI activation video (neutral background)
# ============================

from matplotlib import animation


def create_roi_activation_video(
    dfof,
    label_mat,
    save_path="roi_activation.mp4",
    fps=10,
    dpi=60,
    frame_step=5,
    start_frame=0,
    end_frame=None,
    normalize=True,
    cmap="hot",
    figsize=(3, 3),
):
    """
    Create an animated video of ROI activations over time.
    
    Parameters
    ----------
    dfof : ndarray, shape (n_rois, n_timepoints)
        Activity data (dF/F) for each ROI. Rows are ROIs, columns are timepoints.
    label_mat : ndarray, shape (height, width)
        2D array where each pixel is labeled with its ROI ID (0 = background).
    save_path : str
        Path to save the output video (e.g., 'output.mp4').
    fps : int
        Frames per second for the output video.
    dpi : int
        Resolution (dots per inch) for the output video.
    frame_step : int
        Step size between frames (e.g., 5 means every 5th frame).
    start_frame : int
        First frame to include in the animation.
    end_frame : int or None
        Last frame to include (exclusive). If None, uses all frames.
    normalize : bool
        If True, z-score normalize each ROI's activity.
    cmap : str
        Colormap for ROI activation visualization.
    figsize : tuple
        Figure size in inches (width, height).
        
    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The animation object (also saved to save_path).
    """
    dfof = np.array(dfof, dtype=float)
    label_mat = np.squeeze(np.array(label_mat))
    
    # Ensure dfof is (n_rois, n_timepoints)
    n_rois = len(np.unique(label_mat)) - 1  # exclude background (0)
    if dfof.shape[0] != n_rois and dfof.shape[1] == n_rois:
        dfof = dfof.T
    
    n_timepoints = dfof.shape[1]
    
    if end_frame is None:
        end_frame = n_timepoints
    
    print(f"Creating video: {n_rois} ROIs, frames {start_frame}-{end_frame} (step={frame_step})")
    
    # Normalize per ROI if requested
    if normalize:
        mean = dfof.mean(axis=1, keepdims=True)
        std = dfof.std(axis=1, keepdims=True)
        dfof = (dfof - mean) / (std + 1e-8)
    
    # ROI masks
    roi_ids = np.unique(label_mat)
    roi_ids = roi_ids[roi_ids > 0]
    
    roi_masks = {roi: (label_mat == roi) for roi in roi_ids}
    roi_mask_all = label_mat > 0
    
    # Figure setup
    fig, ax = plt.subplots(figsize=figsize)
    
    # Background (neutral gray)
    ax.imshow(
        np.zeros_like(label_mat),
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    
    # Foreground ROI activation (masked outside ROIs)
    img = ax.imshow(
        np.ma.masked_where(~roi_mask_all, np.zeros_like(label_mat, dtype=float)),
        cmap=cmap,
        vmin=np.percentile(dfof, 5),
        vmax=np.percentile(dfof, 95),
    )
    
    ax.axis("off")
    cbar = plt.colorbar(img, ax=ax, fraction=0.046)
    cbar.set_label("ΔF/F" + (" (z-scored)" if normalize else ""))
    
    title = ax.set_title("")
    
    # Update function
    def update(frame):
        activation = np.zeros_like(label_mat, dtype=float)
        
        for roi in roi_ids:
            roi_idx = int(roi) - 1
            if roi_idx < dfof.shape[0]:
                activation[roi_masks[roi]] = dfof[roi_idx, frame]
        
        img.set_data(np.ma.masked_where(~roi_mask_all, activation))
        title.set_text(f"Frame {frame}")
        
        return (img,)
    
    # Create animation
    frames = range(start_frame, end_frame, frame_step)
    interval = 1000 / fps  # milliseconds between frames
    
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=interval,
        blit=True,
    )
    
    # Save video
    if save_path:
        ani.save(
            save_path,
            writer="ffmpeg",
            dpi=dpi,
            fps=fps,
            extra_args=["-preset", "ultrafast", "-crf", "28"],
        )
        print(f"Saved video to: {save_path}")
    
    plt.close(fig)
    return ani


def setup_logging(dir_name='rnn_training', level=logging.INFO):
    os.makedirs(dir_name, exist_ok=True)
    log_path = os.path.join(dir_name, f'console.log')

    logger = logging.getLogger()  # root logger


    logger.setLevel(level)
    # remove existing handlers to avoid duplicate messages
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logging.info(f'Logging started. Output file: {log_path}')
    return log_path


def debug_stim_alignment(start_offset, stim_delay, snippet_length, snippet_overlap, trials_df, train_df):
    # Debug: Check stim alignment
    print("=== DEBUG: Checking stim alignment ===")
    print(f"START_OFFSET = {start_offset}")
    print(f"STIM_DELAY = {stim_delay}")
    print(f"SNIPPET_LENGTH = {snippet_length}")
    print(f"SNIPPET_OVERLAP = {snippet_overlap}")

    # Check first trial - stim_time should be trial_start_time + STIM_DELAY
    first_trial = trials_df.iloc[0]
    print (first_trial)
    print(f"\nFirst trial: session={first_trial['session']}, config={first_trial['config']}")
    print(f"  trial_start_time = {first_trial['trial_start_time']}")
    print(f"  stim_time = {first_trial['stim_time']} (should be trial_start_time + STIM_DELAY)")
    print(f"  start_time = {first_trial['start_time']} (snippet starts here)")
    print(f"  Expected stim at snippet frame: {first_trial['stim_time'] - first_trial['start_time']}")

    # Check first snippet in train_df
    print(f"\n--- First train snippet from _df ---")
    first_snippet = train_df.iloc[0]
    print (first_snippet)
    print(f"  snippet_start = {first_snippet['snippet_start']}")
    print(f"  first_config = {first_snippet['first_config']}")
    print(f"  first_stim_time = {first_snippet['first_stim_time']} (relative to snippet)")
    print(f"  stim_snippet sum = {first_snippet['stim_snippet'].sum()}")
    nonzero_stim = np.nonzero(first_snippet['stim_snippet'])
    if len(nonzero_stim[0]) > 0:
        print(f"  Stim events in snippet:")
        for t, e in zip(nonzero_stim[0], nonzero_stim[1]):
            print(f"    t={t}: electrode {e}, current {first_snippet['stim_snippet'][t, e]:.0f}")
    else:
        print("  NO STIM in snippet!")

def create_split_dfs(
    split_strategy,
    valid_df,
    trials_df,
    sessions,
    snippet_length,
    val_size=0.2,
    test_size=None,
    seed=None,
    holdout_sessions=None,
    holdout_trials=None,
    holdout_configs=None,
    holdout_electrodes=None,
    holdout_currents=None,
    ):
    '''
    Split the dataframe into train, val, and test by the splitting strategy. Note that if multiple trials belong to a snippet, the 
    first one is used to determine the split. 
    
    :param split_strategy: Description
    :param valid_df: Description
    :param trials_df: Description
    :param sessions: Description
    :param snippet_length: Description
    :param val_size: must be specified for all splits 
    :param test_size: only relevant if split_strategy == 'random'
    :param seed: specify random seed
    :param holdout_sessions: if split_strategy == 'session', which session are test
    :param holdout_trials: if split_strategy == 'trial', which trials from all sessions are test
    :param holdout_configs: if split_strategy == 'config', which configs from all sessions are test
    :param holdout_electrodes: Descripif split_strategy == 'electrode', which electrodes from all sessions are testtion
    :param holdout_currents: if split_strategy == 'current', which currents from all sessions are test
    '''
    holdout_sessions = holdout_sessions or []
    holdout_trials = holdout_trials or []
    holdout_configs = holdout_configs or []
    holdout_electrodes = holdout_electrodes or []
    holdout_currents = holdout_currents or []


    if split_strategy == 'random':
        remaining_df, test_df = train_test_split(valid_df, test_size=test_size, random_state=seed)
    elif split_strategy == 'session':
        test_df = valid_df[valid_df['session'].isin(holdout_sessions)]
        remaining_df = valid_df[~valid_df['session'].isin(holdout_sessions)]
    elif split_strategy == 'trial':
        test_df = valid_df[valid_df['first_trial'].isin(holdout_trials)]
        remaining_df = valid_df[~valid_df['first_trial'].isin(holdout_trials)]
    elif split_strategy == 'config':
        test_df = valid_df[valid_df['first_config'].isin(holdout_configs)]
        remaining_df = valid_df[~valid_df['first_config'].isin(holdout_configs)]
    elif split_strategy == 'electrode':
        test_df = valid_df[valid_df['first_electrode'].isin(holdout_electrodes)]
        remaining_df = valid_df[~valid_df['first_electrode'].isin(holdout_electrodes)]
    elif split_strategy == 'current':
        test_df = valid_df[valid_df['first_current'].isin(holdout_currents)]
        remaining_df = valid_df[~valid_df['first_current'].isin(holdout_currents)]
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")
    
    # Remove snippets that temporally overlap with test frames (important if overlap set to True)
    overlapping_with_test = find_overlapping_snippets(remaining_df, test_df, snippet_length)
    logging.info(f"Removing {len(overlapping_with_test)} additional snippets that overlap with test frames")
    train_val_df = remaining_df.drop(index=list(overlapping_with_test))
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=seed)

    logging.info(f"=== Split Summary ({split_strategy}) ===")
    logging.info(f"Train: {len(train_df)} samples ({100*len(train_df)/len(valid_df):.1f}%)")
    logging.info(f"  Sessions: {sorted(train_df['session'].unique())}")
    logging.info(f"Val: {len(val_df)} samples ({100*len(val_df)/len(valid_df):.1f}%)")
    logging.info(f"Test: {len(test_df)} samples ({100*len(test_df)/len(valid_df):.1f}%)")
    logging.info(f"  Sessions: {sorted(test_df['session'].unique())}")
    return train_df, val_df, test_df

# Helper function to find snippets near specific stim times
def get_stim_times_for_holdout(trials_df, holdout_values, holdout_column, sessions):
    """Get stim times for holdout trials/configs/etc, grouped by session"""
    holdout_stims = trials_df[trials_df[holdout_column].isin(holdout_values)]
    stim_times_by_session = {}
    for session in sessions:
        # Use stim_time (actual stim_time) for temporal matching
        session_stims = holdout_stims[holdout_stims['session'] == session]['stim_time'].values
        stim_times_by_session[session] = set(session_stims)
    return stim_times_by_session

def find_snippets_near_stim_times(source_df, stim_times_by_session, snippet_length):
    """
    Find indices in source_df whose frames overlap with or are within snippet_length of stim times.
    A snippet [start, start+L) is matched if any of its frames overlaps with the window around a stim time.
    """
    matching_indices = set()
    
    for idx, row in source_df.iterrows():
        session = row['session']
        snippet_start = row['snippet_start']
        snippet_end = snippet_start + snippet_length
        
        if session not in stim_times_by_session:
            continue
            
        # Check if this snippet's time window overlaps with any holdout stim time window
        for stim_time in stim_times_by_session[session]:
            # Stim window is [stim_time - L, stim_time + L)
            # Snippet window is [snippet_start, snippet_end)
            # They overlap if: snippet_start < stim_time + L AND snippet_end > stim_time - L
            if snippet_start < stim_time + snippet_length and snippet_end > stim_time - snippet_length:
                matching_indices.add(idx)
                break
        
    return matching_indices

def find_overlapping_snippets(source_df, test_df, snippet_length):
    """
    Find indices in source_df that share ANY frames with test_df snippets.
    This ensures complete temporal separation between train/val and test.
    
    A snippet overlaps if any of its frames (including initial condition at snippet_start-1)
    overlaps with any test snippet frames (including their initial conditions).
    """
    overlapping_indices = set()
    
    # Build set of frames used by test snippets (including initial condition frame)
    test_frames_by_session = {}
    for _, row in test_df.iterrows():
        session = row['session']
        if session not in test_frames_by_session:
            test_frames_by_session[session] = set()
        # Include all frames in snippet AND the initial condition frame (snippet_start - 1)
        start = row['snippet_start']
        init_cond_frame = max(0, start - 1)
        test_frames_by_session[session].update(range(init_cond_frame, start + snippet_length))
    
    for idx, row in source_df.iterrows():
        session = row['session']
        if session not in test_frames_by_session:
            continue
        
        snippet_start = row['snippet_start']
        init_cond_frame = max(0, snippet_start - 1)
        # Check if any frame in this snippet (including init cond) overlaps with test frames
        snippet_frames = set(range(init_cond_frame, snippet_start + snippet_length))
        if snippet_frames & test_frames_by_session[session]:
            overlapping_indices.add(idx)
    
    return overlapping_indices


def make_trials_df(trial_times, n_trials=8, n_sessions=3, n_configs=31, start_offset=0, stim_delay=10):

    # START_OFFSET: how many frames before trial_start_time the snippet should begin
    # If START_OFFSET=0, snippet starts at trial_start_time
    # If START_OFFSET=10, snippet starts 10 frames BEFORE trial_start_time
    # If START_OFFSET=-5, snippet starts 5 frames AFTER trial_start_time
    START_OFFSET = start_offset  # frames before trial_start_time that snippet begins

    # STIM_DELAY: how many frames after trial_start_time the stimulation occurs
    # If STIM_DELAY=0, stim occurs at trial_start_time
    # If STIM_DELAY=10, stim occurs 10 frames after trial_start_time
    STIM_DELAY = stim_delay  # frames after trial_start_time when stim occurs

    logging.info(f"Creating trials DataFrame with START_OFFSET={START_OFFSET} (snippet starts {START_OFFSET} frames before trial_start_time)")
    logging.info(f"Creating trials DataFrame with STIM_DELAY={STIM_DELAY} (stim occurs {STIM_DELAY} frames after trial_start_time)")

    rows = []
    for session in range(n_sessions):
        # Add stimulation trials (configs 1-30)
        for config in range(1, n_configs):
            electrode = (config - 1) // 3
            current = (config - 1) % 3 + 3  # 3, 4, or 5
            for trial in range(8):
                # trial_start_time: when the trial begins (from times array)
                trial_start_time = int(trial_times[session][trial, config - 1])
                # stim_time: when stim occurs (STIM_DELAY frames after trial_start_time)
                stim_time = trial_start_time + STIM_DELAY
                # start_time: when snippet starts (START_OFFSET frames before trial_start_time)
                start_time = max(trial_start_time - START_OFFSET, 0)
                rows.append({
                    'session': session,
                    'trial': trial,
                    'config': config,
                    'electrode': electrode,
                    'current': current,
                    'trial_start_time': trial_start_time,  # when trial begins
                    'stim_time': stim_time,  # when stim occurs
                    'start_time': int(start_time),  # snippet start time
                    'is_stim': True
                })
        
        # Add no-stim trials (last config)
        for trial in range(n_trials):
            trial_start_time = int(trial_times[session][trial, n_configs - 1])  # config 31 is index 30
            stim_time = np.nan  # no stim
            start_time = max(trial_start_time - START_OFFSET, 0)
            rows.append({
                'session': session,
                'trial': trial,
                'config': n_configs - 1,
                'electrode': np.nan,  # no electrode for no-stim
                'current': 0,
                'trial_start_time': trial_start_time,
                'stim_time': stim_time,
                'start_time': int(start_time),
                'is_stim': False
            })

    trials_df = pd.DataFrame(rows)
    return trials_df
    logging.info(f"Final trials_df size: {len(trials_df)}")


def filter_trials (trials_df, filter_configs=None, filter_electrodes=None,filter_currents=None, filter_trials=None, filter_sessions=None):
    '''
    Docstring for filter_trials
    
    :param trials_df: Description
    :param filter_configs: Description
    :param filter_electrodes: Description
    :param filter_currents: Description
    :param filter_trials: Description
    :param filter_sessions: Description
    '''
    if filter_sessions is not None:
        # Keep no-stim trials (current=0) and trials with matching currents
        trials_df = trials_df[(trials_df['session'].isin(filter_sessions))]
        logging.info(f"Filtered to sessions {filter_sessions}: {len(trials_df)} trials remaining")

    # Apply trials  filter
    if filter_trials is not None:
        # Keep no-stim trials (current=0) and trials with matching currents
        trials_df = trials_df[(trials_df['trial'].isin(filter_trials))]
        logging.info(f"Filtered to currents {filter_trials}: {len(trials_df)} trials remaining")


    # Apply config filter
    if filter_configs is not None:
        trials_df = trials_df[trials_df['config'].isin(filter_configs)]
        logging.info(f"Filtered to configs {filter_configs}: {len(trials_df)} trials remaining")

    # Apply electrode filter  
    if filter_electrodes is not None:
        # Keep no-stim trials (electrode=-1) and trials with matching electrodes
        trials_df = trials_df[(trials_df['electrode'].isin(filter_electrodes))]
        logging.info(f"Filtered to electrodes {FILTER_ELECfilter_electrodesTRODES}: {len(trials_df)} trials remaining")

    # Apply current filter
    if filter_currents is not None:
        # Keep no-stim trials (current=0) and trials with matching currents
        trials_df = trials_df[(trials_df['current'].isin(filter_currents))]
        logging.info(f"Filtered to currents {filter_currents}: {len(trials_df)} trials remaining")

    return trials_df

def get_model_error(model, loader, criterion, device, LOSS_TYPE):
    test_running = 0.0
    with torch.no_grad():
        for (inputs, activity_initial), targets in loader:
            inputs = inputs.to(device)
            activity_initial = activity_initial.to(device)
            targets = targets.to(device)
            outputs = model((inputs, activity_initial))
            # Use weighted loss if available, else standard loss
            if LOSS_TYPE == 'weighted_mae':
                loss = criterion(outputs, targets, inputs)
            else:
                loss = criterion(outputs, targets)
            test_running += loss.item() * inputs.size(0)

    test_loss = test_running / len(test_loader.dataset)
    return test_loss


def get_model_error(model, loader, criterion, device, LOSS_TYPE):
    output_loss = 0.0
    outputs_collection = []
    with torch.no_grad():
        for (inputs, activity_initial), targets in loader:
            inputs = inputs.to(device)
            activity_initial = activity_initial.to(device)
            targets = targets.to(device)
            outputs = model((inputs, activity_initial))
            outputs_collection.append(outputs.cpu().numpy())
            # Use weighted loss if available, else standard loss
            if LOSS_TYPE == 'weighted_mae':
                loss = criterion(outputs, targets, inputs)
            else:
                loss = criterion(outputs, targets)
            output_loss += loss.item() * inputs.size(0)
    test_loss = output_loss / len(loader.dataset)
    outputs_collection = np.concatenate(outputs_collection, axis=0)
    return test_loss, outputs_collection

def get_inputs_outputs_targets(model, loader, device):
        all_inputs = []
        all_initconds = []
        all_targets = []
        all_outputs = []
        with torch.no_grad():
            for (inputs, activity_initial), targets in loader:
                inputs = inputs.to(device)
                activity_initial = activity_initial.to(device)
                targets = targets.to(device)
                outputs = model((inputs, activity_initial))
                all_inputs.append(inputs.cpu().numpy())
                all_initconds.append(activity_initial.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())
        all_inputs = np.concatenate(all_inputs, axis=0)
        all_initconds = np.concatenate(all_initconds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)
        return all_inputs, all_initconds, all_targets, all_outputs


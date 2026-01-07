import os
import sys
import datetime
import logging
import random


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

def plot_stim_ticks(stim_times, session_ids=None, tick_width=3,
                    cmap_name='Reds', figsize=(15, 5),
                    out_path=None, invert_y=True):
    """
    stim_times: ndarray, shape (n_sessions, n_times, n_electrodes)
    session_ids: list of session indices to plot (default: all)
    """

    n_sessions, n_times, n_electrodes = stim_times.shape

    if session_ids is None:
        session_ids = list(range(n_sessions))

    cmap = cm.get_cmap(cmap_name)

    # global amplitude range (ignore zeros)
    global_max = np.max(stim_times)
    print(f'Global max amplitude across sessions: {global_max}')
    print("Unique current values:",
          np.unique(stim_times[stim_times > 0]))

    norm = colors.Normalize(
        vmin=0.5,
        vmax=max(1.0, float(global_max))
    )

    f, axes = plt.subplots(
        1, len(session_ids),
        figsize=figsize,
        squeeze=False
    )

    for ax, sid in zip(axes[0], session_ids):
        data = stim_times[sid]  # (n_times, n_electrodes)

        for elec in range(n_electrodes):
            stim_idx = np.nonzero(data[:, elec])[0]
            if stim_idx.size == 0:
                continue

            amps = data[stim_idx, elec]
            for t, a in zip(stim_idx, amps):
                ax.vlines(
                    x=t,
                    ymin=elec - 0.5,
                    ymax=elec + 0.5,
                    color=cmap(norm(a)),
                    linewidth=1,
                    alpha=0.9
                )

        ax.set_xlim(-0.5, n_times - 0.5)
        ax.set_ylim(-0.5, n_electrodes - 0.5)
        if invert_y:
            ax.invert_yaxis()

        ax.set_yticks(np.arange(n_electrodes))
        ax.set_yticklabels(np.arange(1, n_electrodes + 1))
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Electrode')

        n_stims = int((data > 0).sum())
        ax.set_title(f'Session {sid + 1}, n_stims={n_stims}')

    # shared colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    f.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        label='Current Amplitude',
        fraction=0.02,
        pad=0.02
    )

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight')

    plt.show()

def plot_activity_per_session(activity, session_ids=None,
                              figsize=(15, 5), out_path=None):
    """
    activity: ndarray, shape (n_sessions, n_times, n_rois)
    """

    n_sessions, n_times, n_rois = activity.shape

    if session_ids is None:
        session_ids = list(range(n_sessions))

    f, axes = plt.subplots(
        1, len(session_ids),
        figsize=figsize,
        squeeze=False
    )

    for ax, sid in zip(axes[0], session_ids):
        data = activity[sid]  # (n_times, n_rois)

        im = ax.imshow(
            data.T,
            aspect='auto',
            cmap='Blues',
            origin='lower'
        )

        ax.set_title(f'Session {sid + 1}')
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('ROI')

        ax.set_yticks(
            np.arange(0, n_rois, 10),
            labels=np.arange(1, n_rois + 1, 10)
        )

        ax.grid(False)
        f.colorbar(im, ax=ax, label='Activity (dF/F)')

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight')

    plt.show()

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
            for i in range(num_snippets):
                if i == 0: # TODO: How to deal with initial condition at start of series?
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
    import pandas as pd
    
    # Build a lookup from (session, stim_time) -> trial metadata
    # stim_time = trial_start_time + STIM_DELAY, so this is the actual time when stim occurs
    stim_lookup = {}
    for _, row in trials_df.iterrows():
        if row['is_stim']:  # Only add actual stim trials
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
                
                stim_snippet[t_rel, electrode] = current
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
                # Get activity snippet
                activity_snippet = activity[session][i:i + length, :]
                
                # Build stim snippet from trials_df
                stim_snippet, all_configs, all_electrodes, all_currents, all_trials, all_stim_times, all_trial_start_times = \
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
                    first_trial_start_time = all_trial_start_times[first_idx]
                    
                    # Initial condition: frame preceding the first stim's trial_start_time
                    if first_trial_start_time > 0:
                        initial_cond = activity[session][first_trial_start_time - 1, :]
                    else:
                        initial_cond = activity[session][0, :]
                else:
                    first_config = 0
                    first_electrode = -1
                    first_current = 0
                    first_trial = -1
                    first_stim_time = -1
                    first_trial_start_time = -1
                    
                    # No stim in snippet: fall back to frame before snippet start
                    if i > 0:
                        initial_cond = activity[session][i - 1, :]
                    else:
                        initial_cond = activity[session][0, :]
                
                results.append({
                    'session': session,
                    'snippet_start': i,
                    # First stim metadata (for primary labeling)
                    'first_config': first_config,
                    'first_electrode': first_electrode,
                    'first_current': first_current,
                    'first_trial': first_trial,
                    'first_stim_time': first_stim_time,  # relative to snippet start
                    'first_trial_start_time': first_trial_start_time,  # absolute frame of trial start
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
                'first_config': row['config'] if row['is_stim'] else (all_configs[0] if has_stim else 0),
                'first_electrode': row['electrode'] if row['is_stim'] else (all_electrodes[0] if has_stim else -1),
                'first_current': row['current'] if row['is_stim'] else (all_currents[0] if has_stim else 0),
                'first_trial': row['trial'],
                'first_stim_time': stim_time_rel if row['is_stim'] else first_stim_time,
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


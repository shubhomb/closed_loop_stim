import numpy as np
import pandas as pd
import os
import logging
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import matplotlib.colors as colors

from modules import *



def plot_psth_grid(
    model,
    valid_df,
    test_indices,
    dfof,
    configs_to_plot,
    neurons_by_config,
    default_neurons,
    target_neurons,
    neuron_to_dfof_idx,
    neuron_to_output_idx,
    device,
    snippet_length,
    stim_delay,
    stim_offset,
    post_stim_frames,
    max_trials_per_session,
    psth_dir,
    line_alpha=0.9,
    gt_lw=2.0,
    pred_lw=1.2,
    n_sessions=3,
    subplot_size=4,
):
    """
    Plot PSTH grid with ground truth and predictions for multiple configs.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Trained RNN model
    valid_df : pd.DataFrame
        DataFrame containing all valid snippets
    test_indices : set
        Set of indices that are in the test set
    dfof : np.ndarray
        Neural activity data (sessions x time x neurons)
    configs_to_plot : list
        List of config IDs to plot
    neurons_by_config : dict
        Dict mapping config to list of neurons of interest
    default_neurons : list
        Default neurons to use if config not in neurons_by_config
    target_neurons : list
        List of all target neurons
    neuron_to_dfof_idx : dict
        Mapping from neuron ID to dfof index
    neuron_to_output_idx : dict
        Mapping from neuron ID to model output index
    device : torch.device
        Device to run model on
    snippet_length : int
        Length of input snippets
    stim_delay : int
        Delay before stimulation in snippet
    stim_offset : int
        Offset for display time axis
    post_stim_frames : int
        Number of frames to display after stimulation
    max_trials_per_session : int
        Maximum number of trials to plot per session
    psth_dir : str
        Directory to save plots
    line_alpha : float
        Alpha for trace lines
    gt_lw : float
        Line width for ground truth
    pred_lw : float
        Line width for predictions
    n_sessions : int
        Number of sessions to plot
    subplot_size : float
        Size of each subplot
    """
    
    def blue_to_red_8():
        cmap = cm.get_cmap("coolwarm")
        return cmap(np.linspace(0.05, 0.95, 8))
    
    trial_colors_8 = blue_to_red_8()
    source_df = valid_df
    total_display_frames = stim_offset + post_stim_frames
    pred_time_axis = np.arange(snippet_length) - stim_offset
    current_colors = {3: 'red', 4: 'blue', 5: 'green'}
    
    for config in configs_to_plot:
        neurons_of_interest = neurons_by_config.get(str(config), default_neurons) or default_neurons
        neurons_of_interest = [n for n in neurons_of_interest if n in target_neurons] or target_neurons
        
        matching_df = source_df[
            (source_df["first_config"] == config) &
            (source_df["first_stim_time"] == stim_delay)
        ]
        
        logging.info(
            f"Config {config} has {len(matching_df)} matching snippets (stim at t={stim_delay}), "
            f"neurons: {neurons_of_interest}"
        )
        if len(matching_df) == 0:
            continue
        
        # --- figure: rows=neurons, cols=3 sessions ---
        f, ax = plt.subplots(
            len(neurons_of_interest), n_sessions,
            figsize=(subplot_size * n_sessions, subplot_size * len(neurons_of_interest)),
            sharex=True, sharey=True, squeeze=False
        )
        
        for row_idx, neuron in enumerate(neurons_of_interest):
            dfof_idx = neuron_to_dfof_idx[neuron]
            output_idx = neuron_to_output_idx[neuron]
            
            for col_idx in range(n_sessions):
                current_ax = ax[row_idx, col_idx]
                
                # session subset
                session_df = matching_df[matching_df["session"] == col_idx]
                
                # Pick up to max_trials_per_session trials deterministically (by index order)
                session_df = session_df.sort_index().head(max_trials_per_session)
                
                stim_lines_added = {3: False, 4: False, 5: False}
                
                for trial_num, (idx, row) in enumerate(session_df.iterrows(), start=1):
                    trial_color = trial_colors_8[trial_num - 1]
                    
                    # linestyle encodes split: train dashed, test/val solid
                    
                    split_ls = "--" if is_train_or_val else "-"
                    
                    # --- Ground truth ---
                    session = row["session"]
                    snippet_start = row["snippet_start"]
                    gt_end = min(snippet_start + total_display_frames, dfof[session].shape[0])
                    gt_length = gt_end - snippet_start
                    extended_gt = dfof[session][snippet_start:gt_end, dfof_idx]
                    extended_gt_time = np.arange(gt_length) - stim_offset
                    
                    current_ax.plot(
                        extended_gt_time, extended_gt,
                        color=trial_color, alpha=line_alpha,
                        linestyle=split_ls, linewidth=gt_lw,
                        label=f"Trial {trial_num} GT" if trial_num == 1 else None
                    )
                    
                    # --- Prediction ---
                    stim_input = np.expand_dims(row["stim_snippet"].astype(np.float32), axis=0)
                    activity_input = np.expand_dims(row["initial_condition"].astype(np.float32), axis=0)
                    stim_input_t = torch.from_numpy(stim_input).to(device)
                    activity_input_t = torch.from_numpy(activity_input).to(device)
                    
                    with torch.no_grad():
                        outputs = model((stim_input_t, activity_input_t))
                    roi_activity = outputs[0, :, output_idx].detach().cpu().numpy()
                    
                    current_ax.plot(
                        pred_time_axis, roi_activity,
                        color=trial_color, alpha=line_alpha,
                        linestyle=split_ls, linewidth=pred_lw,
                        label=f"Trial {trial_num} Pred" if trial_num == 1 else None
                    )
                    
                    # --- stimulation vertical lines ---
                    stim_snippet = row["stim_snippet"]
                    stim_times_idx = np.nonzero(stim_snippet)[0]
                    electrodes_idx = np.nonzero(stim_snippet)[1]
                    for t_idx, e_idx in zip(stim_times_idx, electrodes_idx):
                        current = int(stim_snippet[t_idx, e_idx])
                        vcol = current_colors.get(current, "gray")
                        vlab = f"Stim (I={current})" if (current in stim_lines_added and not stim_lines_added[current]) else None
                        current_ax.axvline(
                            x=t_idx - stim_offset, color=vcol, linestyle=":", alpha=0.7, label=vlab
                        )
                        if current in stim_lines_added:
                            stim_lines_added[current] = True
                
                # axes formatting
                current_ax.set_ylim(-0.5, 1.0)
                current_ax.set_xlim(-stim_offset, post_stim_frames)
                current_ax.set_yticks([-0.5, 0, 0.5, 1.0])
                current_ax.set_xticks([-stim_offset, 0, 50])
                current_ax.set_aspect("auto")
                
                current_ax.set_title(f"Config {config}, ROI {neuron}, Session {col_idx + 1}")
                current_ax.set_xlabel("Time (frames)")
                current_ax.set_ylabel("Activity")
                
                # One legend for the whole figure (only once, top-left panel)
                if row_idx == 0 and col_idx == 0:
                    legend_elements = [
                        Line2D([0], [0], color="gray", linestyle="-", linewidth=1.5, label="Test (solid)"),
                        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5, label="Train/Val (dashed)"),
                        Line2D([0], [0], color="gray", linestyle="-", linewidth=gt_lw, label="Ground Truth (thick)"),
                        Line2D([0], [0], color="gray", linestyle="-", linewidth=pred_lw, label="Predicted (thin)"),
                    ]
                    current_ax.legend(handles=legend_elements, loc="upper right", fontsize="small")
        
        plt.tight_layout()
        outpath = os.path.join(psth_dir, f"config_{config}_a.png")
        plt.savefig(outpath, dpi=200)
        plt.close(f)
        logging.info(f"Saved combined PSTH: {outpath}")





def plot_split_visualization(
    train_df,
    val_df,
    test_df,
    dfof,
    snippet_length,
    split_strategy,
    snippet_overlap,
    holdout_trials,
    trials_df,
    output_dir,
    num_sessions=3,
    figure_size=(18, 8),
    bar_height=0.8,
    dpi=150,
    colors=None,
    ):
    """
    Visualize train/val/test snippet allocations across sessions and print overlap stats.

    Parameters
    ----------
    train_df, val_df, test_df : pandas.DataFrame
        DataFrames containing snippet metadata with at least 'session' and 'snippet_start'.
    dfof : sequence
        Array-like where dfof[session].shape[0] gives total frames for that session.
    snippet_length : int
        Number of frames per snippet (width of the plotted bars).
    split_strategy : str
        Name of the strategy used to build the split (for the title).
    snippet_overlap : bool
        Whether snippets were generated with overlap (for the title).
    holdout_trials : iterable
        Iterable of trial indices held out for testing (used when strategy is 'trial').
    trials_df : pandas.DataFrame
        DataFrame describing trials; must include 'session', 'trial', and 'stim_time'.
    output_dir : str
        Directory where the visualization PNG will be written.
    num_sessions : int, optional
        How many sessions to visualize. Default is 3.
    figure_size : tuple, optional
        Matplotlib figsize for the full plot. Default is (18, 8).
    bar_height : float, optional
        Height of each bar row. Default is 0.8.
    dpi : int, optional
        Resolution for the saved figure. Default is 150.
    colors : dict, optional
        Mapping for 'train', 'val', 'test' entries to RGBA tuples.
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    os.makedirs(output_dir, exist_ok=True)
    
    default_colors = {
        'train': (0.2, 0.6, 1.0, 0.5),
        'val': (0.2, 0.8, 0.2, 0.5),
        'test': (1.0, 0.3, 0.3, 0.5),
    }
    colors = colors or default_colors
    
    fig, axes = plt.subplots(num_sessions, 1, figsize=figure_size, sharex=True)
    if num_sessions == 1:
        axes = [axes]
    
    max_time = max(dfof[s].shape[0] for s in range(num_sessions))
    
    for session in range(num_sessions):
        ax = axes[session]
        session_train = train_df[train_df['session'] == session]
        session_val = val_df[val_df['session'] == session]
        session_test = test_df[test_df['session'] == session]
        
        for _, row in session_train.iterrows():
            ax.barh(y=0, width=snippet_length, left=row['snippet_start'], height=bar_height,
                    color=colors['train'], edgecolor='none')
        
        for _, row in session_val.iterrows():
            ax.barh(y=1, width=snippet_length, left=row['snippet_start'], height=bar_height,
                    color=colors['val'], edgecolor='none')
        
        for _, row in session_test.iterrows():
            ax.barh(y=2, width=snippet_length, left=row['snippet_start'], height=bar_height,
                    color=colors['test'], edgecolor='none')
        
        if split_strategy == 'trial' and len(holdout_trials) > 0:
            stim_times = trials_df[(trials_df['session'] == session) &
                                   (trials_df['trial'].isin(holdout_trials))]['stim_time'].values
            for stim_time in stim_times:
                ax.axvline(x=stim_time, color='red', linestyle='--', alpha=0.7, linewidth=0.5)
        
        ax.set_ylabel(f'Session {session + 1}')
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Train', 'Val', 'Test'])
        ax.set_xlim(0, max_time)
        ax.set_ylim(-0.5, 2.5)
        ax.text(
            0.99, 0.95,
            f"Train: {len(session_train)}, Val: {len(session_val)}, Test: {len(session_test)}",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    axes[-1].set_xlabel('Time (frames)')
    fig.suptitle(
        f'Train/Val/Test Split Visualization\nStrategy: {split_strategy}, '
        f'Snippet Length: {snippet_length}, Overlap: {snippet_overlap}',
        fontsize=12
    )
    legend_elements = [
        Patch(facecolor=colors['train'], label=f"Train ({len(train_df)})"),
        Patch(facecolor=colors['val'], label=f"Val ({len(val_df)})"),
        Patch(facecolor=colors['test'], label=f"Test ({len(test_df)})"),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    out_path = os.path.join(output_dir, 'split_visualization.png')
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    
    print("\n=== Overlap Analysis ===")
    for session in range(num_sessions):
        train_times = set()
        test_times = set()
        
        session_train = train_df[train_df['session'] == session]
        session_test = test_df[test_df['session'] == session]
        
        for _, row in session_train.iterrows():
            train_times.update(range(row['snippet_start'], row['snippet_start'] + snippet_length))
        for _, row in session_test.iterrows():
            test_times.update(range(row['snippet_start'], row['snippet_start'] + snippet_length))
        
        overlap = train_times & test_times
        overlap_pct = 100 * len(overlap) / max(len(test_times), 1)
        print(
            f"Session {session}: Train frames: {len(train_times)}, Test frames: {len(test_times)}, "
            f"Overlap: {len(overlap)} frames ({overlap_pct:.1f}% of test)"
        )
    
    print(f"Visualization saved to: {out_path}")


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



def show_trials_snippets_sample(trials_df, train_df, dfof, snippet_length, filter_neurons=None, seed=42, savepath=None):
    # ===== Figure 1: Sample TRIALS from trials_df =====
    fig1, axes1 = plt.subplots(3, 3, figsize=(15, 10))
    axes1 = axes1.flatten()

    # Sample 9 trials (3 per session)
    sample_trials = trials_df.groupby('session').apply(lambda x: x.sample(min(3, len(x)), random_state=seed)).reset_index(drop=True)

    for i, (_, trial_row) in enumerate(sample_trials.head(9).iterrows()):
        ax = axes1[i]
        session = trial_row['session']
        trial_num = trial_row['trial']
        config = trial_row['config']
        start_time = trial_row['start_time']
        stim_time = trial_row['stim_time']
        
        # Plot activity for a sample neuron around the trial
        window_start = max(0, start_time - 20)
        window_end = min(dfof[session].shape[0], start_time + snippet_length + 20)
        time_axis = np.arange(window_start, window_end)
        
        # Plot first target neuron's activity
        neuron_idx = 0 if filter_neurons is None else 0
        activity = dfof[session][window_start:window_end, neuron_idx]
        
        ax.plot(time_axis, activity, 'b-', linewidth=1.5, label='Activity')
        ax.axvline(x=start_time, color='green', linestyle='-', linewidth=2, label=f'start_time={start_time}')
        ax.axvline(x=stim_time, color='red', linestyle='--', linewidth=2, label=f'stim_time={stim_time}')
        ax.axvspan(start_time, start_time + snippet_length, alpha=0.2, color='yellow', label='Snippet window')
        
        ax.set_title(f'Session {session}, Trial {trial_num}, Config {config}')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Activity')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if savepath: 
        plt.savefig(f'{savepath}/debug_trials.png', dpi=150)
    plt.show()

    # ===== Figure 2: Sample SNIPPETS from valid_df =====
    fig2, axes2 = plt.subplots(3, 3, figsize=(15, 10))
    axes2 = axes2.flatten()

    sample_snippets = train_df.sample(min(9, len(train_df)), random_state=seed)

    for i, (idx, snip_row) in enumerate(sample_snippets.iterrows()):
        ax = axes2[i]
        session = snip_row['session']
        snippet_start = snip_row['snippet_start']
        first_config = snip_row['first_config']
        first_trial = snip_row['first_trial']
        first_stim_time = snip_row['first_stim_time']  # relative to snippet start
        stim_at_t0 = snip_row['stim_at_t0'] # aligned trial
        
        # Plot snippet activity
        # random_neuron = np.random.randint(snip_row['initial_condition'].shape[1])
        random_neuron = 0
        initial_condition = snip_row['initial_condition'][random_neuron]
        activity_snippet = snip_row['activity_snippet'][:, random_neuron]
        stim_snippet = snip_row['stim_snippet']
        time_axis = np.arange(snippet_length)
        
        # Plot  neuron's activity within snippet
        ax.plot(time_axis, activity_snippet, 'b-', linewidth=1.5, label=f'Activity (neuron {random_neuron})')
        
        # Plot neuron's initial condition immediately before
        ax.scatter(-1, initial_condition, marker="*")
        # Mark stim times from stim_snippet
        stim_times_in_snippet = np.nonzero(stim_snippet)[0]
        for st in np.unique(stim_times_in_snippet):
            ax.axvline(x=st, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Mark first_stim_time (should match stim_snippet nonzero)
        if not np.isnan(first_stim_time):
            ax.axvline(x=first_stim_time, color='orange', linestyle=':', linewidth=2, 
                    label=f'first_stim_time={int(first_stim_time)}')
        
        ax.set_title(f'Snip@{snippet_start}, Config {int(first_config)}, Trial {int(first_trial)}\n'
                    f'first_stim_time={first_stim_time}', fontsize=9)
        ax.set_xlabel('Frame within snippet')
        ax.set_ylabel('Activity')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_xlim(-2, snippet_length)

    fig2.suptitle(f'Sample SNIPPETS (stim times marked)\nSNIPPET_LENGTH={snippet_length}', fontsize=12)
    plt.tight_layout()
    if savepath:
        plt.savefig(f'{savepath}/debug_snippets.png', dpi=150)
    plt.show()

def random_timeseries_overlay(plot_outputs, plot_targets, target_neurons):
    # Random overlay of timeseries for each target neuron overlaid
    plt.figure(figsize=(10, 5))
    n_target_neurons = plot_outputs.shape[2]  # Number of TARGET_NEURONS
    for i, neuron in enumerate(target_neurons):
        plt.plot(plot_outputs[0, :, i], 'r', linewidth=2.0, label=f'Predicted (N{neuron})' if i == 0 else '')
        plt.plot(plot_targets[0, :, i], 'k', linewidth=2.0, label=f'Ground Truth (N{neuron})' if i == 0 else '')
    plt.xlabel('Time (frames)')
    plt.ylabel('Activity')
    plt.title(f'Test sample 0: {len(target_neurons)} target neuron(s)')
    plt.legend()
    plt.grid(True)

def plot_losses(history, test_loss, savepath):
    # ----- Plot training, validation, and test loss -----
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2.0)
    if history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss', linewidth=2.0)
    plt.axhline(y=test_loss, color='r', linestyle='--', linewidth=1.5, label='Test Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    if savepath:
        plt.savefig(f'{savepath}/loss_over_epochs.png')
    plt.show()


def plot_stim_raster(trials_df, dfof, neuron, config, threshold=0.3, pre_stim_frames=10, post_stim_frames=60, 
                     sessions=None, save_path=None):
    """
    Plot a raster of threshold-crossing events for a given neuron and config.
    
    Parameters
    ----------
    neuron : int
        Neuron index (1-indexed, MATLAB style)
    config : int
        Config number (1-31)
    threshold : float
        Activity threshold for marking events (default 0.3)
    pre_stim_frames : int
        Frames before stim to show (default 10)
    post_stim_frames : int
        Frames after stim to show (default 60)
    sessions : list or None
        Which sessions to include (0-indexed). If None, uses all 3.
    save_path : str or None
        If provided, saves figure to this path
    """
    if sessions is None:
        sessions = [0, 1, 2]
    
    # Get all trials for this config
    config_trials = trials_df[trials_df['config'] == config]
    
    # Calculate electrode and current from config
    if config <= 30:
        electrode = (config - 1) // 3
        current = (config - 1) % 3 + 3
    else:
        electrode = -1
        current = 0
    
    # Create figure: one column per session
    n_sessions = len(sessions)
    fig, axes = plt.subplots(1, n_sessions, figsize=(4*n_sessions, 6), sharey=True, squeeze=False)
    axes = axes[0]  # Flatten to 1D
    
    # Time axis relative to stim
    time_axis = np.arange(-pre_stim_frames, post_stim_frames)
    
    for ax_idx, session in enumerate(sessions):
        ax = axes[ax_idx]
        session_trials = config_trials[config_trials['session'] == session].sort_values('trial')
        
        y_offset = 0  # Track vertical position for each trial
        trial_labels = []
        
        for _, trial_row in session_trials.iterrows():
            trial_num = trial_row['trial']
            start_time = trial_row['start_time']
            
            # Get activity window around start time
            start_idx = max(0, start_time - pre_stim_frames)
            end_idx = min(dfof[session].shape[0], start_time + post_stim_frames)
            
            # Get activity for this neuron (neuron-1 for 0-indexing)
            activity = dfof[session][start_idx:end_idx, neuron - 1]
            
            # Adjust time axis based on actual data available
            actual_pre = start_time - start_idx
            actual_time = np.arange(-actual_pre, len(activity) - actual_pre)
            
            # Find threshold crossings
            above_threshold = activity > threshold
            crossing_times = actual_time[above_threshold]
            
            # Determine if this is a test trial (from HOLDOUT_TRIALS)
            is_test = trial_num in HOLDOUT_TRIALS
            marker = '|' if is_test else '|'  # Square for test, line for train
            color = 'blue' if is_test else 'black'
            
            # Plot raster ticks for this trial
            if len(crossing_times) > 0:
                ax.scatter(crossing_times, np.full_like(crossing_times, y_offset, dtype=float), 
                          marker=marker, s=20 if is_test else 50, color=color, linewidths=1)
            
            trial_labels.append(f'Trial {trial_num + 1}' + (' (test)' if is_test else ''))
            y_offset += 1
        
        # Add stim time marker
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Stim time')
        
        # Formatting
        ax.set_xlim(-pre_stim_frames, post_stim_frames)
        ax.set_ylim(-0.5, len(session_trials) - 0.5)
        ax.set_yticks(range(len(session_trials)))
        ax.set_yticklabels(trial_labels)
        ax.set_xlabel('Time relative to stim (frames)')
        ax.set_title(f'Session {session + 1}')
        ax.set_xticks([-pre_stim_frames, 0, 50])
        
        if ax_idx == 0:
            ax.set_ylabel('Trials')
    
    # Add overall title
    fig.suptitle(f'Raster: Neuron {neuron}, Config {config} (E{electrode}, I{current}), threshold={threshold}', 
                 fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f'Saved to {save_path}')
    
    # plt.show()
    return fig


def blue_to_red_8():
    cmap = cm.get_cmap("RdBu_r")
    # Avoid the central white region (~0.45–0.55)
    vals = np.concatenate([
        np.linspace(0.2, 0.40, 4),
        np.linspace(0.6, 0.8, 4)
    ])[::-1]   # <-- flip order
    return cmap(vals)


def compute_roi_percentiles(dfof, percentile=80, filter_neurons=None):
    """
    Compute the kth percentile df/F value for each ROI across all sessions.
    
    Parameters
    ----------
    dfof : list or np.ndarray
        Neural activity data. Shape: (n_sessions,) list of (n_times, n_rois) arrays
        or (n_sessions, n_times, n_rois) array
    percentile : float
        Percentile to compute (0-100). Default 80.
    filter_neurons : list or None
        If provided, only compute for these neurons (1-indexed).
        
    Returns
    -------
    percentiles : dict
        Mapping from neuron number (1-indexed) to percentile value
    """
    # Concatenate all sessions
    all_activity = np.concatenate([dfof[s] for s in range(len(dfof))], axis=0)
    
    n_rois = all_activity.shape[1]
    percentiles = {}
    
    if filter_neurons is not None:
        # Map filtered neuron numbers to their indices in dfof
        for i, neuron in enumerate(filter_neurons):
            roi_activity = all_activity[:, i]
            percentiles[neuron] = np.percentile(roi_activity, percentile)
    else:
        # Use all ROIs (1-indexed)
        for roi_idx in range(n_rois):
            neuron = roi_idx + 1  # 1-indexed
            roi_activity = all_activity[:, roi_idx]
            percentiles[neuron] = np.percentile(roi_activity, percentile)
    
    return percentiles


def plot_stim_raster_cv(
    cv_results,
    cv_hparams,
    valid_df,
    dfof,
    neuron,
    config,
    filter_neurons,
    target_neurons,
    device,
    model_dir,
    input_size,
    output_size,
    initial_cond_size,
    snippet_length,
    threshold=0.3,
    pre_stim_frames=10,
    post_stim_frames=60,
    sessions=None,
    save_path=None,
):
    """
    Plot a raster of threshold-crossing events comparing predictions vs ground truth
    using cross-validated models.
    
    For each snippet, uses the model from the fold where that snippet was held out.
    
    Parameters
    ----------
    cv_results : list of dict
        Cross-validation results
    cv_hparams : dict
        CV hyperparameters (must include split_strategy, n_units, num_layers)
    valid_df : pd.DataFrame
        DataFrame with all valid snippets
    dfof : np.ndarray
        Neural activity data
    neuron : int
        Neuron index (1-indexed)
    config : int
        Config number (1-31)
    filter_neurons : list or None
        Filtered neuron indices
    target_neurons : list/set
        Target neurons for prediction
    device : torch.device
        Device for inference
    model_dir : str
        Directory with CV checkpoints
    input_size, output_size, initial_cond_size : int
        Model dimensions
    snippet_length : int
        Length of snippets
    threshold : float
        Activity threshold for marking events
    pre_stim_frames : int
        Frames before stim to show
    post_stim_frames : int
        Frames after stim to show
    sessions : list or None
        Which sessions to plot (0-indexed)
    save_path : str or None
        Path to save figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if sessions is None:
        sessions = [0, 1, 2]
    
    if neuron not in target_neurons:
        logging.warning(f"Neuron {neuron} not in target_neurons, cannot plot predictions")
        return None
    
    split_strategy = cv_hparams['split_strategy']
    n_units = cv_hparams['n_units']
    num_layers = cv_hparams['num_layers']
    
    # Build holdout -> checkpoint mapping
    holdout_to_checkpoint = {}
    for r in cv_results:
        holdout_elem = r['holdout_element']
        fold_idx = r['fold'] - 1
        checkpoint_path = os.path.join(model_dir, f"cv_fold_{fold_idx}_best.pth")
        holdout_to_checkpoint[holdout_elem] = checkpoint_path
    
    strategy_to_column = {
        'trial': 'first_trial',
        'session': 'session',
        'config': 'first_config',
        'electrode': 'first_electrode',
        'current': 'first_current',
    }
    split_column = strategy_to_column[split_strategy]
    
    # Create neuron index mappings
    if filter_neurons is not None:
        neuron_to_dfof_idx = {n: i for i, n in enumerate(filter_neurons)}
    else:
        neuron_to_dfof_idx = {n: n - 1 for n in range(1, dfof[0].shape[1] + 1)}
    neuron_to_output_idx = {n: i for i, n in enumerate(target_neurons)}
    
    dfof_idx = neuron_to_dfof_idx[neuron]
    output_idx = neuron_to_output_idx[neuron]
    
    # Model cache
    model_cache = {}
    
    def get_model_for_holdout(holdout_elem):
        if holdout_elem not in model_cache:
            if holdout_elem not in holdout_to_checkpoint:
                return None
            checkpoint = holdout_to_checkpoint[holdout_elem]
            if not os.path.exists(checkpoint):
                return None
            model = RNNModel(
                input_size=input_size,
                units=n_units,
                output_size=output_size,
                num_layers=num_layers,
                initial_cond_size=initial_cond_size
            )
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            model.to(device)
            model.eval()
            model_cache[holdout_elem] = model
        return model_cache[holdout_elem]
    
    # Get snippets for this config
    config_df = valid_df[valid_df['first_config'] == config]
    
    # Calculate electrode and current from config
    if config <= 30:
        electrode = (config - 1) // 3
        current = (config - 1) % 3 + 3
    else:
        electrode = -1
        current = 0
    
    # Create figure: two rows (GT and Pred), one column per session
    n_sessions = len(sessions)
    fig, axes = plt.subplots(2, n_sessions, figsize=(4*n_sessions, 8), sharey='row', squeeze=False)
    
    time_axis = np.arange(-pre_stim_frames, post_stim_frames)
    
    for ax_idx, session in enumerate(sessions):
        ax_gt = axes[0, ax_idx]
        ax_pred = axes[1, ax_idx]
        
        session_df = config_df[config_df['session'] == session].sort_values('first_trial')
        
        y_offset = 0
        trial_labels = []
        
        for _, row in session_df.iterrows():
            trial_num = int(row['first_trial']) if not pd.isna(row['first_trial']) else 0
            snippet_start = row['snippet_start']
            stim_time_in_snippet = row['first_stim_time'] if not pd.isna(row['first_stim_time']) else 0
            
            # Get holdout element for this snippet
            holdout_elem = row[split_column]
            if pd.isna(holdout_elem):
                holdout_elem = None
            elif isinstance(holdout_elem, float):
                holdout_elem = int(holdout_elem)
            
            is_test = True  # In CV, we use the held-out model, so it's always "test" for that model
            
            # Get ground truth activity
            gt_start = max(0, snippet_start + int(stim_time_in_snippet) - pre_stim_frames)
            gt_end = min(dfof[session].shape[0], snippet_start + int(stim_time_in_snippet) + post_stim_frames)
            gt_activity = dfof[session][gt_start:gt_end, dfof_idx]
            
            actual_pre = snippet_start + int(stim_time_in_snippet) - gt_start
            gt_time = np.arange(-actual_pre, len(gt_activity) - actual_pre)
            
            # Ground truth threshold crossings
            gt_above = gt_activity > threshold
            gt_crossing_times = gt_time[gt_above]
            
            # Plot GT raster
            if len(gt_crossing_times) > 0:
                ax_gt.scatter(gt_crossing_times, np.full_like(gt_crossing_times, y_offset, dtype=float),
                             marker='|', s=50, color='black', linewidths=1)
            
            # Get prediction using CV model
            model = get_model_for_holdout(holdout_elem)
            if model is not None:
                stim_input = np.expand_dims(row['stim_snippet'].astype(np.float32), axis=0)
                activity_input = np.expand_dims(row['initial_condition'].astype(np.float32), axis=0)
                stim_input_t = torch.from_numpy(stim_input).to(device)
                activity_input_t = torch.from_numpy(activity_input).to(device)
                
                with torch.no_grad():
                    outputs = model((stim_input_t, activity_input_t))
                pred_activity = outputs[0, :, output_idx].cpu().numpy()
                
                # Align prediction to stim time
                pred_start_idx = max(0, int(stim_time_in_snippet) - pre_stim_frames)
                pred_end_idx = min(len(pred_activity), int(stim_time_in_snippet) + post_stim_frames)
                pred_segment = pred_activity[pred_start_idx:pred_end_idx]
                
                pred_time = np.arange(pred_start_idx - int(stim_time_in_snippet), 
                                      pred_end_idx - int(stim_time_in_snippet))
                
                # Prediction threshold crossings
                pred_above = pred_segment > threshold
                pred_crossing_times = pred_time[pred_above]
                
                if len(pred_crossing_times) > 0:
                    ax_pred.scatter(pred_crossing_times, np.full_like(pred_crossing_times, y_offset, dtype=float),
                                   marker='|', s=50, color='blue', linewidths=1)
            
            trial_labels.append(f'Trial {trial_num + 1}')
            y_offset += 1
        
        # Add stim time markers
        ax_gt.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax_pred.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Formatting
        for ax in [ax_gt, ax_pred]:
            ax.set_xlim(-pre_stim_frames, post_stim_frames)
            ax.set_ylim(-0.5, len(session_df) - 0.5)
            ax.set_yticks(range(len(session_df)))
            ax.set_yticklabels(trial_labels)
            ax.set_xticks([-pre_stim_frames, 0, 50])
        
        ax_gt.set_title(f'Session {session + 1} - Ground Truth')
        ax_pred.set_title(f'Session {session + 1} - Predicted')
        ax_pred.set_xlabel('Time relative to stim (frames)')
        
        if ax_idx == 0:
            ax_gt.set_ylabel('Trials (GT)')
            ax_pred.set_ylabel('Trials (Pred)')
    
    fig.suptitle(f'CV Raster: ROI {neuron}, Config {config} (E{electrode}, I{current}), thresh={threshold}', 
                 fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        logging.info(f"Saved CV raster to {save_path}")
    
    model_cache.clear()
    return fig


def compute_cv_confusion_metrics(
    cv_results,
    cv_hparams,
    valid_df,
    dfof,
    good_pairs,
    filter_neurons,
    target_neurons,
    device,
    model_dir,
    input_size,
    output_size,
    initial_cond_size,
    snippet_length,
    percentile=80,
    save_dir=None,
):
    """
    Compute confusion matrix metrics (precision, recall, accuracy) for CV results.
    
    For each ROI, uses the kth percentile of df/F as threshold. For each timepoint
    in predictions and ground truth, determines if both exceed threshold (TP),
    neither do (TN), only GT does (FN), or only pred does (FP).
    
    Parameters
    ----------
    cv_results : list of dict
        Cross-validation results from training loop
    cv_hparams : dict
        Hyperparameters (must include split_strategy, n_units, num_layers)
    valid_df : pd.DataFrame
        DataFrame with all valid snippets
    dfof : np.ndarray
        Neural activity data
    good_pairs : list of tuples
        List of (config, roi) pairs to evaluate
    filter_neurons : list or None
        Filtered neuron indices
    target_neurons : list/set
        Target neurons for prediction
    device : torch.device
        Device for model inference
    model_dir : str
        Directory containing CV model checkpoints
    input_size, output_size, initial_cond_size : int
        Model dimensions
    snippet_length : int
        Length of snippets
    percentile : float
        Percentile to use as threshold (default 80)
    save_dir : str or None
        Directory to save results
        
    Returns
    -------
    cv_results : list of dict
        Updated cv_results with confusion metrics added
    summary : dict
        Summary statistics across all folds
    """
    split_strategy = cv_hparams['split_strategy']
    n_units = cv_hparams['n_units']
    num_layers = cv_hparams['num_layers']
    
    strategy_to_column = {
        'trial': 'first_trial',
        'session': 'session',
        'config': 'first_config',
        'electrode': 'first_electrode',
        'current': 'first_current',
    }
    split_column = strategy_to_column[split_strategy]
    
    # Compute percentile thresholds for each ROI
    roi_thresholds = compute_roi_percentiles(dfof, percentile=percentile, filter_neurons=filter_neurons)
    logging.info(f"Computed {percentile}th percentile thresholds for {len(roi_thresholds)} ROIs")
    
    # Create output directory
    if save_dir is None:
        save_dir = os.path.join(model_dir, 'confusion_metrics')
    os.makedirs(save_dir, exist_ok=True)
    
    # Build holdout -> checkpoint mapping
    holdout_to_checkpoint = {}
    for r in cv_results:
        holdout_elem = r['holdout_element']
        fold_idx = r['fold'] - 1
        checkpoint_path = os.path.join(model_dir, f"cv_fold_{fold_idx}_best.pth")
        holdout_to_checkpoint[holdout_elem] = checkpoint_path
    
    # Create neuron index mappings
    if filter_neurons is not None:
        neuron_to_dfof_idx = {n: i for i, n in enumerate(filter_neurons)}
    else:
        neuron_to_dfof_idx = {n: n - 1 for n in range(1, dfof[0].shape[1] + 1)}
    neuron_to_output_idx = {n: i for i, n in enumerate(target_neurons)}
    
    # Model cache
    model_cache = {}
    
    def get_model_for_holdout(holdout_elem):
        if holdout_elem not in model_cache:
            if holdout_elem not in holdout_to_checkpoint:
                return None
            checkpoint = holdout_to_checkpoint[holdout_elem]
            if not os.path.exists(checkpoint):
                return None
            model = RNNModel(
                input_size=input_size,
                units=n_units,
                output_size=output_size,
                num_layers=num_layers,
                initial_cond_size=initial_cond_size
            )
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            model.to(device)
            model.eval()
            model_cache[holdout_elem] = model
        return model_cache[holdout_elem]
    
    # Get unique ROIs from good_pairs that are in target_neurons
    rois_to_evaluate = list(set([roi for _, roi in good_pairs if roi in target_neurons]))
    logging.info(f"Evaluating {len(rois_to_evaluate)} ROIs: {rois_to_evaluate}")
    
    # Track metrics across folds
    all_fold_metrics = {
        'test': {'precision': [], 'recall': [], 'accuracy': [], 'f1': []},
        'train': {'precision': [], 'recall': [], 'accuracy': [], 'f1': []},
    }
    
    for fold_result in cv_results:
        holdout_elem = fold_result['holdout_element']
        fold_idx = fold_result['fold']
        
        logging.info(f"\n{'='*60}")
        logging.info(f"FOLD {fold_idx}: Holdout={holdout_elem}")
        logging.info(f"{'='*60}")
        
        # Get train and test snippets for this fold
        if pd.isna(holdout_elem):
            test_df = valid_df[valid_df[split_column].isna()]
            train_df = valid_df[~valid_df[split_column].isna()]
        else:
            test_df = valid_df[valid_df[split_column] == holdout_elem]
            train_df = valid_df[valid_df[split_column] != holdout_elem]
        
        model = get_model_for_holdout(holdout_elem)
        if model is None:
            logging.warning(f"Fold {fold_idx}: Could not load model")
            fold_result['confusion_metrics'] = {}
            continue
        
        # Initialize fold-level confusion counts
        fold_confusion = {
            'test': {roi: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0} for roi in rois_to_evaluate},
            'train': {roi: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0} for roi in rois_to_evaluate},
        }
        
        for split_name, split_df in [('test', test_df), ('train', train_df)]:
            if len(split_df) == 0:
                continue
            
            for roi in rois_to_evaluate:
                threshold = roi_thresholds[roi]
                dfof_idx = neuron_to_dfof_idx[roi]
                output_idx = neuron_to_output_idx[roi]
                
                # Get configs associated with this ROI from good_pairs
                roi_configs = [c for c, r in good_pairs if r == roi]
                
                for config in roi_configs:
                    config_df = split_df[split_df['first_config'] == config]
                    
                    for idx, row in config_df.iterrows():
                        # Get ground truth
                        session = row['session']
                        snippet_start = row['snippet_start']
                        gt = dfof[session][snippet_start:snippet_start + snippet_length, dfof_idx]
                        
                        # Get prediction
                        stim_input = np.expand_dims(row['stim_snippet'].astype(np.float32), axis=0)
                        activity_input = np.expand_dims(row['initial_condition'].astype(np.float32), axis=0)
                        stim_input_t = torch.from_numpy(stim_input).to(device)
                        activity_input_t = torch.from_numpy(activity_input).to(device)
                        
                        with torch.no_grad():
                            outputs = model((stim_input_t, activity_input_t))
                        pred = outputs[0, :, output_idx].cpu().numpy()
                        
                        # Ensure same length
                        min_len = min(len(pred), len(gt))
                        pred = pred[:min_len]
                        gt = gt[:min_len]
                        
                        # Apply threshold
                        pred_above = pred > threshold
                        gt_above = gt > threshold
                        
                        # Count confusion matrix elements (per timepoint)
                        fold_confusion[split_name][roi]['TP'] += np.sum(pred_above & gt_above)
                        fold_confusion[split_name][roi]['TN'] += np.sum(~pred_above & ~gt_above)
                        fold_confusion[split_name][roi]['FP'] += np.sum(pred_above & ~gt_above)
                        fold_confusion[split_name][roi]['FN'] += np.sum(~pred_above & gt_above)
        
        # Compute metrics for this fold
        fold_metrics = {'test': {}, 'train': {}}
        
        for split_name in ['test', 'train']:
            split_tp, split_tn, split_fp, split_fn = 0, 0, 0, 0
            
            for roi in rois_to_evaluate:
                cm = fold_confusion[split_name][roi]
                tp, tn, fp, fn = cm['TP'], cm['TN'], cm['FP'], cm['FN']
                
                split_tp += tp
                split_tn += tn
                split_fp += fp
                split_fn += fn
                
                # Per-ROI metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                fold_metrics[split_name][roi] = {
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy,
                    'f1': f1,
                    'confusion': cm,
                }
            
            # Aggregate metrics across ROIs for this split
            agg_precision = split_tp / (split_tp + split_fp) if (split_tp + split_fp) > 0 else 0.0
            agg_recall = split_tp / (split_tp + split_fn) if (split_tp + split_fn) > 0 else 0.0
            agg_accuracy = (split_tp + split_tn) / (split_tp + split_tn + split_fp + split_fn) if (split_tp + split_tn + split_fp + split_fn) > 0 else 0.0
            agg_f1 = 2 * agg_precision * agg_recall / (agg_precision + agg_recall) if (agg_precision + agg_recall) > 0 else 0.0
            
            fold_metrics[split_name]['aggregate'] = {
                'precision': agg_precision,
                'recall': agg_recall,
                'accuracy': agg_accuracy,
                'f1': agg_f1,
                'total_tp': split_tp,
                'total_tn': split_tn,
                'total_fp': split_fp,
                'total_fn': split_fn,
            }
            
            # Add to all-fold tracking
            all_fold_metrics[split_name]['precision'].append(agg_precision)
            all_fold_metrics[split_name]['recall'].append(agg_recall)
            all_fold_metrics[split_name]['accuracy'].append(agg_accuracy)
            all_fold_metrics[split_name]['f1'].append(agg_f1)
        
        fold_result['confusion_metrics'] = fold_metrics
        
        # Log fold results
        test_agg = fold_metrics['test'].get('aggregate', {})
        train_agg = fold_metrics['train'].get('aggregate', {})
        logging.info(f"Fold {fold_idx} TEST:  Prec={test_agg.get('precision', 0):.4f}, "
                    f"Recall={test_agg.get('recall', 0):.4f}, Acc={test_agg.get('accuracy', 0):.4f}, "
                    f"F1={test_agg.get('f1', 0):.4f}")
        logging.info(f"Fold {fold_idx} TRAIN: Prec={train_agg.get('precision', 0):.4f}, "
                    f"Recall={train_agg.get('recall', 0):.4f}, Acc={train_agg.get('accuracy', 0):.4f}, "
                    f"F1={train_agg.get('f1', 0):.4f}")
    
    # Compute summary statistics
    summary = {
        'percentile': percentile,
        'thresholds': roi_thresholds,
        'n_folds': len(cv_results),
        'rois_evaluated': rois_to_evaluate,
    }
    
    for split_name in ['test', 'train']:
        metrics = all_fold_metrics[split_name]
        summary[split_name] = {
            'mean_precision': np.mean(metrics['precision']),
            'std_precision': np.std(metrics['precision']),
            'mean_recall': np.mean(metrics['recall']),
            'std_recall': np.std(metrics['recall']),
            'mean_accuracy': np.mean(metrics['accuracy']),
            'std_accuracy': np.std(metrics['accuracy']),
            'mean_f1': np.mean(metrics['f1']),
            'std_f1': np.std(metrics['f1']),
        }
    
    # Print summary
    print("\n" + "="*80)
    print(f"CONFUSION MATRIX METRICS SUMMARY ({percentile}th percentile threshold)")
    print("="*80)
    print(f"\nROIs evaluated: {rois_to_evaluate}")
    print(f"Thresholds: {', '.join([f'ROI {k}: {v:.4f}' for k, v in roi_thresholds.items() if k in rois_to_evaluate])}")
    print("\n" + "-"*80)
    print("PER-FOLD RESULTS:")
    print("-"*80)
    print(f"{'Fold':<6} {'Split':<6} {'Precision':<12} {'Recall':<12} {'Accuracy':<12} {'F1':<12}")
    print("-"*80)
    
    for r in cv_results:
        fold_idx = r['fold']
        cm = r.get('confusion_metrics', {})
        for split_name in ['test', 'train']:
            agg = cm.get(split_name, {}).get('aggregate', {})
            print(f"{fold_idx:<6} {split_name:<6} {agg.get('precision', 0):<12.4f} "
                  f"{agg.get('recall', 0):<12.4f} {agg.get('accuracy', 0):<12.4f} "
                  f"{agg.get('f1', 0):<12.4f}")
    
    print("\n" + "-"*80)
    print("AGGREGATE RESULTS (mean ± std across folds):")
    print("-"*80)
    for split_name in ['test', 'train']:
        s = summary[split_name]
        print(f"\n{split_name.upper()}:")
        print(f"  Precision: {s['mean_precision']:.4f} ± {s['std_precision']:.4f}")
        print(f"  Recall:    {s['mean_recall']:.4f} ± {s['std_recall']:.4f}")
        print(f"  Accuracy:  {s['mean_accuracy']:.4f} ± {s['std_accuracy']:.4f}")
        print(f"  F1 Score:  {s['mean_f1']:.4f} ± {s['std_f1']:.4f}")
    print("="*80)
    
    # Add summary to cv_results for convenience
    for r in cv_results:
        r['confusion_summary'] = summary
    
    # Clear model cache
    model_cache.clear()
    
    # Save summary to file
    import json
    summary_path = os.path.join(save_dir, f'confusion_summary_p{percentile}.json')
    # Convert numpy types for JSON serialization
    summary_serializable = {}
    for k, v in summary.items():
        if isinstance(v, dict):
            summary_serializable[k] = {
                str(kk): float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                for kk, vv in v.items()
            }
        elif isinstance(v, (list, np.ndarray)):
            summary_serializable[k] = [int(x) if isinstance(x, (np.integer,)) else x for x in v]
        else:
            summary_serializable[k] = v
    
    with open(summary_path, 'w') as f:
        json.dump(summary_serializable, f, indent=2, default=str)
    logging.info(f"Confusion metrics summary saved to {summary_path}")
    
    return cv_results, summary


def plot_cv_confusion_summary(cv_results, summary, save_path=None):
    """
    Plot bar charts summarizing confusion matrix metrics across CV folds.
    
    Parameters
    ----------
    cv_results : list of dict
        CV results with confusion_metrics added
    summary : dict
        Summary statistics from compute_cv_confusion_metrics
    save_path : str or None
        Path to save figure
    """
    n_folds = len(cv_results)
    fold_labels = [f"Fold {r['fold']}" for r in cv_results]
    
    metrics = ['precision', 'recall', 'accuracy', 'f1']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    x = np.arange(n_folds)
    width = 0.35
    
    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        
        test_vals = [r.get('confusion_metrics', {}).get('test', {}).get('aggregate', {}).get(metric, 0) 
                     for r in cv_results]
        train_vals = [r.get('confusion_metrics', {}).get('train', {}).get('aggregate', {}).get(metric, 0) 
                      for r in cv_results]
        
        bars1 = ax.bar(x - width/2, test_vals, width, label='Test', color='coral', alpha=0.8)
        bars2 = ax.bar(x + width/2, train_vals, width, label='Train', color='skyblue', alpha=0.8)
        
        # Add mean lines
        test_mean = summary['test'][f'mean_{metric}']
        train_mean = summary['train'][f'mean_{metric}']
        ax.axhline(y=test_mean, color='red', linestyle='--', linewidth=2, 
                   label=f'Test Mean: {test_mean:.3f}')
        ax.axhline(y=train_mean, color='blue', linestyle='--', linewidth=2,
                   label=f'Train Mean: {train_mean:.3f}')
        
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} by Fold')
        ax.set_xticks(x)
        ax.set_xticklabels(fold_labels, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right', fontsize='small')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f"Confusion Matrix Metrics ({summary['percentile']}th Percentile Threshold)", 
                 fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved confusion summary plot to {save_path}")
    
    plt.show()
    return fig






def plot_psth_from_cv_results(
    cv_results,
    cv_hparams,
    valid_df,
    dfof,
    configs_to_plot,
    neurons_by_config,
    filter_neurons,
    target_neurons,
    device,
    model_dir,
    input_size,
    output_size,
    initial_cond_size,
    snippet_length,
    stim_delay=10,
    post_stim_frames=50,
    show_ground_truth=False,
    subplot_size=2.5,
    line_alpha=0.9,
    pred_lw=1.2,
    gt_lw=2.0,
    n_sessions=3,
    save_dir=None,
):
    """
    Plot PSTHs using cross-validation results.
    
    For each snippet, loads the model from the fold where that snippet was held out,
    ensuring all predictions shown are truly out-of-sample.
    
    Works for any split strategy: trial, session, config, electrode, current.
    
    Parameters
    ----------
    show_ground_truth : bool
        If True, creates adjacent subplots showing ground truth with identical axes.
        Colors for corresponding GT and prediction traces are identical.
    """
    split_strategy = cv_hparams['split_strategy']
    n_units = cv_hparams['n_units']
    num_layers = cv_hparams['num_layers']
    
    # Build mapping: holdout_element -> fold model checkpoint
    holdout_to_checkpoint = {}
    for r in cv_results:
        holdout_elem = r['holdout_element']
        fold_idx = r['fold'] - 1  # fold is 1-indexed in results
        checkpoint_path = os.path.join(model_dir, f"cv_fold_{fold_idx}_best.pth")
        holdout_to_checkpoint[holdout_elem] = checkpoint_path
    
    logging.info(f"CV split strategy: {split_strategy}")
    logging.info(f"Holdout elements: {list(holdout_to_checkpoint.keys())}")
    
    # Map split strategy to DataFrame column
    strategy_to_column = {
        'trial': 'first_trial',
        'session': 'session',
        'config': 'first_config',
        'electrode': 'first_electrode',
        'current': 'first_current',
    }
    split_column = strategy_to_column[split_strategy]
    
    # Cache for loaded models
    model_cache = {}
    
    def get_model_for_holdout(holdout_elem):
        """Load model for a given holdout element, with caching."""
        if holdout_elem not in model_cache:
            if holdout_elem not in holdout_to_checkpoint:
                logging.warning(f"No model found for holdout element {holdout_elem}")
                return None
            checkpoint = holdout_to_checkpoint[holdout_elem]
            if not os.path.exists(checkpoint):
                logging.warning(f"Checkpoint not found: {checkpoint}")
                return None
            model = RNNModel(
                input_size=input_size, 
                units=n_units, 
                output_size=output_size,
                num_layers=num_layers, 
                initial_cond_size=initial_cond_size
            )
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            model.to(device)
            model.eval()
            model_cache[holdout_elem] = model
        return model_cache[holdout_elem]
    
    # Create output directory
    if save_dir is None:
        save_dir = os.path.join(model_dir, 'psths_cv')
    os.makedirs(save_dir, exist_ok=True)
    
    # Time axes
    total_display_frames = stim_delay + post_stim_frames
    pred_time_axis = np.arange(snippet_length) - stim_delay
    gt_time_axis = np.arange(total_display_frames) - stim_delay
    
    # Color schemes
    current_colors = {3: 'red', 4: 'blue', 5: 'green'}
    trial_cmap = cm.get_cmap('tab10')
    
    for config in configs_to_plot:
        # Get neurons of interest for this config. Plot all target neurons if none specified.
        neurons_of_interest = neurons_by_config.get(str(config), target_neurons)
        if neurons_of_interest is None or len(neurons_of_interest) == 0:
            neurons_of_interest = list(target_neurons)
        neurons_of_interest = [n for n in neurons_of_interest if n in target_neurons]
        if len(neurons_of_interest) == 0:
            neurons_of_interest = list(target_neurons)
        
        # Create neuron index mapping: original neuron number -> index in filtered dfof
        # If FILTER_NEURONS is None, identity mapping; otherwise map to filtered indices
        if filter_neurons is not None:
            neuron_to_dfof_idx = {n: i for i, n in enumerate(filter_neurons)}
        else:
            neuron_to_dfof_idx = {n: n - 1 for n in range(1, dfof[0].shape[1] + 1)}  # 1-indexed to 0-indexed

        # Create mapping for model output: neuron number -> index in TARGET_NEURONS
        # Model output only has TARGET_NEURONS columns
        neuron_to_output_idx = {n: i for i, n in enumerate(target_neurons)}

        # Filter matching snippets
        if config != 31:
            matching_df = valid_df[
                (valid_df['first_config'] == config) &
                (valid_df['first_stim_time'] == stim_delay)
            ]
        else:
            matching_df = valid_df[valid_df['first_config'] == config]
        
        if len(matching_df) == 0:
            logging.info(f"Config {config}: no matching snippets, skipping")
            continue
        
        logging.info(f"Config {config}: {len(matching_df)} snippets, neurons: {neurons_of_interest}")
        
        # Create figure - double columns if showing ground truth
        n_cols = n_sessions * 2 if show_ground_truth else n_sessions
        f, ax = plt.subplots(
            len(neurons_of_interest), n_cols,
            figsize=(subplot_size * n_cols, subplot_size * len(neurons_of_interest)),
            sharex=True, sharey=True, squeeze=False
        )
        
        for row_idx, neuron in enumerate(neurons_of_interest):
            dfof_idx = neuron_to_dfof_idx[neuron]
            output_idx = neuron_to_output_idx[neuron]
            
            for sess_idx in range(n_sessions):
                # Column indices: if show_ground_truth, pred is at 2*sess_idx, GT is at 2*sess_idx+1
                if show_ground_truth:
                    pred_col = sess_idx * 2
                    gt_col = sess_idx * 2 + 1
                    pred_ax = ax[row_idx, pred_col]
                    gt_ax = ax[row_idx, gt_col]
                else:
                    pred_col = sess_idx
                    pred_ax = ax[row_idx, pred_col]
                    gt_ax = None
                
                session_df = matching_df[matching_df['session'] == sess_idx]
                
                if len(session_df) == 0:
                    pred_ax.set_title(f"Config {config}, ROI {neuron}, Sess {sess_idx+1}\nPredicted (no data)")
                    if gt_ax is not None:
                        gt_ax.set_title(f"Config {config}, ROI {neuron}, Sess {sess_idx+1}\nGround Truth (no data)")
                    continue
                
                stim_lines_added_pred = {3: False, 4: False, 5: False}
                stim_lines_added_gt = {3: False, 4: False, 5: False}
                
                for trial_idx, (idx, row) in enumerate(session_df.iterrows()):
                    # Determine which holdout element this snippet belongs to
                    holdout_elem = row[split_column]
                    
                    # Handle None values (e.g., electrode=None for config 31)
                    if pd.isna(holdout_elem):
                        holdout_elem = None
                    elif isinstance(holdout_elem, float):
                        holdout_elem = int(holdout_elem)
                    
                    # Get the model that held out this element
                    model = get_model_for_holdout(holdout_elem)
                    if model is None:
                        continue
                    
                    # Use same color for both GT and prediction
                    trial_color = trial_cmap(trial_idx % trial_cmap.N)
                    
                    # --- Ground truth (extended) ---
                    session = row['session']
                    snippet_start = row['snippet_start']
                    gt_end = min(snippet_start + total_display_frames, dfof[session].shape[0])
                    gt_length = gt_end - snippet_start
                    extended_gt = dfof[session][snippet_start:gt_end, dfof_idx]
                    extended_gt_time = np.arange(gt_length) - stim_delay
                    
                    # Plot ground truth on separate subplot if enabled
                    if gt_ax is not None:
                        gt_ax.plot(
                            extended_gt_time, extended_gt,
                            color=trial_color, alpha=line_alpha,
                            linestyle='-', linewidth=gt_lw,
                        )
                    
                    # --- Prediction (from held-out model) ---
                    stim_input = np.expand_dims(row['stim_snippet'].astype(np.float32), axis=0)
                    activity_input = np.expand_dims(row['initial_condition'].astype(np.float32), axis=0)
                    stim_input_t = torch.from_numpy(stim_input).to(device)
                    activity_input_t = torch.from_numpy(activity_input).to(device)
                    
                    with torch.no_grad():
                        outputs = model((stim_input_t, activity_input_t))
                    roi_activity = outputs[0, :, output_idx].detach().cpu().numpy()
                    
                    pred_ax.plot(
                        pred_time_axis, roi_activity,
                        color=trial_color, alpha=line_alpha,
                        linestyle='-', linewidth=pred_lw,
                    )
                    
                    # --- Stim lines on prediction subplot ---
                    stim_snippet = row['stim_snippet']
                    stim_times_idx = np.nonzero(stim_snippet)[0]
                    electrodes_idx = np.nonzero(stim_snippet)[1]
                    for t_idx, e_idx in zip(stim_times_idx, electrodes_idx):
                        current = int(stim_snippet[t_idx, e_idx])
                        vcol = current_colors.get(current, 'gray')
                        vlab = f"Stim (I={current})" if not stim_lines_added_pred.get(current, False) else None
                        pred_ax.axvline(x=t_idx - stim_delay, color=vcol, linestyle=':', alpha=0.7, label=vlab)
                        stim_lines_added_pred[current] = True
                        
                        # Also add stim lines to GT subplot
                        if gt_ax is not None:
                            vlab_gt = f"Stim (I={current})" if not stim_lines_added_gt.get(current, False) else None
                            gt_ax.axvline(x=t_idx - stim_delay, color=vcol, linestyle=':', alpha=0.7, label=vlab_gt)
                            stim_lines_added_gt[current] = True
                
                # Axis formatting for prediction subplot
                pred_ax.set_ylim(-0.5, 1.0)
                pred_ax.set_yticks([-0.5, 0, 0.5, 1.0])
                pred_ax.set_xticks([0, 50])
                pred_ax.set_title(f"Config {config}, ROI {neuron}, Sess {sess_idx+1}\nPredicted")
                pred_ax.set_xlabel("Time (frames)")
                pred_ax.set_ylabel("Activity")
                
                # Axis formatting for ground truth subplot (if enabled)
                if gt_ax is not None:
                    gt_ax.set_ylim(-0.5, 1.0)
                    gt_ax.set_yticks([-0.5, 0, 0.5, 1.0])
                    gt_ax.set_xticks([0, 50])
                    gt_ax.set_title(f"Config {config}, ROI {neuron}, Sess {sess_idx+1}\nGround Truth")
                    gt_ax.set_xlabel("Time (frames)")
                    gt_ax.set_ylabel("Activity")
        
        plt.tight_layout()
        outpath = os.path.join(save_dir, f"config_{config}_cv.png")
        plt.savefig(outpath, dpi=200)
        plt.close(f)
        logging.info(f"Saved CV PSTH: {outpath}")
    
    # Clear model cache
    model_cache.clear()
    logging.info(f"CV PSTH plots saved to {save_dir}")




def plot_cv_test_loss_bar(cv_results, split_strategy, savepath=None):
    """
    Plots a bar chart of test losses for each held-out element in cross-validation.
    """
    # Extract heldout elements and test losses from cv_results
    heldout_elements = [r['holdout_element'] for r in cv_results]
    test_losses = [r['test_loss'] for r in cv_results]

    # Sort by heldout element for consistent plotting
    sorted_indices = np.argsort(heldout_elements)
    heldout_sorted = np.array(heldout_elements)[sorted_indices]
    test_losses_sorted = np.array(test_losses)[sorted_indices]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(heldout_sorted)), test_losses_sorted, color='skyblue')
    plt.xlabel(f"Held-out {split_strategy.capitalize()}")
    plt.ylabel("Test Loss")
    plt.grid()
    plt.title(f"Test Loss by Held-out {split_strategy.capitalize()}")
    plt.xticks(range(len(heldout_sorted)), heldout_sorted, rotation=45)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
        logging.info(f"CV test loss bar plot saved to {savepath}")
    plt.show()


def rolling_z_score(series, window):
    """Normalizes a series using a rolling mean and standard deviation."""
    s = pd.Series(series)
    rolling_mean = s.rolling(window=window, min_periods=1).mean()
    rolling_std = s.rolling(window=window, min_periods=1).std()
    normalized = (s - rolling_mean) / (rolling_std + 1e-8)
    return normalized.fillna(0).values


def compute_cv_zscore_correlations(
    cv_results,
    cv_hparams,
    valid_df,
    dfof,
    good_pairs,
    filter_neurons,
    target_neurons,
    device,
    model_dir,
    input_size,
    output_size,
    initial_cond_size,
    snippet_length,
    stim_delay=10,
    window_size=50,
    save_dir=None,
    save_plots=True,
):
    """
    Compute rolling z-score correlations for CV results.
    
    For each fold, loads the held-out model and computes z-score normalized
    correlations between predictions and ground truth on the test set.
    
    Parameters
    ----------
    cv_results : list of dict
        Cross-validation results from training loop
    cv_hparams : dict
        Hyperparameters used for CV (must include split_strategy, n_units, num_layers)
    valid_df : pd.DataFrame
        DataFrame with all valid snippets
    dfof : np.ndarray
        Neural activity data
    good_pairs : list of tuples
        List of (config, roi) pairs to evaluate
    filter_neurons : list or None
        Filtered neuron indices
    target_neurons : list
        Target neurons for prediction
    device : torch.device
        Device for model inference
    model_dir : str
        Directory containing CV model checkpoints
    input_size, output_size, initial_cond_size : int
        Model dimensions
    snippet_length : int
        Length of snippets
    stim_delay : int
        Delay before stimulation
    window_size : int
        Window size for rolling z-score (default 50)
    save_dir : str or None
        Directory to save correlation plots (if None, uses model_dir/zscore_corr_cv)
    save_plots : bool
        Whether to save individual correlation plots
        
    Returns
    -------
    cv_results : list of dict
        Updated cv_results with 'zscore_correlations' and 'mean_zscore_corr' added
    overall_mean_corr : float
        Overall mean z-score correlation across all folds
    """
    split_strategy = cv_hparams['split_strategy']
    n_units = cv_hparams['n_units']
    num_layers = cv_hparams['num_layers']
    
    # Map split strategy to DataFrame column
    strategy_to_column = {
        'trial': 'first_trial',
        'session': 'session',
        'config': 'first_config',
        'electrode': 'first_electrode',
        'current': 'first_current',
    }
    split_column = strategy_to_column[split_strategy]
    
    # Create output directory
    if save_dir is None:
        save_dir = os.path.join(model_dir, 'zscore_corr_cv')
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)
    
    # Build holdout -> checkpoint mapping
    holdout_to_checkpoint = {}
    for r in cv_results:
        holdout_elem = r['holdout_element']
        fold_idx = r['fold'] - 1
        checkpoint_path = os.path.join(model_dir, f"cv_fold_{fold_idx}_best.pth")
        holdout_to_checkpoint[holdout_elem] = checkpoint_path
    
    # Create neuron index mappings
    if filter_neurons is not None:
        neuron_to_dfof_idx = {n: i for i, n in enumerate(filter_neurons)}
    else:
        neuron_to_dfof_idx = {n: n - 1 for n in range(1, dfof[0].shape[1] + 1)}
    neuron_to_output_idx = {n: i for i, n in enumerate(target_neurons)}
    
    # Model cache
    model_cache = {}
    
    def get_model_for_holdout(holdout_elem):
        if holdout_elem not in model_cache:
            if holdout_elem not in holdout_to_checkpoint:
                return None
            checkpoint = holdout_to_checkpoint[holdout_elem]
            if not os.path.exists(checkpoint):
                return None
            model = RNNModel(
                input_size=input_size,
                units=n_units,
                output_size=output_size,
                num_layers=num_layers,
                initial_cond_size=initial_cond_size
            )
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            model.to(device)
            model.eval()
            model_cache[holdout_elem] = model
        return model_cache[holdout_elem]
    
    # Track correlations per fold
    all_fold_correlations = []
    
    for fold_result in cv_results:
        holdout_elem = fold_result['holdout_element']
        fold_idx = fold_result['fold']
        
        # Get test snippets for this fold
        if pd.isna(holdout_elem):
            test_df = valid_df[valid_df[split_column].isna()]
        else:
            test_df = valid_df[valid_df[split_column] == holdout_elem]
        
        if len(test_df) == 0:
            logging.warning(f"Fold {fold_idx}: No test samples for holdout={holdout_elem}")
            fold_result['zscore_correlations'] = {}
            fold_result['mean_zscore_corr'] = np.nan
            continue
        
        model = get_model_for_holdout(holdout_elem)
        if model is None:
            logging.warning(f"Fold {fold_idx}: Could not load model for holdout={holdout_elem}")
            fold_result['zscore_correlations'] = {}
            fold_result['mean_zscore_corr'] = np.nan
            continue
        
        fold_correlations = {}
        fold_all_corrs = []
        
        for config, roi in good_pairs:
            if roi not in target_neurons:
                continue
            
            if roi not in fold_correlations:
                fold_correlations[roi] = []
            
            output_idx = neuron_to_output_idx[roi]
            dfof_idx = neuron_to_dfof_idx[roi]
            
            # Filter test_df for this config
            config_df = test_df[test_df['first_config'] == config]
            
            for idx, row in config_df.iterrows():
                # Get prediction
                stim_input = np.expand_dims(row['stim_snippet'].astype(np.float32), axis=0)
                activity_input = np.expand_dims(row['initial_condition'].astype(np.float32), axis=0)
                stim_input_t = torch.from_numpy(stim_input).to(device)
                activity_input_t = torch.from_numpy(activity_input).to(device)
                
                with torch.no_grad():
                    outputs = model((stim_input_t, activity_input_t))
                y_pred_raw = outputs[0, :, output_idx].cpu().numpy()
                
                # Get ground truth
                session = row['session']
                snippet_start = row['snippet_start']
                y_true_raw = dfof[session][snippet_start:snippet_start + snippet_length, dfof_idx]
                
                # Ensure same length
                min_len = min(len(y_pred_raw), len(y_true_raw))
                y_pred_raw = y_pred_raw[:min_len]
                y_true_raw = y_true_raw[:min_len]
                
                # Apply rolling z-score normalization
                y_pred_norm = rolling_z_score(y_pred_raw, window_size)
                y_true_norm = rolling_z_score(y_true_raw, window_size)
                
                # Compute correlation
                corr_value = np.corrcoef(y_pred_norm, y_true_norm)[0, 1]
                if not np.isnan(corr_value):
                    fold_correlations[roi].append(corr_value)
                    fold_all_corrs.append(corr_value)
                
                # Save plot if requested
                if save_plots:
                    plt.figure(figsize=(10, 5))
                    
                    # Plot stim times
                    input_data = row['stim_snippet']
                    stim_times_idx = np.nonzero(input_data)[0]
                    electrodes_idx = np.nonzero(input_data)[1]
                    colors_map = {3: 'r', 4: 'b', 5: 'g'}
                    for t_idx, e_idx in zip(stim_times_idx, electrodes_idx):
                        current = int(input_data[t_idx, e_idx])
                        if current in colors_map:
                            plt.axvline(x=t_idx, color=colors_map[current], linestyle='--', alpha=0.3)
                    
                    plt.plot(y_pred_raw, 'blue', linewidth=1.0, label="predicted (original scale)", alpha=0.8)
                    plt.plot(y_true_raw, 'orange', linewidth=1.0, label="ground truth (original scale)", alpha=0.8)
                    
                    plt.title(f"Fold {fold_idx} | ROI {roi}, Config {config} | Z-Corr={corr_value:.3f}")
                    plt.ylim(-0.2, 0.5)
                    plt.ylabel("df/F")
                    plt.xlabel("Time (frames)")
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.legend(loc='lower right')
                    
                    plt.savefig(os.path.join(save_dir, f"fold{fold_idx}_roi{roi}_cfg{config}_idx{idx}.png"))
                    plt.close()
        
        # Store fold-level results
        fold_result['zscore_correlations'] = fold_correlations
        fold_mean = np.nanmean(fold_all_corrs) if fold_all_corrs else np.nan
        fold_result['mean_zscore_corr'] = fold_mean
        all_fold_correlations.extend(fold_all_corrs)
        
        logging.info(f"Fold {fold_idx} | Holdout={holdout_elem} | Mean Z-Corr={fold_mean:.4f} (n={len(fold_all_corrs)})")
    
    # Overall statistics
    overall_mean_corr = np.nanmean(all_fold_correlations) if all_fold_correlations else np.nan
    overall_std_corr = np.nanstd(all_fold_correlations) if all_fold_correlations else np.nan
    
    logging.info(f"Overall Mean Z-Score Correlation: {overall_mean_corr:.4f} ± {overall_std_corr:.4f} (n={len(all_fold_correlations)})")
    
    # Print summary
    print("\n" + "="*70)
    print(f"Z-SCORE CORRELATION RESULTS ({split_strategy} strategy)")
    print("="*70)
    for r in cv_results:
        print(f"  Fold {r['fold']:2d} | Holdout={str(r['holdout_element']):>4s} | "
              f"Z-Corr={r.get('mean_zscore_corr', np.nan):.4f}")
    print("-"*70)
    print(f"  OVERALL Mean Z-Score Correlation: {overall_mean_corr:.4f} ± {overall_std_corr:.4f}")
    print("="*70)
    
    # Clear model cache
    model_cache.clear()
    
    return cv_results, overall_mean_corr
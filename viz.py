import numpy as np
import pandas as pd
import os
import logging
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import matplotlib.colors as colors


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


def plot_stim_raster(trials_df, neuron, config, threshold=0.3, pre_stim_frames=10, post_stim_frames=60, 
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
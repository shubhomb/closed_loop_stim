def analyze_visual_oracle_trajectory_sem(
    traj_results,
    save_folder,
    plot_dim='2d',
    visual_cmap='twilight_shifted',
    oracle_color_mode='colormap',   # 'colormap' or 'black'
    oracle_cmap='YlGn',
    oracle_single_color='black',
    trial_alpha=0.12,
    trial_linewidth=0.9,
    avg_alpha=0.95,
    avg_linewidth=2.8,
    sem_alpha=0.16,
    dpi=180,
    fig_size=(7.2, 7.2),
    summary_fig_size=(8.0, 6.2),
    axis_label_size=16,
    tick_label_size=13,
    title_size=15,
    console_verbose=True,
    subsample_trial_count=None,
    random_seed=42,
    manual_xlim=None,
    manual_ylim=None,
):
    """
    Analyze visual and ORACLE ICMS trajectories from traj_results.

    This function:
    1. Finds ORACLE ICMS trial types:
       ORACLE trials are defined as ICMS conditions whose `condition_value`
       appears more than once among traj_results['icms_trajectory_records'].

    2. For each visual orientation and each ORACLE ICMS condition, plots:
       - all single-trial trajectories
       - average trajectory
       - SEM ribbon orthogonal to the trajectory

    3. For each class, computes the maximum normal-direction SEM along the
       trajectory as the representative SEM value.

    4. Creates a scatter + violin summary plot comparing:
       - all visual orientation representative SEM values
       - all ORACLE ICMS representative SEM values

    New option
    ----------
    subsample_trial_count : None or int
        If None:
            use all available trials in each visual orientation / ORACLE condition.

        If int:
            for each visual orientation and each ORACLE condition, randomly sample
            exactly subsample_trial_count trials before all calculations.

            If any visual orientation or ORACLE condition has fewer than
            subsample_trial_count trials, the function prints a warning and raises
            a ValueError before plotting.

    random_seed : int
        Random seed used for subsampling.

    Returns
    -------
    results : dict
    """

    # =========================================================
    # Validation
    # =========================================================
    if plot_dim.lower() != '2d':
        raise NotImplementedError(
            "This function currently supports only plot_dim='2d'. "
            "The plot_dim argument is kept for future extension."
        )

    if 'visual_trajectory_records' not in traj_results:
        raise ValueError("traj_results is missing key 'visual_trajectory_records'.")
    if 'icms_trajectory_records' not in traj_results:
        raise ValueError("traj_results is missing key 'icms_trajectory_records'.")

    visual_records = list(traj_results['visual_trajectory_records'])
    icms_records = list(traj_results['icms_trajectory_records'])

    if len(visual_records) == 0:
        raise ValueError("traj_results['visual_trajectory_records'] is empty.")
    if len(icms_records) == 0 and console_verbose:
        print("Warning: traj_results['icms_trajectory_records'] is empty.")

    if oracle_color_mode not in ['colormap', 'black']:
        raise ValueError("oracle_color_mode must be either 'colormap' or 'black'.")

    if subsample_trial_count is not None:
        if not isinstance(subsample_trial_count, int) or subsample_trial_count <= 0:
            raise ValueError("subsample_trial_count must be None or a positive integer.")

    rng = np.random.default_rng(random_seed)

    # =========================================================
    # Output folders
    # =========================================================
    root_folder = os.path.join(save_folder, 'Visual_Oracle_Trajectory_SEM_Analysis')
    visual_folder = os.path.join(root_folder, 'Visual_Trajectories')
    oracle_folder = os.path.join(root_folder, 'ORACLE_Trajectories')
    summary_folder = os.path.join(root_folder, 'Summary')

    os.makedirs(root_folder, exist_ok=True)
    os.makedirs(visual_folder, exist_ok=True)
    os.makedirs(oracle_folder, exist_ok=True)
    os.makedirs(summary_folder, exist_ok=True)

    # =========================================================
    # Helper functions
    # =========================================================
    def _try_float(x):
        try:
            return float(x), True
        except Exception:
            return x, False

    def _sorted_keys(keys):
        keys_list = list(keys)
        converted = []
        all_numeric = True

        for k in keys_list:
            val, ok = _try_float(k)
            converted.append((k, val, ok))
            if not ok:
                all_numeric = False

        if all_numeric:
            return [x[0] for x in sorted(converted, key=lambda z: z[1])]

        seen = OrderedDict()
        for k in keys_list:
            if k not in seen:
                seen[k] = None
        return list(seen.keys())

    def _sanitize_filename(s):
        s = str(s)
        s = re.sub(r'[\\/*?:"<>|]+', '_', s)
        s = s.replace(' ', '_')
        return s

    def _get_plot2d_from_record(rec):
        if 'lda_trajectory_plot2d' in rec:
            arr = np.asarray(rec['lda_trajectory_plot2d'], dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError("Record contains invalid 'lda_trajectory_plot2d'.")
            return arr[:, :2]

        if 'lda_trajectory_full' in rec:
            arr = np.asarray(rec['lda_trajectory_full'], dtype=float)
            if arr.ndim != 2:
                raise ValueError("Record contains invalid 'lda_trajectory_full'.")
            if arr.shape[1] >= 2:
                return arr[:, :2]
            raise ValueError("Record 'lda_trajectory_full' has fewer than 2 dimensions.")

        raise ValueError("Record has neither 'lda_trajectory_plot2d' nor 'lda_trajectory_full'.")

    def _truncate_and_stack(traj_list):
        lengths = [traj.shape[0] for traj in traj_list]
        min_len = int(np.min(lengths))
        arr = np.stack([traj[:min_len, :2] for traj in traj_list], axis=0)
        return arr, min_len

    def _compute_mean_normal_sem(traj_stack):
        """
        traj_stack: shape (N, T, 2)

        SEM is computed along the local normal direction of the mean trajectory.
        """
        n_trials, T, _ = traj_stack.shape
        mean_traj = np.mean(traj_stack, axis=0)

        tangent = np.zeros_like(mean_traj)

        if T == 1:
            tangent[0] = np.array([1.0, 0.0])
        else:
            tangent[0] = mean_traj[1] - mean_traj[0]
            tangent[-1] = mean_traj[-1] - mean_traj[-2]
            if T > 2:
                tangent[1:-1] = mean_traj[2:] - mean_traj[:-2]

        tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        tangent_norm[tangent_norm < 1e-12] = 1.0
        tangent_unit = tangent / tangent_norm

        normal = np.column_stack([-tangent_unit[:, 1], tangent_unit[:, 0]])

        diff = traj_stack - mean_traj[None, :, :]
        signed_normal_dev = np.sum(diff * normal[None, :, :], axis=2)

        if n_trials > 1:
            sem_scalar = np.std(signed_normal_dev, axis=0, ddof=1) / np.sqrt(n_trials)
        else:
            sem_scalar = np.zeros(T, dtype=float)

        upper = mean_traj + sem_scalar[:, None] * normal
        lower = mean_traj - sem_scalar[:, None] * normal

        return mean_traj, sem_scalar, upper, lower, tangent_unit, normal

    def _set_square_limits(ax, xs, ys, pad_ratio=0.08):
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)

        x_range = x_max - x_min
        y_range = y_max - y_min
        max_range = max(x_range, y_range, 1e-6)

        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)

        half = 0.5 * max_range * (1.0 + pad_ratio)

        ax.set_xlim(x_center - half, x_center + half)
        ax.set_ylim(y_center - half, y_center + half)
        ax.set_aspect('equal', adjustable='box')

    def _plot_group_trajectory(
        traj_stack,
        color,
        save_path,
        group_title,
        line_label_text,
    ):
        """
        Plot all single trials + mean + orthogonal SEM ribbon for one group.
        """
        n_trials, T, _ = traj_stack.shape

        mean_traj, sem_scalar, upper, lower, tangent_unit, normal = _compute_mean_normal_sem(traj_stack)

        rep_sem = float(np.nanmax(sem_scalar))
        rep_sem_idx = int(np.nanargmax(sem_scalar)) if len(sem_scalar) > 0 else 0

        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

        # SEM ribbon
        poly_x = np.concatenate([upper[:, 0], lower[::-1, 0]])
        poly_y = np.concatenate([upper[:, 1], lower[::-1, 1]])
        ax.fill(
            poly_x,
            poly_y,
            color=color,
            alpha=sem_alpha,
            linewidth=0,
            zorder=1
        )

        # Single trials
        for i in range(n_trials):
            ax.plot(
                traj_stack[i, :, 0],
                traj_stack[i, :, 1],
                color=color,
                alpha=trial_alpha,
                linewidth=trial_linewidth,
                zorder=2
            )

        # Mean trajectory
        ax.plot(
            mean_traj[:, 0],
            mean_traj[:, 1],
            color=color,
            alpha=avg_alpha,
            linewidth=avg_linewidth,
            zorder=3
        )

        # Mark representative max-SEM point
        ax.scatter(
            mean_traj[rep_sem_idx, 0],
            mean_traj[rep_sem_idx, 1],
            s=40,
            color=color,
            edgecolor='black',
            linewidth=0.5,
            zorder=4
        )

        if manual_xlim is not None:
            if len(manual_xlim) != 2:
                raise ValueError("manual_xlim must be None or a tuple/list like (xmin, xmax).")
            ax.set_xlim(tuple(manual_xlim))
        else:
            all_x = np.concatenate([
                traj_stack[:, :, 0].ravel(),
                upper[:, 0].ravel(),
                lower[:, 0].ravel()
            ])
        
        if manual_ylim is not None:
            if len(manual_ylim) != 2:
                raise ValueError("manual_ylim must be None or a tuple/list like (ymin, ymax).")
            ax.set_ylim(tuple(manual_ylim))
        else:
            all_y = np.concatenate([
                traj_stack[:, :, 1].ravel(),
                upper[:, 1].ravel(),
                lower[:, 1].ravel()
            ])
        
        if manual_xlim is None or manual_ylim is None:
            if manual_xlim is None and manual_ylim is None:
                _set_square_limits(ax, all_x, all_y, pad_ratio=0.10)
            else:
                if manual_xlim is None:
                    x_min, x_max = np.min(all_x), np.max(all_x)
                    x_pad = 0.10 * max(1e-9, x_max - x_min)
                    ax.set_xlim(x_min - x_pad, x_max + x_pad)
        
                if manual_ylim is None:
                    y_min, y_max = np.min(all_y), np.max(all_y)
                    y_pad = 0.10 * max(1e-9, y_max - y_min)
                    ax.set_ylim(y_min - y_pad, y_max + y_pad)
        
        ax.set_aspect('equal', adjustable='box')
        
        ax.set_xlabel("LD 1", fontsize=axis_label_size, fontweight='bold')
        ax.set_ylabel("LD 2", fontsize=axis_label_size, fontweight='bold')
        ax.tick_params(axis='both', labelsize=tick_label_size, width=1.4)
        ax.grid(True, alpha=0.22)

        title_text = (
            f"{group_title}\n"
            f"{line_label_text} | n = {n_trials} | Max normal SEM = {rep_sem:.4f}"
        )
        ax.set_title(title_text, fontsize=title_size, fontweight='bold')

        for spine in ax.spines.values():
            spine.set_linewidth(1.4)

        plt.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        return {
            'n_trials': n_trials,
            'n_timepoints': T,
            'mean_traj': mean_traj,
            'sem_scalar': sem_scalar,
            'upper': upper,
            'lower': lower,
            'tangent_unit': tangent_unit,
            'normal': normal,
            'rep_sem': rep_sem,
            'rep_sem_idx': rep_sem_idx,
            'save_path': save_path,
        }

    def _subsample_group_records(group_dict, group_name):
        """
        group_dict: OrderedDict-like or dict of key -> list[records]
        Returns new OrderedDict with subsampled records if subsample_trial_count is not None.
        """
        if subsample_trial_count is None:
            return OrderedDict((k, list(v)) for k, v in group_dict.items())

        too_few = []
        for k, recs in group_dict.items():
            if len(recs) < subsample_trial_count:
                too_few.append((k, len(recs)))

        if len(too_few) > 0:
            print("\n" + "=" * 80)
            print(f"ERROR: Cannot subsample {group_name}.")
            print(f"Requested subsample_trial_count = {subsample_trial_count}")
            print("The following classes have fewer trials than requested:")
            for k, n in too_few:
                print(f"  {group_name} class {k}: only {n} trials")
            print("=" * 80)
            raise ValueError(
                f"Some {group_name} classes have fewer trials than "
                f"subsample_trial_count={subsample_trial_count}."
            )

        sampled = OrderedDict()
        for k, recs in group_dict.items():
            recs = list(recs)
            idx = rng.choice(np.arange(len(recs)), size=subsample_trial_count, replace=False)
            idx = np.sort(idx)
            sampled[k] = [recs[i] for i in idx]

        return sampled

    # =========================================================
    # Group visual records by label_deg
    # =========================================================
    visual_groups_raw = defaultdict(list)
    for rec in visual_records:
        if 'label_deg' not in rec:
            raise ValueError("A visual trajectory record is missing 'label_deg'.")
        visual_groups_raw[rec['label_deg']].append(rec)

    visual_labels_sorted = _sorted_keys(visual_groups_raw.keys())
    visual_groups_raw = OrderedDict((lab, visual_groups_raw[lab]) for lab in visual_labels_sorted)

    # =========================================================
    # Group ICMS records by condition_value, define ORACLE conditions
    # =========================================================
    icms_groups = defaultdict(list)
    for rec in icms_records:
        if 'condition_value' not in rec:
            raise ValueError("An ICMS trajectory record is missing 'condition_value'.")
        icms_groups[rec['condition_value']].append(rec)

    icms_conditions_sorted = _sorted_keys(icms_groups.keys())

    oracle_groups_raw = OrderedDict()
    oracle_counts = OrderedDict()

    for cond in icms_conditions_sorted:
        cnt = len(icms_groups[cond])
        if cnt > 1:
            oracle_groups_raw[cond] = icms_groups[cond]
            oracle_counts[cond] = cnt

    oracle_conditions_sorted = list(oracle_groups_raw.keys())

    # =========================================================
    # Console report: ORACLE trial summary
    # =========================================================
    if console_verbose:
        print("\n" + "=" * 80)
        print("ORACLE Trial Detection Summary")
        print("=" * 80)
        print(f"Total ICMS trajectory records: {len(icms_records)}")
        print(f"Total unique ICMS conditions: {len(icms_groups)}")
        print(f"Detected ORACLE conditions (repeated > 1 time): {len(oracle_groups_raw)}")

        if len(oracle_groups_raw) > 0:
            print("\nORACLE condition repetition counts:")
            for cond in oracle_conditions_sorted:
                print(f"  Condition {cond}: {oracle_counts[cond]} trials")

            unique_repeat_counts = sorted(set(oracle_counts.values()))
            if len(unique_repeat_counts) == 1:
                print(
                    f"\nAll ORACLE conditions have the same repetition count: "
                    f"{unique_repeat_counts[0]}"
                )
            else:
                print("\nWARNING:")
                print("Different ORACLE conditions have different repetition counts.")
                print("Please double-check whether this is expected.")
                print(f"Observed repetition counts: {unique_repeat_counts}")
        else:
            print("\nNo ORACLE conditions were found (no repeated ICMS condition_value).")

        print("=" * 80)

    # =========================================================
    # Subsample if requested
    # =========================================================
    if subsample_trial_count is not None:
        print("\n" + "=" * 80)
        print("Subsampling mode is ON")
        print(f"subsample_trial_count = {subsample_trial_count}")
        print(f"random_seed = {random_seed}")
        print("Each visual orientation and each ORACLE ICMS condition will be")
        print("randomly subsampled to exactly this number of trials.")
        print("If any class has fewer trials, the function will stop with an error.")
        print("=" * 80)

    visual_groups = _subsample_group_records(visual_groups_raw, "Visual")

    if len(oracle_groups_raw) > 0:
        oracle_groups = _subsample_group_records(oracle_groups_raw, "ORACLE")
    else:
        oracle_groups = OrderedDict()

    # Updated sorted keys after subsampling
    visual_labels_sorted = list(visual_groups.keys())
    oracle_conditions_sorted = list(oracle_groups.keys())

    # Oracle counts after subsampling
    oracle_counts_used = OrderedDict((cond, len(oracle_groups[cond])) for cond in oracle_conditions_sorted)
    visual_counts_used = OrderedDict((lab, len(visual_groups[lab])) for lab in visual_labels_sorted)

    if console_verbose and subsample_trial_count is not None:
        print("\nSubsampling finished successfully.")
        print("Used visual trial counts:")
        for lab, cnt in visual_counts_used.items():
            print(f"  Visual {lab}: {cnt} trials")
        print("Used ORACLE trial counts:")
        for cond, cnt in oracle_counts_used.items():
            print(f"  ORACLE {cond}: {cnt} trials")

    # =========================================================
    # Colors
    # =========================================================
    visual_norm = colors.Normalize(vmin=0.0, vmax=360.0)
    visual_cmap_obj = cm.get_cmap(visual_cmap)

    if len(oracle_conditions_sorted) > 0:
        if oracle_color_mode == 'black':
            oracle_color_map = {cond: oracle_single_color for cond in oracle_conditions_sorted}
        else:
            oracle_cmap_obj = cm.get_cmap(oracle_cmap, max(2, len(oracle_conditions_sorted)))
            oracle_color_map = {
                cond: oracle_cmap_obj(i / max(1, len(oracle_conditions_sorted) - 1))
                for i, cond in enumerate(oracle_conditions_sorted)
            }
    else:
        oracle_color_map = {}

    # =========================================================
    # Plot per-visual-class figures
    # =========================================================
    if console_verbose:
        print("\nGenerating visual trajectory plots...")

    visual_results = OrderedDict()
    visual_rep_sem_values = []
    visual_rep_sem_labels = []

    for lab in visual_labels_sorted:
        recs = visual_groups[lab]
        traj_list = [_get_plot2d_from_record(r) for r in recs]
        traj_stack, min_len = _truncate_and_stack(traj_list)

        try:
            lab_float = float(lab)
            color = visual_cmap_obj(visual_norm(lab_float))
            label_text = f"{lab_float:.1f}°"
            file_stub = f"visual_{lab_float:.1f}deg"
        except Exception:
            color = visual_cmap_obj(0.5)
            label_text = str(lab)
            file_stub = f"visual_{_sanitize_filename(lab)}"

        save_path = os.path.join(visual_folder, f"{_sanitize_filename(file_stub)}.png")

        res = _plot_group_trajectory(
            traj_stack=traj_stack,
            color=color,
            save_path=save_path,
            group_title="Visual Stimulus Trajectories",
            line_label_text=f"Orientation = {label_text}",
        )

        visual_results[lab] = res
        visual_rep_sem_values.append(res['rep_sem'])
        visual_rep_sem_labels.append(lab)

    # =========================================================
    # Plot per-ORACLE-class figures
    # =========================================================
    if console_verbose:
        print("Generating ORACLE trajectory plots...")

    oracle_results = OrderedDict()
    oracle_rep_sem_values = []
    oracle_rep_sem_labels = []

    for cond in oracle_conditions_sorted:
        recs = oracle_groups[cond]
        traj_list = [_get_plot2d_from_record(r) for r in recs]
        traj_stack, min_len = _truncate_and_stack(traj_list)

        color = oracle_color_map[cond]
        save_path = os.path.join(
            oracle_folder,
            f"oracle_{_sanitize_filename(cond)}.png"
        )

        res = _plot_group_trajectory(
            traj_stack=traj_stack,
            color=color,
            save_path=save_path,
            group_title="ORACLE ICMS Trajectories",
            line_label_text=f"Condition = {cond}",
        )

        oracle_results[cond] = res
        oracle_rep_sem_values.append(res['rep_sem'])
        oracle_rep_sem_labels.append(cond)

    # =========================================================
    # Summary scatter + violin plot
    # =========================================================
    if console_verbose:
        print("Generating representative SEM summary plot...")

    if subsample_trial_count is None:
        summary_name = "Representative_Max_SEM_Comparison.png"
    else:
        summary_name = f"Representative_Max_SEM_Comparison_subsampleN{subsample_trial_count}.png"

    summary_path = os.path.join(summary_folder, summary_name)

    fig, ax = plt.subplots(figsize=summary_fig_size, dpi=dpi)

    data_for_violin = []
    positions = []
    group_names = []

    if len(visual_rep_sem_values) > 0:
        data_for_violin.append(np.asarray(visual_rep_sem_values, dtype=float))
        positions.append(1)
        group_names.append("Visual")

    if len(oracle_rep_sem_values) > 0:
        data_for_violin.append(np.asarray(oracle_rep_sem_values, dtype=float))
        positions.append(2 if len(positions) == 1 else 1)
        group_names.append("ORACLE")

    if len(data_for_violin) > 0:
        violin = ax.violinplot(
            data_for_violin,
            positions=positions,
            widths=0.65,
            showmeans=False,
            showmedians=True,
            showextrema=False
        )

        violin_facecolors = [
            (0.35, 0.55, 0.90, 0.28),
            (0.35, 0.75, 0.45, 0.28),
        ]

        for i, body in enumerate(violin['bodies']):
            fc = violin_facecolors[min(i, len(violin_facecolors) - 1)]
            body.set_facecolor(fc)
            body.set_edgecolor('none')
            body.set_alpha(fc[3])

        if 'cmedians' in violin:
            violin['cmedians'].set_color('black')
            violin['cmedians'].set_linewidth(2.0)

    scatter_rng = np.random.default_rng(42)

    if len(visual_rep_sem_values) > 0:
        x = 1 + scatter_rng.uniform(-0.08, 0.08, size=len(visual_rep_sem_values))
        ax.scatter(
            x,
            visual_rep_sem_values,
            s=55,
            color=(0.20, 0.35, 0.80, 0.78),
            edgecolor=(0.10, 0.10, 0.35, 0.95),
            linewidth=0.8,
            zorder=3,
            label='Visual classes'
        )

    if len(oracle_rep_sem_values) > 0:
        oracle_x_pos = 2 if len(visual_rep_sem_values) > 0 else 1
        x = oracle_x_pos + scatter_rng.uniform(-0.08, 0.08, size=len(oracle_rep_sem_values))
        ax.scatter(
            x,
            oracle_rep_sem_values,
            s=55,
            color=(0.15, 0.60, 0.22, 0.78),
            edgecolor=(0.05, 0.25, 0.08, 0.95),
            linewidth=0.8,
            zorder=3,
            label='ORACLE classes'
        )

    if len(visual_rep_sem_values) > 0:
        ax.scatter(
            [1],
            [np.mean(visual_rep_sem_values)],
            s=140,
            marker='D',
            color=(0.10, 0.25, 0.70, 1.0),
            edgecolor='black',
            linewidth=1.0,
            zorder=4
        )

    if len(oracle_rep_sem_values) > 0:
        oracle_x_pos = 2 if len(visual_rep_sem_values) > 0 else 1
        ax.scatter(
            [oracle_x_pos],
            [np.mean(oracle_rep_sem_values)],
            s=140,
            marker='D',
            color=(0.05, 0.45, 0.12, 1.0),
            edgecolor='black',
            linewidth=1.0,
            zorder=4
        )

    if len(visual_rep_sem_values) > 0 and len(oracle_rep_sem_values) > 0:
        xticks = [1, 2]
        xticklabels = ['Visual', 'ORACLE']
    elif len(visual_rep_sem_values) > 0:
        xticks = [1]
        xticklabels = ['Visual']
    elif len(oracle_rep_sem_values) > 0:
        xticks = [1]
        xticklabels = ['ORACLE']
    else:
        xticks = [1]
        xticklabels = ['No data']

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=axis_label_size, fontweight='bold')
    ax.set_ylabel("Representative max normal SEM", fontsize=axis_label_size, fontweight='bold')

    if subsample_trial_count is None:
        title_extra = ""
    else:
        title_extra = f" | subsample n={subsample_trial_count}"

    ax.set_title(
        f"Representative SEM Comparison{title_extra}",
        fontsize=title_size + 1,
        fontweight='bold'
    )

    ax.tick_params(axis='y', labelsize=tick_label_size, width=1.4)
    ax.grid(True, axis='y', alpha=0.25)

    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    text_lines = []
    if len(visual_rep_sem_values) > 0:
        text_lines.append(f"Visual classes: {len(visual_rep_sem_values)}")
        text_lines.append(f"Visual mean: {np.mean(visual_rep_sem_values):.4f}")
    if len(oracle_rep_sem_values) > 0:
        text_lines.append(f"ORACLE classes: {len(oracle_rep_sem_values)}")
        text_lines.append(f"ORACLE mean: {np.mean(oracle_rep_sem_values):.4f}")
    if subsample_trial_count is not None:
        text_lines.append(f"Subsample n/class: {subsample_trial_count}")

    if len(text_lines) > 0:
        ax.text(
            0.98, 0.98,
            "\n".join(text_lines),
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=tick_label_size - 1,
            bbox=dict(
                boxstyle='round,pad=0.35',
                facecolor='white',
                alpha=0.75,
                edgecolor='0.7'
            )
        )

    plt.tight_layout()
    fig.savefig(summary_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    # =========================================================
    # Final result dict
    # =========================================================
    results = {
        'root_folder': root_folder,
        'visual_folder': visual_folder,
        'oracle_folder': oracle_folder,
        'summary_folder': summary_folder,

        'subsample_trial_count': subsample_trial_count,
        'random_seed': random_seed,
        'visual_counts_raw': OrderedDict((lab, len(visual_groups_raw[lab])) for lab in visual_groups_raw),
        'visual_counts_used': visual_counts_used,
        'oracle_counts_raw': oracle_counts,
        'oracle_counts_used': oracle_counts_used,

        'oracle_conditions': oracle_conditions_sorted,
        'oracle_repeat_count_consistent': (
            len(set(oracle_counts.values())) <= 1 if len(oracle_counts) > 0 else True
        ),

        'visual_results': visual_results,
        'oracle_results': oracle_results,

        'visual_rep_sem_values': np.asarray(visual_rep_sem_values, dtype=float),
        'visual_rep_sem_labels': visual_rep_sem_labels,
        'oracle_rep_sem_values': np.asarray(oracle_rep_sem_values, dtype=float),
        'oracle_rep_sem_labels': oracle_rep_sem_labels,

        'summary_plot_path': summary_path,
    }

    # =========================================================
    # Console summary
    # =========================================================
    if console_verbose:
        print("\n" + "=" * 80)
        print("Analysis finished.")
        print("=" * 80)
        print(f"Visual figures saved to: {visual_folder}")
        print(f"ORACLE figures saved to: {oracle_folder}")
        print(f"Summary figure saved to: {summary_path}")
        print(f"Number of visual classes plotted: {len(visual_results)}")
        print(f"Number of ORACLE classes plotted: {len(oracle_results)}")
        if subsample_trial_count is not None:
            print(f"Subsampled trial count per class: {subsample_trial_count}")
            print(f"Random seed: {random_seed}")
        print("=" * 80)

    return results


%% Read in results from 1/30 mouse session
clear all; 
close all;
exp_data = load("data/1-30-2026/Complete_Experiment_Data.mat").experiment_results;
N_stim_periods = exp_data.metadata.N;
K_lookback_periods = exp_data.metadata.K;
section_duration_mins = exp_data.metadata.T_mins;
recordedChannels = exp_data.metadata.channels; 


% Loop through each of the 23 experiment sections
for s = 1:length(exp_data.sections)
    fprintf('--- Section %d ---\n', s);
    raw_data = exp_data.sections(s).raw_data; % size (n_iterations x n_electrodes)
    % Access the events for this specific section
    current_events = exp_data.sections(s).events;
    
    % Check if the events array is empty
    if isempty(current_events)
        fprintf('  No events recorded.\n');
        continue;
    end
    
    % Iterate through each event/action in the struct array
    for e = 1:length(current_events)
        rel_time = current_events(e).rel_time;
        action   = current_events(e).action;
        channel  = current_events(e).channel;
        iter     = current_events(e).iteration;
        
        % Print a formatted summary of the event
        fprintf('  [T+%.2fs] Iter: %d | Action: %s | Channel: %d\n', ...
            rel_time, iter, action, channel);
    end
    fprintf('\n');
end

% Analysis of Spike Threshold Crossings vs. Stim Events
results = struct();

for s = 1:length(exp_data.sections)
    sec = exp_data.sections(s);
    
    % 1. Skip sections with no thresholds or no data
    if isempty(sec.thresholds_used) || any(isnan(sec.thresholds_used(:)))
        continue;
    end
    
    % thresholds_used is 2 x N_electrodes
    % Row 1: Low Threshold, Row 2: High Threshold
    low_thresh = sec.thresholds_used(1, :);
    high_thresh = sec.thresholds_used(2, :);
    
    [num_iters, num_electrodes] = size(sec.raw_data);
    
    % 2. Find every instance of a threshold crossing
    % These logical masks find where activity is outside the [Low, High] bounds
    cross_low = sec.raw_data < low_thresh;
    cross_high = sec.raw_data > high_thresh;
    
    % 3. Augment the existing events structure
    if ~isempty(sec.events)
        % Add new fields to store our findings
        for e = 1:length(sec.events)
            iter = sec.events(e).iteration;
            
            % Find which channels crossed at this specific iteration
            ch_low = recordedChannels(find(cross_low(iter, :)));
            ch_high = recordedChannels(find(cross_high(iter, :)));
            
            % Store the results back into the struct
            exp_data.sections(s).events(e).crossed_low_channels = ch_low;
            exp_data.sections(s).events(e).crossed_high_channels = ch_high;
            
            % Print Verification
            if ~isempty(ch_low) || ~isempty(ch_high)
                fprintf('Sec %d | Iter %d: Action %s on Ch %d. Threshold triggers: [Low: %s] [High: %s]\n', ...
                    s, iter, sec.events(e).action, sec.events(e).channel, ...
                    num2str(ch_low), num2str(ch_high));
            end
        end
    end
end



%% Figure: Closed-Loop Performance with Rank-Sum-Test Verification Violin
% Generates one Figure per Channel with subplots and p-values.

% --- Parameters ---
preWindowIter  = 0;
postWindowIter = 5;
isSum = 1; 


% Colors & Styling
pureRed  = [1.0, 0.0, 0.0];
pureBlue = [0.0, 0.0, 0.5];
alphaLineStim   = 0.8;
alphaLineNoStim = 0.4;
pointSize = 10;
pointAlpha = 0.15; % Slightly lower to see mean lines better
jitterWidth = 0.12;
maxHalfWidth = 0.35;

% Violin smoothing settings
binEdges = -0.5:1:40.5;
binCenters = 0:40;
kk = -1:1;
g = exp(-(kk.^2) / (2*2.0^2)); g = g / sum(g);

stim_sections = find([exp_data.sections.is_stim_enabled] == 1);
num_subplots = length(stim_sections);
sub_rows = ceil(sqrt(num_subplots));
sub_cols = ceil(num_subplots / sub_rows);

for cIdx = 1:length(recordedChannels)
    ch_num = recordedChannels(cIdx);
    figure('Name', sprintf('Ch %d Stat Analysis', ch_num), 'Color', 'w', ...
           'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);
    
    for s_idx = 1:num_subplots
        s = stim_sections(s_idx);
        sec = exp_data.sections(s);
        
        l_thresh = sec.thresholds_used(1, cIdx);
        h_thresh = sec.thresholds_used(2, cIdx);
        
        spikeValuesByMask = cell(1,4);
        
        for e = 1:length(sec.events)
            ev = sec.events(e);
            iter = ev.iteration;
            val = sec.raw_data(iter, cIdx);
            
            isLow  = val < l_thresh;
            isHigh = val > h_thresh;
            isTargetStim = strcmpi(ev.action, 'STIM') && (ev.channel == ch_num);
            isGlobalSham = strcmpi(ev.action, 'SHAM_CONTROL');            
            mIdx = 0;
            % Channel Low + Global Sham (Strictly NO stimulation anywhere)
            if isLow && isGlobalSham,  mIdx = 1; end
            % Channel Low + Targeted Stim (Closed-loop success)
            if isLow && isTargetStim,  mIdx = 2; end
            % Channel High + Global Sham (Strictly NO stimulation anywhere)
            if isHigh && isGlobalSham, mIdx = 3; end
            % Channel High + Targeted Stim (Closed-loop success)
            if isHigh && isTargetStim, mIdx = 4; end
            
            % NOTE: if a DIFFERENT channel was stimulated while this channel 
            % crossed a threshold, mIdx remains 0 and the data is ignored.
            
            if mIdx > 0
                idxStart = max(1, iter - preWindowIter);
                idxEnd   = min(size(sec.raw_data,1), iter + postWindowIter);
                windowData = sec.raw_data(idxStart:idxEnd, cIdx);
                if isSum == 1
                    spikeValuesByMask{mIdx} = [spikeValuesByMask{mIdx}; sum(windowData(:))];
                else
                    spikeValuesByMask{mIdx} = [spikeValuesByMask{mIdx}; windowData(:)];
                end
            end
        end
        subplot(sub_rows, sub_cols, s_idx);
        hold on;
        xPos = [0.8, 1.2, 2.3, 2.7];
        
        % --- Statistical Testing ---
        % Test Low State (Mask 1 vs 2)
        pLow = NaN;
        if ~isempty(spikeValuesByMask{1}) && ~isempty(spikeValuesByMask{2})
            pLow = ranksum(spikeValuesByMask{1}, spikeValuesByMask{2});        
        end
       
        % Test High State (Mask 3 vs 4)
        pHigh = NaN;
        if ~isempty(spikeValuesByMask{3}) && ~isempty(spikeValuesByMask{4})
            pHigh = ranksum(spikeValuesByMask{3}, spikeValuesByMask{4});        
        end

        % --- Plotting Logic ---
        colors = {pureRed, pureRed, pureBlue, pureBlue};
        alphas = {alphaLineNoStim, alphaLineStim, alphaLineNoStim, alphaLineStim};

        for sp = 1:4
            v = spikeValuesByMask{sp};
            n = length(v); % Sample size
            if isempty(v), continue; end
            
            scatter(xPos(sp) + (rand(size(v))-0.5)*jitterWidth, v, pointSize, ...
                colors{sp}, 'filled', 'MarkerFaceAlpha', pointAlpha);
            
            counts = histcounts(v, binEdges, 'Normalization', 'pdf');
            dens = conv(counts, g, 'same');
            w = (dens / (max(dens)+1e-6)) * maxHalfWidth;
            
            patch([xPos(sp)-w, fliplr(xPos(sp)+w)], [binCenters, fliplr(binCenters)], ...
                colors{sp}, 'FaceAlpha', 0.25 * alphas{sp}, 'EdgeColor', colors{sp});
            
            plot([xPos(sp)-0.15, xPos(sp)+0.15], [median(v), median(v)], 'k-', 'LineWidth', 2);
            % PRINT COUNT (n)
            text(xPos(sp), -3, sprintf('n=%d', n), 'HorizontalAlignment', 'center', ...
                'FontSize', 7, 'FontWeight', 'normal', 'Color', [0.4 0.4 0.4]);
        end
        
        % --- Annotate P-Values ---
        text(1.0, 38, sprintf('p=%.3f', pLow), 'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
        text(2.5, 38, sprintf('p=%.3f', pHigh), 'HorizontalAlignment', 'center', 'FontSize', 8, 'FontWeight', 'bold');
        
        title(sprintf('Sec %d', s));
        xlim([0.3 3.2]); ylim([0 40]);
        set(gca, 'XTick', [1, 2.5], 'XTickLabel', {'Low', 'High'}, 'FontSize', 7);
        grid on;
    end
    if isSum == 1
        sgtitle(sprintf('Ch %d: Sham vs Stim Spike Sum t=[%d, %d] Summed', ch_num, preWindowIter, postWindowIter));
    else
        sgtitle(sprintf('Ch %d: Sham vs Stim Spike Sum t=[%d, %d] Separate Iters', ch_num, preWindowIter, postWindowIter));
    end
    saveas(gcf, sprintf("figures/1-30-2026/section_violins/channel_%d_section_all.png", ch_num));
end



%% Figure: Global Session Analysis (Pooled Sections)
% Collapses all stimulation-enabled sections into a single comprehensive plot per channel.

% --- Parameters (consistent with previous steps) ---
recordedChannels = exp_data.metadata.channels; 

% Style Definitions
pureRed  = [1.0, 0.0, 0.0];
pureBlue = [0.0, 0.0, 0.5];
alphaLineStim   = 0.8;
alphaLineNoStim = 0.4;
pointSize = 8;     % Smaller points because we have much more data now
pointAlpha = 0.1;   % Lower alpha to handle overlap/density
jitterWidth = 0.1;
maxHalfWidth = 0.5;
sigma = 5; 
% Smoothing
binEdges = -0.5:1:40.5;
binCenters = 0:40;
kk = -3:3; 
g = exp(-(kk.^2)/(2*sigma^2)); 
g = g/sum(g);

stim_sections = find([exp_data.sections.is_stim_enabled] == 1);

for cIdx = 1:length(recordedChannels)
    ch_num = recordedChannels(cIdx);
    figure('Name', sprintf('Global Ch %d Analysis', ch_num), 'Color', 'w');
    hold on;
    
    % Initialize Pooled Buckets
    pooledSpikes = cell(1,4); % 1:LowSham, 2:LowStim, 3:HighSham, 4:HighStim
    
    % --- POOLING DATA ---
    for s = stim_sections
        sec = exp_data.sections(s);
        l_thresh = sec.thresholds_used(1, cIdx);
        h_thresh = sec.thresholds_used(2, cIdx);
        
        for e = 1:length(sec.events)
            ev = sec.events(e);
            iter = ev.iteration;
            val = sec.raw_data(iter, cIdx);
            
            isLow  = val < l_thresh;
            isHigh = val > h_thresh;
            isTargetStim = strcmpi(ev.action, 'STIM') && (ev.channel == ch_num);
            isGlobalSham = strcmpi(ev.action, 'SHAM_CONTROL');            
            mIdx = 0;
            % Channel Low + Global Sham (Strictly NO stimulation anywhere)
            if isLow && isGlobalSham,  mIdx = 1; end
            % Channel Low + Targeted Stim (Closed-loop success)
            if isLow && isTargetStim,  mIdx = 2; end
            % Channel High + Global Sham (Strictly NO stimulation anywhere)
            if isHigh && isGlobalSham, mIdx = 3; end
            % Channel High + Targeted Stim (Closed-loop success)
            if isHigh && isTargetStim, mIdx = 4; end
            
            % NOTE: if a DIFFERENT channel was stimulated while this channel 
            % crossed a threshold, mIdx remains 0 and the data is ignored.
            
            if mIdx > 0
                idxStart = max(1, iter - preWindowIter);
                idxEnd   = min(size(sec.raw_data,1), iter + postWindowIter);
                windowData = sec.raw_data(idxStart:idxEnd, cIdx);
                if isSum == 1
                    pooledSpikes{mIdx} = [pooledSpikes{mIdx}; sum(windowData(:))];    
                else
                    pooledSpikes{mIdx} = [pooledSpikes{mIdx}; windowData(:)];    
                end
            end
        end
    end
    
    % --- STATS & PLOTTING ---
    xPos = [0.8, 1.2, 2.3, 2.7];
    colors = {pureRed, pureRed, pureBlue, pureBlue};
    alphas = {alphaLineNoStim, alphaLineStim, alphaLineNoStim, alphaLineStim};
    % 1. Find the dynamic maximum for this channel to set bin edges
    maxSpikeVal = 40; % Default minimum height
    for sp = 1:4
        if ~isempty(pooledSpikes{sp})
            maxSpikeVal = max(maxSpikeVal, max(pooledSpikes{sp}));
        end
    end
    binEdges = -0.5:1:(ceil(maxSpikeVal) + 0.5);
    binCenters = 0:ceil(maxSpikeVal);
    for sp = 1:4
        v = pooledSpikes{sp};
        n = length(v); % Sample size
        if isempty(v), continue; end
        
        % Jittered points
        scatter(xPos(sp) + (rand(size(v))-0.5)*jitterWidth, v, pointSize, ...
            colors{sp}, 'filled', 'MarkerFaceAlpha', pointAlpha, 'HandleVisibility', 'off');
        
        % Violin density
        counts = histcounts(v, binEdges, 'Normalization', 'pdf');
        dens = conv(counts, g, 'same');
        w = (dens / (max(dens)+1e-6)) * maxHalfWidth;
        
        patch([xPos(sp)-w, fliplr(xPos(sp)+w)], [binCenters, fliplr(binCenters)], ...
            colors{sp}, 'FaceAlpha', 0.25 * alphas{sp}, 'EdgeColor', colors{sp}, 'LineWidth', 1.5);
        
        % Mean line
        plot([xPos(sp)-0.15, xPos(sp)+0.15], [median(v), median(v)], 'k-', 'LineWidth', 3);
        text(xPos(sp), -3, sprintf('n=%d', n), 'HorizontalAlignment', 'center', ...
        'FontSize', 7, 'FontWeight', 'normal', 'Color', [0.4 0.4 0.4]);
    end
    
    
    % Calculate Global P-values
    pL = NaN; pH = NaN;
    if ~isempty(pooledSpikes{1}) && ~isempty(pooledSpikes{2})
        % Check if there is any variation in the data
         pL = ranksum(pooledSpikes{1}, pooledSpikes{2});
         pH = ranksum(pooledSpikes{3}, pooledSpikes{4});
    end
   
    
    % Annotations
    text(1.0, maxSpikeVal+6, sprintf('Low State p = %.4f', pL), 'Horiz','center', 'FontWeight','bold');
    text(2.5, maxSpikeVal+6, sprintf('High State p = %.4f', pH), 'Horiz','center', 'FontWeight','bold');
    
    % Formatting
    ylabel('Spike Count (Windowed)');
    xlim([0.3 3.2]); 
    ylim([0, maxSpikeVal + 5]); % Give it a little breathing room at the top
%     set(gca, 'XTick', [1, 2.5], 'XTickLabel', {'Low Threshold State', 'High Threshold State'});
    grid on;
    hold off;
    if isSum == 1
        sgtitle(sprintf('Ch %d: Sham vs Stim Spike Sum t=[%d, %d] Summed, All Sections', ch_num, preWindowIter, postWindowIter));
    else
        sgtitle(sprintf('Ch %d: Sham vs Stim Spike Sum t=[%d, %d] Separate Iters, All Sections', ch_num, preWindowIter, postWindowIter));
    end

    saveas(gcf, sprintf("figures/1-30-2026/section_violins/channel_%d_sections_added.png", ch_num));
end



%% Plot Threshold Evolution - Strict Stim-Only Logic
num_sections = length(exp_data.sections);
num_channels = length(exp_data.metadata.channels);
channels     = exp_data.metadata.channels;

% Pre-allocate with NaNs
low_thresholds  = nan(num_channels, num_sections);
high_thresholds = nan(num_channels, num_sections);
stim_mask       = [exp_data.sections.is_stim_enabled]; 

for s = 1:num_sections
    t_used = exp_data.sections(s).thresholds_used;
    % Only extract if stimulation was enabled AND data isn't NaN
    if stim_mask(s) && ~any(isnan(t_used(:)))
        low_thresholds(:, s)  = t_used(1, :);
        high_thresholds(:, s) = t_used(2, :);
    end
end

figure('Name', 'Strict Threshold Evolution', 'Color', 'w', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);

for c = 1:num_channels
    subplot(2, 5, c);
    hold on;
    
    y_limits = [0 50]; 
    
    % 1. BACKGROUND DELINEATION
    for s = 1:num_sections
        if stim_mask(s)
            % Stimulation Active: Very light blue to show "Closed Loop ON"
            fill([s-0.5 s+0.5 s+0.5 s-0.5], [y_limits(1) y_limits(1) y_limits(2) y_limits(2)], ...
                [0.9 0.95 1.0], 'EdgeColor', 'none', 'FaceAlpha', 0.6);
        else
            % Baseline/Washout: Blank white or very faint gray
            fill([s-0.5 s+0.5 s+0.5 s-0.5], [y_limits(1) y_limits(1) y_limits(2) y_limits(2)], ...
                [0.98 0.98 0.98], 'EdgeColor', 'none', 'FaceAlpha', 1.0);
        end
    end

    % 2. PLOT ACTIVE THRESHOLDS ONLY
    % Because baseline sections are now NaN in our local arrays, 
    % the plot will naturally break/disconnect between stim sessions.
    h_high = plot(1:num_sections, high_thresholds(c, :), 'b-s', ...
        'LineWidth', 2, 'MarkerFaceColor', 'b', 'MarkerSize', 5);
    h_low  = plot(1:num_sections, low_thresholds(c, :), 'r-o', ...
        'LineWidth', 2, 'MarkerFaceColor', 'r', 'MarkerSize', 5);

    % Aesthetics
    title(sprintf('Channel %d', channels(c)), 'FontWeight', 'bold');
    if c > 5, xlabel('Section Index'); end
    if mod(c, 5) == 1, ylabel('Spike Count'); end
    
    ylim(y_limits);
    xlim([0.5 num_sections+0.5]);
    grid on;
    set(gca, 'Layer', 'top', 'TickDir', 'out'); 
end

% Legend and Title
legend([h_high, h_low], {'Active High Bound', 'Active Low Bound'}, ...
    'Orientation', 'horizontal', 'Position', [0.4 0.02 0.2 0.05]);
sgtitle('Adaptive Threshold Evolution: Closed-Loop Sections Only', 'FontSize', 16);

% Save to the correct folder
fig_dir = 'figures/1-30-2026';
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
saveas(gcf, fullfile(fig_dir, 'strict_threshold_evolution.png'));



%% Figure: Spike Count Time-Series with Thresholds and Stim Markers
% Generates one Figure per Channel.
% Each Figure contains 23 subplots (one for each experimental section).

recordedChannels = exp_data.metadata.channels; 
postWindowIter = 2; % The window to highlight after a stimulus

for cIdx = 1:length(recordedChannels)
    ch_num = recordedChannels(cIdx);
    
    % Create a large figure for 23 subplots
    fig_h = figure('Name', sprintf('Ch %d Full Time-Series', ch_num), 'Color', 'w', ...
                   'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);
    
    % 23 subplots (e.g., 5 rows x 5 columns layout)
    tiledlayout(5, 5, 'Padding', 'compact', 'TileSpacing', 'tight');
    
    for s = 1:23
        nexttile;
        hold on;
        
        sec = exp_data.sections(s);
        raw_spikes = sec.raw_data(:, cIdx);
        num_iters = length(raw_spikes);
        iter_vec = 1:num_iters;
        
        % 1. Plot the Raw Spike Count Line
        plot(iter_vec, raw_spikes, 'k-', 'LineWidth', 0.5, 'Color', [0.4 0.4 0.4]);
        
        % 2. Plot Thresholds if they exist (is_stim_enabled)
        if sec.is_stim_enabled && ~any(isnan(sec.thresholds_used(:, cIdx)))
            l_thresh = sec.thresholds_used(1, cIdx);
            h_thresh = sec.thresholds_used(2, cIdx);
            
            % Horizontal threshold lines
            yline(l_thresh, 'r--', 'LineWidth', 1, 'Alpha', 0.6);
            yline(h_thresh, 'b--', 'LineWidth', 1, 'Alpha', 0.6);
            
            % 3. Highlight Stim Events on THIS channel
            for e = 1:length(sec.events)
                ev = sec.events(e);
                % Only highlight if it was a STIM action on this specific channel
                if strcmpi(ev.action, 'STIM') && (ev.channel == ch_num)
                    t_trigger = ev.iteration;
                    t_end = min(num_iters, t_trigger + postWindowIter);
                    
                    % Draw a small vertical "patch" area for the post-stim window
                    % If it was a high-cross, make it light blue; low-cross, light red
                    if raw_spikes(t_trigger) > h_thresh
                        pColor = [0.0 0.0 1.0]; % Blue
                    else
                        pColor = [1.0 0.0 0.0]; % Red
                    end
                    
                    % Highlight the window [Trigger : Trigger+postWindow]
                    patch([t_trigger t_trigger t_end t_end], [0 50 50 0], pColor, ...
                        'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
                    
                    % Marker at the exact trigger point
                    plot(t_trigger, raw_spikes(t_trigger), 'mo', 'MarkerSize', 4, 'MarkerFaceColor', 'm');
                end
            end
        end
        
        % Formatting Subplot
        title(sprintf('Sec %d', s), 'FontSize', 8);
        ylim([0 50]);
        xlim([1 300]);
        set(gca, 'FontSize', 7);
        if s < 21, xticklabels({}); end % Hide x-labels except for bottom row
        if mod(s, 5) ~= 1, yticklabels({}); end % Hide y-labels except for left column
        grid on;
    end
    
    % Global Title and Save
    sgtitle(sprintf('Channel %d: Iteration-by-Iteration Spike Counts and Threshold Crossings', ch_num));
    
    fig_dir = 'figures/1-30-2026/time_series';
    if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
    saveas(fig_h, fullfile(fig_dir, sprintf('channel_%d_all_sections_ts.png', ch_num)));
end


%% Figure: Stimulation Efficiency (Bar Plots per Channel)
% This script pools all stimulation-enabled sections and counts 
% the occurrences of Stim vs. Sham for each crossing type.

recordedChannels = exp_data.metadata.channels;
num_channels = length(recordedChannels);
stim_sections = find([exp_data.sections.is_stim_enabled] == 1);

% Prepare the main figure (10 subplots for 10 channels)
main_fig = figure('Name', 'Threshold Crossing Action Counts', 'Color', 'w', ...
                  'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);
tlo = tiledlayout(2, 5, 'Padding', 'compact', 'TileSpacing', 'compact');

for cIdx = 1:num_channels
    ch_num = recordedChannels(cIdx);
    
    % Initialize counts for this channel
    % [Low_Sham, Low_Stim, High_Sham, High_Stim]
    counts = zeros(1, 4); 
    
    for s = stim_sections
        sec = exp_data.sections(s);
        l_thresh = sec.thresholds_used(1, cIdx);
        h_thresh = sec.thresholds_used(2, cIdx);
        
        for e = 1:length(sec.events)
            ev = sec.events(e);
            val = sec.raw_data(ev.iteration, cIdx);
            
            % Logic to categorize the event
            isLow  = val < l_thresh;
            isHigh = val > h_thresh;
            isTargetStim = strcmpi(ev.action, 'STIM') && (ev.channel == ch_num);
            isGlobalSham = strcmpi(ev.action, 'SHAM_CONTROL');
            
            if isLow && isGlobalSham,  counts(1) = counts(1) + 1; end
            if isLow && isTargetStim,  counts(2) = counts(2) + 1; end
            if isHigh && isGlobalSham, counts(3) = counts(3) + 1; end
            if isHigh && isTargetStim, counts(4) = counts(4) + 1; end
        end
    end
    
    % --- Plotting for this channel ---
    nexttile;
    
    % Group data for bar chart: [Sham, Stim] for Low; [Sham, Stim] for High
    bar_data = [counts(1), counts(2); counts(3), counts(4)];
    
    b = bar(bar_data, 'grouped');
    
    % Styling
    b(1).FaceColor = [0.6 0.6 0.6]; % Sham: Gray
    b(2).FaceColor = [0.2 0.6 0.2]; % Stim: Green
    
    title(sprintf('Channel %d', ch_num));
    set(gca, 'XTickLabel', {'Low Cross', 'High Cross'}, 'FontSize', 9);
    ylabel('Total Event Count');
    grid on;
    
    % Add text labels on top of bars for clarity
    xtips1 = b(1).XEndPoints; ytips1 = b(1).YData;
    labels1 = string(b(1).YData);
    text(xtips1, ytips1, labels1, 'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize', 8);
    
    xtips2 = b(2).XEndPoints; ytips2 = b(2).YData;
    labels2 = string(b(2).YData);
    text(xtips2, ytips2, labels2, 'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize', 8);
end

% Global legend and formatting
lg = legend(b, {'Global Sham', 'Targeted Stim'}, 'Orientation', 'horizontal');
lg.Layout.Tile = 'north';
title(tlo, 'Comparison of Stimulated vs. Sham Threshold Crossings across Stim Sessions', 'FontSize', 16, 'FontWeight', 'bold');

% Save the figure
fig_dir = 'figures/1-30-2026/crossing_counts/';
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
saveas(main_fig, fullfile(fig_dir, 'threshold_action_counts.png'));


%% Figure: Action-Grouped Session Counts (Uniform Colors)
recordedChannels = exp_data.metadata.channels;
num_channels = length(recordedChannels);
stim_sections = find([exp_data.sections.is_stim_enabled] == 1); 
num_sessions = length(stim_sections);

main_fig = figure('Name', 'Action Grouped Counts', 'Color', 'w', ...
                  'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);

for cIdx = 1:num_channels
    ch_num = recordedChannels(cIdx);
    subplot(2, 5, cIdx);
    hold on;
    
    % session_data stores [Sham_Count, Stim_Count] for each of the 11 sessions
    session_data = zeros(num_sessions, 2); 
    
    for sess_idx = 1:num_sessions
        s = stim_sections(sess_idx);
        sec = exp_data.sections(s);
        l_thresh = sec.thresholds_used(1, cIdx);
        h_thresh = sec.thresholds_used(2, cIdx);
        
        for e = 1:length(sec.events)
            ev = sec.events(e);
            val = sec.raw_data(ev.iteration, cIdx);
            
            % Check if a crossing occurred on this channel
            isCross = (val < l_thresh) || (val > h_thresh);
            isTargetStim = strcmpi(ev.action, 'STIM') && (ev.channel == ch_num);
            isGlobalSham = strcmpi(ev.action, 'SHAM_CONTROL');
            
            if isCross && isGlobalSham, session_data(sess_idx, 1) = session_data(sess_idx, 1) + 1; end
            if isCross && isTargetStim, session_data(sess_idx, 2) = session_data(sess_idx, 2) + 1; end
        end
    end
    
    % Grouped plotting: 
    % We pass the data so that all sessions are grouped under 'Sham' and 'Stim'
    % To get two distinct blocks of 11 bars, we use the session_data directly
    % but we must format it as a large grouped bar.
    
    % Create the x-positions for the two clusters
    b = bar([1, 2], [session_data(:,1)'; session_data(:,2)'], 'grouped');
    
    % Styling: Apply uniform colors to match your previous plot
    % b is an array of 11 bar handles (one for each session)
    for sess_idx = 1:num_sessions
        b(sess_idx).FaceColor = 'flat';
        % Set left cluster (Index 1) to Gray, right cluster (Index 2) to Green
        b(sess_idx).CData(1,:) = [0.6 0.6 0.6]; % Gray
        b(sess_idx).CData(2,:) = [0.2 0.6 0.2]; % Green
    end
    
    title(sprintf('Channel %d', ch_num));
    set(gca, 'XTick', [1, 2], 'XTickLabel', {'Global Sham', 'Targeted Stim'}, 'FontSize', 8);
    ylabel('Crossing Count');
    grid on;
    
    % Adjust x-axis limits to show the clusters clearly
    xlim([0.5 2.5]);
end

sgtitle('Session Evolution: Sham Sessions (Left) vs Stim Sessions (Right)', 'FontSize', 16, 'FontWeight', 'bold');

% Save
fig_dir = 'figures/1-30-2026/session_counts/';
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
saveas(main_fig, fullfile(fig_dir, 'uniform_action_evolution.png'));

%% Figure: Population Coincidence Analysis (Grouped Bars)
num_channels = length(exp_data.metadata.channels);
stim_sections = find([exp_data.sections.is_stim_enabled] == 1);

% Initialize frequency bins
freq_low  = zeros(1, num_channels + 1); 
freq_high = zeros(1, num_channels + 1);
freq_any  = zeros(1, num_channels + 1);

for s = stim_sections
    sec = exp_data.sections(s);
    l_thresh = sec.thresholds_used(1, :);
    h_thresh = sec.thresholds_used(2, :);
    raw = sec.raw_data;
    
    % Logical masks
    is_low  = raw < l_thresh;
    is_high = raw > h_thresh;
    is_any  = is_low | is_high;
    
    % Sum across channels per iteration
    counts_low  = sum(is_low, 2);
    counts_high = sum(is_high, 2);
    counts_any  = sum(is_any, 2);
    
    % Accumulate
    for n = 0:num_channels
        freq_low(n+1)  = freq_low(n+1)  + sum(counts_low == n);
        freq_high(n+1) = freq_high(n+1) + sum(counts_high == n);
        freq_any(n+1)  = freq_any(n+1)  + sum(counts_any == n);
    end
end

x_bins = 1:num_channels;
data_low  = freq_low(2:end)';  % Transpose to column for matrix grouping
data_high = freq_high(2:end)';
data_any  = freq_any(2:end)';

figure('Name', 'Population Synchrony Analysis', 'Color', 'w', 'Position', [100 100 900 700]);

% --- Subplot 1: Low and High Side-by-Side ---
subplot(2, 1, 1);
% Create a matrix where each row is a category (1-10 channels) 
% and each column is a series (Low, High)
b_group = bar(x_bins, [data_low, data_high], 'grouped');

% Set specific colors
b_group(1).FaceColor = [1.0 0.4 0.4]; % Red for Low
b_group(2).FaceColor = [0.4 0.4 1.0]; % Blue for High

title('Low vs. High State', 'FontSize', 14);
xlabel('Number of Simultaneous Channel Crossings');
ylabel('N Total Iterations');
legend({'Low Threshold Crossings', 'High Threshold Crossings'}, 'Location', 'northeast');
grid on;
set(gca, 'XTick', 1:num_channels);

% --- Subplot 2: Aggregate (Any Crossing) ---
subplot(2, 1, 2);
b_any = bar(x_bins, data_any, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', 'k');

title('Aggregate Synchronous Crossings (Any Threshold Hit)', 'FontSize', 14);
xlabel('Number of Simultaneous Channel Crossings');
ylabel('N Total Iterations');
grid on;
set(gca, 'XTick', 1:num_channels);

% Add labels on top of the aggregate bars
for i = 1:num_channels
    if data_any(i) > 0
        text(x_bins(i), data_any(i), num2str(data_any(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 9, 'FontWeight', 'bold');
    end
end

sgtitle('Frequency of Multi-Channel Threshold Hits in Stim Sessions', 'FontSize', 16);

% Save results
fig_dir = 'figures/1-30-2026/population_coincidence/';
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
saveas(gcf, fullfile(fig_dir, 'population_coincidence_grouped.png'));



%% Figure: Threshold Evolution (Stim-Enabled Sections Only)
% This plot ignores non-stim sections and shows the trajectory of bounds.

recordedChannels = exp_data.metadata.channels;
num_channels = length(recordedChannels);
stim_sections = find([exp_data.sections.is_stim_enabled] == 1);
num_stim_sessions = length(stim_sections);

figure('Name', 'Threshold Trajectories', 'Color', 'w', ...
       'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8]);

for cIdx = 1:num_channels
    ch_num = recordedChannels(cIdx);
    subplot(2, 5, cIdx); % Assuming 10 channels; adjust grid as needed
    hold on;
    
    % Pre-allocate arrays for this channel's thresholds
    low_vals  = zeros(1, num_stim_sessions);
    high_vals = zeros(1, num_stim_sessions);
    
    % Extract thresholds from stim-enabled sections
    for sess_idx = 1:num_stim_sessions
        s = stim_sections(sess_idx);
        low_vals(sess_idx)  = exp_data.sections(s).thresholds_used(1, cIdx);
        high_vals(sess_idx) = exp_data.sections(s).thresholds_used(2, cIdx);
    end
    
    % Plot the lines
    % Using 'o-' to show the individual session points clearly
    plot(1:num_stim_sessions, high_vals, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'High Bound');
    plot(1:num_stim_sessions, low_vals, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', 'Low Bound');
    
    % Aesthetics per subplot
    title(sprintf('Ch %d', ch_num));
    if cIdx > 5, xlabel('Stim Session #'); end
    if mod(cIdx, 5) == 1, ylabel('Spike Threshold'); end
    
    grid on;
    % Match the Y-axis across all subplots for easier comparison
    ylim([0, max([high_vals, 10]) + 5]); 
    set(gca, 'XTick', 1:num_stim_sessions);
end

% Add a global legend
lgd = legend({'High Threshold', 'Low Threshold'}, 'Orientation', 'horizontal');
lgd.Position = [0.45 0.02 0.1 0.03]; % Bottom center

sgtitle('Adaptive Threshold Evolution (Stimulation Sections Only)', 'FontSize', 16, 'FontWeight', 'bold');

% Save
fig_dir = 'figures/1-30-2026/threshold_evolution/';
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
saveas(gcf, fullfile(fig_dir, 'stim_threshold_trajectories.png'));
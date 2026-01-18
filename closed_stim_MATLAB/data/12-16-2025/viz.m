%% Read in results from 12/16 mouse session

% Baseline  (20 min) threshold calculation session
% note alpha was 0.18 (~5s memory) for EMA filter, and 5s cooldown after each
% stimulation. 
clear all; 
close all;

warmup = load('data/12-16-2025/warmup_Xts.mat').warmup_Xts;
jsonName = "percentiles_2025-12-16.json";
jsonStruct = jsondecode(fileread(jsonName));

% Electrodes
electrodeIDs = {jsonStruct.electrodeID}; 
disp (electrodeIDs)
lowThresholds = [jsonStruct.fifthPercentile]; 
highThresholds = [jsonStruct.ninetyFifthPercentile];

% First stimulation session
stimulation_1 = load('data/12-16-2025/stimulation_Xts_1.mat').stimulation_Xts;
tc1 = load('threshold_crossings_1.mat').threshold_crossings;

% Second stimulation session
tc2 = load('data/12-16-2025/threshold_crossings_2.mat').threshold_crossings;
stimulation_2 = load('stimulation_Xts_2.mat').stimulation_Xts;

recordedChannels = electrodeIDs;

stimSession = 1;

if stimSession == 1
    tc = tc1; 
    stimulation_readout = stimulation_1;
    % Specify and run EMA filter over each channel
elseif stimSession == 2
    tc = tc2;
    stimulation_readout = stimulation_2;
else
    disp ("Choose stim session!");
end


% see moving average calculation used for experiment
emaFiltered = zeros(size(stimulation_readout)); % Initialize matrix for EMA filtered data
alpha = 0.18; % Smoothing factor for EMA

chIdx = 1;
for ch = recordedChannels
    emaFiltered(:, chIdx) = filter(alpha, [1, alpha - 1], stimulation_readout(:, chIdx));
    chIdx = chIdx + 1; 
end

%% Main histogram: per channel, spike counts in specified window pre-stim and post-stim, in low and high thresholds
% Calculate spike counts in specified windows
preStimWindow = 5; % seconds before stimulation
postStimWindow = 5; % seconds after stimulation





%% Channel-level crossing & stimulation summary

clearvars -except tc electrodeIDs
close all;

numChannels = length(electrodeIDs);

% Preallocate
numCrossings    = zeros(numChannels,1);
numStimulations = zeros(numChannels,1);
numLowCrossings = zeros(numChannels,1);
numHighCrossings= zeros(numChannels,1);

for chIdx = 1:numChannels
    ch_num = electrodeIDs{chIdx};

    if isempty(tc{ch_num})
        continue
    end

    events = tc{ch_num};

    % Total crossings
    numCrossings(chIdx) = length(events);

    % Crossing types
    crossingTypes = [events.crossingType];
    numLowCrossings(chIdx)  = sum(crossingTypes == -1);
    numHighCrossings(chIdx) = sum(crossingTypes == 1);

    % Stimulations
    stimulated = [events.channelStimulated];
    numStimulations(chIdx) = sum(stimulated);
end

% Print summary table
summaryTable = table( ...
    electrodeIDs(:), ...
    numCrossings, ...
    numStimulations, ...
    numLowCrossings, ...
    numHighCrossings, ...
    'VariableNames', {'Channel','TotalCrossings','Stimulations','LowCrossings','HighCrossings'});

disp(summaryTable)

%  Scatter plot: stimulations vs crossings (one dot per channel)
figure('Name','Stimulations vs Channel Crossings','Color','w');
scatter(numCrossings, numStimulations, 80, 'filled');
grid on;

xlabel('Number of Threshold Crossings');
ylabel('Number of Stimulations');
title('Stimulations vs Crossings (One Dot per Channel)');

% Label each point with channel ID
hold on;
for chIdx = 1:numChannels
    if numCrossings(chIdx) > 0
        text(numCrossings(chIdx)+0.2, ...
             numStimulations(chIdx), ...
             sprintf('%d', electrodeIDs{chIdx}), ...
             'FontSize', 9);
    end
end
saveas(gcf, 'Stimulations_vs_Crossings_AllChannels.png');
writetable(summaryTable, 'Channel_Crossing_Stim_Summary.csv');

hold off;

%% Full channel data plot separated by channel
% Channel iteration loop
chIdx = 1;
for ch = recordedChannels
    ch_num = ch{1};
    
    if ~isempty(tc{ch_num})
        events = tc{ch_num};
        eventIters = [events.iter];  % Get the iteration indices for crossings
        stimulated    = [events.channelStimulated]; 
        crossingTypes = [events.crossingType];
        stimulatedIterations = eventIters(stimulated);  % Select the stimulated iterations from eventIters
        % Find length of stimulated iterations
        numStimulatedIterations = length(stimulatedIterations);
        disp(['Number of Stimulated Iterations for Channel ', num2str(ch_num), ': ', num2str(numStimulatedIterations)]);
        % Plot the entire stimulation_readout for the current channel
        figure('Name', sprintf('Readout for Channel %d', ch_num));
        hold on;
        plot(stimulation_readout(:, chIdx), 'Color', [0.6, 0.6, 0.6], 'LineWidth', 1.5); % Plot the entire stimulation readout
        plot(emaFiltered(:, chIdx), 'b', 'LineWidth', 2); % Plot the entire stimulation readout

        ylim([0 40]); % Adjust the limits as needed
        % Plot high and low thresholds as dashed red lines
        yLimits = ylim; % Get current y-limits for proper scaling
        hold on;
        plot(xlim, [lowThresholds(chIdx), lowThresholds(chIdx)], 'r--', 'LineWidth', 1.5); % Low threshold line
        plot(xlim, [highThresholds(chIdx), highThresholds(chIdx)], 'r--', 'LineWidth', 1.5); % High threshold line
        hold on;
       
        for k = 1:length(events)
            if events(k).channelStimulated
                stimTime = events(k).iter;
                if crossingTypes(k) == 1
                    yShade = [highThresholds(chIdx) 40 40 highThresholds(chIdx)];
                    color = 'b';
                else
                    yShade = [0 lowThresholds(chIdx) lowThresholds(chIdx) 0];
                    color = 'y'; 
                end
                xShade = [stimTime stimTime stimTime+5 stimTime+5];
                fill(xShade, yShade, color, 'FaceAlpha',0.3,'EdgeColor','none');
            end
        end
        
        xlabel('Time (s)');
        ylabel('Spike Count (spikes/s)');
        title(sprintf('Stimulation Readout with Windows for Channel %d', ch_num));
        grid on;
        hold off;
        saveas(gcf, sprintf('Stimulation_Readout_Channel_%d.png', ch_num));
    end
    chIdx = chIdx + 1;
end
%% Stacked full channel data plot (shared axes) 

numCh = length(recordedChannels);

figure('Name','Stacked Stimulation Readouts','Color','w');
tiledlayout(numCh, 1, ...
    'TileSpacing','compact', ...
    'Padding','compact');

chIdx = 1;
for ch = recordedChannels
    ch_num = ch{1};
    events = tc{ch_num};  % Retrieve events for the current channel
    crossingTypes = [events.crossingType];   
    nexttile;
    hold on;

    % Plot signals
    plot(stimulation_readout(:, chIdx), 'Color', [0.6, 0.6, 0.6], 'LineWidth', 1);
    plot(emaFiltered(:, chIdx), 'b--', 'LineWidth', 1);

    % Lock axes BEFORE shading
    ylim([0 40]);
    xlim([1 size(stimulation_readout,1)]);

    % Thresholds
    yline(lowThresholds(chIdx),  'r--', 'LineWidth', 1);
    yline(highThresholds(chIdx), 'r--', 'LineWidth', 1);
    events = tc{ch_num};
    for k = 1:length(events)
        if events(k).channelStimulated
            stimTime = events(k).iter;
            if crossingTypes(k) == 1
                yShade = [highThresholds(chIdx) 40 40 highThresholds(chIdx)];
                color = 'b';
            else
                yShade = [0 lowThresholds(chIdx) lowThresholds(chIdx) 0];
                color = 'y'; 
            end
            xShade = [stimTime stimTime stimTime+5 stimTime+5];
            fill(xShade, yShade, color, 'FaceAlpha',0.3,'EdgeColor','none');
        end
        end

    % Channel label (top-right)
    text(0.98, 0.90, sprintf('Ch %d', ch_num), ...
        'Units','normalized', ...
        'HorizontalAlignment','right', ...
        'FontWeight','bold');

    % Clean look
    grid on;
    set(gca,'Box','off');

    % Only bottom plot gets x-label
    if chIdx == numCh
        xlabel('Time (s)');
    else
        set(gca,'XTickLabel',[]);
    end

    % Only middle plot gets y-label (optional)
    if chIdx == ceil(numCh/2)
        ylabel('Spike Count (spikes/s)');
    end

    hold off;
    chIdx = chIdx + 1;
end

% Save figure
saveas(gcf,'Stacked_Stimulation_Readouts_AllChannels.png');

%% Time-series aggregation
preWindowIter = 2;  % Define the number of iterations before the event
postWindowIter = 2; % Define the number of iterations after the event
chIdx = 1; 
for ch = recordedChannels
    ch_num = ch{1};
      
    if ~isempty(tc{ch_num})
        events = tc{ch_num};
        iterIndices = [events.iter];      
        
        % Print the indices of channel crossings
        numCrossings = length(iterIndices);
        crossingTypes = [events.crossingType];   
        lowCount = sum(crossingTypes == -1);
        highCount = sum(crossingTypes == 1);
        stimulated    = [events.channelStimulated];     
        numStimulations = sum(stimulated);  % Count the number of times the channel was stimulated
        coin_flip_prob = [events.coinFlip]; 

        % Plot number of stimulations as a function of number of crossings
        % figure('Name', sprintf('Stimulations vs Crossings for Channel %d', ch_num));
        % bar([numCrossings, numStimulations]);
        % set(gca, 'XTickLabel', {'Number of Crossings', 'Number of Stimulations'});
        % ylabel('Count');
        % title(sprintf('Channel %d: Stimulations vs Crossings', ch_num));
        % grid on;

        figure('Name', sprintf('Channel %d', ch_num));

        % Define subplot types
        subplotMasks = {
            (crossingTypes == -1 & ~stimulated & coin_flip_prob <= 0.5),  % Low threshold, no stim on any channel
            (crossingTypes == -1 & stimulated),                            % Low threshold, stim
            (crossingTypes == 1 & ~stimulated & coin_flip_prob <= 0.5),  % High threshold, no stim on any channel
            (crossingTypes == 1 & stimulated)                             % High threshold, stim
        };
        subplotTitles = {
            'Low Threshold, No Stim', ...
            'Low Threshold, Stim', ...
            'High Threshold, No Stim', ...
            'High Threshold, Stim'
        };
        thresholds = [lowThresholds(chIdx), lowThresholds(chIdx), highThresholds(chIdx), highThresholds(chIdx)];

        t_common = -preWindowIter:postWindowIter;

        for sp = 1:4
            mask = subplotMasks{sp};
            eventIters = iterIndices(mask);
            if isempty(eventIters)
                continue
            end
        
            subplot(2,2,sp);
            hold on;
        
            allTraces = nan(length(eventIters), length(t_common));
    
            for k = 1:length(eventIters)
                iterCross = eventIters(k);
        
                % Compute actual bounds in data
                idxStart = max(1, iterCross - preWindowIter);
                idxEnd   = min(size(stimulation_readout,1), iterCross + postWindowIter);
        
                Xtrace = stimulation_readout(idxStart:idxEnd, chIdx);
        
                % Where this trace lands in t_common
                tStartIdx = (idxStart - iterCross) + preWindowIter + 1;
                tEndIdx   = tStartIdx + length(Xtrace) - 1;
        
                % Fill into full-length vector
                Xfull = nan(1, length(t_common));
                Xfull(tStartIdx:tEndIdx) = Xtrace;
        
                allTraces(k,:) = Xfull;
        
                % Plot individual trace (continuous)
                plot(t_common, Xfull, 'Color', [0.6 0.6 0.6], 'LineWidth', 1);
            end
        
            % Plot mean trace
            meanTrace = mean(allTraces, 1, 'omitnan');
            plot(t_common, meanTrace, 'k', 'LineWidth', 2);
        
            % Threshold + alignment lines
            yline(thresholds(sp), 'r--', 'LineWidth', 1.5);
            xline(0, 'r--', 'LineWidth', 2);
        
            xlabel('Iterations relative to crossing');
            ylabel('X_t');
            title(sprintf('%s (Ch %d)', subplotTitles{sp}, ch_num));
            grid on;

            % Set common y-limits
            ylim([0, 30]);
            hold off;
        end
        % Save the figure
        saveas(gcf, sprintf('Channel_%d_Stimulation_Analysis.png', ch_num));
        chIdx = chIdx + 1; 
    end
end


%% Aggregate threshold crossings across all channels by iteration

clearvars -except tc stimulation_readout electrodeIDs
close all;

numIters    = size(stimulation_readout, 1);
numChannels = length(electrodeIDs);

% Preallocate count vector
crossingsPerIter = zeros(numIters, 1);

% Loop over channels
for chIdx = 1:numChannels
    ch_num = electrodeIDs{chIdx};

    if isempty(tc{ch_num})
        continue
    end

    events = tc{ch_num};
    iterIndices = [events.iter];

    % Accumulate counts
    for k = 1:length(iterIndices)
        iter = iterIndices(k);
        if iter >= 1 && iter <= numIters
            crossingsPerIter(iter) = crossingsPerIter(iter) + 1;
        end
    end
end



maxCrossings = max(crossingsPerIter);

% Define integer bins: 0,1,2,...
binEdges = -0.5 : 1 : (maxCrossings + 0.5);

figure('Name','Histogram of Threshold Crossings per Iteration','Color','w');

histogram(crossingsPerIter, binEdges, ...
    'Normalization','count', ...
    'FaceColor',[0.4 0.4 0.4], ...
    'EdgeColor','k');

grid on;
xlabel('Number of Threshold Crossings in an Iteration');
ylabel('Number of Iterations');
title('Distribution of Threshold Crossings per Iteration');

xticks(0:maxCrossings);

saveas(gcf, 'Histogram_Crossings_Per_Iteration.png');
fprintf('Total iterations: %d\n', numIters);
fprintf('Iterations with zero crossings: %d (%.2f%%)\n', ...
    sum(crossingsPerIter == 0), ...
    100 * mean(crossingsPerIter == 0));

fprintf('Max crossings in a single iteration: %d\n', maxCrossings);

%% SANITY-CHECK AUDIT PLOT FOR 12/16 SESSION
% Plots threshold crossings vs stimulations for each channel

clearvars -except tc stimulation_readout electrodeIDs lowThresholds highThresholds
close all;

numChannels = length(electrodeIDs);
numIters = size(stimulation_readout,1);

figure('Name','Sanity Check: Stim vs Threshold Crossings','Color','w');
hold on;

violations = [];

for chIdx = 1:numChannels
    ch_num = electrodeIDs{chIdx};

    if isempty(tc{ch_num})
        continue
    end

    events = tc{ch_num};

    iters        = [events.iter];             
    stimulated   = [events.channelStimulated]; 
    crossingType = [events.crossingType];
    coins = [events.coinFlip];

    % Plot all threshold crossings (gray)
    scatter(iters, chIdx*ones(size(iters)), 25, [0.6 0.6 0.6], 'filled');

    % Plot stimulations (red circles)
    stimIters = iters(stimulated);
    scatter(stimIters, chIdx*ones(size(stimIters)), 80, 'r', 'o', 'LineWidth', 1.5);

    % Logical sanity check: stim must occur at a threshold crossing
    for k = find(stimulated)
        if (crossingType(k) ~= -1 && crossingType(k) ~= 1) || (coins(k) < 0.5)
            violations(end+1,:) = [iters(k), chIdx];
        end
    end
end

xlabel('Iteration');
ylabel('Channel Index');
yticks(1:numChannels);
yticklabels(electrodeIDs);
title('Sanity Check: Threshold Crossings (gray) vs Stimulations (red)');

% Count the number of violations
numViolations = size(violations, 1);
fprintf('Number of threshold-crossing violations: %d\n', numViolations);
grid on;

% Legend
scatter(nan,nan,25,[0.6 0.6 0.6],'filled');
scatter(nan,nan,80,'r','o','LineWidth',1.5);
legend({'Threshold Crossing','Stimulation Delivered'},'Location','best');

hold off;

saveas(gcf,'SanityCheck_Stim_vs_Threshold_1216Session.png');

% Optional hard assertion
if ~isempty(violations)
    warning('Stim delivered without threshold crossing! First violation: Iter=%d, Ch=%s', ...
            violations(1,1), electrodeIDs{violations(1,2)});
else
    disp('✓ Sanity check passed: all stimulations coincide with threshold crossings');
end


%% Compare warmup and stimulation periods

figure('Name','Overlay Warmup and Stimulation Plots','Color','w');

warmupColor = [0.2 0.6 0.2];
stimColor   = [0.6 0.2 0.2];

numChannels = length(electrodeIDs);
startPoint  = 5;

for chIdx = 1:numChannels
    ch_num = electrodeIDs{chIdx};  % label only

    subplot(numChannels, 1, chIdx);
    hold on;

    % ---- Warmup ----
    yWarm = warmup(startPoint:end, chIdx);
    xWarm = startPoint : startPoint + numel(yWarm) - 1;
    plot(xWarm, yWarm, 'Color', warmupColor, 'LineWidth', 1.5);

    % ---- Stimulation ----
    yStim = stimulation_readout(startPoint:end, chIdx);
    xStim = startPoint : startPoint + numel(yStim) - 1;
    plot(xStim, yStim, 'Color', stimColor, 'LineWidth', 1.5);

    ylabel(sprintf('Ch %d', ch_num));
    grid on;
    box off;

    if chIdx == numChannels
        xlabel('Iteration');
    else
        set(gca, 'XTickLabel', []);
    end
end

sgtitle('Overlay of Warmup and Stimulation Threshold Crossings');
saveas(gcf, 'Overlay_Warmup_Stimulation_Plots.png');


%% Proportion of crossings that resulted in stimulation bar plot

numChannels = length(electrodeIDs);

propStimulated = nan(numChannels,1);
numTotalCross  = zeros(numChannels,1);
numStimCoin    = zeros(numChannels,1);

for chIdx = 1:numChannels
    ch_num = electrodeIDs{chIdx};

    if isempty(tc{ch_num})
        continue
    end

    events = tc{ch_num};

    stimulated = [events.channelStimulated];
    coinFlip   = [events.coinFlip];

    numTotalCross(chIdx) = numel(events);

    numStimCoin(chIdx) = sum(stimulated);

    % Proportion
    propStimulated(chIdx) = numStimCoin(chIdx) / numTotalCross(chIdx);

end

hold on;
for chIdx = 1:numChannels
    if numTotalCross(chIdx) > 0
        text(chIdx, propStimulated(chIdx) + 0.03, ...
            sprintf('%d/%d', numStimCoin(chIdx), numTotalCross(chIdx)), ...
            'HorizontalAlignment','center', ...
            'FontSize',9);
    end
end
hold off;


figure('Name','Proportion of Crossings Resulting in Stimulation','Color','w');

bar(propStimulated, 'FaceColor',[0.3 0.3 0.7]);
ylim([0 1]);
yline(0.5, 'r--','LabelHorizontalAlignment','left', 'LabelVerticalAlignment','middle', 'LineWidth', 1.5);

xlabel('Channel');
ylabel('Proportion of Crossings Stimulated');
title('Fraction of Threshold Crossings That Resulted in Stimulation');

xticks(1:numChannels);
xticklabels(electrodeIDs);
xtickangle(45);

grid on;
box off;

saveas(gcf,'Proportion_Stimulated_Per_Channel.png');


%% Overlaid histograms per channel + normalized density (yyaxis)

% -------------------------------
% Disable axes toolbar globally
% -------------------------------
set(groot, 'defaultAxesToolbarVisible', 'off');

% -------------------------------
% Compute GLOBAL robust x-range
% -------------------------------
allValsGlobal = [
    warmup(:);
    stimulation_1(:);
    stimulation_2(:)
];

allValsGlobal = allValsGlobal(~isnan(allValsGlobal));

xLow  = prctile(allValsGlobal, 1);
xHigh = prctile(allValsGlobal, 99.5);

if xLow >= xHigh
    xLow  = min(allValsGlobal);
    xHigh = max(allValsGlobal);
end

nBins = 30;
edges = linspace(xLow, xHigh, nBins + 1);

% -------------------------------
% Plot per channel
% -------------------------------
numChannels = length(recordedChannels);

for chIdx = 1:numChannels
    ch_num = recordedChannels{chIdx};

    % Extract channel data
    warm_vals  = warmup(:, chIdx);
    stim1_vals = stimulation_1(:, chIdx);
    stim2_vals = stimulation_2(:, chIdx);

    % Remove NaNs
    warm_vals  = warm_vals(~isnan(warm_vals));
    stim1_vals = stim1_vals(~isnan(stim1_vals));
    stim2_vals = stim2_vals(~isnan(stim2_vals));

    % Clip to plotting range
    warm_plot  = warm_vals(warm_vals  >= 0 & warm_vals  <= xHigh);
    stim1_plot = stim1_vals(stim1_vals >= 0 & stim1_vals <= xHigh);
    stim2_plot = stim2_vals(stim2_vals >= 0 & stim2_vals <= xHigh);

    if isempty(warm_plot) && isempty(stim1_plot) && isempty(stim2_plot)
        continue
    end

    % -------------------------------
    % Plot
    % -------------------------------
    fig = figure('Name', sprintf('Channel %d', ch_num), ...
                 'Color', 'w', ...
                 'Toolbar', 'none');
    hold on;

    % ---- LEFT AXIS: HISTOGRAMS ----
    yyaxis left

    histogram(warm_plot, ...
        'BinEdges', edges, ...
        'FaceColor', [0.6 0.6 0.6], ...
        'FaceAlpha', 0.7, ...
        'EdgeColor', 'none');

    histogram(stim1_plot, ...
        'BinEdges', edges, ...
        'FaceColor', [0.2 0.6 0.2], ...
        'FaceAlpha', 0.6, ...
        'EdgeColor', 'none');

    histogram(stim2_plot, ...
        'BinEdges', edges, ...
        'FaceColor', [0.2 0.2 0.8], ...
        'FaceAlpha', 0.6, ...
        'EdgeColor', 'none');

    ylabel('Count');
    xlim([xLow xHigh]);
    xticks(floor(xLow):ceil(xHigh));

    % ---- RIGHT AXIS: NORMALIZED DENSITY ----
    yyaxis right
    ylim([0 1]);
    ylabel('Normalized density');

    sigma = 1.0;   % smoothing width (bins)

    % SOLID = no stimulation
    plotSmoothHistLine(warm_plot,  edges, [0.4 0.4 0.4], 0.9, '-',  sigma);
    plotSmoothHistLine(stim1_plot, edges, [0.2 0.6 0.2], 0.9, '--', sigma);
    plotSmoothHistLine(stim2_plot, edges, [0.2 0.2 0.8], 0.9, '--', sigma);

    % -------------------------------
    % Labels & export
    % -------------------------------
    xlabel('X_t value');
    title(sprintf('Channel %d: Warmup vs Stimulation', ch_num));

    legend({'Warmup', 'Stim 1', 'Stim 2', ...
            'Warmup density', 'Stim 1 density', 'Stim 2 density'}, ...
           'Location', 'best');

    grid on;
    box off;

    exportgraphics(fig, ...
        fullfile(figuresdir, ...
        sprintf('Channel_%d_Warmup_Stim_OverlayHist.png', ch_num)), ...
        'Resolution', 300);
end


function plotSmoothHistLine(data, edges, color, alpha, lineStyle, sigma)
    if isempty(data)
        return
    end

    % Histogram counts
    counts = histcounts(data, edges);
    centers = edges(1:end-1) + diff(edges)/2;

    % ---- Manual Gaussian kernel (no toolboxes) ----
    halfWidth = 2;      % kernel radius
    x = -halfWidth:halfWidth;
    kernel = exp(-(x.^2)/(2*sigma^2));
    kernel = kernel / sum(kernel);

    smoothCounts = conv(counts, kernel, 'same');

    % ---- Normalize to [0, 1] ----
    if max(smoothCounts) > 0
        smoothCounts = smoothCounts / max(smoothCounts);
    end

    % Plot line
    plot(centers, smoothCounts, ...
        'Color', [color alpha], ...
        'LineWidth', 2, ...
        'LineStyle', lineStyle);
end

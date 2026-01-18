%% Hypothetical rule effect on baseline
% if baseline was to be stimulated a certain way, what would happen?
clear all; 
close all;

jsonName = "data/1-13-2026/percentiles_2026-01-13.json";
jsonStruct = jsondecode(fileread(jsonName));

% Electrodes
electrodeIDs = {jsonStruct.electrodeID}; 
disp (electrodeIDs)
lowThresholds = [jsonStruct.fifthPercentile]; 
highThresholds = [jsonStruct.ninetyFifthPercentile];

warmup = readmatrix('data/1-13-2026/warmup_Xts.csv');

% Verify the dimensions match your original data
[rows, cols] = size(warmup);
stimulation_readout = warmup; 
selectedChannels = electrodeIDs; % can be all channels or a subset that seems ideal from inspection 
numChannels = length(electrodeIDs);
threshold_crossings = cell(1, numChannels);
stim_cool_down = 5;
recordedChannels = electrodeIDs;


% see moving average calculation used for experiment
emaFiltered = zeros(size(stimulation_readout)); % Initialize matrix for EMA filtered data
alpha = 0.18; % Smoothing factor for EMA

chIdx = 1;
for ch = recordedChannels
    emaFiltered(:, chIdx) = filter(alpha, [1, alpha - 1], stimulation_readout(:, chIdx));
    chIdx = chIdx + 1; 
end

for ch = selectedChannels
    ch_num = ch{1}; 
    threshold_crossings{ch_num} = struct( ...
        'iter', {}, ...
        'channel', {}, ...
        'crossingType', {}, ...   % -1 = low, +1 = high
        'wasSelected', {}, ...    % true if selected for stim
        'channelStimulated', {}, ...       % true if stim delivered
        'anyStimThisIter', {}...
    );
end


last_stim_iter = -1; 
for i = 1:size(stimulation_readout, 1) % iterate over  timepoints in warmup
    iterStart=tic; 
    X_t = stimulation_readout(i, :);
    stim_channels = [];
    crossings = []; 
    for chIdx = 1:numChannels
        channelID = selectedChannels(chIdx);
        thresholdLow  = lowThresholds(chIdx);
        thresholdHigh = highThresholds(chIdx);

        if X_t(chIdx) < thresholdLow
            stim_channels(end+1) = channelID{1};
            crossings(end+1) = -1;
        elseif X_t(chIdx) > thresholdHigh
            stim_channels(end+1) = channelID{1};
            crossings(end+1) = +1;
        end
    end

    stim_channel = 0;
    if i - last_stim_iter <= stim_cool_down       % Ignore cooldown periods altogether
        continue; 
    end

    % choose stim channel or no stimulation with new rule
    if ~isempty(stim_channels)
        idx = randi(numel(stim_channels) + 1);
        if idx == numel(stim_channels) + 1 % don't stimulate with probability equal to 1/(n_threshold_cross_channels+1)
            stim_channel = 0;
        else
            stim_channel = stim_channels(idx);
            stim_params.train_stim_channels = stim_channel;
            last_stim_iter = i; 
        end
    end
    if stim_channel ~= 0 
        stim_delivered_iter = true;
        last_stim_iter = i;
        fprintf('Stim at iter %d | ch %d \n', ...
            i, stim_channel);
    else 
        stim_delivered_iter = false;
    end

    for k = 1:numel(stim_channels)
        % if any stim_channel crossed a threshold, then this is updated
        channelID = stim_channels(k);
        event.iter = i;
        event.channel = channelID;
        event.crossingType = crossings(k); 
        event.wasSelected = (channelID == stim_channel);
        event.channelStimulated = (channelID == stim_channel);
        event.anyStimThisIter = stim_delivered_iter;
        threshold_crossings{channelID}(end+1) = event;
    end
    
end
disp ("Fake run! Changing channel crossings variable");
tc = threshold_crossings; % overwrite with newly calculated threshold crossing and simulated stimulations

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
        saveas(gcf, sprintf('Warmup_Readout_Channel_%d.png', ch_num));
    end
    chIdx = chIdx + 1;
end

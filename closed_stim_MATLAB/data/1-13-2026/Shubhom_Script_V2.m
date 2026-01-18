%% Closed loop stimulation experiment 1-13-26
% Ran 30 min baseline, 30 min stim sessions x2  w 5 uA current. We also
% used a different stimulation rule such that given n channels cross their
% baseline-derived stim threshold, we stimulate any of them with
% probability 1/(n+1) and, with same probability, perform no stimulation.
% We did this to balance no stim/stim more from the 12-16-25 experiment. 

% This script should be run section by section. The first section collects
% baseline data for a specified number of runtime_mins. The second section
% stimulates based on calculated thresholds per channel. 
% 
% Four outputs are produced from this script: warmup_Xts.csv and
% warmup_Xts.mat (TODO: make less redundant), stimulation_Xts.mat and threshold_crossings.mat. The
% _Xts files contain recorded channel spikes from Ripple software (so they
% are size [n_timepoints, n_channels] from each phase) while
% threshold_crossings.mat contains a data structure for each threshold
% crossing event for each channel, specifying stimulation timepoints as
% well. 
%
% Some quick plots are generated, though more up-to-date plots are in viz.py.

%% Section 1 Collect data
clear all; 
close all;


%%
runtime_mins = 30 %30;
runtime_mins_stim = runtime_mins; 
refresh_rate = 1; % 1s refresh
% identify names of recording channels
rec_channels =   [6,12,22,... % Jan 13, 2026
                32+6, 32+7, 32+19, 32+28, 32+32,...
                64+9, 64+26]%1:10;
% rec_channels = elecs(6, 7); % TODO: change based on recording electrode



DEBUG_MODE = 0;


if ~DEBUG_MODE % initialize on Yuxuan computer
   % This script is used to automate the ICMS + imaging channel survey procedure 
    % for the Ripple Neural Interface Processor.   
    addpath('C:\Users\XieLu\Desktop\Yuxuan_Exp_Code_20251014\Ripple Exp Pipeline\StimPack128')
    addpath(genpath('C:\Users\XieLu\Desktop\Yuxuan_Exp_Code_20251014\Ripple Exp Pipeline\StimPack128\xippmex'));
    addpath('C:\Users\XieLu\Desktop\Yuxuan_Exp_Code_20251014\Ripple Exp Pipeline\StimPack128\functions');
    addpath(genpath('C:\Users\XieLu\Desktop\Yuxuan_Exp_Code_20251014\Ripple Exp Pipeline\StimPack128\PulsePal'));
    
    
    % NIP Hardware Initialization 
    disp('Initializing Ripple Neural Interface Processor...'); fprintf('\n');
    % Initialize xippmex
    status = xippmex;
    if status ~= 1; error('Xippmex Did Not Initialize'); end
    % Give the NIP some time to process any commands we have sent
    pause(0.5)
    elecs = xippmex('elec', 'micro');
    fprintf('Initialization complete!\n');
    
    xippmex('stim', 'enable', 0);
    xippmex('stim', 'res', elecs, 1); % step size of 1 uA
    xippmex('stim', 'enable', 1);
    
    
    % Create directory to save data
    parent_dir = 'C:/Data';
    
    animal_dir = uigetdir(parent_dir,'Select animal folder');
    date_str = char(date);
    save_path = fullfile(animal_dir,date_str,['Closed_Loop_Test']); % automatically creates date folder
    if ~exist(save_path,'dir')
        mkdir(save_path)
    end
    fprintf('Data will be saved in %s\n',save_path);

    t = datetime;
    time_str = strcat('H',num2str(t.Hour),'M',num2str(t.Minute));

    ephys_filename = fullfile(save_path, strcat('ephys_', time_str));
    fprintf('\nEphys file name: %s\n', ephys_filename);

    xippmex('trial', 'recording', ephys_filename); 

end




warmup_Xts = nan(runtime_mins * 60, length(rec_channels));

% CSV settings
outfile = 'warmup_Xts.csv'; 
% header = arrayfun(@(x) ['Ch_' num2str(x)], rec_channels, 'UniformOutput', false);
% 
% % Create the initial file and write the header (WriteMode is 'overwrite' by default)
% writecell(header, outfile, 'Delimiter', ',', 'FileType', 'text'); 
% 
% disp(['Initialized file: ' outfile]);
% disp('---');

disp(['Starting data collection for ' num2str(runtime_mins) ' minutes']);
% Initialize data collection parameters
i = 1; 
tStart = tic;
if ~DEBUG_MODE
    X_ref = xippmex('spike', rec_channels);
end
pause(1)
while toc(tStart) < runtime_mins * 60
    if DEBUG_MODE
        X_t = randn(1, length(rec_channels)) * 2 + 5; % mean 5 std 2
    else 
        X_t = xippmex('spike', rec_channels);
    end 
    % writematrix(X_t, outfile, 'WriteMode', 'append', 'Delimiter', ',',
    % 'FileType', 'text');    only need for realtime writing
    warmup_Xts(i, :) = X_t;
    i = i + 1;
    pause(refresh_rate); % 
end

% Print the 
% Save warmup_Xts as a CSV with columns labeled by rec_channels
header = arrayfun(@(x) ['Ch_' num2str(x)], rec_channels, 'UniformOutput', false);
writecell(header, outfile, 'Delimiter', ',', 'FileType', 'text'); 
writematrix(warmup_Xts, outfile, 'WriteMode', 'append', 'Delimiter', ',', 'FileType', 'text');

disp(['Finished and saved to ', 'warmup_Xts.csv']);
fprintf('warmup_Xts shape: %d timepoints, %d channels\n', size(warmup_Xts, 1), size(warmup_Xts, 2));
save('warmup_Xts.mat', 'warmup_Xts');

%% Section 2 Calculate filters and set thresholds

% Define the specific channels to select
% TODO during experimenet: select specific channels
selectedChannels = rec_channels; % can be all channels or a subset that seems ideal from inspection 


dataStruct = readtable('warmup_Xts.csv');
recordingChannels = str2double(strrep(dataStruct.Properties.VariableNames, 'Ch_', '')); % column index of recording channel maps to channel name
dataset = table2array(dataStruct);


% choose indices of recordingChannels corresponding to selectedChannels
selectedIndices = ismember(recordingChannels, selectedChannels);
selectedDataIndices = find(selectedIndices);

if size(dataset, 2) ~= length(selectedChannels)
    dataset = dataset(:, selectedDataIndices); % Select only the columns corresponding to selectedChannels
    nChannels = size(dataset, 2); % Update the number of channels after selection
else
    nChannels = length(selectedChannels); % Update the number of channels after selection
end

% Specify and run EMA filter over each channel
emaFiltered = zeros(size(dataset)); % Initialize matrix for EMA filtered data
alpha = 0.18; % Smoothing factor for EMA. The filter history is ~1/alpha seconds

for ch = 1:nChannels
    emaFiltered(:, ch) = filter(alpha, [1, alpha - 1], dataset(:, ch));
end

% Find 5th and 95th percentile on each channel and save it
percentiles = zeros(2, nChannels); % Initialize matrix for percentiles
for ch = 1:nChannels
    percentiles(1, ch) = prctile(emaFiltered(:, ch), 5); % 5th percentile
    percentiles(2, ch) = prctile(emaFiltered(:, ch), 95); % 95th percentile
end
stimulation_window = 1; % ms we stimulate in low/high periods
jsonData = struct('electrodeID', [], 'fifthPercentile', [], 'ninetyFifthPercentile', []);

for ch = 1:nChannels
    jsonData(ch).electrodeID = selectedChannels(ch);
    jsonData(ch).fifthPercentile = percentiles(1, ch);
    jsonData(ch).ninetyFifthPercentile = percentiles(2, ch);
end

jsonOutput = jsonencode(jsonData);

% Write JSON data to a file
dateStr = datestr(now, 'yyyy-mm-dd'); % Get current date and time as string
fileID = fopen(['percentiles_' dateStr '.json'], 'w'); % Add date to JSON file name
if fileID == -1
    error('Cannot open file for writing.');
end
fwrite(fileID, jsonOutput, 'char');
fclose(fileID);

% Print JSON data electrode IDs and fifth/ninetyfifth percentile fields
for ch = 1:nChannels
    fprintf('Electrode ID: %d, Fifth Percentile: %.2f, Ninety Fifth Percentile: %.2f\n', ...
        jsonData(ch).electrodeID, jsonData(ch).fifthPercentile, jsonData(ch).ninetyFifthPercentile);
end

% Plot the 5th and 95th percentiles for each channel
figure('Position', [100, 100, 800, 1500]); % Set figure size to be taller

for idx = 1:nChannels
    ch = selectedChannels(idx);
    sb = subplot(nChannels, 1, idx); % Create a subplot for each channel

    hold on;
    
    % Plot original data (k-- dashed black)
    plot(1:length(dataset(:, idx)), dataset(:, idx), 'r--', 'DisplayName', sprintf('Original Channel %d', ch)); 
    
    % Plot EMA filtered data (b- solid blue)
    plot(1:length(emaFiltered(:, idx)), emaFiltered(:, idx), 'b-', 'DisplayName', 'EMA Filtered');    
    yline(percentiles(1, idx), 'r--'); % Dashed line for 5th percentile
    yline(percentiles(2, idx), 'g--'); % Dashed line for 95th percentile

    % Identify points exceeding the 95th percentile and below the 5th percentile
    belowThreshold = emaFiltered(:, idx) < percentiles(1, idx);
    aboveThreshold = emaFiltered(:, idx) > percentiles(2, idx);

    % Find the *start* of periods
    startAbove = find(diff([0; aboveThreshold]) == 1); 
    startBelow = find(diff([0; belowThreshold]) == 1);
    
    % Get current y-limits for the fill function
    currentYLims = ylim;

    % Initialize variables to track the end time of the last stimulation window
    lastHighStimTime = -inf;
    lastLowStimTime = -inf;

    % --- Create stimulation windows for above 95th percentile (Blue fill) ---
    for t = startAbove'
        if t >= lastHighStimTime + stimulation_window 
            fill([t, t + stimulation_window, t + stimulation_window, t], ...
                 [currentYLims(1), currentYLims(1), currentYLims(2), currentYLims(2)], ...
                 [1, 0.65, 0], 'FaceAlpha', 0.7, 'EdgeColor', 'none', 'DisplayName', 'Stim (High)');
            lastHighStimTime = t;
        end
    end
    
    % --- Create stimulation windows for below 5th percentile (Orange fill) ---
    for t = startBelow'
        if t >= lastLowStimTime + stimulation_window
            fill([t, t + stimulation_window, t + stimulation_window, t], ...
                 [currentYLims(1), currentYLims(1), currentYLims(2), currentYLims(2)], ...
                 'b', 'FaceAlpha', 0.4, 'EdgeColor', 'none', 'DisplayName', 'Stim (Low)');
            lastLowStimTime = t;
        end
    end
    
    % Bring the data lines back to the front so they are visible over the fill patches
    uistack(findobj(gca, 'Type', 'line'), 'top');    
    hold off;
    xlabel('Time');
    ylabel('Amplitude');
    title(sprintf('EMA Filtered Data for Channel %d', ch));
end

% Save the entire figure as a single file at the end
saveas(gcf, 'EMA_Filtered_All_Channels.png');

%% Section 3 Stimulate based on thresholds
if ~DEBUG_MODE
    % Stimulation parameters 
    % Set on/off durations and other pulse parameters
    stim_params.train_time = 0.1;        % 2 seconds ON
    stim_params.intertrain_time = 0.4;   % 2 seconds OFF
    stim_params.save_images_time = 5;  % Time for imaging system to save images
    
    % The pulse frequency (and derived parameters) will be updated per trial.
    % Default values (to be overwritten in each trial):
    stim_params.pulse_freq = 10;  
    stim_params.interpulse_interval = 1 / stim_params.pulse_freq;
    stim_params.stim_duration_NIP_clock = 30000 / stim_params.pulse_freq;
    stim_params.pulses_per_train = stim_params.pulse_freq * stim_params.train_time;
    stim_params.pulse_width = 6;  % (33 us step, as before)
    stim_params.IPI = 3;        % Interphase interval
    stim_params.train_stim_channels = 6;  
    stim_params.train_stim_currents = 6; 
    stim_params.train_delay_mask = 0;
    disp(stim_params);
end


jsonName = ['percentiles_', dateStr, '.json']; % or manually set
jsonStruct = jsondecode(fileread(jsonName));
electrodeIDs = {jsonStruct.electrodeID}; 

lowThresholds = [jsonStruct.fifthPercentile]; 
highThresholds = [jsonStruct.ninetyFifthPercentile];

disp('Electrode IDs (Cell Array):');
disp(electrodeIDs'); % Display as a column

disp('Low Thresholds (Numeric Vector):');
disp(lowThresholds');

disp('High Thresholds (Numeric Vector):');
disp(highThresholds');

% Keep track of whether threshold was crossed and whether stimulation was
% performed 
nChannels = max(selectedChannels);
threshold_crossings = cell(1, nChannels);

for ch = selectedChannels
    threshold_crossings{ch} = struct( ...
        'iter', {}, ...
        'time', {}, ...
        'channel', {}, ...
        'crossingType', {}, ...   % -1 = low, +1 = high
        'wasSelected', {}, ...    % true if selected for stim
        'channelStimulated', {}, ...       % true if stim delivered
        'stimAllowed', {},...
        'anyStimThisIter', {}...
    );
end

stimulation_Xts = nan(runtime_mins_stim * 60, length(selectedChannels));
iter = 1; % Initialize iter to 1 to avoid invalid index error

tStart = tic;
last_stim_time = tic; 
stim_cool_down = 5;
stim_delivered_iter = false(runtime_mins_stim * 60, 1);
stim_allowed_iter   = false(runtime_mins_stim * 60, 1);

disp("Starting stimulation loop");

if ~DEBUG_MODE
    X_ref = xippmex('spike', rec_channels);
    pause(1)
end

while toc(tStart) < runtime_mins_stim * 60

    if DEBUG_MODE
        X_t = randn(1, length(selectedChannels)) * 2 + 5;
    else

        X_t = xippmex('spike', selectedChannels);
    end
    stimulation_Xts(iter, :) = X_t;    
    

    stim_allowed = toc(last_stim_time) >= stim_cool_down;
    stim_allowed_iter(iter) = stim_allowed;
    
    if ~stim_allowed
        iter = iter + 1;
        pause(refresh_rate);
        continue;
    end

    
    
    stim_channels = [];
    crossings = [];

    for i = 1:length(selectedChannels)
        channelID = selectedChannels(i);
        thresholdLow  = lowThresholds(i);
        thresholdHigh = highThresholds(i);

        if X_t(i) < thresholdLow
            stim_channels(end+1) = channelID;
            crossings(end+1) = -1;

        elseif X_t(i) > thresholdHigh
            stim_channels(end+1) = channelID;
            crossings(end+1) = +1;
        end
    end

    stim_channel = 0;
    if toc(last_stim_time) < stim_cool_down
        % Ignore cooldown periods altogether
        continue; 
    end

    % choose stim channel or no stimulation with new rule
    if ~isempty(stim_channels) && (toc(last_stim_time) >= stim_cool_down)
        idx = randi(numel(stim_channels) + 1);
        if idx == numel(stim_channels) + 1 % don't stimulate with probability equal to 1/(n_threshold_cross_channels+1)
            stim_channel = 0;
        else
            stim_channel = stim_channels(idx);
            stim_params.train_stim_channels = stim_channel;
        end
    end


    if stim_channel ~= 0 
        stim_delivered_iter(iter) = true;
        last_stim_time = tic;

        if ~DEBUG_MODE
            cmd = stim_command(stim_params);
            xippmex('stimseq', cmd);
            pause(stim_params.train_time);
            fprintf('Stim at approx. %.2f s | iter %d | ch %d | crossing %d\n', ...
                toc(tStart), iter, stim_channel, crossings(idx));
        else
            fprintf('Stim at approx. %.2f s | iter %d | ch %d | crossing %d\n', ...
                toc(tStart), iter, stim_channel, crossings(idx));
        end
    
    elseif stim_channel == 0 % % completely ignore if not stimulating
        fprintf('No stim at approx. %.2f s | iter %d | ch %d \n ', ...
            toc(tStart), iter, stim_channel);
    end

    for k = 1:numel(stim_channels)
        % if any stim_channel crossed a threshold, then this is updated
        channelID = stim_channels(k);
        event.iter = iter;
        event.time = toc(tStart);
        event.channel = channelID;
        event.crossingType = crossings(k); 
        event.wasSelected = (channelID == stim_channel);
        event.channelStimulated = (channelID == stim_channel);
        event.stimAllowed = stim_allowed;
        event.anyStimThisIter = stim_delivered_iter(iter);
        threshold_crossings{channelID}(end+1) = event;
    end
    
    iter = iter + 1; 
    pause(refresh_rate);
end

save('threshold_crossings.mat', 'threshold_crossings', '-v7.3')
save('stimulation_Xts.mat', 'stimulation_Xts', '-v7.3');
fprintf('\nClosed-loop complete!\n');



%%
% ---- Final Ends
if ~DEBUG_MODE
    pause(0.5);
    xippmex('trial', 'stopped', ephys_filename);

    % Generate a new filename with a date and "_backup" suffix
    % Ensure a unique name if running multiple times on the same day
    %backupFileName = sprintf('%s_%s_saved.m', currentScriptName, datestr(now, 'yyyy_mm_dd_HHMMSS'));
    
    % Copy the current file to the new backup file name
    %copyfile([currentScriptName, '.m'], backupFileName);
end



%% Plot figures based on threshold crossings from stimulation 1
% each channel ch has k threshold crossings, accessed via tc{ch} struct
% with fields     iter, time, channel, crossingType, wasSelected, 
% and stimulated
fprintf('\nNow plotting figures based on threshold crossings and stimulations...\n');


tc_struct = load('threshold_crossings.mat'); 
threshold_crossings = tc_struct.threshold_crossings;

sx_struct = load('stimulation_Xts.mat');
stimulation_Xts = sx_struct.stimulation_Xts;  % size: iter x length(selectedChannels)

recordedChannels = selectedChannels;  % must match columns of stimulation_Xts
preWindowIter  = 0;  % 0 iterations before (pre-stim shouldn't be in the same histogram)
postWindowIter = 10;  % 10 iterations after

for chIdx = 1:length(recordedChannels)
    ch = recordedChannels(chIdx);
    if ~isempty(threshold_crossings{ch})
        events = threshold_crossings{ch};
        iterIndices = [events.iter];             
        crossingTypes = [events.crossingType];   
        thisStimulated    = [events.channelStimulated];    
        anyStimulated = [events.anyStimThisIter];
        % if this channel is not stimulated for a
        % given iteration, it means a different channel was stimulated
        
        subplotMasks = {
            (crossingTypes == -1 & ~anyStimulated),  % Low threshold, no stim on any channel
            (crossingTypes == -1 & thisStimulated),                            % Low threshold, stim
            (crossingTypes == 1 & ~anyStimulated ),  % High threshold, no stim on any channel
            (crossingTypes == 1 & thisStimulated)                             % High threshold, stim
        };

        
        % Collect stimulation_Xts for the lookback and lookahead
        stimVals = [];
        for iterCross = iterIndices
            idxStart = max(1, iterCross - preWindowIter);
            idxEnd   = min(size(stimulation_Xts, 1), iterCross + postWindowIter);
            stimVals = [stimVals; stimulation_Xts(idxStart:idxEnd, chIdx)];
        end
        
        % Get channel thresholds
        thresholdLow  = lowThresholds(chIdx);
        thresholdHigh = highThresholds(chIdx);
        
        figure;

        % Low threshold, no stimulation
        subplot(2,2,1);
        mask = subplotMasks{1};  % Use the mask from subplotMasks
        histogram(stimVals(mask), 'Normalization', 'probability');
        hold on;
        scatter(stimVals(mask), 0.01*ones(sum(mask),1), 50, 'b', 'filled'); % dots at bottom
        hold off;
        xlabel('Stimulation Xts'); ylabel('Probability');
        title(sprintf('Low Threshold, No Stim (Ch %d)', ch));
        grid on;

        % Low threshold, stimulation
        subplot(2,2,2);
        mask = subplotMasks{2};  % Use the mask from subplotMasks
        histogram(stimVals(mask), 'Normalization', 'probability');
        hold on;
        scatter(stimVals(mask), 0.01*ones(sum(mask),1), 50, 'g', 'filled'); 
        hold off;
        xlabel('Stimulation Xts'); ylabel('Probability');
        title(sprintf('Low Threshold, Stim (Ch %d)', ch));
        grid on;

        % High threshold, no stimulation
        subplot(2,2,3);
        mask = subplotMasks{3};  % Use the mask from subplotMasks
        histogram(stimVals(mask), 'Normalization', 'probability');
        hold on;
        scatter(stimVals(mask), 0.01*ones(sum(mask),1), 50, 'b', 'filled'); 
        hold off;
        xlabel('Stimulation Xts'); ylabel('Probability');
        title(sprintf('High Threshold, No Stim (Ch %d)', ch));
        grid on;

        % High threshold, stimulation
        subplot(2,2,4);
        mask = subplotMasks{4};  % Use the mask from subplotMasks
        histogram(stimVals(mask), 'Normalization', 'probability');
        hold on;
        scatter(stimVals(mask), 0.01*ones(sum(mask),1), 50, 'g', 'filled'); 
        hold off;
        xlabel('Stimulation Xts'); ylabel('Probability');
        title(sprintf('High Threshold, Stim (Ch %d)', ch));
        grid on;
    end
end

%% Plot figures 2 (looking at traces pre/post stim over 4 conditions)
tc_struct = load('threshold_crossings.mat'); 
threshold_crossings = tc_struct.threshold_crossings;

sx_struct = load('stimulation_Xts.mat');
stimulation_Xts = sx_struct.stimulation_Xts;  % size: iter x length(selectedChannels)

recordedChannels = selectedChannels;  % must match columns of stimulation_Xts
preWindowIter  = 10;  % 10 iterations before
postWindowIter = 10;  % 10 iterations after

for chIdx = 1:length(recordedChannels)
    ch = recordedChannels(chIdx);
    if ~isempty(threshold_crossings{ch})
        events = threshold_crossings{ch};
        iterIndices = [events.iter];             
        crossingTypes = [events.crossingType];   
        thisStimulated    = [events.channelStimulated];    
        anyStimulated = [events.anyStimThisIter];
        % if this channel was not stimulated, a
        % different channel was stimulated. we don't want to include these
        % in no stim

        figure('Name', sprintf('Channel %d', ch));

        % Define subplot types
        subplotMasks = {
            (crossingTypes == -1 & ~anyStimulated),  % Low threshold, no stim on any channel
            (crossingTypes == -1 & thisStimulated),                            % Low threshold, stim
            (crossingTypes == 1 & ~anyStimulated ),  % High threshold, no stim on any channel
            (crossingTypes == 1 & thisStimulated)                             % High threshold, stim
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
        idxEnd   = min(size(stimulation_Xts,1), iterCross + postWindowIter);

        Xtrace = stimulation_Xts(idxStart:idxEnd, chIdx);

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
    title(sprintf('%s (Ch %d)', subplotTitles{sp}, ch));
    grid on;
    hold off;
end
   end
end
%% Neuro-Stimulation Experiment Planner (V5 - Reproducible)
clear; clc; close all;

% --- 1. User-Set Parameters ---
random_seed       = 42;               % Set any integer for reproducibility
num_electrodes    = 10;               
current_levels    = [3, 4, 5];     
pulse_counts      = [1, 2, 3, 4, 5, 10];       
ipi_levels        = [0.002, 0.010, 0.100]; 

trial_runtime_ms  = 2000;           
max_total_runtime = 180;             % Minutes

% --- 2. Repeat Logic ---
percent_to_repeat = 1.0;             
num_repeats       = 10;              

% --- 3. Set Random Seed ---
% 'twister' is the standard Mersenne Twister algorithm
rng(random_seed, 'twister'); 

% --- 4. Generate Grid and Filter Redundancy ---
[E, C, P, I] = ndgrid(1:num_electrodes, current_levels, pulse_counts, ipi_levels);
all_combos = [E(:), C(:), P(:), I(:)];

% Remove redundant IPIs for single-pulse (1P) trials
is_1p = (all_combos(:,3) == 1);
is_first_ipi = (all_combos(:,4) == ipi_levels(1));
unique_trials = all_combos((~is_1p) | (is_1p & is_first_ipi), :);

% --- 5. Replication and Shuffle ---
num_unique = size(unique_trials, 1);
num_to_repeat = round(num_unique * percent_to_repeat);
idx_to_repeat = randperm(num_unique, num_to_repeat);

extra_trials = repmat(unique_trials(idx_to_repeat, :), num_repeats - 1, 1);
final_trial_matrix = [unique_trials; extra_trials];
total_trials = size(final_trial_matrix, 1);

% Final Random Shuffle (governed by the seed)
final_trial_matrix = final_trial_matrix(randperm(total_trials), :);
T = array2table(final_trial_matrix, 'VariableNames', {'Electrode', 'Current', 'Pulses', 'IPI'});

% --- 6. Sanity Checks ---
fprintf('--- VALIDATION REPORT (Seed: %d) ---\n', random_seed);

% Check: Every trial has a partner (except if repeats < 1)
[~, ~, group_idx] = unique(T, 'rows');
counts = histcounts(group_idx, 1:max(group_idx)+1);

if all(counts == num_repeats)
    fprintf('[PASS] Each of the %d unique conditions appears exactly %d times.\n', num_unique, num_repeats);
else
    fprintf('[FAIL] Count mismatch! Check repeat logic.\n');
end

% --- 7. Visualization ---
figure('Color', 'w', 'Name', sprintf('Seed %d Visualizer', random_seed));

% Subplot 1: Parameter Coverage
subplot(1,2,1);
scatter3(T.Electrode, T.Current, T.Pulses, 40, T.IPI, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Electrode'); ylabel('Current (uA)'); zlabel('Pulses');
title('Parameter Space Coverage'); colorbar; grid on;

% Subplot 2: Randomization Sequence
subplot(1,2,2);
stairs(T.Electrode(1:min(100, total_trials)), 'LineWidth', 1.5);
hold on;
stairs(T.Pulses(1:min(100, total_trials)), 'LineWidth', 1.5);
xlabel('Trial Index'); ylabel('Value');
title('Shuffle Pattern (First 100 Trials)');
legend('Electrode ID', 'Pulse Count');
grid on;

% Runtime Calculation
total_min = (total_trials * (trial_runtime_ms/1000)) / 60;
fprintf('Total Trials: %d | Expected Runtime: %.2f mins\n', total_trials, total_min);
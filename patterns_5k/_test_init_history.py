"""Test that __getitem__ produces correct shapes with init_state + history."""
from utils import BinnedStimSpikeDataset
from run_experiment import load_raw_data
import yaml

cfg = yaml.safe_load(open('results/hist_cnn_v3_2026-02-19_14-35-39/trial_0047_history=20/config.yaml'))
raw = load_raw_data(cfg, logger=None)
ks = cfg['kernel_sizes']
n_init = sum(k - 1 for k in ks)
print(f'kernel_sizes={ks}, n_initial_state_bins={n_init}, history={cfg["history"]}')

trials = list(raw['spike_responses'].keys())
# Use indices 1-2 (not 0, since init_state needs timing_idx - 1)
ds = BinnedStimSpikeDataset(
    trial_indices=trials[1:3],
    pattern_df=raw['pattern_df'], spike_responses=raw['spike_responses'],
    channel_to_index=raw['channel_to_index'], timing_to_pattern=raw['timing_to_pattern'],
    input_bin_size_ms=cfg['input_bin_size_ms'], output_bin_size_ms=cfg['output_bin_size_ms'],
    n_input_bins=cfg['n_input_bins'], n_output_bins=cfg['n_output_bins'],
    max_time_ms=cfg['max_time_ms'], output_offset=cfg['output_offset'],
    encoding_mode=cfg['encoding_mode'], init_state=cfg['init_state'],
    n_initial_state_bins=n_init, history=cfg['history'])

x, y = ds[0]
print(f'x.shape={x.shape}, y.shape={y.shape}')
expected_width = n_init + cfg['n_input_bins']
print(f'Expected x width = {expected_width} = n_init({n_init}) + n_input({cfg["n_input_bins"]})')
print(f'Conv output after valid conv = {x.shape[1] - n_init} (should be {cfg["n_output_bins"]})')

# Check history channel has data (not all zeros) in early columns
hist_ch = x[42:, :20]  # first 20 columns of history channels
has_data = (hist_ch != 0).any().item()
print(f'History ch early cols nonzero: {has_data}')

assert x.shape[1] == expected_width, f"x width {x.shape[1]} != expected {expected_width}"
assert y.shape[1] == cfg['n_output_bins'], f"y width {y.shape[1]} != {cfg['n_output_bins']}"
assert x.shape[1] - n_init == cfg['n_output_bins'], "Conv output would not match n_output_bins"
print("ALL CHECKS PASSED")

"""Quick shape + causality tests for SimpleCausalSpikeCNN."""
import torch, sys
sys.path.insert(0, ".")
from models import SimpleCausalSpikeCNN

def test(label, model, x_shape, expected_out):
    x = torch.randn(*x_shape)
    y = model(x)
    ok = tuple(y.shape) == tuple(expected_out)
    print(f"{'PASS' if ok else 'FAIL'} {label}: input {tuple(x.shape)} -> output {tuple(y.shape)}  (expected {tuple(expected_out)})")
    if not ok:
        sys.exit(1)

# Test 1: Causal padding K=60, init_state=False
m = SimpleCausalSpikeCNN(n_stim_channels=105, n_neurons=63, n_input_bins=60, n_output_bins=60,
                         embedding_dim=0, conv_channels=[128], kernel_sizes=[60],
                         fc_dims=[256], use_init_state=False)
test("causal K=60", m, (4, 105, 60), (4, 63, 60))

# Test 2: Valid padding K=60, init_state=True  (input = 60 + 59 context = 119)
m2 = SimpleCausalSpikeCNN(n_stim_channels=105, n_neurons=63, n_input_bins=60, n_output_bins=60,
                          embedding_dim=0, conv_channels=[128], kernel_sizes=[60],
                          fc_dims=[256], use_init_state=True)
test("valid K=60", m2, (4, 105, 119), (4, 63, 60))

# Test 3: Causal multi-layer K=[30,10]
m3 = SimpleCausalSpikeCNN(n_stim_channels=42, n_neurons=63, n_input_bins=60, n_output_bins=60,
                          embedding_dim=0, conv_channels=[64, 128], kernel_sizes=[30, 10],
                          fc_dims=[256], use_init_state=False)
test("causal K=[30,10]", m3, (4, 42, 60), (4, 63, 60))

# Test 4: Causality check - output at position 0 must NOT change when future input changes
print("\n--- Causality check ---")
m4 = SimpleCausalSpikeCNN(n_stim_channels=42, n_neurons=63, n_input_bins=60, n_output_bins=60,
                          embedding_dim=0, conv_channels=[128], kernel_sizes=[60],
                          fc_dims=[256], use_init_state=False)
m4.eval()
x1 = torch.randn(1, 42, 60)
x2 = x1.clone()
x2[:, :, 30:] = torch.randn(1, 42, 30)  # change future bins 30-59
with torch.no_grad():
    y1 = m4(x1)
    y2 = m4(x2)
# Output at position 0 should be identical (only depends on input at position 0)
diff_pos0 = (y1[:, :, 0] - y2[:, :, 0]).abs().max().item()
# Output at position 59 SHOULD differ (depends on different late inputs)
diff_pos59 = (y1[:, :, 59] - y2[:, :, 59]).abs().max().item()
causal_ok = diff_pos0 < 1e-6
print(f"{'PASS' if causal_ok else 'FAIL'} causality: output[0] diff={diff_pos0:.2e} (should be ~0), output[59] diff={diff_pos59:.2e} (should be >0)")
if not causal_ok:
    sys.exit(1)

print("\nAll tests PASSED")

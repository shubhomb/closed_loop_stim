import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset


class RNNModel(nn.Module):
    def __init__(self, input_size: int, units: int, output_size: int, num_layers: int = 2, initial_cond_size: int = None):
        """
        RNN model for predicting neural activity from stimulation input.
        
        Parameters
        ----------
        input_size : int
            Number of input features (e.g., electrodes)
        units : int
            Hidden state size
        output_size : int
            Number of output neurons to predict (TARGET_NEURONS)
        num_layers : int
            Number of GRU layers
        initial_cond_size : int, optional
            Size of initial condition input (FILTER_NEURONS). If None, defaults to output_size.
            This allows using more neurons for initial state than for prediction targets.
        """
        super(RNNModel, self).__init__()
        self.num_layers = num_layers
        self.units = units
        # Use initial_cond_size if provided, otherwise default to output_size
        self.initial_cond_size = initial_cond_size if initial_cond_size is not None else output_size
        self.initial_state_projection = nn.Linear(self.initial_cond_size, units)  # activity_initial -> hidden for layer 0
        self.rnn = nn.GRU(input_size, units, 
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=0.2 if num_layers > 1 else 0.0,
                          bidirectional=False,
                          )  # returns (batch, seq, hidden)
        self.dense = nn.Sequential(
            nn.Linear(units, output_size)
        )  # maps hidden -> ROI outputs with nonlinearity

    def forward(self, inp):
        # inp is a tuple: (inputs, activity_initial)
        # inputs: Tensor shape (batch, seq_len, input_size)
        # activity_initial: Tensor shape (batch, initial_cond_size) - uses all FILTER_NEURONS
        inputs, activity_initial = inp

        # Ensure tensors and device alignment
        device = next(self.parameters()).device
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
        if not torch.is_tensor(activity_initial):
            activity_initial = torch.tensor(activity_initial, dtype=torch.float32, device=device)

        batch_size = inputs.size(0)
        
        # Build h0 for all layers: (num_layers, batch, units)
        # Layer 0: project from initial activity
        # Layers 1+: initialize to zeros
        h0_layer0 = self.initial_state_projection(activity_initial)  # (batch, units)
        
        if self.num_layers == 1:
            h0 = h0_layer0.unsqueeze(0)  # (1, batch, units)
        else:
            # Initialize remaining layers with zeros
            h0_remaining = torch.zeros(self.num_layers - 1, batch_size, self.units, 
                                       device=device, dtype=h0_layer0.dtype)
            h0 = torch.cat([h0_layer0.unsqueeze(0), h0_remaining], dim=0)  # (num_layers, batch, units)

        # GRU forward
        out, _ = self.rnn(inputs, h0)                           # out: (batch, seq_len, units)

        # per-timestep hidden to output ROIs
        return self.dense(out)                                  # (batch, seq_len, output_size)


# --- Dataset that returns ((inputs, activity_initial), target) ---
class SeqDataset(Dataset):
    """
    Dataset for RNN training that returns ((stim_inputs, activity_initial), activity_target).
    
    Can be initialized from arrays or from a DataFrame with snippet columns.
    """
    def __init__(self, activity_init_conds=None, stims=None, activity=None, df=None, target_indices=None):
        """
        Initialize from arrays OR from a DataFrame.
        
        Parameters
        ----------
        activity_init_conds : ndarray, optional
            Initial conditions, shape (n_samples, n_rois)
        stims : ndarray, optional
            Stimulation snippets, shape (n_samples, seq_len, n_electrodes)
        activity : ndarray, optional
            Activity snippets, shape (n_samples, seq_len, n_rois)
        df : pd.DataFrame, optional
            DataFrame with columns: initial_condition, stim_snippet, activity_snippet, valid
            If provided, arrays are extracted from the DataFrame.
        target_indices : list of int, optional
            Indices of neurons to use as targets (columns in activity_snippet).
            If None, all neurons are used as targets. Initial conditions always use all neurons.
        """
        self.target_indices = target_indices
        
        if df is not None:
            # Initialize from DataFrame
            valid_df = df[df['valid']].copy()
            self.initial_conds = torch.tensor(
                np.stack(valid_df['initial_condition'].values), dtype=torch.float32
            )
            self.X = torch.tensor(
                np.stack(valid_df['stim_snippet'].values), dtype=torch.float32
            )
            # Full activity for reference (used for initial conditions dimension)
            full_activity = torch.tensor(
                np.stack(valid_df['activity_snippet'].values), dtype=torch.float32
            )
            # Subset to target neurons if specified
            if target_indices is not None:
                self.Y = full_activity[:, :, target_indices]
            else:
                self.Y = full_activity
            # Store metadata for reference
            self.metadata = valid_df.drop(
                columns=['initial_condition', 'stim_snippet', 'activity_snippet', 'valid']
            ).reset_index(drop=True)
        else:
            # Initialize from arrays
            self.initial_conds = torch.tensor(activity_init_conds, dtype=torch.float32)
            self.X = torch.tensor(stims, dtype=torch.float32)
            full_activity = torch.tensor(activity, dtype=torch.float32)
            # Subset to target neurons if specified
            if target_indices is not None:
                self.Y = full_activity[:, :, target_indices]
            else:
                self.Y = full_activity
            self.metadata = None
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        inp = self.X[idx]                     # (seq_len, input_size)
        target = self.Y[idx]                 # (seq_len, output_size)
        activity_initial = self.initial_conds[idx]
        return (inp, activity_initial), target
    
    def get_metadata(self, idx):
        """Get metadata for a specific sample (session, trial, config, etc.)"""
        if self.metadata is not None:
            return self.metadata.iloc[idx].to_dict()
        return None



class ARkModel(nn.Module):
    """
    Linear AR[k] control model. Predicts y_t as a linear function of the
    last k activity values and the last k stim values (no nonlinearity).

        y_t = sum_{i=1..k} A_i y_{t-i} + sum_{i=0..k-1} B_i u_{t-i} + b

    Initial activity (pre-stim history) is taken from `activity_initial`, which
    must contain the same neurons that are being predicted (use target_indices
    for both `target_indices` and as the slice of the initial condition).

    Parameters
    ----------
    input_size : int
        Number of stim input channels (electrodes).
    output_size : int
        Number of output neurons to predict.
    k : int
        AR order (lag length). Same lag used for activity and stim.
    init_cond_indices : list of int, optional
        If `activity_initial` from the dataset contains FILTER_NEURONS but the
        AR model only predicts TARGET_NEURONS, pass the indices that map
        FILTER -> TARGET so the history is sliced correctly. If None, assumes
        `activity_initial` already has `output_size` columns.
    """
    def __init__(self, input_size: int, output_size: int, k: int = 5,
                 init_cond_indices=None):
        super().__init__()
        self.k = k
        self.input_size = input_size
        self.output_size = output_size
        if init_cond_indices is not None:
            self.register_buffer(
                'init_cond_indices',
                torch.tensor(init_cond_indices, dtype=torch.long),
            )
        else:
            self.init_cond_indices = None
        # Split the linear map into A (activity-lag) and B (stim-lag) blocks so we can
        # apply the stim contribution in one shot and only recurse on the activity side.
        #   y_t = A @ vec(y_{t-1..t-k}) + B @ vec(u_{t..t-k+1}) + bias
        self.A = nn.Linear(k * output_size, output_size, bias=True)   # carries the bias
        self.B = nn.Linear(k * input_size, output_size, bias=False)

    def forward(self, inp):
        inputs, activity_initial = inp
        device = next(self.parameters()).device
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
        if not torch.is_tensor(activity_initial):
            activity_initial = torch.tensor(activity_initial, dtype=torch.float32, device=device)

        batch, seq_len, _ = inputs.shape

        if self.init_cond_indices is not None:
            y_init = activity_initial.index_select(-1, self.init_cond_indices)  # (batch, output_size)
        else:
            y_init = activity_initial

        # ---- Vectorized stim contribution ----
        # Build per-timestep stim-lag features: at each t, [u_t, u_{t-1}, ..., u_{t-k+1}].
        # unfold gives a (batch, n_windows, input_size, k) view; rearrange to (batch, T, k*input_size).
        if self.k > 1:
            pad = torch.zeros(batch, self.k - 1, self.input_size,
                              device=inputs.device, dtype=inputs.dtype)
            u_padded = torch.cat([pad, inputs], dim=1)  # (batch, T+k-1, input_size)
        else:
            u_padded = inputs
        # unfold along time dim: (batch, T, input_size, k); flip to put most recent first
        u_lags = u_padded.unfold(dimension=1, size=self.k, step=1)              # (batch, T, input_size, k)
        u_lags = u_lags.flip(dims=[-1])                                          # most recent first
        u_feat = u_lags.reshape(batch, seq_len, self.k * self.input_size)
        stim_contrib = self.B(u_feat)                                            # (batch, T, output_size)

        # ---- Recursive activity contribution (must stay sequential) ----
        # Carry a rolling history of size k with most-recent-first ordering.
        y_hist = y_init.unsqueeze(1).expand(batch, self.k, self.output_size).contiguous()
        outs = []
        for t in range(seq_len):
            y_t = self.A(y_hist.reshape(batch, -1)) + stim_contrib[:, t, :]
            outs.append(y_t)
            y_hist = torch.cat([y_t.unsqueeze(1), y_hist[:, :-1, :]], dim=1)

        return torch.stack(outs, dim=1)  # (batch, seq_len, output_size)


class WeightedMAELoss(nn.Module):
    def __init__(self, stim_weight=5.0, window=15, decay='none', linear_end=0.2, exp_tau=5.0):
        """
        Weighted MAE loss that applies higher weight to frames around stimulation.
        
        Parameters
        ----------
        stim_weight : float
            Multiplier for frames around stim (e.g., 5x weight)
        window : int
            Number of frames after stim to weight higher
        decay : str
            'none' (flat), 'linear', or 'exp' for weight decay after stim
        linear_end : float
            End value for linear decay (start=1.0, end=linear_end). Default 0.2.
        exp_tau : float
            Time constant for exponential decay: exp(-t/tau). Default 5.0 frames.
        """
        super().__init__()
        self.stim_weight = stim_weight
        self.window = window
        self.decay = decay
        self.linear_end = linear_end
        self.exp_tau = exp_tau
    
    def forward(self, pred, target, stim_input):
        batch, seq_len, n_out = pred.shape
        device = pred.device
        
        # Detect stim at each timestep (any electrode with current > 0)
        stim_any = (stim_input.sum(dim=-1) > 0).float()  # (batch, seq_len)
        
        # Create convolution kernel to spread weight forward from stim times
        kernel = torch.ones(1, 1, self.window, device=device)
        if self.decay == 'linear':
            kernel = torch.linspace(1, self.linear_end, self.window, device=device).view(1, 1, -1)
        elif self.decay == 'exp':
            kernel = torch.exp(-torch.arange(self.window, device=device).float() / self.exp_tau).view(1, 1, -1)
        
        # Convolve to spread weights forward from stim times
        stim_padded = torch.nn.functional.pad(stim_any.unsqueeze(1), (0, self.window - 1))  # pad right
        weight_boost = torch.nn.functional.conv1d(stim_padded, kernel).squeeze(1)  # (batch, seq_len)
        
        # Combine: base weight 1 + boost where stim occurred
        weights = 1.0 + (self.stim_weight - 1.0) * (weight_boost > 0).float()
        weights = weights.unsqueeze(-1)  # (batch, seq_len, 1)
        
        loss = (weights * torch.abs(pred - target)).mean()
        return loss
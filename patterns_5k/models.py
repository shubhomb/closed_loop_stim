"""
Neural stimulation to spike prediction models.

Models take categorical stimulation input and predict spike rates/counts.
Input: (batch, n_channels, n_input_bins) - categorical indices [0-4]
Output: (batch, n_neurons, n_output_bins) - predicted spike rates (log scale for Poisson)
"""

from pyexpat import features
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from collections import defaultdict
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from metrics import compute_correlation

        
def get_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: One of 'mlp', 'cnn', 'causal_cnn'
        **kwargs: Model-specific arguments
    
    Returns:
        Instantiated model
    """
    models = {
        'cnn': SimpleCausalSpikeCNN,
        'cache_cnn': HistoryCacheCausalCNN
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](**kwargs)


class SimpleCausalSpikeCNN(nn.Module):
    def __init__(self,
                 n_stim_channels: int,
                 n_neurons: int,
                 n_input_bins: int = 60,
                 n_output_bins: int = 10,
                 embedding_dim: int = 8,
                 conv_channels: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] = [3, 3, 3],
                 fc_dims: List[int] = [128],
                 dropout: float = 0.2,
                 num_stim_levels: int = 5,
                 pooling: str = 'flatten',
                 use_batch_norm: bool = True,
                 use_init_state: bool = False):
        super().__init__()
        
        self.n_input_bins = n_input_bins
        self.n_output_bins = n_output_bins
        self.use_init_state = use_init_state

        if len(conv_channels) != len(kernel_sizes):
            raise ValueError(
                f"conv_channels ({len(conv_channels)} layers) and kernel_sizes "
                f"({len(kernel_sizes)} layers) must have the same length. "
                f"Got conv_channels={conv_channels}, kernel_sizes={kernel_sizes}"
            )
        
        # 1. Embedding Setup
        self.embedding_dim = max(embedding_dim, 1)
        if self.embedding_dim <= 1:
            self.embedding = nn.Identity()
            in_channels = n_stim_channels
        else:
            self.embedding = nn.Embedding(num_embeddings=num_stim_levels, embedding_dim=embedding_dim)
            in_channels = n_stim_channels * self.embedding_dim

        # 2. Build CNN.
        #    - valid convolution when using init_state (dataset prepends context
        #      from the previous trial so output length == n_input_bins).
        #    - causal (left-only) padding otherwise: pad K-1 on the left so each
        #      output position only depends on current & past inputs.  This
        #      preserves the temporal dimension (like 'same') but prevents
        #      forward leakage when spike-history channels are present.
        self.kernel_sizes_list = list(kernel_sizes)
        layers = []
        for out_channels, k_size in zip(conv_channels, kernel_sizes):
            if use_init_state:
                # Valid conv — no padding; dataset prepends context bins
                layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=k_size))
            else:
                # Causal conv — left-only padding
                layers.append(nn.ConstantPad1d((k_size - 1, 0), 0))
                layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=k_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            in_channels = out_channels
            
        self.conv_stack = nn.Sequential(*layers)

        # 3. FC Layers (applied per timestep — no pooling)
        fc_layers = []
        curr_dim = conv_channels[-1]
        
        for fc_dim in fc_dims:
            fc_layers.extend([
                nn.Linear(curr_dim, fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            curr_dim = fc_dim
            
        fc_layers.append(nn.Linear(curr_dim, n_neurons))
        self.fc = nn.Sequential(*fc_layers)

    @property
    def total_conv_reduction(self):
        """Time bins consumed by valid convolution (sum of kernel_size-1 per layer).
        When using init_state, prepend this many bins from the previous trial."""
        return sum(k - 1 for k in self.kernel_sizes_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_channels, n_input_bins = x.shape
        
        # 1. Embed & Reshape: (Batch, Channels * Emb, Time)
        if self.embedding_dim > 1:
            x = self.embedding(x)
            x = x.permute(0, 1, 3, 2).reshape(batch_size, -1, n_input_bins)
        
        # 2. Conv over the sequence
        #    valid: output T = input_bins - total_conv_reduction  (== n_input_bins with prepended context)
        #    same:  output T = input_bins  (== n_input_bins)
        features = self.conv_stack(x)  # (B, conv_ch, T)
        
        # 3. FC per timestep (no pooling)
        features = features.transpose(1, 2)  # (B, T, conv_ch)
        y = self.fc(features)                # (B, T, n_neurons)
        # Final Shape: (Batch, n_neurons, n_output_bins)
        return y.transpose(1, 2)




class HistoryCacheCausalCNN(nn.Module):
    """Causal CNN with a per-neuron cache of recent outputs.

    The cache stores the last ``cache_size`` output bins (predicted firing
    rates) for every neuron and is embedded into additional input channels
    that are concatenated with the stimulus before the conv stack.

    Three forward modes
    -------------------
    * **teacher_forcing** (default) — cache at each timestep is filled
      entirely from ground-truth spike history.  The full input is
      pre-built in parallel and processed in one pass (fast).
    * **semi_ar** — sequential loop over timesteps.  The most recent
      cache bins come from the model's own predictions; the oldest bin
      is replaced with ground truth once it becomes available (models
      a fixed observation delay of ``cache_size − 1`` bins).  Usable
      at both train and inference time.
    * **ar** — fully autoregressive.  All cache bins are the model's
      own predictions (no ground truth at any lag).

    Input channels to the conv: ``n_stim_channels + n_neurons * cache_embed_dim``.
    """

    def __init__(self,
                 n_stim_channels: int,
                 n_neurons: int,
                 n_input_bins: int = 60,
                 n_output_bins: int = 10,
                 conv_channels: List[int] = [256],
                 kernel_sizes: List[int] = [3],
                 fc_dims: List[int] = [256],
                 dropout: float = 0.2,
                 cache_size: int = 5,
                 cache_embed_dim: int = 8,
                 use_batch_norm: bool = True,
                 use_init_state: bool = True):
        super().__init__()

        self.n_stim_channels = n_stim_channels
        self.n_neurons = n_neurons
        self.n_input_bins = n_input_bins
        self.n_output_bins = n_output_bins
        self.use_init_state = use_init_state
        self.cache_size = cache_size
        self.cache_embed_dim = cache_embed_dim

        if len(conv_channels) != len(kernel_sizes):
            raise ValueError(
                f"conv_channels ({len(conv_channels)} layers) and kernel_sizes "
                f"({len(kernel_sizes)} layers) must have the same length. "
                f"Got conv_channels={conv_channels}, kernel_sizes={kernel_sizes}"
            )

        # ---- Per-neuron cache embedding (shared weights across neurons) ----
        # Maps (cache_size,) → (cache_embed_dim,) for each neuron.
        self.cache_embedding = nn.Sequential(
            nn.Linear(cache_size, cache_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ---- Conv stack ----
        # Input channels = stim channels + embedded cache channels.
        in_channels = n_stim_channels + n_neurons * cache_embed_dim
        self.kernel_sizes_list = list(kernel_sizes)
        layers: list[nn.Module] = []
        for out_channels, k_size in zip(conv_channels, kernel_sizes):
            if use_init_state:
                # Valid conv — no padding; dataset prepends context bins
                layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=k_size))
            else:
                # Causal conv — left-only padding
                layers.append(nn.ConstantPad1d((k_size - 1, 0), 0))
                layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=k_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.conv_stack = nn.Sequential(*layers)

        # ---- FC head (applied per timestep) ----
        fc_layers: list[nn.Module] = []
        curr_dim = conv_channels[-1]
        for fc_dim in fc_dims:
            fc_layers.extend([
                nn.Linear(curr_dim, fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            curr_dim = fc_dim
        fc_layers.append(nn.Linear(curr_dim, n_neurons))
        self.fc = nn.Sequential(*fc_layers)

    # ------------------------------------------------------------------
    @property
    def total_conv_reduction(self) -> int:
        """Time bins consumed by valid convolution (sum of kernel_size-1 per layer)."""
        return sum(k - 1 for k in self.kernel_sizes_list)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _embed_cache(self, cache: torch.Tensor) -> torch.Tensor:
        """Embed a cache tensor into channel features.

        Parameters
        ----------
        cache : (B, n_neurons, cache_size)

        Returns
        -------
        (B, n_neurons * cache_embed_dim)
        """
        B = cache.shape[0]
        # (B * n_neurons, cache_size) → embed → (B * n_neurons, embed_dim)
        flat = cache.reshape(B * self.n_neurons, self.cache_size)
        embedded = self.cache_embedding(flat)
        return embedded.reshape(B, self.n_neurons * self.cache_embed_dim)

    def _build_cache_channels_tf(self, spike_context: torch.Tensor,
                                 T: int) -> torch.Tensor:
        """Build cache input channels for **all** timesteps from GT (teacher forcing).

        Parameters
        ----------
        spike_context : (B, n_neurons, T_ctx)
            Ground-truth spike counts covering [0 … T_ctx).  At output
            timestep *t*, the cache contains
            ``spike_context[:, :, t-cache_size : t]`` (zero-padded when
            *t* < cache_size).
        T : int
            Number of input timesteps to produce cache channels for.

        Returns
        -------
        cache_channels : (B, n_neurons * cache_embed_dim, T)
        """
        B = spike_context.shape[0]
        device = spike_context.device

        # Pad left so indexing never goes negative
        padded = F.pad(spike_context, (self.cache_size, 0), value=0.0)
        # padded[:, :, t : t + cache_size] gives cache at input time t

        # Use unfold to get all sliding windows at once
        # (B, n_neurons, T_padded) → unfold(dim=2, size=cs, step=1)
        # → (B, n_neurons, n_windows, cache_size)
        windows = padded.unfold(2, self.cache_size, 1)  # (B, N, ?, cs)
        windows = windows[:, :, :T, :]                  # (B, N, T, cs)

        # Embed: flatten (B*N*T, cs) → embed → reshape
        BNT = B * self.n_neurons * T
        flat = windows.reshape(BNT, self.cache_size)
        embedded = self.cache_embedding(flat)            # (BNT, embed_dim)
        embedded = embedded.reshape(B, self.n_neurons, T, self.cache_embed_dim)
        # → (B, N * embed_dim, T)
        return embedded.permute(0, 1, 3, 2).reshape(
            B, self.n_neurons * self.cache_embed_dim, T)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, stim_input: torch.Tensor,
                spike_context: Optional[torch.Tensor] = None,
                mode: str = 'teacher_forcing') -> torch.Tensor:
        """
        Parameters
        ----------
        stim_input : (B, n_stim_channels, T_in)
            Stimulus-only input (may include init-state padding bins on
            the left when ``use_init_state=True``).
        spike_context : (B, n_neurons, T_ctx) or None
            Ground-truth spike counts used for building the cache.

            * ``teacher_forcing``: required.  Cache at each timestep is
              sliced from this tensor.
            * ``semi_ar``: required.  Only the oldest cache bin is
              replaced with GT once it becomes observable (delay =
              ``cache_size − 1`` bins); more recent bins are the
              model's own predictions.
            * ``ar``: ignored — cache is filled entirely from
              predictions.
        mode : {'teacher_forcing', 'semi_ar', 'ar'}

        Returns
        -------
        (B, n_neurons, T_out)  predicted log-rates.
        """
        if mode == 'teacher_forcing':
            return self._forward_teacher_forcing(stim_input, spike_context)
        elif mode in ('semi_ar', 'ar'):
            return self._forward_sequential(stim_input, spike_context, mode)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'teacher_forcing', 'semi_ar', or 'ar'.")

    # ------------------------------------------------------------------
    def _forward_teacher_forcing(self, stim_input: torch.Tensor,
                                  spike_context: torch.Tensor) -> torch.Tensor:
        """Parallel teacher-forced forward pass."""
        B, _, T_in = stim_input.shape
        cache_ch = self._build_cache_channels_tf(spike_context, T_in)
        x = torch.cat([stim_input, cache_ch], dim=1)   # (B, in_ch, T_in)
        features = self.conv_stack(x)                   # (B, conv_ch, T_out)
        features = features.transpose(1, 2)             # (B, T_out, conv_ch)
        y = self.fc(features)                           # (B, T_out, n_neurons)
        return y.transpose(1, 2)                        # (B, n_neurons, T_out)

    # ------------------------------------------------------------------
    def _forward_sequential(self, stim_input: torch.Tensor,
                             spike_context: Optional[torch.Tensor],
                             mode: str) -> torch.Tensor:
        """Sequential forward with streaming conv buffer (semi_ar / ar).

        Processes one input timestep at a time:
        1. Embed current cache → extra channels for this timestep.
        2. Append ``[stim_t, cache_channels_t]`` to a sliding conv buffer.
        3. Once the buffer has ``receptive_field`` frames, run the conv
           stack to produce one output timestep.
        4. Run FC → prediction.
        5. Update cache: shift left, newest = prediction.  In semi_ar
           mode, replace oldest bin with GT once observable.
        """
        B, _, T_in = stim_input.shape
        device = stim_input.device
        rf = self.total_conv_reduction + 1               # receptive field (frames needed for 1 output)
        T_out = T_in - self.total_conv_reduction if self.use_init_state else T_in
        in_ch = self.n_stim_channels + self.n_neurons * self.cache_embed_dim

        outputs = torch.zeros(B, self.n_neurons, T_out, device=device)

        # Cache: (B, n_neurons, cache_size) — initialised to zeros
        cache = torch.zeros(B, self.n_neurons, self.cache_size, device=device)

        # Streaming conv buffer: (B, in_ch, rf)
        buf = torch.zeros(B, in_ch, rf, device=device)

        out_idx = 0
        for t in range(T_in):
            # 1. Embed current cache → (B, n_neurons * embed_dim)
            cache_emb = self._embed_cache(cache)

            # 2. Build input frame: (B, in_ch, 1)
            stim_t = stim_input[:, :, t]                 # (B, stim_ch)
            frame = torch.cat([stim_t, cache_emb], dim=1).unsqueeze(-1)

            # 3. Shift buffer left, append frame on the right
            buf = torch.cat([buf[:, :, 1:], frame], dim=-1)

            # 4. Produce output once we have enough context
            if t >= self.total_conv_reduction:
                # Run conv stack but skip any ConstantPad1d layers —
                # the buffer already provides the receptive-field context.
                h = buf
                for layer in self.conv_stack:
                    if isinstance(layer, nn.ConstantPad1d):
                        continue
                    h = layer(h)
                conv_out = h.squeeze(-1)                 # (B, conv_ch)
                pred_log = self.fc(conv_out)              # (B, n_neurons)
                outputs[:, :, out_idx] = pred_log

                # 5. Update cache
                pred_rate = torch.exp(pred_log).detach()
                cache = torch.cat([cache[:, :, 1:],
                                   pred_rate.unsqueeze(-1)], dim=-1)

                # Semi-AR: replace oldest bin with GT (delayed observation)
                if mode == 'semi_ar' and spike_context is not None:
                    gt_t = out_idx - self.cache_size + 1
                    if 0 <= gt_t < spike_context.shape[2]:
                        cache[:, :, 0] = spike_context[:, :, gt_t]

                out_idx += 1

        return outputs
        
    

# =====================
# Single-trial sliding-window prediction helpers
# =====================

def sliding_window_predict_trial(model_tuple, cfg, test_loader, timing_idx,
                                 stim_binned=None, spikes_binned=None):
    """Sliding-window, teacher-forced prediction for a **single trial**.

    Returns a ``(n_neurons, total_output_bins)`` rate array covering the
    whole trial timeline.  Overlapping windows are averaged.

    Parameters
    ----------
    model_tuple : tuple  ``(model, _, device)``
    cfg : dict           experiment config (needs input/output bin sizes, offsets, etc.)
    test_loader : DataLoader  (only ``.dataset`` is used for defaults)
    timing_idx : int     trial timing index
    stim_binned : ndarray, optional   ``(n_stim_ch, total_input_bins)``; defaults to
        the dataset's stored stim for this trial's pattern.
    spikes_binned : ndarray, optional  ``(n_neurons, total_output_bins)``; ground-truth
        spike counts for this trial (used as history).  Defaults to dataset's stored version.
    """
    model, _, device = model_tuple
    ds = test_loader.dataset

    n_neurons         = ds.n_neurons
    n_input_bins      = cfg['n_input_bins']
    n_output_bins     = cfg['n_output_bins']
    output_bin_size   = cfg['output_bin_size_ms']
    input_bin_size    = cfg['input_bin_size_ms']
    max_time_ms       = cfg['max_time_ms']
    output_offset     = cfg.get('output_offset', 0)
    history           = cfg.get('history', 0)
    init_state_flag   = cfg.get('init_state', False)
    n_init            = getattr(ds, 'n_initial_state_bins', 0)

    tot_out = max_time_ms // output_bin_size
    tot_in  = max_time_ms // input_bin_size
    max_s   = min(tot_out - output_offset - n_output_bins, tot_in - n_input_bins)

    if stim_binned is None:
        stim_binned = ds.pattern_stims[ds.timing_to_pattern[timing_idx]]
    if spikes_binned is None:
        spikes_binned = ds.spike_responses_binned[timing_idx]

    p_sum = np.zeros((n_neurons, tot_out), dtype=np.float32)
    p_cnt = np.zeros(tot_out, dtype=np.float32)

    _init = None
    if init_state_flag and n_init > 0:
        prev = timing_idx - 1
        if prev in ds.spike_responses_binned:
            _init = ds.spike_responses_binned[prev][:, -n_init:]
        else:
            _init = np.zeros((n_neurons, n_init), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for t in range(max_s + 1):
            se = t + n_input_bins
            actual_bins = stim_binned.shape[1]
            if se <= actual_bins:
                xs = stim_binned[:, t:se].copy()
            else:
                xs = np.zeros((stim_binned.shape[0], n_input_bins), dtype=np.float32)
                av = max(0, actual_bins - t)
                if av > 0:
                    xs[:, :av] = stim_binned[:, t:t + av]

            if init_state_flag and _init is not None:
                xs = np.pad(xs, ((0, 0), (n_init, 0)), mode='constant')

            if history and history > 0:
                if init_state_flag and _init is not None:
                    sfl = np.concatenate([_init, spikes_binned[:, t:t + n_input_bins]], axis=1)
                else:
                    sfl = spikes_binned[:, t:t + n_input_bins]
                tt = xs.shape[1]
                yh = np.zeros((n_neurons, tt), dtype=np.float32)
                if tt > history:
                    src = sfl[:, :tt - history]
                    yh[:, history:history + src.shape[1]] = src
                xs = np.concatenate([xs.astype(np.float32), yh], axis=0)

            bx = torch.tensor(xs, dtype=torch.float32).unsqueeze(0).to(device)
            pr = model(bx).cpu().numpy()
            for o in range(n_output_bins):
                tb = t + output_offset + o
                if tb < tot_out:
                    p_sum[:, tb] += pr[0, :, o]
                    p_cnt[tb] += 1

    msk = p_cnt > 0
    pa = np.zeros_like(p_sum)
    pa[:, msk] = p_sum[:, msk] / p_cnt[msk]
    return np.exp(pa)


def sliding_window_predict_trial_ar(model_tuple, cfg, test_loader, timing_idx,
                                     stim_binned=None, chunk_length=None):
    """Single-trial **autoregressive** sliding-window prediction.

    Same interface as :func:`sliding_window_predict_trial` but the model's
    own predicted rates are fed back as spike-history instead of ground-truth.
    The init-state context from the previous trial is always ground-truth.

    Parameters
    ----------
    chunk_length : int or None
        If given, every *chunk_length* output bins the AR history buffer is
        replaced by ground-truth spikes (periodic correction).
    """
    model, _, device = model_tuple
    ds = test_loader.dataset

    n_neurons         = ds.n_neurons
    n_input_bins      = cfg['n_input_bins']
    n_output_bins     = cfg['n_output_bins']
    output_bin_size   = cfg['output_bin_size_ms']
    input_bin_size    = cfg['input_bin_size_ms']
    max_time_ms       = cfg['max_time_ms']
    output_offset     = cfg.get('output_offset', 0)
    history           = cfg.get('history', 0)
    init_state_flag   = cfg.get('init_state', False)
    n_init            = getattr(ds, 'n_initial_state_bins', 0)

    if history is None or history == 0:
        raise ValueError("AR prediction requires history > 0.")

    tot_out = max_time_ms // output_bin_size
    tot_in  = max_time_ms // input_bin_size
    max_s   = min(tot_out - output_offset - n_output_bins, tot_in - n_input_bins)

    if stim_binned is None:
        stim_binned = ds.pattern_stims[ds.timing_to_pattern[timing_idx]]
    gt_spikes = ds.spike_responses_binned[timing_idx] # ground truth spikes for the trial

    p_sum  = np.zeros((n_neurons, tot_out), dtype=np.float32)
    p_cnt  = np.zeros(tot_out, dtype=np.float32)
    ar_sum = np.zeros((n_neurons, tot_out), dtype=np.float32)
    ar_cnt = np.zeros(tot_out, dtype=np.float32)

    _init = None
    if init_state_flag and n_init > 0:
        prev = timing_idx - 1
        if prev in ds.spike_responses_binned:
            _init = ds.spike_responses_binned[prev][:, -n_init:]
        else:
            _init = np.zeros((n_neurons, n_init), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for t in range(max_s + 1):
            if chunk_length is not None:
                t_out = t + output_offset
                if t_out > 0 and t_out % chunk_length == 0:
                    ar_sum[:, :t_out] = gt_spikes[:, :t_out].astype(np.float32)
                    ar_cnt[:t_out]    = 1.0

            se = t + n_input_bins
            actual_bins = stim_binned.shape[1]
            if se <= actual_bins:
                xs = stim_binned[:, t:se].copy()
            else:
                xs = np.zeros((stim_binned.shape[0], n_input_bins), dtype=np.float32)
                av = max(0, actual_bins - t)
                if av > 0:
                    xs[:, :av] = stim_binned[:, t:t + av]

            if init_state_flag and _init is not None:
                xs = np.pad(xs, ((0, 0), (n_init, 0)), mode='constant')

            # -- build AR history channel --
            curr_ar = np.zeros((n_neurons, tot_out), dtype=np.float32)
            msk = ar_cnt > 0
            if msk.any():
                curr_ar[:, msk] = ar_sum[:, msk] / ar_cnt[msk]

            ar_slice = curr_ar[:, t:t + n_input_bins]
            if ar_slice.shape[1] < n_input_bins:
                ar_slice = np.pad(ar_slice, ((0, 0), (0, n_input_bins - ar_slice.shape[1])))

            if init_state_flag and _init is not None:
                sfl = np.concatenate([_init, ar_slice], axis=1)
            else:
                sfl = ar_slice

            tt = xs.shape[1]
            yh = np.zeros((n_neurons, tt), dtype=np.float32)
            if tt > history:
                src = sfl[:, :tt - history]
                yh[:, history:history + src.shape[1]] = src
            xs = np.concatenate([xs.astype(np.float32), yh], axis=0)

            bx = torch.tensor(xs, dtype=torch.float32).unsqueeze(0).to(device)
            pr = model(bx).cpu().numpy()
            for o in range(n_output_bins):
                tb = t + output_offset + o
                if tb < tot_out:
                    p_sum[:, tb] += pr[0, :, o]
                    p_cnt[tb] += 1
                    ar_sum[:, tb] += np.exp(pr[0, :, o])
                    ar_cnt[tb] += 1

    msk = p_cnt > 0
    pa = np.zeros_like(p_sum)
    pa[:, msk] = p_sum[:, msk] / p_cnt[msk]
    return np.exp(pa)


# =====================
# Autoregressive Prediction
# =====================

def predict_autoregressive(model_tuple, loader, chunk_length=None, coarse_factor=0):
    """Generate autoregressive (AR) predictions for every sample in *loader*.

    Instead of ground-truth spike history the model receives its own
    previous predicted rates as the history input.  This lets you evaluate
    how well the model performs without access to observed neural activity.

    Parameters
    ----------
    model_tuple : tuple
        ``(model, cfg_or_criterion, device)`` as returned by
        ``open_model_and_data``.
    loader : DataLoader
        Test ``DataLoader`` whose underlying dataset is a
        ``BinnedStimSpikeDataset`` with ``history > 0``.
    chunk_length : int or None
        Maximum number of **output time-bins** the model predicts
        autoregressively before its history buffer is reset to ground-truth.
        
        * ``None`` (default) – fully autoregressive: the model never sees
          ground-truth history for the current trial (init-state from the
          previous trial is always ground-truth).
        * An integer *L* – every *L* output bins within a trial the AR
          history buffer is overwritten with the actual spike counts,
          simulating periodic closed-loop correction.
    coarse_factor : int
        If > 0 the returned arrays are temporally coarsened (averaged)
        by this factor before being returned.

    Returns
    -------
    all_pred : np.ndarray, shape ``(N_samples, n_neurons, n_output_bins')``
        Predicted firing rates (rate space, **not** log) per sample.
    all_true : np.ndarray, shape ``(N_samples, n_neurons, n_output_bins')``
        Ground-truth spike counts per sample (matching ``all_pred``).
    """
    model, _, device = model_tuple
    dataset = loader.dataset
    model.eval()

    history = dataset.history if dataset.history else 0
    if history == 0:
        raise ValueError("Autoregressive prediction requires a model with history > 0.")

    n_neurons     = dataset.n_neurons
    n_input_bins  = dataset.n_input_bins
    n_output_bins = dataset.n_output_bins
    output_offset = dataset.output_offset
    use_init_state        = dataset.init_state
    n_initial_state_bins  = getattr(dataset, 'n_initial_state_bins', 0)
    total_out_bins        = dataset.total_bins_output

    # ---- group samples by trial, sorted in time order ----
    trial_to_samples = defaultdict(list)
    for sample_idx, (timing_idx, t) in enumerate(dataset.samples):
        trial_to_samples[timing_idx].append((t, sample_idx))
    for timing_idx in trial_to_samples:
        trial_to_samples[timing_idx].sort()

    n_samples = len(dataset)
    all_pred = np.zeros((n_samples, n_neurons, n_output_bins), dtype=np.float32)
    all_true = np.zeros((n_samples, n_neurons, n_output_bins), dtype=np.float32)

    with torch.no_grad():
        for timing_idx, time_samples in tqdm(trial_to_samples.items(),
                                             desc="AR predict", leave=False):
            gt_spikes = dataset.spike_responses_binned[timing_idx]   # (neurons, total_out_bins)
            stim = dataset.pattern_stims[dataset.timing_to_pattern[timing_idx]]

            # ---- init-state context from previous trial (always GT) ----
            if use_init_state:
                spike_prepend = n_initial_state_bins + history
                prev_trial = dataset.spike_responses_binned.get(
                    timing_idx - 1,
                    np.zeros((n_neurons, total_out_bins), dtype=np.float32),
                )
                available = prev_trial.shape[1]
                if spike_prepend <= available:
                    init_ctx = prev_trial[:, -spike_prepend:]
                else:
                    init_ctx = np.zeros((n_neurons, spike_prepend), dtype=np.float32)
                    init_ctx[:, -available:] = prev_trial

                stim_padded = np.pad(stim, ((0, 0), (n_initial_state_bins, 0)),
                                     mode='constant')
                x_width      = n_initial_state_bins + n_input_bins
                spike_offset = history
            else:
                init_ctx     = None
                stim_padded  = stim
                x_width      = n_input_bins
                spike_offset = 0

            # ---- AR rate buffer (running sum & count per output bin) ----
            ar_sum   = np.zeros((n_neurons, total_out_bins), dtype=np.float32)
            ar_count = np.zeros(total_out_bins, dtype=np.float32)

            for t, sample_idx in time_samples:
                # -- chunk correction: reset AR buffer to GT --
                if chunk_length is not None:
                    t_out = t + output_offset
                    if t_out > 0 and t_out % chunk_length == 0:
                        ar_sum[:, :t_out]   = gt_spikes[:, :t_out].astype(np.float32)
                        ar_count[:t_out]    = 1.0

                # -- stim input (always GT) --
                x = stim_padded[:, t : t + x_width].copy().astype(np.float32)

                # -- build AR history channel --
                ar_avg = np.zeros((n_neurons, total_out_bins), dtype=np.float32)
                mask = ar_count > 0
                if mask.any():
                    ar_avg[:, mask] = ar_sum[:, mask] / ar_count[mask]

                if use_init_state:
                    spikes_for_hist = np.concatenate([init_ctx, ar_avg], axis=1)
                else:
                    spikes_for_hist = ar_avg

                # replicate __getitem__ lag logic
                n_time    = x_width
                lag_start = t + spike_offset - history
                y_history = np.zeros((n_neurons, n_time), dtype=np.float32)
                if lag_start >= 0:
                    y_history[:] = spikes_for_hist[:, lag_start : lag_start + n_time]
                else:
                    valid_from = -lag_start
                    avail = n_time - valid_from
                    if avail > 0:
                        y_history[:, valid_from:] = spikes_for_hist[:, 0 : avail]

                x_input = np.concatenate([x, y_history], axis=0)   # (channels, time)

                # -- forward pass --
                bx = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(device)
                pred_log = model(bx).cpu().numpy()[0]              # (neurons, n_output_bins)
                pred_rates = np.exp(pred_log)

                # -- store per-sample result --
                all_pred[sample_idx] = pred_rates

                if use_init_state:
                    out_start = t + output_offset + n_initial_state_bins + spike_offset
                    gt_full = np.concatenate([init_ctx, gt_spikes], axis=1)
                else:
                    out_start = t + output_offset
                    gt_full = gt_spikes
                all_true[sample_idx] = gt_full[:, out_start : out_start + n_output_bins]

                # -- update AR buffer with new predictions --
                for o in range(n_output_bins):
                    tb = t + output_offset + o
                    if 0 <= tb < total_out_bins:
                        ar_sum[:, tb]  += pred_rates[:, o]
                        ar_count[tb]   += 1

    # ---- optional coarsening ----
    if coarse_factor > 0:
        def _coarsen_3d(arr, factor):
            N, neurons, fine = arr.shape
            coarse = fine // factor
            return arr[:, :, :coarse * factor].reshape(N, neurons, coarse, factor).mean(axis=3)
        all_pred = _coarsen_3d(all_pred, coarse_factor)
        all_true = _coarsen_3d(all_true, coarse_factor)

    return all_pred, all_true


# =====================
# Perturbation Analysis
# =====================

def perturbation_analysis(model_tuple, loader, n_stim_channels,
                          n_shuffles=10, coarse_factor=0, seed=42):
    """Perturbation analysis for a stimulus-to-spike model.

    Evaluates how model predictions degrade under input perturbations.
    Automatically detects whether the model uses spike-history channels
    (i.e. input has more than ``n_stim_channels`` channels).

    Perturbations applied:

    1. **stim_shuffle** – stimulus channels are randomly permuted (i.e. which
       channel carries which pattern is shuffled), preserving the temporal
       structure and the number of simultaneously active channels at each
       timestep.

    *History-model only (skipped when input has no history channels):*

    2. **history_mean** – every history channel is replaced by that
       neuron's mean firing rate averaged over all test samples and
       time bins. 
    3. **history_shuffle** – for each sample the history channels are
       replaced with those from a randomly chosen *different* trial of
       the same stimulation pattern (at the same time offset), testing
       whether predictions depend on the specific pre-stimulation
       activity or only on the stimulus itself.

    Parameters
    ----------
    model_tuple : tuple  ``(model, _, device)``
    loader : DataLoader  test loader.
    n_stim_channels : int
        Number of stimulus channels in the input tensor (remaining
        channels are assumed to be spike-history).
    n_shuffles : int
        How many independent random shuffles for stochastic perturbations.
    coarse_factor : int
        Temporal coarsening factor applied before computing FEV (0 = none).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict  with keys ``'original'``, ``'stim_shuffle'``, and (when history
    channels are present) ``'history_mean'``, ``'history_shuffle'``.
    Each value is a dict:

    * ``fve_neurons`` – per-neuron FEV array.  Shape ``(n_neurons,)`` for
      deterministic conditions; ``(n_shuffles, n_neurons)`` for stochastic.
    * ``fve_mean`` – scalar (deterministic) or ``(n_shuffles,)`` array.
    """
    from metrics import fraction_variance_explained, _coarsen

    model, _, device = model_tuple
    model.eval()
    rng = np.random.RandomState(seed)

    n_neurons = loader.dataset.n_neurons

    # Detect whether the model has history channels
    sample_x = next(iter(loader))[0]
    total_channels = sample_x.shape[1]
    has_history = total_channels > n_stim_channels

    # ---- 1. Original predictions + collect per-neuron mean history ----
    all_pred_orig, all_true = [], []
    if has_history:
        hist_sum = np.zeros(n_neurons, dtype=np.float64)
        hist_count = 0
        all_hist_channels = []  # for history-swap perturbation

    with torch.no_grad():
        for bx, by in tqdm(loader, desc="Original pass", leave=False):
            pred = torch.exp(model(bx.to(device))).cpu().numpy()
            all_pred_orig.append(pred)
            all_true.append(by.numpy())

            if has_history:
                hist = bx[:, n_stim_channels:, :].numpy()   # (B, n_neurons, T)
                hist_sum += hist.sum(axis=(0, 2))
                hist_count += hist.shape[0] * hist.shape[2]
                all_hist_channels.append(bx[:, n_stim_channels:].clone())

    all_pred_orig = np.concatenate(all_pred_orig, axis=0)
    all_true = np.concatenate(all_true, axis=0)

    if has_history:
        all_hist_channels = torch.cat(all_hist_channels, dim=0)  # (N, n_neurons, T)
        # Map (pattern_name, time_offset) → list of sample indices
        dataset = loader.dataset
        pattern_t_to_indices = defaultdict(list)
        for sample_idx, (timing_idx, t) in enumerate(dataset.samples):
            pname = dataset.timing_to_pattern[timing_idx]
            pattern_t_to_indices[(pname, t)].append(sample_idx)

    # ---- 2. History-mean perturbation (deterministic, history only) ----
    if has_history:
        mean_hist_per_neuron = hist_sum / hist_count          # (n_neurons,)
        all_pred_hmean = []
        with torch.no_grad():
            for bx, by in tqdm(loader, desc="History-mean pass", leave=False):
                bx_p = bx.clone()
                for n in range(n_neurons):
                    bx_p[:, n_stim_channels + n, :] = mean_hist_per_neuron[n]
                pred = torch.exp(model(bx_p.to(device))).cpu().numpy()
                all_pred_hmean.append(pred)
        all_pred_hmean = np.concatenate(all_pred_hmean, axis=0)

    # ---- 3. Stochastic perturbations ----
    stim_fve_n     = np.zeros((n_shuffles, n_neurons))
    stim_fve_m     = np.zeros(n_shuffles)
    if has_history:
        hshuffle_fve_n = np.zeros((n_shuffles, n_neurons))
        hshuffle_fve_m = np.zeros(n_shuffles)

    for s in tqdm(range(n_shuffles), desc="Shuffle iterations"):
        pred_stim_all = []
        if has_history:
            pred_hshuffle_all = []
            batch_cursor = 0
        with torch.no_grad():
            for bx, by in loader:
                B, C, T = bx.shape

                # --- stim shuffle: permute channels, preserve timepoints ---
                # Shuffling channels (not time) preserves how many channels
                # are stimulated at each timestep while breaking channel identity.
                bx_stim = bx.clone()
                for b in range(B):
                    perm = torch.from_numpy(rng.permutation(n_stim_channels).copy())
                    bx_stim[b, :n_stim_channels] = bx_stim[b, :n_stim_channels][perm]
                pred_stim_all.append(
                    torch.exp(model(bx_stim.to(device))).cpu().numpy())

                if has_history:
                    # --- history swap: use history from a different trial
                    # of the same pattern at the same time offset ---
                    bx_hs = bx.clone()
                    for b in range(B):
                        global_idx = batch_cursor + b
                        timing_idx, t = dataset.samples[global_idx]
                        pname = dataset.timing_to_pattern[timing_idx]
                        candidates = [
                            si for si in pattern_t_to_indices[(pname, t)]
                            if dataset.samples[si][0] != timing_idx
                        ]
                        if candidates:
                            other_idx = rng.choice(candidates)
                            bx_hs[b, n_stim_channels:] = all_hist_channels[other_idx]
                    pred_hshuffle_all.append(
                        torch.exp(model(bx_hs.to(device))).cpu().numpy())
                    batch_cursor += B

        pred_stim_all = np.concatenate(pred_stim_all, axis=0)

        if coarse_factor > 0:
            sc = _coarsen(pred_stim_all, coarse_factor)
            tc = _coarsen(all_true, coarse_factor)
        else:
            sc, tc = pred_stim_all, all_true

        stim_fve_n[s], stim_fve_m[s] = fraction_variance_explained(
            tc, sc, global_variance=True)

        if has_history:
            pred_hshuffle_all = np.concatenate(pred_hshuffle_all, axis=0)
            if coarse_factor > 0:
                hc = _coarsen(pred_hshuffle_all, coarse_factor)
            else:
                hc = pred_hshuffle_all
            hshuffle_fve_n[s], hshuffle_fve_m[s] = fraction_variance_explained(
                tc, hc, global_variance=True)

    # ---- FEV for original ----
    if coarse_factor > 0:
        oc = _coarsen(all_pred_orig, coarse_factor)
        tc = _coarsen(all_true, coarse_factor)
    else:
        oc, tc = all_pred_orig, all_true

    orig_fve_n, orig_fve_m = fraction_variance_explained(tc, oc, global_variance=True)

    out = {
        'original':     {'fve_neurons': orig_fve_n,  'fve_mean': orig_fve_m},
        'stim_shuffle': {'fve_neurons': stim_fve_n,  'fve_mean': stim_fve_m},
    }

    if has_history:
        if coarse_factor > 0:
            hmc = _coarsen(all_pred_hmean, coarse_factor)
        else:
            hmc = all_pred_hmean
        hmean_fve_n, hmean_fve_m = fraction_variance_explained(tc, hmc, global_variance=True)
        out['history_mean']    = {'fve_neurons': hmean_fve_n,    'fve_mean': hmean_fve_m}
        out['history_shuffle'] = {'fve_neurons': hshuffle_fve_n, 'fve_mean': hshuffle_fve_m}

    return out




# =====================
# Training and Validation Functions
# =====================

def train_epoch(model, loader, criterion, optimizer, device, grad_clip=True, max_norm=1.0, sum_loss=False, weight_loss=1, use_init_state=False, is_cache_model=False, cache_mode='teacher_forcing', scaler=None, use_amp=False):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: torch device
        grad_clip: Whether to clip gradients
        max_norm: Maximum gradient norm for clipping
        use_init_state: Whether the dataset returns initial state (for RNN models)
        is_cache_model: Whether the model is a HistoryCacheCausalCNN
        cache_mode: Forward mode for cache model ('teacher_forcing', 'semi_ar', 'ar')
        scaler: GradScaler instance for AMP (created automatically if use_amp=True and scaler is None)
        use_amp: Whether to use automatic mixed precision
    
    Returns:
        Average loss over the epoch
    """
    if use_amp and scaler is None:
        scaler = GradScaler()
    
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        if use_init_state:
            batch_x, batch_y, batch_init = batch
            batch_x, batch_y, batch_init = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True), batch_init.to(device, non_blocking=True)
        else:
            batch_x, batch_y = batch
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            batch_init = None
        
        optimizer.zero_grad()
        
        with autocast(device.type, enabled=use_amp):
            # Forward pass - handle models with initial_spikes argument
            if is_cache_model:
                predictions = model(batch_x, batch_y, mode=cache_mode)
            elif use_init_state and hasattr(model, 'forward') and 'initial_spikes' in model.forward.__code__.co_varnames:
                predictions = model(batch_x, initial_spikes=batch_init)
            else:
                predictions = model(batch_x)
            
            if sum_loss:
                    # 1. Convert log-rates to rates (counts)
                    rates = torch.exp(predictions)
                    # 2. Sum the rates to get total predicted count
                    summed_rates = rates.sum(dim=-1)
                    # 3. Convert back to log-space for PoissonNLLLoss(log_input=True) w a tiny eps
                    predictions = torch.log(summed_rates + 1e-8)
                    batch_y = batch_y.sum(dim=-1)
            weights = torch.ones_like(batch_y)
            weights[batch_y > 0] = weight_loss # weight nonzeros more
            loss = (weights * criterion(predictions, batch_y)).mean()
        
        # Backward pass with AMP scaling
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device, sum_loss=False, weight_loss=1, use_init_state=False, is_cache_model=False, cache_mode='teacher_forcing', use_amp=False):
    """
    Validate the model on a dataset.
    
    Args:
        model: PyTorch model
        loader: DataLoader for validation/test data
        criterion: Loss function
        device: torch device
        sum_loss: Whether to sum the loss over output bins
        weight_loss: Weight for non-zero samples in loss
        use_init_state: Whether the dataset returns initial state (for RNN models)
        is_cache_model: Whether the model is a HistoryCacheCausalCNN
        cache_mode: Forward mode for cache model ('teacher_forcing', 'semi_ar', 'ar')
        use_amp: Whether to use automatic mixed precision
    
    Returns:
        Average loss over the dataset
    """
    total_corr = 0
    
    model.eval()
    total_loss = 0
    pbar = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for batch in pbar:
            if use_init_state:
                batch_x, batch_y, batch_init = batch
                batch_x, batch_y, batch_init = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True), batch_init.to(device, non_blocking=True)
            else:
                batch_x, batch_y = batch
                batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
                batch_init = None
            
            with autocast(device.type, enabled=use_amp):
                # Forward pass - handle models with initial_spikes argument
                if is_cache_model:
                    predictions = model(batch_x, batch_y, mode=cache_mode)
                elif use_init_state and hasattr(model, 'forward') and 'initial_spikes' in model.forward.__code__.co_varnames:
                    predictions = model(batch_x, initial_spikes=batch_init)
                else:
                    predictions = model(batch_x)
                
                if sum_loss:
                    rates = torch.exp(predictions)
                    summed_rates = rates.sum(dim=-1)
                    predictions = torch.log(summed_rates + 1e-8)
                    batch_y = batch_y.sum(dim=-1)
                weights = torch.ones_like(batch_y)
                weights[batch_y > 0] = weight_loss # weight nonzeros more
                loss = (weights * criterion(predictions, batch_y)).mean()

            # Calculate correlation for this batch
            batch_corr = compute_correlation(predictions, batch_y)
            total_corr += batch_corr * batch_x.size(0)
            total_loss += loss.item() * batch_x.size(0)
            pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(loader.dataset), total_corr / len(loader.dataset)





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
import warnings
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
                                 T: int,
                                 initial_cache: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        initial_cache : (B, n_neurons, cache_size) or None
            Spike counts to seed the cache with (e.g. the previous trial's
            last ``cache_size`` bins).  When ``None`` the cache is left-padded
            with zeros (the default cold start).

        Returns
        -------
        cache_channels : (B, n_neurons * cache_embed_dim, T)
        """
        B = spike_context.shape[0]
        device = spike_context.device

        # Pad left so indexing never goes negative.  Either seed with the
        # previous trial's tail (initial_cache) or with zeros.
        if initial_cache is not None:
            padded = torch.cat([initial_cache.to(spike_context.dtype), spike_context], dim=2)
        else:
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
                mode: str = 'teacher_forcing',
                initial_cache: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        stim_input : (B, n_stim_channels, T_in)
            Stimulus-only input (may include init-state padding bins on
            the left when ``use_init_state=True``).
        initial_cache : (B, n_neurons, cache_size) or None
            Optional spike counts to seed the spike-history cache at t=0
            (e.g. the previous trial's last ``cache_size`` bins).  When
            ``None`` the cache starts cold (all zeros).  Applies to all modes.
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
            return self._forward_teacher_forcing(stim_input, spike_context, initial_cache)
        elif mode in ('semi_ar', 'ar'):
            return self._forward_sequential(stim_input, spike_context, mode, initial_cache)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'teacher_forcing', 'semi_ar', or 'ar'.")

    # ------------------------------------------------------------------
    def _forward_teacher_forcing(self, stim_input: torch.Tensor,
                                  spike_context: torch.Tensor,
                                  initial_cache: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Parallel teacher-forced forward pass."""
        B, _, T_in = stim_input.shape
        cache_ch = self._build_cache_channels_tf(spike_context, T_in, initial_cache)
        x = torch.cat([stim_input, cache_ch], dim=1)   # (B, in_ch, T_in)
        features = self.conv_stack(x)                   # (B, conv_ch, T_out)
        features = features.transpose(1, 2)             # (B, T_out, conv_ch)
        y = self.fc(features)                           # (B, T_out, n_neurons)
        return y.transpose(1, 2)                        # (B, n_neurons, T_out)

    # ------------------------------------------------------------------
    def _forward_sequential(self, stim_input: torch.Tensor,
                             spike_context: Optional[torch.Tensor],
                             mode: str,
                             initial_cache: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        # Cache: (B, n_neurons, cache_size) — seeded from the previous trial's
        # tail when initial_cache is given, otherwise a cold (zeros) start.
        if initial_cache is not None:
            cache = initial_cache.to(device=device, dtype=stim_input.dtype).clone()
        else:
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
                sampled_spikes = torch.poisson(pred_rate)  # (B, n_neurons)
                cache = torch.cat([cache[:, :, 1:],
                                   sampled_spikes.unsqueeze(-1)], dim=-1)

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

    # Mirror BinnedStimSpikeDataset.__getitem__ alignment exactly so that
    # prediction indexing matches training indexing.
    history_val = history if (history is not None and history > 0) else 0
    if init_state_flag:
        spike_prepend = n_init + history_val
        prev = timing_idx - 1
        prev_trial = ds.spike_responses_binned.get(
            prev, np.zeros((n_neurons, tot_out), dtype=np.float32)
        )
        available = prev_trial.shape[1]
        if spike_prepend <= available:
            prev_ctx = prev_trial[:, -spike_prepend:]
        else:
            prev_ctx = np.zeros((n_neurons, spike_prepend), dtype=np.float32)
            if available > 0:
                prev_ctx[:, -available:] = prev_trial

        stim_full    = np.pad(stim_binned, ((0, 0), (n_init, 0)), mode='constant')
        spikes_full  = np.concatenate([prev_ctx, spikes_binned], axis=1)
        x_width      = n_init + n_input_bins
        spike_offset = history_val
    else:
        stim_full    = stim_binned
        spikes_full  = spikes_binned
        x_width      = n_input_bins
        spike_offset = 0

    model.eval()
    with torch.no_grad():
        for t in range(max_s + 1):
            se = t + x_width
            actual_bins = stim_full.shape[1]
            if se <= actual_bins:
                xs = stim_full[:, t:se].copy().astype(np.float32)
            else:
                xs = np.zeros((stim_full.shape[0], x_width), dtype=np.float32)
                av = max(0, actual_bins - t)
                if av > 0:
                    xs[:, :av] = stim_full[:, t:t + av]

            if history_val > 0:
                n_time    = x_width
                lag_start = t + spike_offset - history_val
                yh = np.zeros((n_neurons, n_time), dtype=np.float32)
                if lag_start >= 0:
                    end = min(lag_start + n_time, spikes_full.shape[1])
                    yh[:, :end - lag_start] = spikes_full[:, lag_start:end]
                else:
                    valid_from = -lag_start
                    avail      = min(n_time - valid_from, spikes_full.shape[1])
                    if avail > 0:
                        yh[:, valid_from:valid_from + avail] = spikes_full[:, 0:avail]
                xs = np.concatenate([xs, yh], axis=0)

            bx = torch.tensor(xs, dtype=torch.float32).unsqueeze(0).to(device)
            pr = model(bx).cpu().numpy()
            for o in range(n_output_bins):
                tb = t + output_offset + o
                if 0 <= tb < tot_out:
                    p_sum[:, tb] += pr[0, :, o]
                    p_cnt[tb] += 1

    msk = p_cnt > 0
    pa = np.zeros_like(p_sum)
    pa[:, msk] = p_sum[:, msk] / p_cnt[msk]
    return np.exp(pa)


def predict_trial_ar_binwise(model_tuple, cfg, test_loader, timing_idx,
                             stim_binned=None, chunk_length=None,
                             sample=False, seed=None, max_rate=2.0):
    """True **bin-by-bin** autoregressive rollout for a single-pass model.

    Use this when the model predicts the whole trial in one forward pass
    (``n_input_bins == n_output_bins``, ``output_offset == 0``).  The model
    is re-run once per output bin; after each step the freshly predicted
    bin (optionally Poisson-sampled to an integer count) is written into
    the spike buffer so it becomes the ``history`` input for the next bin,
    exactly mirroring the dataset's training-time layout
    (``BinnedStimSpikeDataset.__getitem__``).

    Parameters
    ----------
    chunk_length : int or None
        If given, every *chunk_length* output bins the committed AR buffer
        is overwritten with ground-truth spikes (periodic correction).
    sample : bool
        If True, each committed bin is a Poisson **sample** drawn from the
        predicted rate (discrete integer counts, matching the training-time
        history distribution).  If False, the continuous rate is fed back.
    seed : int or None
        Seed for the Poisson RNG; a fixed seed makes the rollout
        reproducible.  Ignored when ``sample=False``.
    max_rate : float
        Spikes/bin cap (default 2).  The predicted rate is clipped to
        ``[0, max_rate]`` (NaN/inf sanitised) and, when sampling, the
        Poisson draw is also clamped to ``<= max_rate`` before being fed
        back as history, so a diverging AR rollout cannot blow up.

    Returns
    -------
    np.ndarray, shape ``(n_neurons, tot_out)``
        Predicted firing rates (rate space, not log).
    """
    model, _, device = model_tuple
    ds = test_loader.dataset

    n_neurons       = ds.n_neurons
    n_input_bins    = cfg['n_input_bins']
    n_output_bins   = cfg['n_output_bins']
    output_offset   = cfg.get('output_offset', 0)
    history         = cfg.get('history', 0)
    init_state_flag = cfg.get('init_state', False)
    n_init          = getattr(ds, 'n_initial_state_bins', 0)

    if history is None or history == 0:
        raise ValueError("AR prediction requires history > 0.")
    if not (n_input_bins == n_output_bins and output_offset == 0):
        raise ValueError(
            "predict_trial_ar_binwise requires a single-pass model "
            f"(n_input_bins == n_output_bins, output_offset == 0); got "
            f"n_input_bins={n_input_bins}, n_output_bins={n_output_bins}, "
            f"output_offset={output_offset}.")

    rng = np.random.default_rng(seed) if sample else None

    if stim_binned is None:
        stim_binned = ds.pattern_stims[ds.timing_to_pattern[timing_idx]]
    gt_spikes = ds.spike_responses_binned[timing_idx].astype(np.float32)
    tot_out   = gt_spikes.shape[1]

    # --- stim input (always ground truth), padded for valid-conv context ---
    if init_state_flag and n_init > 0:
        stim_full = np.pad(stim_binned, ((0, 0), (n_init, 0)), mode='constant')
        spike_prepend = n_init + history
        prev = ds.spike_responses_binned.get(
            timing_idx - 1, np.zeros((n_neurons, spike_prepend), dtype=np.float32))
        if prev.shape[1] >= spike_prepend:
            prev_ctx = prev[:, -spike_prepend:].astype(np.float32)
        else:
            prev_ctx = np.zeros((n_neurons, spike_prepend), dtype=np.float32)
            prev_ctx[:, -prev.shape[1]:] = prev
        x_width      = n_init + n_input_bins
        spike_offset = history
    else:
        stim_full    = stim_binned
        prev_ctx     = np.zeros((n_neurons, history), dtype=np.float32)
        x_width      = n_input_bins
        spike_offset = 0

    x = stim_full[:, 0:x_width].astype(np.float32)            # (n_chan, x_width)

    # spikes_full = [prev-trial context | running AR buffer for this trial]
    ar_buf      = np.zeros((n_neurons, tot_out), dtype=np.float32)
    spikes_full = np.concatenate([prev_ctx, ar_buf], axis=1)
    lag_start   = spike_offset - history                       # = 0 for these models

    model.eval()
    with torch.no_grad():
        for k in range(tot_out):
            # periodic ground-truth correction of already-committed bins
            if chunk_length is not None and k > 0 and k % chunk_length == 0:
                spikes_full[:, prev_ctx.shape[1]:prev_ctx.shape[1] + k] = \
                    gt_spikes[:, :k]

            # history channel aligned exactly as in __getitem__
            y_hist = np.zeros((n_neurons, x_width), dtype=np.float32)
            if lag_start >= 0:
                y_hist[:] = spikes_full[:, lag_start:lag_start + x_width]
            else:
                vf = -lag_start
                y_hist[:, vf:] = spikes_full[:, 0:x_width - vf]

            xs = np.concatenate([x, y_hist], axis=0)
            bx = torch.tensor(xs, dtype=torch.float32).unsqueeze(0).to(device)
            pr = model(bx).cpu().numpy()[0]                    # (n_neurons, tot_out)

            # clip to <= max_rate spikes/bin: a diverging AR rollout can blow
            # exp(pr) up unboundedly and feed the blow-up back as history.
            rate_k = np.exp(pr[:, k])
            rate_k = np.clip(np.nan_to_num(rate_k, nan=0.0, posinf=max_rate),
                             0.0, max_rate)
            if sample:
                fed = np.minimum(rng.poisson(rate_k), max_rate).astype(np.float32)
            else:
                fed = rate_k
            ar_buf[:, k] = fed
            spikes_full[:, prev_ctx.shape[1] + k] = ar_buf[:, k] # Rewrite spikes full input for next step
            # Last step writes the final predicted bin into spikes_full,

    # final pass already gives full-trial rates; recompute clean rate output
    with torch.no_grad():
        y_hist = np.zeros((n_neurons, x_width), dtype=np.float32)
        if lag_start >= 0:
            y_hist[:] = spikes_full[:, lag_start:lag_start + x_width]
        else:
            vf = -lag_start
            y_hist[:, vf:] = spikes_full[:, 0:x_width - vf]
        xs = np.concatenate([x, y_hist], axis=0)
        bx = torch.tensor(xs, dtype=torch.float32).unsqueeze(0).to(device)
        pr = model(bx).cpu().numpy()[0]
    out = np.clip(np.nan_to_num(np.exp(pr), nan=0.0, posinf=max_rate),
                  0.0, max_rate)
    return out.astype(np.float32)


def sliding_window_predict_trial_ar(model_tuple, cfg, test_loader, timing_idx,
                                     stim_binned=None, chunk_length=None,
                                     sample=False, seed=None, n_sampling_trials=None):
    """Single-trial **autoregressive** sliding-window prediction.

    Same interface as :func:`sliding_window_predict_trial` but the model's
    own predicted output is fed back as spike-history instead of ground-truth.
    The init-state context from the previous trial is always ground-truth.

    Parameters
    ----------
    chunk_length : int or None
        If given, every *chunk_length* output bins the AR history buffer is
        replaced by ground-truth spikes (periodic correction).
    sample : bool
        If True, the fed-back history is a Poisson **sample** drawn from the
        predicted rate (discrete counts, matching the training-time history
        distribution) instead of the continuous rate lambda itself. The
        returned array is still the (sampled) spike-count rollout.
    seed : int or None
        Seed for the Poisson sampling RNG. With ``sample=True`` a fixed seed
        makes the predicted spike train fully reproducible for a given input.
        Ignored when ``sample=False``.
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

    rng = np.random.default_rng(seed) if sample else None
    if sample and n_output_bins > 1:
        warnings.warn(
            f"sample=True with n_output_bins={n_output_bins}: overlapping "
            "windows average multiple Poisson draws per bin, so the fed-back "
            "history is not strictly integer. Use n_output_bins=1 for a "
            "purely discrete AR rollout.",
            stacklevel=2,
        )

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
                    rate = np.exp(pr[0, :, o])
                    fed = rng.poisson(rate).astype(np.float32) if sample else rate
                    ar_sum[:, tb] += fed
                    ar_cnt[tb] += 1

    msk = p_cnt > 0
    pa = np.zeros_like(p_sum)
    pa[:, msk] = p_sum[:, msk] / p_cnt[msk]
    return np.exp(pa)


def sliding_window_predict_trial_ar_sampled(model_tuple, cfg, test_loader,
                                             timing_idx, stim_binned=None,
                                             chunk_length=None, n_samples=20,
                                             seed=0, return_samples=False):
    """Average of :func:`predict_trial_ar_binwise`.

    Runs the bin-by-bin autoregressive rollout *n_samples* times, each with
    discrete Poisson-sampled spike history (so the fed-back history matches
    the discrete training-time distribution), and averages the per-sample
    predicted-rate trajectories into an empirical expectation.

    Each sample uses a distinct, deterministic seed derived from *seed*
    (``seed + i``), so the whole estimate is reproducible while individual
    samples explore different stochastic rollouts.

    Parameters
    ----------
    n_samples : int
        Number of independent sampled rollouts to average.
    seed : int
        Base seed; sample ``i`` uses ``seed + i``.
    return_samples : bool
        If True, also return the stacked per-sample predictions
        ``(n_samples, n_neurons, tot_out)``.

    Returns
    -------
    np.ndarray
        Mean predicted-rate trajectory ``(n_neurons, tot_out)``, or
        ``(mean, samples)`` if ``return_samples`` is True.
    """
    samples = []
    for i in range(n_samples):
        pred = predict_trial_ar_binwise(
            model_tuple, cfg, test_loader, timing_idx,
            stim_binned=stim_binned, chunk_length=chunk_length,
            sample=True, seed=seed + i,
        )
        samples.append(pred)
    samples = np.stack(samples, axis=0)
    mean_pred = samples.mean(axis=0)
    if return_samples:
        return mean_pred, samples
    return mean_pred


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



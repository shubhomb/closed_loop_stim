class CausalSpikeCNN(nn.Module):
    """
    Causal 1-D CNN model that predicts spike counts autoregressively.
    
    For each output bin, the model only sees input bins up to and including
    the corresponding time window (right-aligned with left-padding).
    
    Unlike regular CNN, this model applies causal masking internally:
    - Input: (batch, n_channels, n_input_bins) - same as regular CNN
    - For output bin o, only input bins [0, (o+1)*input_bins_per_output) are visible
    - The visible bins are right-aligned with left-padding using pad_value
    
    Each output bin is processed by the same shared CNN, producing:
        (batch, n_neurons, n_output_bins)
    """
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
                 init_bias: Optional[float] = None,
                 num_stim_levels: int = 5,
                 use_batch_norm: bool = True,
                 pooling: str = 'adaptive_avg',
                 pad_value: int = 0):
        """
        Args:
            n_stim_channels: Number of stimulation channels
            n_neurons: Number of neurons to predict
            n_input_bins: Total number of input time bins (across all output bins)
            n_output_bins: Number of output time bins to predict
            embedding_dim: Dimension of stim category embeddings
            conv_channels: List of output channels for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            fc_dims: List of fully connected layer dimensions after conv
            dropout: Dropout rate
            init_bias: Initial bias for output layer (mean spike rate)
            num_stim_levels: Number of stim categories
            use_batch_norm: Whether to use batch normalization
            pooling: Pooling type - 'adaptive_avg', 'adaptive_max', or 'flatten'
            pad_value: Value to use for padding in causal masking (default 4 = NO_STIM_INDEX)
        """
        super().__init__()
        
        self.n_stim_channels = n_stim_channels
        self.n_neurons = n_neurons
        self.n_input_bins = n_input_bins
        self.n_output_bins = n_output_bins
        self.embedding_dim = max(embedding_dim, 1)
        self.pooling = pooling
        self.pad_value = pad_value
        
        # Compute how many input bins correspond to one output bin
        # This is used for causal masking
        assert n_input_bins % n_output_bins == 0, \
            f"n_input_bins ({n_input_bins}) must be divisible by n_output_bins ({n_output_bins})"
        self.input_bins_per_output = n_input_bins // n_output_bins
        
        assert len(conv_channels) == len(kernel_sizes), \
            f"conv_channels and kernel_sizes must have same length"
        
        if self.embedding_dim <= 1:
            self.embedding = nn.Identity()
            in_channels = 1
        else:
            self.embedding = nn.Embedding(num_embeddings=num_stim_levels, embedding_dim=embedding_dim)
            in_channels = self.embedding_dim

        # Input sequence length after embedding
        seq_len = n_stim_channels * n_input_bins
        
        # Build 1D CNN layers (shared across all output bins)
        conv_layers = []
        current_seq_len = seq_len
        
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            padding = kernel_size // 2
            conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
            )
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        
        self.conv = nn.Sequential(*conv_layers)
        

        if pooling == 'flatten' or pooling == 'none' or pooling is None:
            self.pool = nn.Flatten()
            fc_input_dim = conv_channels[-1] * current_seq_len
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")
        
        # Fully connected layers (outputs single neuron predictions per output bin)
        fc_layers = []
        prev_dim = fc_input_dim
        for fc_dim in fc_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = fc_dim
        
        # Output layer: predicts n_neurons for ONE output bin
        fc_layers.append(nn.Linear(prev_dim, n_neurons))
        
        self.fc = nn.Sequential(*fc_layers)
        ()
    
    def _create_causal_input(self, x: torch.Tensor, output_bin: int) -> torch.Tensor:
        """
        Create causally-masked input for a specific output bin.
        
        For output bin o, only input bins [0, (o+1)*input_bins_per_output) are visible.
        The visible bins are right-aligned; left side is padded with pad_value.
        
        Args:
            x: (batch, n_channels, n_input_bins) - full input
            output_bin: which output bin (0-indexed)
        
        Returns:
            x_causal: (batch, n_channels, n_input_bins) - causally masked input
        """
        batch_size, n_channels, n_input_bins = x.shape
        
        # Number of input bins visible for this output bin
        n_visible = (output_bin + 1) * self.input_bins_per_output
        
        # Create padded tensor filled with pad_value
        x_causal = torch.full_like(x, self.pad_value)
        
        # Right-align: place visible bins at the end
        x_causal[:, :, -n_visible:] = x[:, :, :n_visible]
        
        return x_causal
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, n_input_bins) - LongTensor with category indices
               Same format as regular CNN - causal masking is applied internally
        Returns:
            y: (batch, n_neurons, n_output_bins) - predicted spike counts (log scale for Poisson)
        """
        batch_size, n_channels, n_input_bins = x.shape
        
        # Process each output bin with the shared CNN
        outputs = []
        for o in range(self.n_output_bins):
            # Apply causal masking for this output bin
            x_o = self._create_causal_input(x, o)  # (batch, n_channels, n_input_bins)
            
            # Apply embedding
            x_o = self.embedding(x_o)  # (batch, n_channels, n_input_bins, embedding_dim)
            
            # Reshape for 1D conv
            x_o = x_o.reshape(batch_size, -1, self.embedding_dim)  # (batch, seq_len, embedding_dim)
            x_o = x_o.transpose(1, 2)  # (batch, embedding_dim, seq_len)
            
            # Apply convolutions
            x_o = self.conv(x_o)  # (batch, conv_channels[-1], seq_len)
            
            # Pooling
            if self.pooling in ['adaptive_avg', 'adaptive_max']:
                x_o = self.pool(x_o).squeeze(-1)  # (batch, conv_channels[-1])
            else:
                x_o = self.pool(x_o)
            
            # FC layers -> (batch, n_neurons)
            x_o = self.fc(x_o)
            outputs.append(x_o)
        
        # Stack outputs: (batch, n_neurons, n_output_bins)
        y = torch.stack(outputs, dim=2)
        
        return y


class SpikeRNN(nn.Module):
    """
    GRU-based dynamical model for spike prediction.
    
    The model uses stimulation activity as input at each timestep.
    The initial hidden state is a linear projection from neurons x initial_state_bins.
    Output at each step is spikes per neuron (vector of n_neurons x 1 bin).
    
    Args:
        n_stim_channels: Number of stimulation channels
        n_neurons: Number of neurons to predict
        n_input_bins: Number of input time bins (stimulation)
        n_output_bins: Number of output time bins (predictions)
        latent_dim: Dimension of GRU hidden state
        n_initial_state_bins: Number of bins used to initialize hidden state (from spike history)
        embedding_dim: Dimension for stim level embedding (0 or 1 = no embedding)
        num_stim_levels: Number of discrete stimulation levels
        num_gru_layers: Number of stacked GRU layers
        dropout: Dropout rate
        fc_dims: List of fully connected layer dimensions for output head
        bidirectional: Whether to use bidirectional GRU (default False for causal)
    """
    
    def __init__(self,
                 n_stim_channels: int,
                 n_neurons: int,
                 n_input_bins: int = 60,
                 n_output_bins: int = 10,
                 latent_dim: int = 128,
                 n_initial_state_bins: int = 1,
                 embedding_dim: int = 8,
                 num_stim_levels: int = 5,
                 num_gru_layers: int = 1,
                 dropout: float = 0.2,
                 fc_dims: List[int] = [64],
                 bidirectional: bool = False):
        super().__init__()
        
        self.n_input_bins = n_input_bins
        self.n_output_bins = n_output_bins
        self.n_neurons = n_neurons
        self.n_stim_channels = n_stim_channels
        self.latent_dim = latent_dim
        self.n_initial_state_bins = n_initial_state_bins
        self.num_gru_layers = num_gru_layers
        self.bidirectional = bidirectional
        
        # Calculate how often we sample the output (input to output bin ratio)
        assert n_input_bins % n_output_bins == 0
        self.bins_per_output = n_input_bins // n_output_bins
        
        # 1. Embedding for stimulation levels
        self.embedding_dim = max(embedding_dim, 1)
        if self.embedding_dim <= 1:
            self.embedding = nn.Identity()
            gru_input_dim = n_stim_channels
        else:
            self.embedding = nn.Embedding(num_embeddings=num_stim_levels, embedding_dim=embedding_dim)
            gru_input_dim = n_stim_channels * self.embedding_dim
        
        # 2. Initial hidden state projection
        # Project from (n_neurons * n_initial_state_bins) -> (num_layers * num_directions, latent_dim)
        self.num_directions = 2 if bidirectional else 1
        initial_state_input_dim = n_neurons * n_initial_state_bins
        self.hidden_init_proj = nn.Linear(initial_state_input_dim, num_gru_layers * self.num_directions * latent_dim)
        
        # 3. GRU layer
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=latent_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        # 4. Output FC layers: from latent -> n_neurons
        fc_layers = []
        gru_output_dim = latent_dim * self.num_directions
        curr_dim = gru_output_dim
        
        for fc_dim in fc_dims:
            fc_layers.extend([
                nn.Linear(curr_dim, fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            curr_dim = fc_dim
        
        fc_layers.append(nn.Linear(curr_dim, n_neurons))
        self.fc = nn.Sequential(*fc_layers)
    
    def forward(self, x: torch.Tensor, initial_spikes: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Stimulation input of shape (batch, n_stim_channels, n_input_bins)
            initial_spikes: Optional initial spike state of shape (batch, n_neurons, n_initial_state_bins)
                            Used to compute initial hidden state. If None, uses zeros.
        
        Returns:
            Log spike rates of shape (batch, n_neurons, n_output_bins)
        """
        batch_size, n_channels, n_input_bins = x.shape
        
        # 1. Embed stimulation input & reshape
        # Input: (batch, n_channels, n_input_bins) with integer stim levels
        if self.embedding_dim > 1:
            # x: (batch, n_channels, n_input_bins) -> (batch, n_channels, n_input_bins, embedding_dim)
            x = self.embedding(x)
            # Reshape to (batch, n_input_bins, n_channels * embedding_dim)
            x = x.permute(0, 2, 1, 3).reshape(batch_size, n_input_bins, -1)
        else:
            # Just transpose to (batch, n_input_bins, n_channels)
            x = x.permute(0, 2, 1).float()
        
        # 2. Compute initial hidden state from initial spikes
        if initial_spikes is not None:
            # Flatten: (batch, n_neurons, n_initial_state_bins) -> (batch, n_neurons * n_initial_state_bins)
            h_input = initial_spikes.reshape(batch_size, -1)
        else:
            # Use zeros if no initial spikes provided
            h_input = torch.zeros(batch_size, self.n_neurons * self.n_initial_state_bins, 
                                  device=x.device, dtype=torch.float32)
        
        # Project to hidden state
        h_proj = self.hidden_init_proj(h_input)  # (batch, num_layers * num_directions * latent_dim)
        h0 = h_proj.view(self.num_gru_layers * self.num_directions, batch_size, self.latent_dim)
        
        # 3. Run GRU over all input timesteps
        # x: (batch, n_input_bins, gru_input_dim)
        # gru_out: (batch, n_input_bins, latent_dim * num_directions)
        gru_out, _ = self.gru(x, h0)
        
        # 4. Subsample to output resolution
        # Take every bins_per_output-th timestep, offset to get the END of each output bin
        # This ensures causality: output bin t only sees inputs up to and including bin t
        output_indices = torch.arange(self.bins_per_output - 1, n_input_bins, self.bins_per_output, device=x.device)
        gru_out_subsampled = gru_out[:, output_indices, :]  # (batch, n_output_bins, latent_dim * num_directions)
        
        # 5. FC to get per-neuron outputs
        # (batch, n_output_bins, latent_dim) -> (batch, n_output_bins, n_neurons)
        output = self.fc(gru_out_subsampled)
        
        # 6. Transpose to match expected output shape: (batch, n_neurons, n_output_bins)
        output = output.transpose(1, 2)
        
        return output

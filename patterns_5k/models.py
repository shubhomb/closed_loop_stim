"""
Neural stimulation to spike prediction models.

Models take categorical stimulation input and predict spike rates/counts.
Input: (batch, n_channels, n_input_bins) - categorical indices [0-4]
Output: (batch, n_neurons, n_output_bins) - predicted spike rates (log scale for Poisson)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from tqdm import tqdm
from utils import compute_correlation


class StimToSpikeMLP(nn.Module):
    """
    Simple feedforward model that predicts spike counts from stimulation bins.
    
    Architecture:
    1. Embedding layer: maps each stim category to a learned vector
    2. Flatten: (n_channels * n_input_bins * embedding_dim)
    3. MLP layers with ReLU activation
    4. Output layer: predicts spike counts for all neurons and output bins
    """
    def __init__(self, 
                 n_stim_channels: int,
                 n_neurons: int,
                 n_input_bins: int = 1,
                 n_output_bins: int = 1,
                 embedding_dim: int = 8,
                 hidden_dims: List[int] = [128],
                 dropout: float = 0.2,
                 init_bias: Optional[float] = None,
                 linear: bool = False,
                 num_stim_levels: int = 5):
        """
        Args:
            n_stim_channels: Number of stimulation channels
            n_neurons: Number of neurons to predict
            n_input_bins: Number of input time bins
            n_output_bins: Number of output time bins to predict
            embedding_dim: Dimension of stim category embeddings
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            init_bias: Initial bias for output layer (mean spike rate)
            num_stim_levels: Number of stim categories (e.g., 5: 4 delay modes + no-stim)
        """
        super().__init__()
        
        self.n_stim_channels = n_stim_channels
        self.n_neurons = n_neurons
        self.n_input_bins = n_input_bins
        self.n_output_bins = n_output_bins
        self.embedding_dim = embedding_dim
        self.initial_bias = torch.log(torch.tensor(init_bias) + 1e-6) if init_bias is not None else None
        
        # Embedding for categorical stim encoding
        self.embedding = nn.Embedding(num_embeddings=num_stim_levels, embedding_dim=embedding_dim)
        
        # Calculate input dimension after embedding and flattening
        input_dim = n_stim_channels * n_input_bins * embedding_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if not linear else nn.Identity(), # Fix: Just instantiate the layer
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim        
        # Output layer: predict spike counts for all neurons and output bins
        output_dim = n_neurons * n_output_bins
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        
        if self.initial_bias is not None:
            self._initialize_output_bias()

    def _initialize_output_bias(self):
        """Initialize output layer bias to match mean spike rate."""
        with torch.no_grad():
            self.mlp[-1].bias.fill_(self.initial_bias)
            # Initialize weights to be smaller to keep variance low initially
            self.mlp[-1].weight.normal_(0, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, n_input_bins) - LongTensor with category indices
        Returns:
            y: (batch, n_neurons, n_output_bins) - predicted spike counts (log scale for Poisson)
        """
        batch_size = x.shape[0]
        
        # Apply embedding: (batch, n_channels, n_input_bins) -> (batch, n_channels, n_input_bins, embedding_dim)
        x = self.embedding(x)
        
        # Flatten: (batch, n_channels * n_input_bins * embedding_dim)
        x = x.reshape(batch_size, -1)
        
        # MLP forward pass
        x = self.mlp(x)
        
        # Reshape output: (batch, n_neurons * n_output_bins) -> (batch, n_neurons, n_output_bins)
        x = x.reshape(batch_size, self.n_neurons, self.n_output_bins)
        
        return x


class StimToSpikeCNN(nn.Module):
    """
    1-D CNN model that predicts spike counts from stimulation bins.
    
    Architecture:
    1. Embedding layer: maps each stim category to a learned vector
    2. Reshape to (batch, embedding_dim, n_channels * n_input_bins) for 1D conv
    3. Stack of 1D conv layers with batch norm and ReLU
    4. Global pooling + MLP head
    5. Output layer: predicts spike counts for all neurons and output bins
    
    The CNN learns spatial patterns across channels and temporal patterns across bins.
    """
    def __init__(self,
                 n_stim_channels: int,
                 n_neurons: int,
                 n_input_bins: int = 1,
                 n_output_bins: int = 1,
                 embedding_dim: int = 8,
                 conv_channels: List[int] = [32, 64, 128],
                 kernel_sizes: List[int] = [3, 3, 3],
                 fc_dims: List[int] = [128],
                 dropout: float = 0.2,
                 init_bias: Optional[float] = None,
                 num_stim_levels: int = 5,
                 use_batch_norm: bool = True,
                 pooling: str = 'adaptive_avg'):
        """
        Args:
            n_stim_channels: Number of stimulation channels
            n_neurons: Number of neurons to predict
            n_input_bins: Number of input time bins
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
        """
        super().__init__()
        
        self.n_stim_channels = n_stim_channels
        self.n_neurons = n_neurons
        self.n_input_bins = n_input_bins
        self.n_output_bins = n_output_bins
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        self.initial_bias = torch.log(torch.tensor(init_bias) + 1e-6) if init_bias is not None else None
        
        # Validate inputs
        assert len(conv_channels) == len(kernel_sizes), \
            f"conv_channels and kernel_sizes must have same length, got {len(conv_channels)} and {len(kernel_sizes)}"
        
        # Embedding for categorical stim encoding
        self.embedding = nn.Embedding(num_embeddings=num_stim_levels, embedding_dim=embedding_dim)
        
        # Input sequence length after embedding
        seq_len = n_stim_channels * n_input_bins # will be length stim channels if 1 input bin
        
        # Build 1D CNN layers
        # Input: (batch, embedding_dim, seq_len)
        conv_layers = []
        in_channels = embedding_dim
        current_seq_len = seq_len
        
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            # Padding to maintain sequence length (same padding)
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
        
        # Pooling layer
        if pooling == 'adaptive_avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
            fc_input_dim = conv_channels[-1]
        elif pooling == 'adaptive_max':
            self.pool = nn.AdaptiveMaxPool1d(1)
            fc_input_dim = conv_channels[-1]
        elif pooling == 'flatten':
            self.pool = nn.Flatten()
            fc_input_dim = conv_channels[-1] * current_seq_len
        else:
            raise ValueError(f"Unknown pooling type: {pooling}")
        
        # Fully connected layers
        fc_layers = []
        prev_dim = fc_input_dim
        for fc_dim in fc_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = fc_dim
        
        # Output layer
        output_dim = n_neurons * n_output_bins
        fc_layers.append(nn.Linear(prev_dim, output_dim))
        
        self.fc = nn.Sequential(*fc_layers)
        
        if self.initial_bias is not None:
            self._initialize_output_bias()
    
    def _initialize_output_bias(self):
        """Initialize output layer bias to match mean spike rate."""
        with torch.no_grad():
            self.fc[-1].bias.fill_(self.initial_bias)
            self.fc[-1].weight.normal_(0, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, n_input_bins) - LongTensor with category indices
        Returns:
            y: (batch, n_neurons, n_output_bins) - predicted spike counts (log scale for Poisson)
        """
        batch_size = x.shape[0]
        
        # Apply embedding: (batch, n_channels, n_input_bins) -> (batch, n_channels, n_input_bins, embedding_dim)
        x = self.embedding(x)
        
        # Reshape for 1D conv: (batch, n_channels, n_input_bins, embedding_dim) 
        #                   -> (batch, n_channels * n_input_bins, embedding_dim)
        #                   -> (batch, embedding_dim, n_channels * n_input_bins)
        x = x.reshape(batch_size, -1, self.embedding_dim)  # (batch, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch, embedding_dim, seq_len)
        
        # Apply 1D convolutions
        x = self.conv(x)  # (batch, conv_channels[-1], seq_len)
        
        # Pooling
        if self.pooling in ['adaptive_avg', 'adaptive_max']:
            x = self.pool(x)  # (batch, conv_channels[-1], 1)
            x = x.squeeze(-1)  # (batch, conv_channels[-1])
        else:
            x = self.pool(x)  # (batch, conv_channels[-1] * seq_len)
        
        # FC layers
        x = self.fc(x)  # (batch, n_neurons * n_output_bins)
        
        # Reshape output
        x = x.reshape(batch_size, self.n_neurons, self.n_output_bins)
        
        return x


class StimToSpikeCNNTemporal(nn.Module):
    """
    1-D CNN model with separate processing for channels and time bins.
    
    This model first applies convolutions across channels within each time bin,
    then processes temporal patterns across bins. Better suited when n_input_bins > 1.
    
    Architecture:
    1. Embedding layer: maps each stim category to a learned vector
    2. Channel conv: 1D conv across channels (shared across time bins)
    3. Temporal conv: 1D conv across time bins
    4. FC head with output layer
    """
    def __init__(self,
                 n_stim_channels: int,
                 n_neurons: int,
                 n_input_bins: int = 1,
                 n_output_bins: int = 1,
                 embedding_dim: int = 8,
                 channel_conv_dims: List[int] = [32, 64],
                 temporal_conv_dims: List[int] = [64],
                 kernel_size_channel: int = 5,
                 kernel_size_temporal: int = 3,
                 fc_dims: List[int] = [128],
                 dropout: float = 0.2,
                 init_bias: Optional[float] = None,
                 num_stim_levels: int = 5):
        """
        Args:
            n_stim_channels: Number of stimulation channels
            n_neurons: Number of neurons to predict
            n_input_bins: Number of input time bins
            n_output_bins: Number of output time bins to predict
            embedding_dim: Dimension of stim category embeddings
            channel_conv_dims: Output dims for channel-wise convolutions
            temporal_conv_dims: Output dims for temporal convolutions
            kernel_size_channel: Kernel size for channel convolutions
            kernel_size_temporal: Kernel size for temporal convolutions
            fc_dims: FC layer dimensions
            dropout: Dropout rate
            init_bias: Initial bias for output layer
            num_stim_levels: Number of stim categories
        """
        super().__init__()
        
        self.n_stim_channels = n_stim_channels
        self.n_neurons = n_neurons
        self.n_input_bins = n_input_bins
        self.n_output_bins = n_output_bins
        self.embedding_dim = embedding_dim
        self.initial_bias = torch.log(torch.tensor(init_bias) + 1e-6) if init_bias is not None else None
        
        # Embedding
        self.embedding = nn.Embedding(num_embeddings=num_stim_levels, embedding_dim=embedding_dim)
        
        # Channel-wise 1D conv (applied across channels)
        channel_conv_layers = []
        in_ch = embedding_dim
        for out_ch in channel_conv_dims:
            channel_conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size_channel, padding=kernel_size_channel // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_ch = out_ch
        self.channel_conv = nn.Sequential(*channel_conv_layers)
        
        # Temporal 1D conv (applied across time bins)
        # After channel conv: (batch, channel_conv_dims[-1], n_channels)
        # We pool across channels then apply temporal conv
        self.channel_pool = nn.AdaptiveAvgPool1d(1)
        
        if n_input_bins > 1 and len(temporal_conv_dims) > 0:
            temporal_conv_layers = []
            in_ch = channel_conv_dims[-1]
            for out_ch in temporal_conv_dims:
                temporal_conv_layers.extend([
                    nn.Conv1d(in_ch, out_ch, kernel_size_temporal, padding=kernel_size_temporal // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                in_ch = out_ch
            self.temporal_conv = nn.Sequential(*temporal_conv_layers)
            self.temporal_pool = nn.AdaptiveAvgPool1d(1)
            fc_input_dim = temporal_conv_dims[-1]
        else:
            self.temporal_conv = None
            self.temporal_pool = None
            fc_input_dim = channel_conv_dims[-1]
        
        # FC layers
        fc_layers = []
        prev_dim = fc_input_dim
        for fc_dim in fc_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = fc_dim
        
        output_dim = n_neurons * n_output_bins
        fc_layers.append(nn.Linear(prev_dim, output_dim))
        self.fc = nn.Sequential(*fc_layers)
        
        if self.initial_bias is not None:
            self._initialize_output_bias()
    
    def _initialize_output_bias(self):
        """Initialize output layer bias to match mean spike rate."""
        with torch.no_grad():
            self.fc[-1].bias.fill_(self.initial_bias)
            self.fc[-1].weight.normal_(0, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, n_input_bins) - LongTensor with category indices
        Returns:
            y: (batch, n_neurons, n_output_bins) - predicted spike counts (log scale for Poisson)
        """
        batch_size = x.shape[0]
        
        # (batch, n_channels, n_input_bins) -> (batch, n_channels, n_input_bins, embedding_dim)
        x = self.embedding(x)
        
        # Process each time bin with channel conv, then stack for temporal conv
        if self.n_input_bins > 1 and self.temporal_conv is not None:
            # Process each time bin separately
            time_features = []
            for t in range(self.n_input_bins):
                # (batch, n_channels, embedding_dim) -> (batch, embedding_dim, n_channels)
                x_t = x[:, :, t, :].transpose(1, 2)
                # Apply channel conv
                x_t = self.channel_conv(x_t)  # (batch, channel_conv_dims[-1], n_channels)
                # Pool across channels
                x_t = self.channel_pool(x_t).squeeze(-1)  # (batch, channel_conv_dims[-1])
                time_features.append(x_t)
            
            # Stack time features: (batch, channel_conv_dims[-1], n_input_bins)
            x = torch.stack(time_features, dim=2)
            # Apply temporal conv
            x = self.temporal_conv(x)  # (batch, temporal_conv_dims[-1], n_input_bins)
            x = self.temporal_pool(x).squeeze(-1)  # (batch, temporal_conv_dims[-1])
        else:
            # Single time bin - just process channels
            # (batch, n_channels, 1, embedding_dim) -> (batch, n_channels, embedding_dim)
            x = x.squeeze(2)
            # (batch, n_channels, embedding_dim) -> (batch, embedding_dim, n_channels)
            x = x.transpose(1, 2)
            x = self.channel_conv(x)  # (batch, channel_conv_dims[-1], n_channels)
            x = self.channel_pool(x).squeeze(-1)  # (batch, channel_conv_dims[-1])
        
        # FC layers
        x = self.fc(x)
        
        # Reshape output
        x = x.reshape(batch_size, self.n_neurons, self.n_output_bins)
        
        return x


def get_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: One of 'mlp', 'cnn', 'cnn_temporal'
        **kwargs: Model-specific arguments
    
    Returns:
        Instantiated model
    """
    models = {
        'mlp': StimToSpikeMLP,
        'cnn': StimToSpikeCNN,
        'cnn_temporal': StimToSpikeCNNTemporal,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](**kwargs)


# =====================
# Training and Validation Functions
# =====================

def train_epoch(model, loader, criterion, optimizer, device, grad_clip=True, max_norm=1.0, sum_loss=False, weight_loss=1):
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
    
    Returns:
        Average loss over the epoch
    """
    
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_x, batch_y in pbar:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
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
        # on some bin sizes, we may want to sum
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()        
        total_loss += loss.item() * batch_x.size(0)
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device, sum_loss=False, weight_loss=1):
    """
    Validate the model on a dataset.
    
    Args:
        model: PyTorch model
        loader: DataLoader for validation/test data
        criterion: Loss function
        device: torch device
        sum_loss: Whether to sum the loss over output bins
        weight_loss: Weight for non-zero samples in loss
    
    Returns:
        Average loss over the dataset
    """
    total_corr = 0
    
    model.eval()
    total_loss = 0
    pbar = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
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
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
from typing import List, Optional
from tqdm import tqdm
from utils import compute_correlation
import torch
import torch.nn as nn
from typing import List, Optional


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
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](**kwargs)






# =====================
# Training and Validation Functions
# =====================

def train_epoch(model, loader, criterion, optimizer, device, grad_clip=True, max_norm=1.0, sum_loss=False, weight_loss=1, use_init_state=False):
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
    
    Returns:
        Average loss over the epoch
    """
    
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        if use_init_state:
            batch_x, batch_y, batch_init = batch
            batch_x, batch_y, batch_init = batch_x.to(device), batch_y.to(device), batch_init.to(device)
        else:
            batch_x, batch_y = batch
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_init = None
        
        optimizer.zero_grad()
        
        # Forward pass - handle models with initial_spikes argument
        if use_init_state and hasattr(model, 'forward') and 'initial_spikes' in model.forward.__code__.co_varnames:
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
        # on some bin sizes, we may want to sum
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()        
        total_loss += loss.item() * batch_x.size(0)
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device, sum_loss=False, weight_loss=1, use_init_state=False):
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
                batch_x, batch_y, batch_init = batch_x.to(device), batch_y.to(device), batch_init.to(device)
            else:
                batch_x, batch_y = batch
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_init = None
            
            # Forward pass - handle models with initial_spikes argument
            if use_init_state and hasattr(model, 'forward') and 'initial_spikes' in model.forward.__code__.co_varnames:
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





import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from optlearn.encoder import LSNNMeanVarEncoder

class T5Dataset(Dataset):
    def __init__(self, features, intents):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.intents = torch.tensor(intents, dtype=torch.float32)
        assert self.features.shape[0] == self.intents.shape[0]

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.intents[idx], self.features[idx]
    
if __name__ == "__main__":
    encoder = LSNNMeanVarEncoder(
        num_channels=192,
        snr_dof=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        sd_spontaneous=0.01,
        sd_dof=0.01,
        sd_sign=0.01,
        fc_dims=[32]
    )

    data = np.load('data/t5_combined_dataset.npz')
    neural_all_combined = data['neural']
    kinematic_all_combined = data['kinematic']
    dataset = T5Dataset(neural_all_combined, kinematic_all_combined)
    train_val_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_val_size
    train_size = int(0.8 * train_val_size)
    val_size = train_val_size - train_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print(f'Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    optimizer = Adam(encoder.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10)
    loss_fn = torch.nn.MSELoss()

    encoder.to(device)

    epochs = 5000
    early_stopping_patience = 100
    model_save_path = './models/t5_encoder.pth'

    if os.path.exists(model_save_path):
        encoder.load_state_dict(torch.load(model_save_path, weights_only=True))
        print(f'Loaded model from {model_save_path}')

    best_val_loss, best_epoch = np.inf, 0
    for epoch in range(epochs):
        encoder.train()
        train_loss = 0.0
        for batch_intents, batch_features in train_loader:
            batch_intents = batch_intents.to(device)
            batch_features = batch_features.to(device)
            optimizer.zero_grad()
            y_pred_mean, y_pred_var = encoder(batch_intents)
            loss = encoder.loss_fn(y_pred_mean, y_pred_var, batch_features)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_features.size(0)
        train_loss /= len(train_loader.dataset)

        encoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_intents, batch_features in val_loader:
                batch_intents = batch_intents.to(device)
                batch_features = batch_features.to(device)
                y_pred_mean, y_pred_var = encoder(batch_intents)
                loss = encoder.loss_fn(y_pred_mean, y_pred_var, batch_features)
                val_loss += loss.item() * batch_features.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(encoder.state_dict(), model_save_path)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} (Saved Model)")
        
        if epoch - best_epoch >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
#!/usr/bin/env python3
"""
Hyperparameter tuning script using Optuna for RNN neural activity prediction.

Usage:
    python hp_tuning.py --n_trials 50 --timeout 3600
    python hp_tuning.py --n_trials 100 --study_name my_study --resume

Key hyperparameters tuned:
    - units (hidden size): 8-256
    - num_layers: 1-3
    - learning_rate: 1e-5 to 1e-2
    - batch_size: 1, 4, 8, 16
    - weight_decay (regularization): 0 to 0.1
    - loss_type: mse, mae, weighted_mae
    - snippet_length: 30-100
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time

import numpy as np
import optuna
from optuna.trial import TrialState
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils import make_snippets_df, setup_logging
from modules import RNNModel, SeqDataset, WeightedMAELoss


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    elif torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def load_data(filter_configs=None, filter_neurons=None, filter_electrodes=None, filter_currents=None):
    """Load and preprocess data."""
    data = scipy.io.loadmat('data.mat')
    times = np.stack(data['times'][0, ...])
    dfof = np.stack(data['dfof'][0, ...])
    
    # Filter neurons if specified
    if filter_neurons is not None:
        neuron_indices = [n - 1 for n in filter_neurons]
        dfof = np.array([session_dfof[:, neuron_indices] for session_dfof in dfof])
    
    # Build stim_times_sess
    stim_times_sess = []
    for session in range(3):
        stim_times = np.zeros((dfof[session].shape[0], 10))
        for config in range(1, 31):
            electrode = (config - 1) // 3
            current = (config - 1) % 3 + 3
            for trial in range(8):
                config_time = times[session][trial, config - 1]
                stim_times[config_time, electrode] = current
        stim_times_sess.append(stim_times)
    stim_times_sess = np.array(stim_times_sess)
    
    # Build trials_df
    rows = []
    for session in range(3):
        for config in range(1, 31):
            electrode = (config - 1) // 3
            current = (config - 1) % 3 + 3
            for trial in range(8):
                raw_stim_time = int(times[session][trial, config - 1])
                rows.append({
                    'session': session,
                    'trial': trial,
                    'config': config,
                    'electrode': electrode,
                    'current': current,
                    'stim_time': raw_stim_time,
                    'start_time': raw_stim_time,
                    'is_stim': True
                })
        # No-stim trials
        for trial in range(8):
            raw_stim_time = int(times[session][trial, 30])
            rows.append({
                'session': session,
                'trial': trial,
                'config': 31,
                'electrode': -1,
                'current': 0,
                'stim_time': raw_stim_time,
                'start_time': raw_stim_time,
                'is_stim': False
            })
    
    import pandas as pd
    trials_df = pd.DataFrame(rows)
    
    # Apply filters
    if filter_configs is not None:
        trials_df = trials_df[trials_df['config'].isin(filter_configs)]
    if filter_electrodes is not None:
        trials_df = trials_df[(trials_df['electrode'].isin(filter_electrodes)) | (trials_df['electrode'] == -1)]
    if filter_currents is not None:
        trials_df = trials_df[(trials_df['current'].isin(filter_currents)) | (trials_df['current'] == 0)]
    
    return dfof, stim_times_sess, trials_df


def create_dataloaders(trials_df, dfof, stim_times_sess, snippet_length, batch_size, 
                       holdout_trials=[7], seed=42, snippet_overlap=False):
    """Create train/val/test dataloaders."""
    
    snippets_df = make_snippets_df(trials_df, dfof, stim_times_sess, snippet_length,
                                   overlap=snippet_overlap, stride=1)
    valid_df = snippets_df[snippets_df['valid']].copy()
    
    # Trial-based split
    holdout_mask = valid_df['first_trial'].isin(holdout_trials)
    test_df = valid_df[holdout_mask]
    train_val_df = valid_df[~holdout_mask]
    train_df, val_df = train_test_split(train_val_df, test_size=0.15, random_state=seed)
    
    train_ds = SeqDataset(df=train_df)
    val_ds = SeqDataset(df=val_df)
    test_ds = SeqDataset(df=test_df)
    
    # Use num_workers=0 to avoid multiprocessing issues on macOS
    # (utils.py has module-level code that causes race conditions when workers spawn)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                             num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader, train_ds


def train_epoch(model, train_loader, criterion, optimizer, device, device_type, 
                loss_type, scaler, use_amp=True):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    for (inputs, activity_initial), targets in train_loader:
        inputs = inputs.to(device)
        activity_initial = activity_initial.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            outputs = model((inputs, activity_initial))
            if loss_type == 'weighted_mae':
                loss = criterion(outputs, targets, inputs)
            else:
                loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
    
    return running_loss / len(train_loader.dataset)


def validate(model, val_loader, criterion, device, device_type, loss_type, use_amp=True):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
        for (inputs, activity_initial), targets in val_loader:
            inputs = inputs.to(device)
            activity_initial = activity_initial.to(device)
            targets = targets.to(device)
            
            outputs = model((inputs, activity_initial))
            if loss_type == 'weighted_mae':
                loss = criterion(outputs, targets, inputs)
            else:
                loss = criterion(outputs, targets)
            
            val_loss += loss.item() * inputs.size(0)
    
    return val_loss / len(val_loader.dataset)


def objective(trial, args, dfof, stim_times_sess, trials_df, device, device_type):
    """Optuna objective function."""
    
    # === Hyperparameters to tune ===
    units = trial.suggest_int('units', 8, 256, log=True)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [1, 4, 8, 16, 32])
    weight_decay = trial.suggest_float('weight_decay', 0, 0.1)
    loss_type = trial.suggest_categorical('loss_type', ['mse', 'mae'])
    snippet_length = trial.suggest_int('snippet_length', 30, 100, step=10)
    
    # Optional: tune weighted_mae params if selected
    if loss_type == 'weighted_mae':
        stim_weight = trial.suggest_float('stim_weight', 1.0, 10.0)
        stim_window = trial.suggest_int('stim_window', 5, 30)
    
    # Create dataloaders with this snippet_length
    try:
        train_loader, val_loader, test_loader, train_ds = create_dataloaders(
            trials_df, dfof, stim_times_sess, snippet_length, batch_size,
            holdout_trials=args.holdout_trials, seed=args.seed,
            snippet_overlap=args.snippet_overlap
        )
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        return float('inf')
    
    if len(train_loader.dataset) == 0:
        return float('inf')
    
    # Get dimensions
    sample_stim, _ = train_ds[0]
    input_size = sample_stim[0].shape[1]
    output_size = train_ds.Y.shape[2]
    
    # Create model
    model = RNNModel(input_size=input_size, units=units, output_size=output_size, 
                     num_layers=num_layers).to(device)
    
    # Create criterion
    if loss_type == 'mse':
        criterion = nn.MSELoss()
    elif loss_type == 'mae':
        criterion = nn.L1Loss()
    elif loss_type == 'weighted_mae':
        criterion = WeightedMAELoss(stim_weight=stim_weight, window=stim_window, decay='none')
    else:
        criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device_type, enabled=args.use_amp)
    
    # Training loop with pruning
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.max_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, 
                                 device, device_type, loss_type, scaler, args.use_amp)
        val_loss = validate(model, val_loader, criterion, device, device_type, 
                           loss_type, args.use_amp)
        
        # Report to Optuna for pruning
        trial.report(val_loss, epoch)
        
        # Prune if not promising
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            break
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for RNN model')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of trials')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    parser.add_argument('--study_name', type=str, default='rnn_hpsearch', help='Study name')
    parser.add_argument('--resume', action='store_true', help='Resume existing study')
    parser.add_argument('--max_epochs', type=int, default=30, help='Max epochs per trial')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--holdout_trials', type=int, nargs='+', default=[7], help='Holdout trial indices')
    parser.add_argument('--filter_configs', type=int, nargs='+', default=None, help='Filter configs')
    parser.add_argument('--filter_neurons', type=int, nargs='+', default=None, help='Filter neurons')
    parser.add_argument('--snippet_overlap', action='store_true', default=False, help='Use overlapping snippets')
    parser.add_argument('--output_dir', type=str, default='hp_tuning_results', help='Output directory')
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.study_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'tuning.log')),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting hyperparameter search: {args.study_name}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Args: {args}")
    
    # Get device
    device, device_type = get_device()
    logging.info(f"Using device: {device}")
    
    # Load data once
    logging.info("Loading data...")
    dfof, stim_times_sess, trials_df = load_data(
        filter_configs=args.filter_configs,
        filter_neurons=args.filter_neurons
    )
    logging.info(f"Data loaded: dfof shape={dfof.shape}, trials_df size={len(trials_df)}")
    
    # Create Optuna study
    storage = f"sqlite:///{os.path.join(output_dir, 'optuna.db')}"
    
    if args.resume:
        study = optuna.load_study(study_name=args.study_name, storage=storage)
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            sampler=optuna.samplers.TPESampler(seed=args.seed)
        )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args, dfof, stim_times_sess, trials_df, device, device_type),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
        gc_after_trial=True
    )
    
    # === Generate Report ===
    logging.info("\n" + "="*60)
    logging.info("HYPERPARAMETER SEARCH COMPLETE")
    logging.info("="*60)
    
    # Best trial
    best_trial = study.best_trial
    logging.info(f"\nBest trial: {best_trial.number}")
    logging.info(f"Best validation loss: {best_trial.value:.6f}")
    logging.info(f"Best hyperparameters:")
    for key, value in best_trial.params.items():
        logging.info(f"  {key}: {value}")
    
    # Save results
    results = {
        'study_name': args.study_name,
        'n_trials': len(study.trials),
        'best_trial': best_trial.number,
        'best_value': best_trial.value,
        'best_params': best_trial.params,
        'all_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': str(t.state)
            }
            for t in study.trials
        ]
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualization plots
    try:
        import optuna.visualization as vis
        
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(os.path.join(output_dir, 'optimization_history.html'))
        
        # Parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_html(os.path.join(output_dir, 'param_importances.html'))
        
        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(output_dir, 'parallel_coordinate.html'))
        
        # Slice plot
        fig = vis.plot_slice(study)
        fig.write_html(os.path.join(output_dir, 'slice_plot.html'))
        
        logging.info(f"\nVisualization plots saved to {output_dir}")
    except ImportError:
        logging.warning("Install plotly for visualizations: pip install plotly")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total trials: {len(study.trials)}")
    print(f"Completed: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
    print(f"Pruned: {len([t for t in study.trials if t.state == TrialState.PRUNED])}")
    print(f"Failed: {len([t for t in study.trials if t.state == TrialState.FAIL])}")
    print(f"\nBest validation loss: {best_trial.value:.6f}")
    print(f"\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    print(f"\nResults saved to: {output_dir}")
    
    return study


if __name__ == '__main__':
    main()

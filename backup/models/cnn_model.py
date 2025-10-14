"""
1D Convolutional Neural Network (CNN) Model Implementation

This module contains the CNN1DFitter class for performing 1D Convolutional
Neural Network regression on spectroscopic data for concentration prediction.
"""

import numpy as np
import streamlit as st
import base64
import tempfile
import os
import pickle
from typing import Dict, List
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import random

# Try to import Optuna for Bayesian optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    st.warning("PyTorch not available. CNN models will be disabled.")


class CNN1DModel(nn.Module):
    """PyTorch 1D CNN model for spectroscopic data regression."""
    
    def __init__(self, n_wavelengths, filter_list, kernel_size, dense_units, 
                 dropout_rate, use_pooling=True):
        super(CNN1DModel, self).__init__()
        
        # Store configuration for serialization
        self.config = {
            'n_wavelengths': n_wavelengths,
            'filter_list': filter_list,
            'kernel_size': kernel_size,
            'dense_units': dense_units,
            'dropout_rate': dropout_rate,
            'use_pooling': use_pooling
        }
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        in_channels = 1  # Input has 1 channel
        for i, filters in enumerate(filter_list):
            self.conv_layers.append(
                nn.Conv1d(in_channels, filters, kernel_size, padding='same')
            )
            if use_pooling and i < len(filter_list) - 1:  # Don't pool after last conv layer
                self.pool_layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = filters
        
        # Calculate flattened size after conv layers
        # This is a rough estimate - in practice, we'll use adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Build dense layers
        self.dense_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # First dense layer input size is the last conv layer's output
        dense_input_size = filter_list[-1] if filter_list else 32
        
        for i, units in enumerate(dense_units[:-1]):  # All except the last one
            self.dense_layers.append(nn.Linear(dense_input_size, units))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            dense_input_size = units
        
        # Final output layer
        if dense_units:
            self.dense_layers.append(nn.Linear(dense_input_size, dense_units[-1]))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Handle 2D input by adding channel dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # x shape: (batch_size, n_wavelengths, 1) -> (batch_size, 1, n_wavelengths)
        elif x.dim() == 3 and x.shape[-1] == 1:
            x = x.transpose(1, 2)
        
        # Convolutional layers
        for i, conv_layer in enumerate(self.conv_layers):
            x = torch.relu(conv_layer(x))
            if i < len(self.pool_layers):
                x = self.pool_layers[i](x)
        
        # Adaptive pooling to get fixed size output
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Dense layers
        for i, dense_layer in enumerate(self.dense_layers[:-1]):
            x = torch.relu(dense_layer(x))
            if i < len(self.dropout_layers):
                x = self.dropout_layers[i](x)
        
        # Final output layer (no activation)
        if self.dense_layers:
            x = self.dense_layers[-1](x)
        
        return x


def _optimize_cnn_params(x_data, y_data, param_space, optimization_method, cv_folds, 
                        max_epochs, enable_early_stopping, early_stopping_patience, device, n_trials=20):
    """Optimize CNN hyperparameters using the specified method."""
    
    if optimization_method == "Grid Search":
        print("Performing Grid Search optimization for CNN...")
        return _grid_search_cnn(x_data, y_data, param_space, cv_folds, 
                               max_epochs, enable_early_stopping, early_stopping_patience, device)
    elif optimization_method == "Random Search":
        print("Performing Random Search optimization for CNN...")
        return _random_search_cnn(x_data, y_data, param_space, cv_folds, 
                                 max_epochs, enable_early_stopping, early_stopping_patience, device, n_trials)
    elif optimization_method == "Bayesian":
        if OPTUNA_AVAILABLE:
            print("Performing Bayesian optimization for CNN...")
            return _bayesian_search_cnn(x_data, y_data, param_space, cv_folds, 
                                       max_epochs, enable_early_stopping, early_stopping_patience, device, n_trials)
        else:
            print("Optuna not available, falling back to Random Search...")
            return _random_search_cnn(x_data, y_data, param_space, cv_folds, 
                                     max_epochs, enable_early_stopping, early_stopping_patience, device, n_trials)
    else:
        print(f"Unknown optimization method: {optimization_method}, using Random Search...")
        return _random_search_cnn(x_data, y_data, param_space, cv_folds, 
                                 max_epochs, enable_early_stopping, early_stopping_patience, device, n_trials)


def _random_search_cnn(x_data, y_data, param_space, cv_folds, max_epochs, 
                       enable_early_stopping, early_stopping_patience, device, n_trials=15):
    """Perform random search optimization for CNN."""
    best_score = -float('inf')
    best_params = None
    
    for trial in range(n_trials):
        # Sample random parameters
        params = {
            'filters': random.choice(param_space['filters']),
            'kernel_size': random.choice(param_space['kernel_size']),
            'dense_units': random.choice(param_space['dense_units']),
            'dropout_rate': random.choice(param_space['dropout_rate']),
            'batch_size': random.choice(param_space['batch_size']),
            'learning_rate': random.choice(param_space['learning_rate'])
        }
        
        print(f"Trial {trial + 1}/{n_trials}: Testing params {params}")
        
        # Train and evaluate model
        score = _evaluate_cnn_params(x_data, y_data, params, cv_folds, 
                                    max_epochs, enable_early_stopping, early_stopping_patience, device)
        
        if score > best_score:
            best_score = score
            best_params = params
            print(f"New best score: {best_score:.4f}")
    
    print(f"Best parameters found: {best_params}")
    print(f"Best CV score: {best_score:.4f}")
    return best_params


def _evaluate_cnn_params(x_data, y_data, params, cv_folds, max_epochs, 
                         enable_early_stopping, early_stopping_patience, device):
    """Evaluate CNN parameters using cross-validation."""
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    
    # Use reduced epochs for CV to speed up optimization (40% of max_epochs)
    # This gives models enough time to converge while keeping optimization fast
    cv_epochs = max(100, int(max_epochs * 0.4))
    cv_patience = max(10, int(early_stopping_patience * 0.5))  # Proportional patience
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_data)):
        x_train, x_val = x_data[train_idx], x_data[val_idx]
        y_train, y_val = y_data[train_idx], y_data[val_idx]
        
        # Create model
        model = CNN1DModel(
            n_wavelengths=x_data.shape[1],
            filter_list=params['filters'],
            kernel_size=params['kernel_size'],
            dense_units=params['dense_units'],
            dropout_rate=params['dropout_rate']
        ).to(device)
        
        # Train model
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = nn.MSELoss()
        
        # Convert to tensors and reshape for CNN
        # Reshape: (n_samples, n_wavelengths) -> (n_samples, 1, n_wavelengths)
        x_train_reshaped = x_train.reshape(-1, 1, x_train.shape[1])
        x_val_reshaped = x_val.reshape(-1, 1, x_val.shape[1])
        
        x_train_tensor = torch.FloatTensor(x_train_reshaped).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        x_val_tensor = torch.FloatTensor(x_val_reshaped).to(device)
        y_val_tensor = torch.FloatTensor(y_val).to(device)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(cv_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(x_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation
            if enable_early_stopping and epoch % 5 == 0:  # Check every 5 epochs
                model.eval()
                with torch.no_grad():
                    val_outputs = model(x_val_tensor)
                    val_loss = criterion(val_outputs.squeeze(), y_val_tensor).item()
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = model.state_dict().copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= cv_patience:
                        break
        
        # Restore best model if early stopping was used
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val_tensor)
            val_predictions = val_outputs.squeeze().cpu().numpy()
            r2 = r2_score(y_val, val_predictions)
            scores.append(r2)
    
    return np.mean(scores)


def _bayesian_search_cnn(x_data, y_data, param_space, cv_folds, max_epochs, 
                        enable_early_stopping, early_stopping_patience, device, n_trials=25):
    """Perform Bayesian optimization for CNN using Optuna."""
    if not OPTUNA_AVAILABLE:
        print("Optuna not available, falling back to Random Search...")
        return _random_search_cnn(x_data, y_data, param_space, cv_folds, 
                                 max_epochs, enable_early_stopping, early_stopping_patience, device, n_trials)
    
    def objective(trial):
        # Sample parameters from param_space
        params = {}
        
        # Filters - convert to strings for categorical
        params['filters'] = trial.suggest_categorical('filters', [str(f) for f in param_space['filters']])
        
        # Kernel size
        params['kernel_size'] = trial.suggest_categorical('kernel_size', param_space['kernel_size'])
        
        # Dense units - convert to strings for categorical
        params['dense_units'] = trial.suggest_categorical('dense_units', [str(d) for d in param_space['dense_units']])
        
        # Dropout rate - now always dict format
        dropout_cfg = param_space['dropout_rate']
        params['dropout_rate'] = trial.suggest_float('dropout_rate', dropout_cfg['low'], dropout_cfg['high'])
        
        # Batch size - always categorical
        params['batch_size'] = trial.suggest_categorical('batch_size', param_space['batch_size'])
        
        # Learning rate - now always dict format
        lr_cfg = param_space['learning_rate']
        params['learning_rate'] = trial.suggest_float('learning_rate', lr_cfg['low'], lr_cfg['high'], log=lr_cfg.get('log', False))
        
        # Convert back from strings to lists
        params['filters'] = eval(params['filters']) if isinstance(params['filters'], str) else params['filters']
        params['dense_units'] = eval(params['dense_units']) if isinstance(params['dense_units'], str) else params['dense_units']
        
        # Evaluate parameters
        score = _evaluate_cnn_params(x_data, y_data, params, cv_folds, 
                                    max_epochs, enable_early_stopping, early_stopping_patience, device)
        return score
    
    # Create study with better sampler settings and pruning
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            seed=42, 
            n_startup_trials=min(10, n_trials // 3),
            multivariate=True
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10
        )
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    # Convert back from strings to lists
    if isinstance(best_params.get('filters'), str):
        best_params['filters'] = eval(best_params['filters'])
    if isinstance(best_params.get('dense_units'), str):
        best_params['dense_units'] = eval(best_params['dense_units'])
    
    print(f"Best parameters found: {best_params}")
    print(f"Best CV score: {study.best_value:.4f}")
    
    return best_params


def _grid_search_cnn(x_data, y_data, param_space, cv_folds, max_epochs, 
                    enable_early_stopping, early_stopping_patience, device):
    """Perform grid search optimization for CNN."""
    print("Grid search is computationally expensive for CNN. Using Random Search instead...")
    return _random_search_cnn(x_data, y_data, param_space, cv_folds, 
                             max_epochs, enable_early_stopping, early_stopping_patience, device)


class CNN1DFitter:
    """1D Convolutional Neural Network for concentration prediction using full spectrum."""
    
    @staticmethod
    def fit_cnn1d(x_data_tuple: tuple, y_data_tuple: tuple,
                  cnn_configs: List[Dict] = None,
                  optimization_method: str = "Random Search",
                  cv_folds: int = 5,
                  max_epochs: int = 1000,
                  enable_early_stopping: bool = True,
                  early_stopping_patience: int = 20,
                  n_trials: int = 20,
                  run_cv: bool = True) -> Dict[str, Dict]:
        """
        Fit 1D CNN models with various architectures using full spectrum data.

        Args:
            x_data_tuple: Input spectral data as tuple for caching (n_samples, n_wavelengths)
            y_data_tuple: Output concentration data as tuple for caching (n_samples,)
            cnn_configs: List of CNN configurations to try
            run_cv: Whether to perform cross-validation
            
        Returns:
            Dictionary with config_name -> fit_results mapping
        """
        if not TORCH_AVAILABLE:
            st.warning("PyTorch not available. CNN models disabled.")
            return {}
        
        # Convert tuples back to arrays for computation
        x_data = np.array(x_data_tuple)  # Full spectrum: (n_samples, n_wavelengths)
        y_data = np.array(y_data_tuple)  # Concentrations: (n_samples,)
        
        if len(x_data) < 3:
            st.warning("CNN requires at least 3 data points")
            return {}
        
        # Validate input data
        if np.any(~np.isfinite(x_data)) or np.any(~np.isfinite(y_data)):
            st.error("Data contains invalid values (NaN or Inf)")
            return {}
        
        # Split data for validation during final training (15% holdout)
        from sklearn.model_selection import train_test_split
        X_train_full, X_val_final, y_train_full, y_val_final = train_test_split(
            x_data, y_data, test_size=0.15, random_state=42
        )
        
        # Standardize data first (fit on training, transform both)
        from sklearn.preprocessing import StandardScaler
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        
        x_train_scaled = scaler_x.fit_transform(X_train_full)
        y_train_scaled = scaler_y.fit_transform(y_train_full.reshape(-1, 1)).ravel()
        
        x_val_scaled = scaler_x.transform(X_val_final)
        y_val_scaled = scaler_y.transform(y_val_final.reshape(-1, 1)).ravel()
        
        # For compatibility, also create full scaled versions
        x_scaled = scaler_x.transform(x_data)
        y_scaled = scaler_y.transform(y_data.reshape(-1, 1)).ravel()
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use optimization to find best hyperparameters
        if cnn_configs is None:
            print("No CNN configurations provided. Using hyperparameter optimization...")
            
            # Adaptive parameter space based on dataset size and sequence length
            n = len(x_data)  # Number of samples
            n_features = x_scaled.shape[1]  # Number of wavelengths
            
            # Adaptive kernel sizes based on sequence length
            if n_features < 50:
                kernel_sizes = [3, 5]
            elif n_features < 200:
                kernel_sizes = [3, 5, 7]
            else:
                kernel_sizes = [3, 5, 7, 9]  # Larger kernels for long sequences
            
            # Adaptive filter configurations
            if n_features > 200:
                # Large sequences: use more complex architectures
                filter_configs = [[32], [64], [32, 64], [64, 128], [32, 64, 32]]
            else:
                # Smaller sequences: simpler architectures
                filter_configs = [[32], [64], [32, 64], [64, 64]]
            
            # Adaptive batch size based on sample size
            max_batch = min(n // 2, 32)  # CNNs often work better with smaller batches
            if n < 10:
                batch_sizes = [min(2, n), min(4, n)]
            elif n < 20:
                batch_sizes = [4, 8, min(16, n)]
            else:
                batch_sizes = [4, 8, 16, min(32, max_batch)]
            batch_sizes = [b for b in batch_sizes if b > 0 and b <= n]
            
            # Define parameter space for optimization
            param_space = {
                'filters': filter_configs,
                'kernel_size': kernel_sizes,
                'dense_units': [[32, 16, 1], [64, 32, 1], [32, 32, 16, 1], [64, 64, 32, 1]],
                'dropout_rate': {'low': 0.1, 'high': 0.4},  # Use dict format for continuous
                'batch_size': batch_sizes,
                'learning_rate': {'low': 1e-4, 'high': 5e-3, 'log': True}  # Use dict format
            }
            
            print(f"   CNN optimization: {n} samples, {n_features} wavelengths")
            print(f"   Batch sizes: {batch_sizes}")
            print(f"   Kernel sizes: {kernel_sizes}")
            print(f"   Filter configs: {filter_configs}")
            
            # Perform optimization on training data only
            best_params = _optimize_cnn_params(
                x_train_scaled, y_train_scaled, param_space, optimization_method, cv_folds,
                max_epochs, enable_early_stopping, early_stopping_patience, device, n_trials
            )
            
            # Create single optimized configuration
            cnn_configs = [{
                'name': 'CNN1D_Optimized',
                'filters': best_params['filters'],
                'kernel_size': best_params['kernel_size'],
                'dense_units': best_params['dense_units'],
                'dropout_rate': best_params['dropout_rate'],
                'epochs': max_epochs,
                'batch_size': best_params['batch_size'],
                'learning_rate': best_params['learning_rate']
            }]
        
        # Reshape for CNN (add channel dimension)
        # For full spectrum: (n_samples, n_wavelengths) -> (n_samples, 1, n_wavelengths)
        n_wavelengths = x_scaled.shape[1]
        x_cnn = x_scaled.reshape(-1, 1, n_wavelengths)
        
        fit_results = {}
        n = len(x_data)
        y_mean = np.mean(y_data)
        ss_tot = np.sum((y_data - y_mean) ** 2)
        
        for config in cnn_configs:
            try:
                # Validate config structure
                if not isinstance(config, dict):
                    st.warning(f"Invalid config for CNN: config must be a dictionary")
                    continue
                
                if 'name' not in config:
                    st.warning(f"Invalid config for CNN: missing 'name' field")
                    continue
                
                config_name = config['name']
                
                # Extract parameters with validation
                try:
                    filters_val = config.get('filters', 32)
                    if isinstance(filters_val, list) and len(filters_val) > 0:
                        filter_list = [int(f) for f in filters_val]
                    elif isinstance(filters_val, (int, float)):
                        filter_list = [int(filters_val)]
                    else:
                        filter_list = [32]
                except (ValueError, TypeError, IndexError):
                    filter_list = [32]
                    st.warning(f"Invalid filters value for CNN '{config_name}', using default: [32]")
                
                try:
                    kernel_size_val = config.get('kernel_size', 3)
                    if isinstance(kernel_size_val, list) and len(kernel_size_val) > 0:
                        kernel_size = min(int(kernel_size_val[0]), n_wavelengths)
                    elif isinstance(kernel_size_val, (int, float)):
                        kernel_size = min(int(kernel_size_val), n_wavelengths)
                    else:
                        kernel_size = min(3, n_wavelengths)
                except (ValueError, TypeError, IndexError):
                    kernel_size = min(3, n_wavelengths)
                    st.warning(f"Invalid kernel_size value for CNN '{config_name}', using default: {kernel_size}")
                
                # Ensure kernel_size is at least 1
                kernel_size = max(1, kernel_size)
                
                try:
                    use_pooling_val = config.get('use_pooling', True)
                    if isinstance(use_pooling_val, list) and len(use_pooling_val) > 0:
                        use_pooling = bool(use_pooling_val[0])
                    elif isinstance(use_pooling_val, (bool, int, float)):
                        use_pooling = bool(use_pooling_val)
                    else:
                        use_pooling = True
                except (ValueError, TypeError, IndexError):
                    use_pooling = True
                    st.warning(f"Invalid use_pooling value for CNN '{config_name}', using default: True")
                
                # Handle dense_units parameter from config
                try:
                    dense_units = config.get('dense_units', [50, 1])
                    if isinstance(dense_units, list) and len(dense_units) > 0:
                        dense_units = [int(u) for u in dense_units]
                    else:
                        dense_units = [50, 1]
                except (ValueError, TypeError, IndexError):
                    dense_units = [50, 1]
                    st.warning(f"Invalid dense_units configuration for CNN '{config_name}', using default: [50, 1]")
                
                # Handle dropout_rate parameter
                try:
                    dropout_rate_val = config.get('dropout_rate', 0.2)
                    if isinstance(dropout_rate_val, list) and len(dropout_rate_val) > 0:
                        dropout_rate = float(dropout_rate_val[0])
                    elif isinstance(dropout_rate_val, (int, float)):
                        dropout_rate = float(dropout_rate_val)
                    else:
                        dropout_rate = 0.2
                except (ValueError, TypeError, IndexError):
                    dropout_rate = 0.2
                    st.warning(f"Invalid dropout_rate value for CNN '{config_name}', using default: 0.2")
                
                # Build CNN model
                model = CNN1DModel(
                    n_wavelengths=n_wavelengths,
                    filter_list=filter_list,
                    kernel_size=kernel_size,
                    dense_units=dense_units,
                    dropout_rate=dropout_rate,
                    use_pooling=use_pooling
                ).to(device)
                
                # Training parameters
                try:
                    epochs_val = config.get('epochs', 1000)
                    if isinstance(epochs_val, list) and len(epochs_val) > 0:
                        epochs = int(epochs_val[0])
                    elif isinstance(epochs_val, (int, float)):
                        epochs = int(epochs_val)
                    else:
                        epochs = 1000
                except (ValueError, TypeError, IndexError):
                    epochs = 1000
                    st.warning(f"Invalid epochs value for CNN '{config_name}', using default: 1000")
                
                try:
                    batch_size_val = config.get('batch_size', 32)
                    if isinstance(batch_size_val, list) and len(batch_size_val) > 0:
                        batch_size = min(int(batch_size_val[0]), len(x_cnn))
                    elif isinstance(batch_size_val, (int, float)):
                        batch_size = min(int(batch_size_val), len(x_cnn))
                    else:
                        batch_size = min(32, len(x_cnn))
                except (ValueError, TypeError, IndexError):
                    batch_size = min(32, len(x_cnn))
                    st.warning(f"Invalid batch_size value for CNN '{config_name}', using default: {batch_size}")
                
                # Reshape for CNN with train/val split
                x_train_cnn = x_train_scaled.reshape(-1, 1, n_wavelengths)
                x_val_cnn = x_val_scaled.reshape(-1, 1, n_wavelengths)
                
                # Convert to PyTorch tensors for training and validation
                x_train_tensor = torch.FloatTensor(x_train_cnn).to(device)
                y_train_tensor = torch.FloatTensor(y_train_scaled.reshape(-1, 1)).to(device)
                x_val_tensor = torch.FloatTensor(x_val_cnn).to(device)
                y_val_tensor = torch.FloatTensor(y_val_scaled.reshape(-1, 1)).to(device)
                
                # Create data loader (training only)
                train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                # Setup training
                criterion = nn.MSELoss()
                learning_rate = config.get('learning_rate', 0.001)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                scheduler_patience = config.get('scheduler_patience', 10)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, factor=0.5)
                
                # Training loop with validation-based early stopping
                import time
                training_start_time = time.time()
                best_val_loss = float('inf')
                patience_counter = 0
                patience = early_stopping_patience  # Use the parameter from sidebar
                train_losses = []
                val_losses = []
                
                for epoch in range(epochs):
                    # Training phase
                    model.train()
                    epoch_train_loss = 0.0
                    for batch_x, batch_y in train_dataloader:
                        optimizer.zero_grad()
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        epoch_train_loss += loss.item()
                    
                    avg_train_loss = epoch_train_loss / len(train_dataloader)
                    train_losses.append(avg_train_loss)
                    
                    # Validation phase
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(x_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor).item()
                        val_losses.append(val_loss)
                    
                    scheduler.step(val_loss)
                    
                    # Early stopping based on validation loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model state
                        best_model_state = model.state_dict().copy()
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        # Restore best model (early stopping)
                        model.load_state_dict(best_model_state)
                        break
                
                # Calculate training time
                training_time = time.time() - training_start_time
                
                # Make predictions
                model.eval()
                with torch.no_grad():
                    y_pred_scaled = model(x_tensor).cpu().numpy().ravel()
                
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                
                # Calculate statistics
                residuals = y_data - y_pred
                ss_res = np.sum(residuals ** 2)
                mse = ss_res / n
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(residuals))
                
                # Additional metrics
                mape = np.mean(np.abs((y_data - y_pred) / y_data)) * 100 if np.all(y_data != 0) else np.nan
                max_error = np.max(np.abs(residuals))
                
                # R-squared
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0
                
                # Adjusted R-squared (approximation)
                n_params = sum(p.numel() for p in model.parameters())
                if n > n_params and r_squared < 0.999:
                    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - n_params)
                else:
                    adj_r_squared = r_squared
                
                # Information criteria
                if ss_res > 1e-12:
                    log_likelihood = -n/2 * np.log(2*np.pi) - n/2 * np.log(ss_res/n) - n/2
                    aic = 2 * n_params - 2 * log_likelihood
                    bic = n_params * np.log(n) - 2 * log_likelihood
                else:
                    aic = np.inf
                    bic = np.inf
                
                # Serialize PyTorch model using pickle
                try:
                    # Create serialization data
                    serialization_data = {
                        'state_dict': model.state_dict(),
                        'config': model.config,
                        'scaler_x': scaler_x,
                        'scaler_y': scaler_y,
                        'model_config': config
                    }
                    
                    # Serialize using pickle
                    model_pickle = pickle.dumps(serialization_data)
                    model_serialized = base64.b64encode(model_pickle).decode('utf-8')
                    
                except Exception as e:
                    st.warning(f"Failed to serialize CNN model '{config_name}': {str(e)}")
                    model_serialized = 'PYTORCH_SERIALIZATION_FAILED'
                
                # Also serialize scalers separately for compatibility
                scalers_serialized = None
                try:
                    scalers_data = {
                        'scaler_x': scaler_x,
                        'scaler_y': scaler_y,
                        'config': config
                    }
                    scalers_pickle = pickle.dumps(scalers_data)
                    scalers_serialized = base64.b64encode(scalers_pickle).decode('utf-8')
                except Exception as e:
                    st.warning(f"Failed to serialize CNN scalers for '{config_name}': {str(e)}")
                    scalers_serialized = None
                
                # Create model info for display
                model_info = {
                    'architecture': f"CNN1D({config['filters']}, {config['kernel_size']}, {config['dense_units']})",
                    'filters': config['filters'],
                    'kernel_size': config['kernel_size'],
                    'dense_units': config['dense_units'],
                    'dropout_rate': config['dropout_rate']
                }
                
                fit_results[config_name] = {
                    'model': model,
                    'scaler_x': scaler_x,
                    'scaler_y': scaler_y,
                    'predictions': y_pred,
                    'r_squared': max(0, min(1, r_squared)),
                    'adj_r_squared': max(0, min(1, adj_r_squared)),
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'max_error': max_error
                }

                # Cross-validation score for CNN
                cv_score = np.nan
                cv_rmse = np.nan
                cv_std = np.nan
                if run_cv:
                    try:
                        cv_folds = max(2, min(5, n - 1))

                        # For CNN, we need to create a custom scorer that uses the CNN model
                        from sklearn.model_selection import KFold
                        from sklearn.base import BaseEstimator, RegressorMixin

                        class CNNWrapper(BaseEstimator, RegressorMixin):
                            def __init__(self, model, scaler_x, scaler_y, device):
                                self.model = model
                                self.scaler_x = scaler_x
                                self.scaler_y = scaler_y
                                self.device = device

                            def fit(self, X, y):
                                return self

                            def predict(self, X):
                                self.model.eval()
                                with torch.no_grad():
                                    # Scale input
                                    X_scaled = self.scaler_x.transform(X)
                                    # Convert to tensor and reshape for CNN
                                    X_tensor = torch.FloatTensor(X_scaled).unsqueeze(1).to(self.device)
                                    # Get predictions
                                    outputs = self.model(X_tensor)
                                    # Convert back to numpy and inverse scale
                                    y_pred_scaled = outputs.cpu().numpy().flatten()
                                    y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                                    return y_pred

                        cnn_wrapper = CNNWrapper(model, scaler_x, scaler_y, device)

                        # Perform cross-validation
                        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                        cv_scores = []
                        cv_mse_scores = []

                        for train_idx, val_idx in kf.split(x_scaled):
                            X_train, X_val = x_scaled[train_idx], x_scaled[val_idx]
                            y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]

                            # Create temporary CNN model with same architecture
                            temp_model = CNN1DModel(
                                n_wavelengths=n_wavelengths,
                                filter_list=filter_list,
                                kernel_size=kernel_size,
                                dense_units=dense_units,
                                dropout_rate=dropout_rate
                            ).to(device)

                            # Create data loaders
                            train_dataset = TensorDataset(
                                torch.FloatTensor(X_train).unsqueeze(1),
                                torch.FloatTensor(y_train).unsqueeze(1)
                            )
                            val_dataset = TensorDataset(
                                torch.FloatTensor(X_val).unsqueeze(1),
                                torch.FloatTensor(y_val).unsqueeze(1)
                            )

                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                            # Train temporary model
                            criterion = nn.MSELoss()
                            learning_rate = config.get('learning_rate', 0.001)
                            optimizer = optim.Adam(temp_model.parameters(), lr=learning_rate)

                            temp_model.train()
                            for epoch in range(min(50, max_epochs)):  # Limited epochs for CV
                                for batch_X, batch_y in train_loader:
                                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                    optimizer.zero_grad()
                                    outputs = temp_model(batch_X)
                                    loss = criterion(outputs, batch_y)
                                    loss.backward()
                                    optimizer.step()

                            # Evaluate on validation set
                            temp_model.eval()
                            with torch.no_grad():
                                val_preds = []
                                val_targets = []
                                for batch_X, batch_y in val_loader:
                                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                    outputs = temp_model(batch_X)
                                    val_preds.extend(outputs.cpu().numpy().flatten())
                                    val_targets.extend(batch_y.cpu().numpy().flatten())

                                val_preds = np.array(val_preds)
                                val_targets = np.array(val_targets)

                                # Calculate R²
                                ss_res = np.sum((val_targets - val_preds) ** 2)
                                ss_tot = np.sum((val_targets - np.mean(val_targets)) ** 2)
                                r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0
                                cv_scores.append(r2)

                                # Calculate MSE for RMSE
                                mse_val = np.mean((val_targets - val_preds) ** 2)
                                cv_mse_scores.append(mse_val)

                        if cv_scores:
                            cv_score = float(np.mean(cv_scores))
                            cv_std = float(np.std(cv_scores))
                            cv_rmse = float(np.sqrt(np.mean(cv_mse_scores)))

                    except Exception as e:
                        print(f"CNN CV calculation failed: {e}")
                        cv_score = np.nan
                        cv_rmse = np.nan
                        cv_std = np.nan

                # Add CV results to the fit_results dictionary
                fit_results[config_name]['cv_score'] = cv_score
                fit_results[config_name]['cv_rmse'] = cv_rmse
                fit_results[config_name]['cv_std'] = cv_std
                fit_results[config_name]['training_time'] = training_time
                fit_results[config_name]['aic'] = aic
                fit_results[config_name]['bic'] = bic
                fit_results[config_name]['residuals'] = residuals
                fit_results[config_name]['n_parameters'] = n_params
                fit_results[config_name]['model_serialized'] = model_serialized
                fit_results[config_name]['scalers_serialized'] = scalers_serialized
                fit_results[config_name]['config'] = config
                fit_results[config_name]['model_info'] = model_info
                fit_results[config_name]['train_losses'] = train_losses
                fit_results[config_name]['val_losses'] = val_losses
                
            except Exception as e:
                st.warning(f"Failed to fit CNN '{config_name}': {str(e)}")
                continue
        
        return fit_results
    
    @staticmethod
    def predict_with_cnn1d(model_data: Dict, spectrum: np.ndarray) -> float:
        """
        Predict concentration using trained CNN model.
        
        Args:
            model_data: Dictionary with model, scalers, etc.
            spectrum: Full spectrum absorbance data (n_wavelengths,)
            
        Returns:
            Predicted concentration
        """
        if not TORCH_AVAILABLE:
            return 0.0
            
        try:
            model = model_data['model']
            scaler_x = model_data['scaler_x']
            scaler_y = model_data['scaler_y']
            
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Ensure spectrum is 2D (1, n_wavelengths)
            if spectrum.ndim == 1:
                spectrum = spectrum.reshape(1, -1)
            
            # Scale input spectrum
            x_scaled = scaler_x.transform(spectrum)
            n_wavelengths = x_scaled.shape[1]
            x_cnn = x_scaled.reshape(1, 1, -1)
            
            # Convert to PyTorch tensor
            x_tensor = torch.FloatTensor(x_cnn).to(device)
            
            # Predict concentration
            model.eval()
            with torch.no_grad():
                y_pred_scaled = model(x_tensor).cpu().numpy().ravel()
            
            # Inverse transform
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            return float(y_pred[0])
            
        except Exception as e:
            st.error(f"Error predicting with CNN: {str(e)}")
            return 0.0
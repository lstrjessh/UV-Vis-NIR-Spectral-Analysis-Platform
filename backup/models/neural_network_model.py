"""
Neural Network (MLP) Model Implementation

This module contains the NeuralNetworkFitter class for performing Multi-Layer
Perceptron regression on spectroscopic data for concentration prediction.
"""

import numpy as np
import streamlit as st
import pickle
import base64
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
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
    st.warning("PyTorch not available. Neural network models will be disabled.")


class MLPModel(nn.Module):
    """PyTorch MLP model for spectroscopic data regression."""
    
    def __init__(self, input_size, hidden_layers, activation='relu', dropout_rate=0.0):
        super(MLPModel, self).__init__()
        
        # Store configuration for serialization
        self.config = {
            'input_size': input_size,
            'hidden_layers': hidden_layers,
            'activation': activation,
            'dropout_rate': dropout_rate
        }
        
        # Build layers dynamically
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            if dropout_rate > 0:
                self.dropouts.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        self.layers.append(nn.Linear(prev_size, 1))
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()  # Default to ReLU
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        for i, layer in enumerate(self.layers[:-1]):  # All except the last layer
            x = layer(x)
            x = self.activation(x)
            if i < len(self.dropouts):
                x = self.dropouts[i](x)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        return x


def _optimize_neural_network_params(x_data, y_data, param_space, optimization_method, cv_folds, 
                                   max_epochs, enable_early_stopping, early_stopping_patience, device, n_trials=20):
    """Optimize neural network hyperparameters using the specified method."""
    
    if optimization_method == "Grid Search":
        print("Performing Grid Search optimization for MLP...")
        return _grid_search_neural_network(x_data, y_data, param_space, cv_folds, 
                                         max_epochs, enable_early_stopping, early_stopping_patience, device)
    elif optimization_method == "Random Search":
        print("Performing Random Search optimization for MLP...")
        return _random_search_neural_network(x_data, y_data, param_space, cv_folds, 
                                           max_epochs, enable_early_stopping, early_stopping_patience, device, n_trials)
    elif optimization_method == "Bayesian":
        if OPTUNA_AVAILABLE:
            print("Performing Bayesian optimization for MLP...")
            return _bayesian_search_neural_network(x_data, y_data, param_space, cv_folds, 
                                                 max_epochs, enable_early_stopping, early_stopping_patience, device, n_trials)
        else:
            print("Optuna not available, falling back to Random Search...")
            return _random_search_neural_network(x_data, y_data, param_space, cv_folds, 
                                               max_epochs, enable_early_stopping, early_stopping_patience, device, n_trials)
    else:
        print(f"Unknown optimization method: {optimization_method}, using Random Search...")
        return _random_search_neural_network(x_data, y_data, param_space, cv_folds, 
                                           max_epochs, enable_early_stopping, early_stopping_patience, device, n_trials)


def _random_search_neural_network(x_data, y_data, param_space, cv_folds, max_epochs, 
                                 enable_early_stopping, early_stopping_patience, device, n_trials=20):
    """Perform random search optimization for neural network."""
    best_score = -float('inf')
    best_params = None
    
    for trial in range(n_trials):
        # Sample random parameters
        params = {
            'hidden_layers': random.choice(param_space['hidden_layers']),
            'activation': random.choice(param_space['activation']),
            'dropout_rate': random.choice(param_space['dropout_rate']),
            'learning_rate': random.choice(param_space['learning_rate']),
            'batch_size': random.choice(param_space['batch_size']),
            'weight_decay': random.choice(param_space['weight_decay'])
        }
        
        print(f"Trial {trial + 1}/{n_trials}: Testing params {params}")
        
        # Train and evaluate model
        score = _evaluate_neural_network_params(x_data, y_data, params, cv_folds, 
                                              max_epochs, enable_early_stopping, early_stopping_patience, device)
        
        if score > best_score:
            best_score = score
            best_params = params
            print(f"New best score: {best_score:.4f}")
    
    print(f"Best parameters found: {best_params}")
    print(f"Best CV score: {best_score:.4f}")
    return best_params


def _evaluate_neural_network_params(x_data, y_data, params, cv_folds, max_epochs, 
                                   enable_early_stopping, early_stopping_patience, device):
    """Evaluate neural network parameters using cross-validation."""
    from sklearn.model_selection import KFold
    
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
        model = MLPModel(
            input_size=x_data.shape[1],
            hidden_layers=params['hidden_layers'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate']
        ).to(device)
        
        # Train model
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], 
                              weight_decay=params['weight_decay'])
        criterion = nn.MSELoss()
        
        # Convert to tensors
        x_train_tensor = torch.FloatTensor(x_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        x_val_tensor = torch.FloatTensor(x_val).to(device)
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


def _bayesian_search_neural_network(x_data, y_data, param_space, cv_folds, max_epochs, 
                                   enable_early_stopping, early_stopping_patience, device, n_trials=30):
    """Perform Bayesian optimization for neural network using Optuna."""
    if not OPTUNA_AVAILABLE:
        print("Optuna not available, falling back to Random Search...")
        return _random_search_neural_network(x_data, y_data, param_space, cv_folds, 
                                           max_epochs, enable_early_stopping, early_stopping_patience, device, n_trials)
    
    def objective(trial):
        # Sample parameters from param_space
        params = {}
        
        # Hidden layers - convert to strings for categorical
        hidden_choices = [str(h) for h in param_space['hidden_layers']]
        params['hidden_layers'] = trial.suggest_categorical('hidden_layers', hidden_choices)
        
        # Activation
        params['activation'] = trial.suggest_categorical('activation', param_space['activation'])
        
        # Dropout rate - now always dict format
        dropout_cfg = param_space['dropout_rate']
        params['dropout_rate'] = trial.suggest_float('dropout_rate', dropout_cfg['low'], dropout_cfg['high'])
        
        # Learning rate - now always dict format
        lr_cfg = param_space['learning_rate']
        params['learning_rate'] = trial.suggest_float('learning_rate', lr_cfg['low'], lr_cfg['high'], log=lr_cfg.get('log', False))
        
        # Batch size - always categorical
        params['batch_size'] = trial.suggest_categorical('batch_size', param_space['batch_size'])
        
        # Weight decay - now always dict format
        wd_cfg = param_space['weight_decay']
        params['weight_decay'] = trial.suggest_float('weight_decay', wd_cfg['low'], wd_cfg['high'], log=wd_cfg.get('log', False))
        
        # Convert hidden_layers back to tuple
        params['hidden_layers'] = eval(params['hidden_layers']) if isinstance(params['hidden_layers'], str) else params['hidden_layers']
        
        # Evaluate parameters
        score = _evaluate_neural_network_params(x_data, y_data, params, cv_folds, 
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
    # Convert hidden_layers back to tuple if it's a string
    if isinstance(best_params.get('hidden_layers'), str):
        best_params['hidden_layers'] = eval(best_params['hidden_layers'])
    
    print(f"Best parameters found: {best_params}")
    print(f"Best CV score: {study.best_value:.4f}")
    
    return best_params


def _grid_search_neural_network(x_data, y_data, param_space, cv_folds, max_epochs, 
                               enable_early_stopping, early_stopping_patience, device):
    """Perform grid search optimization for neural network."""
    print("Grid search is computationally expensive for neural networks. Using Random Search instead...")
    return _random_search_neural_network(x_data, y_data, param_space, cv_folds, 
                                       max_epochs, enable_early_stopping, early_stopping_patience, device)


class NeuralNetworkFitter:
    """Multi-Layer Perceptron for concentration prediction using full spectrum."""
    
    @staticmethod
    def fit_neural_networks(x_data_tuple: tuple, y_data_tuple: tuple, 
                            network_configs: List[Dict] = None,
                            optimization_method: str = "Random Search",
                            cv_folds: int = 5,
                            max_epochs: int = 1000,
                            enable_early_stopping: bool = True,
                            early_stopping_patience: int = 20,
                            n_trials: int = 20,
                            run_cv: bool = True) -> Dict[str, Dict]:
        """
        Fit MLP models with various architectures using full spectrum data.
        
        Args:
            x_data_tuple: Input spectral data as tuple for caching (n_samples, n_wavelengths)
            y_data_tuple: Output concentration data as tuple for caching (n_samples,)
            network_configs: List of network configurations to try
            
        Returns:
            Dictionary with network_name -> fit_results mapping
        """
        if not TORCH_AVAILABLE:
            st.warning("PyTorch not available. Neural network models disabled.")
            return {}
        
        # Convert tuples back to arrays for computation
        x_data = np.array(x_data_tuple)  # Full spectrum: (n_samples, n_wavelengths)
        y_data = np.array(y_data_tuple)  # Concentrations: (n_samples,)
        
        # Ensure x_data is 2D
        if x_data.ndim == 1:
            x_data = x_data.reshape(-1, 1)
        
        if len(x_data) < 3:
            st.warning("Neural networks require at least 3 data points")
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
        
        # Standardize data (fit on training, transform both)
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        
        x_train_scaled = scaler_x.fit_transform(X_train_full)
        y_train_scaled = scaler_y.fit_transform(y_train_full.reshape(-1, 1)).ravel()
        
        x_val_scaled = scaler_x.transform(X_val_final)
        y_val_scaled = scaler_y.transform(y_val_final.reshape(-1, 1)).ravel()
        
        # For compatibility, also create full scaled versions
        x_scaled = scaler_x.transform(x_data)
        y_scaled = scaler_y.transform(y_data.reshape(-1, 1)).ravel()
        
        fit_results = {}
        n = len(x_data)
        y_mean = np.mean(y_data)
        ss_tot = np.sum((y_data - y_mean) ** 2)
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use optimization to find best hyperparameters
        if network_configs is None:
            print("No MLP configurations provided. Using hyperparameter optimization...")
            
            # Adaptive parameter space based on dataset size
            n = len(x_data)  # Number of samples
            n_features = x_scaled.shape[1]
            
            # Adaptive hidden layer architectures (scale with features and samples)
            if n_features > 100:
                # Large feature space (spectroscopy): use smaller networks
                hidden_architectures = [(32,), (64,), (32, 16), (64, 32), (32, 32), (64, 64)]
            else:
                # Smaller feature space: can use larger networks
                hidden_architectures = [(50,), (100,), (50, 50), (100, 50), (64, 32)]
            
            # Adaptive batch size based on sample size
            max_batch = min(n // 2, 64)  # At most half the dataset or 64
            if n < 10:
                batch_sizes = [min(2, n), min(4, n)]
            elif n < 20:
                batch_sizes = [4, 8, min(16, n)]
            else:
                batch_sizes = [8, 16, min(32, max_batch), min(64, max_batch)]
            batch_sizes = [b for b in batch_sizes if b > 0 and b <= n]  # Filter valid sizes
            
            # Define parameter space for optimization
            param_space = {
                'hidden_layers': hidden_architectures,
                'activation': ['relu', 'tanh'],  # Removed sigmoid (rarely optimal for regression)
                'dropout_rate': {'low': 0.0, 'high': 0.4},  # Use dict format for continuous
                'learning_rate': {'low': 1e-4, 'high': 1e-2, 'log': True},  # Use dict format
                'batch_size': batch_sizes,
                'weight_decay': {'low': 1e-6, 'high': 1e-3, 'log': True}  # Use dict format
            }
            
            print(f"   MLP optimization: {n} samples, {n_features} features")
            print(f"   Batch sizes: {batch_sizes}")
            print(f"   Hidden architectures: {hidden_architectures}")
            
            # Perform optimization on training data only
            best_params = _optimize_neural_network_params(
                x_train_scaled, y_train_scaled, param_space, optimization_method, cv_folds,
                max_epochs, enable_early_stopping, early_stopping_patience, device, n_trials
            )
            
            # Create single optimized configuration
            network_configs = [{
                'name': 'MLP_Optimized',
                'type': 'pytorch',
                'hidden_layers': best_params['hidden_layers'],
                'activation': best_params['activation'],
                'dropout_rate': best_params['dropout_rate'],
                'learning_rate': best_params['learning_rate'],
                'epochs': max_epochs,
                'batch_size': best_params['batch_size'],
                'weight_decay': best_params['weight_decay']
            }]
        
        for config in network_configs:
            try:
                network_name = config['name']
                network_type = config.get('type', 'pytorch')
                
                if network_type == 'pytorch':
                    # Extract MLP parameters
                    hidden_layers = config.get('hidden_layers', (100,))
                    activation = config.get('activation', 'relu')
                    dropout_rate = config.get('dropout_rate', 0.0)
                    learning_rate = config.get('learning_rate', 0.001)
                    epochs = config.get('epochs', 1000)
                    batch_size = config.get('batch_size', 32)
                    weight_decay = config.get('weight_decay', 0.0001)
                    
                    # Create model
                    input_size = x_scaled.shape[1]
                    model = MLPModel(
                        input_size=input_size,
                        hidden_layers=hidden_layers,
                        activation=activation,
                        dropout_rate=dropout_rate
                    ).to(device)
                    
                    # Convert to PyTorch tensors for training and validation
                    x_train_tensor = torch.FloatTensor(x_train_scaled).to(device)
                    y_train_tensor = torch.FloatTensor(y_train_scaled.reshape(-1, 1)).to(device)
                    x_val_tensor = torch.FloatTensor(x_val_scaled).to(device)
                    y_val_tensor = torch.FloatTensor(y_val_scaled.reshape(-1, 1)).to(device)
                    
                    # Create data loader (training only)
                    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    
                    # Setup training
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
                    
                    # Cross-validation score
                    cv_score = np.nan
                    cv_rmse = np.nan
                    cv_std = np.nan
                    if run_cv:
                        try:
                            from sklearn.model_selection import KFold
                            
                            # Use sklearn KFold for robust cross-validation
                            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                            cv_scores = []
                            cv_mse_scores = []
                            
                            for train_idx, val_idx in kf.split(x_scaled):
                                x_train_fold, x_val_fold = x_scaled[train_idx], x_scaled[val_idx]
                                y_train_fold, y_val_fold = y_scaled[train_idx], y_scaled[val_idx]
                                
                                # Create temporary model for this fold
                                fold_model = MLPModel(
                                    input_size=input_size,
                                    hidden_layers=hidden_layers,
                                    activation=activation,
                                    dropout_rate=dropout_rate
                                ).to(device)
                                
                                fold_optimizer = optim.Adam(fold_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                                
                                # Convert to tensors
                                x_train_tensor = torch.FloatTensor(x_train_fold).to(device)
                                y_train_tensor = torch.FloatTensor(y_train_fold.reshape(-1, 1)).to(device)
                                x_val_tensor = torch.FloatTensor(x_val_fold).to(device)
                                y_val_tensor = torch.FloatTensor(y_val_fold.reshape(-1, 1)).to(device)
                                
                                # Quick training for CV (limited epochs)
                                for _ in range(min(50, max_epochs // 20)):
                                    fold_optimizer.zero_grad()
                                    outputs = fold_model(x_train_tensor)
                                    loss = criterion(outputs, y_train_tensor)
                                    loss.backward()
                                    fold_optimizer.step()
                                
                                # Evaluate on validation set
                                fold_model.eval()
                                with torch.no_grad():
                                    val_pred_scaled = fold_model(x_val_tensor).cpu().numpy().ravel()
                                    val_true_scaled = y_val_fold
                                    
                                    # Calculate R² for this fold
                                    ss_res_fold = np.sum((val_true_scaled - val_pred_scaled) ** 2)
                                    ss_tot_fold = np.sum((val_true_scaled - np.mean(val_true_scaled)) ** 2)
                                    fold_r2 = 1 - (ss_res_fold / ss_tot_fold) if ss_tot_fold > 1e-12 else 0
                                    cv_scores.append(fold_r2)
                                    
                                    # Calculate MSE for RMSE
                                    fold_mse = np.mean((val_true_scaled - val_pred_scaled) ** 2)
                                    cv_mse_scores.append(fold_mse)
                            
                            if cv_scores:
                                cv_score = float(np.mean(cv_scores))
                                cv_std = float(np.std(cv_scores))
                                cv_rmse = float(np.sqrt(np.mean(cv_mse_scores)))
                        except Exception as e:
                            print(f"Neural network CV calculation failed: {e}")
                            cv_score = np.nan
                            cv_rmse = np.nan
                            cv_std = np.nan
                    
                    # Serialize model
                    try:
                        serialization_data = {
                            'state_dict': model.state_dict(),
                            'config': model.config,
                            'scaler_x': scaler_x,
                            'scaler_y': scaler_y,
                            'model_config': config
                        }
                        model_pickle = pickle.dumps(serialization_data)
                        model_serialized = base64.b64encode(model_pickle).decode('utf-8')
                    except Exception as e:
                        st.warning(f"Failed to serialize MLP model '{network_name}': {str(e)}")
                        model_serialized = 'PYTORCH_SERIALIZATION_FAILED'
                    
                    fit_results[network_name] = {
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
                        'max_error': max_error,
                        'cv_score': cv_score,
                        'training_time': training_time,
                        'aic': aic,
                        'bic': bic,
                        'cv_rmse': cv_rmse,
                        'cv_std': cv_std,
                        'residuals': residuals,
                        'model_info': {
                            'type': 'pytorch',
                            'architecture': f"MLP{hidden_layers}",
                            'n_parameters': n_params,
                            'activation': activation,
                            'dropout_rate': dropout_rate,
                            'learning_rate': learning_rate
                        },
                        'model_serialized': model_serialized,
                        'train_losses': train_losses,
                        'val_losses': val_losses
                    }
                
            except Exception as e:
                st.warning(f"Failed to fit neural network '{network_name}': {str(e)}")
                continue
        
        return fit_results
    
    @staticmethod
    def predict_with_neural_network(model_data: Dict, spectrum: np.ndarray) -> float:
        """
        Predict concentration using trained neural network model.
        
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
            
            # Convert to PyTorch tensor
            x_tensor = torch.FloatTensor(x_scaled).to(device)
            
            # Predict concentration
            model.eval()
            with torch.no_grad():
                y_pred_scaled = model(x_tensor).cpu().numpy()
            
            # Inverse transform
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            return float(y_pred[0])
            
        except Exception as e:
            st.error(f"Error predicting with neural network: {str(e)}")
            return 0.0
    
    @staticmethod
    def select_best_neural_network(nn_results: Dict[str, Dict], 
                                  criterion: str = "r_squared") -> Optional[str]:
        """Select best neural network model based on specified criterion."""
        if not nn_results:
            return None
        
        try:
            if criterion == "r_squared":
                return max(nn_results.keys(), 
                          key=lambda k: nn_results[k].get('r_squared', -np.inf))
            elif criterion == "adj_r_squared":
                return max(nn_results.keys(), 
                          key=lambda k: nn_results[k].get('adj_r_squared', -np.inf))
            elif criterion == "rmse":
                return min(nn_results.keys(), 
                          key=lambda k: nn_results[k].get('rmse', np.inf))
            elif criterion == "aic":
                return min(nn_results.keys(), 
                          key=lambda k: nn_results[k].get('aic', np.inf))
            elif criterion == "bic":
                return min(nn_results.keys(), 
                          key=lambda k: nn_results[k].get('bic', np.inf))
            elif criterion == "cv_rmse":
                valid_results = {k: v for k, v in nn_results.items() 
                               if not np.isnan(v['cv_rmse'])}
                if valid_results:
                    return min(valid_results.keys(), 
                              key=lambda k: valid_results[k]['cv_rmse'])
                else:
                    return max(nn_results.keys(), 
                              key=lambda k: nn_results[k].get('r_squared', -np.inf))
            else:
                return max(nn_results.keys(), 
                          key=lambda k: nn_results[k].get('r_squared', -np.inf))
                
        except Exception as e:
            st.error(f"Error selecting best neural network: {str(e)}")
            return list(nn_results.keys())[0] if nn_results else None
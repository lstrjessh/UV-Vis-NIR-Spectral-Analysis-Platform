"""
Neural network model implementations.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import time
import warnings
warnings.filterwarnings('ignore')

from ..core.base_model import BaseModel, ModelResult, ModelConfig
from ..utils.optimization import OptunaOptimizer


class MLPModel(BaseModel):
    """Multi-Layer Perceptron using scikit-learn."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.scaler = StandardScaler()
        
    def _build_model(self, **params) -> MLPRegressor:
        """Build MLP model."""
        return MLPRegressor(
            hidden_layer_sizes=params.get('hidden_layer_sizes', (100,)),
            activation=params.get('activation', 'relu'),
            solver=params.get('solver', 'adam'),
            alpha=params.get('alpha', 0.0001),
            learning_rate=params.get('learning_rate', 'constant'),
            learning_rate_init=params.get('learning_rate_init', 0.001),
            max_iter=params.get('max_iter', 1000),
            early_stopping=self.config.early_stopping,
            validation_fraction=0.1,
            n_iter_no_change=self.config.early_stopping_patience,
            random_state=self.config.random_state
        )
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize MLP hyperparameters."""
        
        X_scaled = self.scaler.fit_transform(X)
        
        optimizer = OptunaOptimizer(
            n_trials=self.config.n_trials,
            random_state=self.config.random_state
        )
        
        def objective(trial):
            # Suggest architecture
            n_layers = trial.suggest_int('n_layers', 1, 3)
            layers = []
            for i in range(n_layers):
                n_units = trial.suggest_int(f'n_units_l{i}', 10, 200)
                layers.append(n_units)
            
            params = {
                'hidden_layer_sizes': tuple(layers),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
                'max_iter': self.config.max_epochs or 1000
            }
            
            model = self._build_model(**params)
            
            try:
                scores = cross_val_score(
                    model, X_scaled, y,
                    cv=min(self.config.cv_folds, len(y) // 2),
                    scoring='r2'
                )
                return np.mean(scores)
            except:
                return -np.inf
        
        return optimizer.optimize(objective, direction='maximize')
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> ModelResult:
        """Train MLP model."""
        start_time = time.time()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Optimize hyperparameters
        if self.config.verbose:
            print(f"Optimizing MLP hyperparameters...")
        
        optimal_params = self._optimize_hyperparameters(X, y)
        
        # Build and train final model
        self.model = self._build_model(**optimal_params)
        self.model.fit(X_scaled, y)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y, y_pred, X)
        
        # Cross-validation
        if self.config.cv_folds > 1:
            cv_scores = cross_val_score(
                self.model, X_scaled, y,
                cv=min(self.config.cv_folds, len(y) // 2),
                scoring='r2'
            )
            metrics.cv_scores = cv_scores.tolist()
            metrics.cv_mean = float(np.mean(cv_scores))
            metrics.cv_std = float(np.std(cv_scores))
        
        metrics.training_time = time.time() - start_time
        metrics.n_iterations = self.model.n_iter_
        
        self.is_fitted = True
        
        return ModelResult(
            model=self.model,
            metrics=metrics,
            config=self.config,
            predictions=y_pred,
            residuals=y - y_pred,
            hyperparameters=optimal_params
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class CNN1DModel(BaseModel):
    """1D Convolutional Neural Network (PyTorch-based if available)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.scaler = StandardScaler()
        self.torch_available = self._check_torch()
        
    def _check_torch(self):
        """Check if PyTorch is available."""
        try:
            import torch
            import torch.nn as nn
            return True
        except ImportError:
            return False
    
    def _build_pytorch_model(self, input_size: int, **params):
        """Build PyTorch CNN model."""
        import torch.nn as nn
        
        class CNN1D(nn.Module):
            def __init__(self, input_size, n_filters, kernel_size, n_dense, dropout_rate):
                super(CNN1D, self).__init__()
                
                # Convolutional layers
                self.conv1 = nn.Conv1d(1, n_filters, kernel_size, padding=kernel_size//2)
                self.bn1 = nn.BatchNorm1d(n_filters)
                self.conv2 = nn.Conv1d(n_filters, n_filters*2, kernel_size, padding=kernel_size//2)
                self.bn2 = nn.BatchNorm1d(n_filters*2)
                
                # Global average pooling
                self.gap = nn.AdaptiveAvgPool1d(1)
                
                # Dense layers
                self.fc1 = nn.Linear(n_filters*2, n_dense)
                self.dropout = nn.Dropout(dropout_rate)
                self.fc2 = nn.Linear(n_dense, 1)
                
                self.relu = nn.ReLU()
                
            def forward(self, x):
                # Add channel dimension
                x = x.unsqueeze(1)
                
                # Convolutional blocks
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.relu(self.bn2(self.conv2(x)))
                
                # Global pooling
                x = self.gap(x)
                x = x.squeeze(-1)
                
                # Dense layers
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                
                return x.squeeze(-1)
        
        return CNN1D(
            input_size,
            params.get('n_filters', 32),
            params.get('kernel_size', 5),
            params.get('n_dense', 64),
            params.get('dropout_rate', 0.2)
        )
    
    def _train_pytorch_model(self, model, X: np.ndarray, y: np.ndarray, **params):
        """Train PyTorch model."""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        batch_size = min(32, len(X) // 2)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001))
        
        # Training loop
        model.train()
        for epoch in range(params.get('n_epochs', 100)):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        return model
    
    def _build_model(self, input_size: int, **params):
        """Build CNN model (PyTorch or fallback to MLP)."""
        if self.torch_available:
            return self._build_pytorch_model(input_size, **params)
        else:
            # Fallback to MLP
            return MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=params.get('alpha', 0.001),
                learning_rate_init=params.get('learning_rate', 0.001),
                max_iter=params.get('n_epochs', 500),
                early_stopping=self.config.early_stopping,
                validation_fraction=0.1,
                n_iter_no_change=self.config.early_stopping_patience,
                random_state=self.config.random_state
            )
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize CNN hyperparameters."""
        
        if not self.torch_available:
            # Fallback to MLP optimization
            optimizer = OptunaOptimizer(
                n_trials=min(self.config.n_trials, 20),
                random_state=self.config.random_state
            )
            
            def objective(trial):
                params = {
                    'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'n_epochs': trial.suggest_int('n_epochs', 200, 1000)
                }
                
                model = self._build_model(X.shape[1], **params)
                X_scaled = self.scaler.fit_transform(X)
                
                try:
                    model.fit(X_scaled, y)
                    y_pred = model.predict(X_scaled)
                    from sklearn.metrics import r2_score
                    return r2_score(y, y_pred)
                except:
                    return -np.inf
            
            return optimizer.optimize(objective, direction='maximize')
        
        # PyTorch optimization
        import torch
        
        optimizer = OptunaOptimizer(
            n_trials=min(self.config.n_trials, 20),
            random_state=self.config.random_state
        )
        
        def objective(trial):
            params = {
                'n_filters': trial.suggest_int('n_filters', 16, 64),
                'kernel_size': trial.suggest_categorical('kernel_size', [3, 5, 7]),
                'n_dense': trial.suggest_int('n_dense', 32, 128),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'n_epochs': trial.suggest_int('n_epochs', 50, 200)
            }
            
            model = self._build_pytorch_model(X.shape[1], **params)
            model = self._train_pytorch_model(model, X, y, **params)
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                y_pred = model(X_tensor).numpy()
                
            from sklearn.metrics import r2_score
            return r2_score(y, y_pred)
        
        return optimizer.optimize(objective, direction='maximize')
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> ModelResult:
        """Train CNN model."""
        start_time = time.time()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Optimize hyperparameters
        if self.config.verbose:
            print(f"Optimizing CNN hyperparameters (PyTorch: {self.torch_available})...")
        
        optimal_params = self._optimize_hyperparameters(X_scaled, y)
        
        # Build and train final model
        if self.torch_available:
            import torch
            
            self.model = self._build_pytorch_model(X_scaled.shape[1], **optimal_params)
            self.model = self._train_pytorch_model(self.model, X_scaled, y, **optimal_params)
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                y_pred = self.model(X_tensor).numpy()
        else:
            self.model = self._build_model(X_scaled.shape[1], **optimal_params)
            self.model.fit(X_scaled, y)
            y_pred = self.model.predict(X_scaled)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y, y_pred, X)
        metrics.training_time = time.time() - start_time
        
        self.is_fitted = True
        
        return ModelResult(
            model=self.model,
            metrics=metrics,
            config=self.config,
            predictions=y_pred,
            residuals=y - y_pred,
            hyperparameters=optimal_params
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        if self.torch_available:
            import torch
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                return self.model(X_tensor).numpy()
        else:
            return self.model.predict(X_scaled)

"""
Neural network model implementations.
"""

import numpy as np
from typing import Dict, Any
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

from ..core.base_model import BaseModel, ModelResult, ModelConfig
from ..utils.optimization import OptunaOptimizer, RandomSearchOptimizer


class MLPModel(BaseModel):
    """Multi-Layer Perceptron implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
    def _build_model(self, **params) -> MLPRegressor:
        """Build MLP model."""
        return MLPRegressor(
            hidden_layer_sizes=params.get('hidden_layer_sizes', (100,)),
            activation=params.get('activation', 'relu'),
            solver=params.get('solver', 'adam'),
            alpha=params.get('alpha', 0.0001),
            batch_size=params.get('batch_size', 'auto'),
            learning_rate=params.get('learning_rate', 'constant'),
            learning_rate_init=params.get('learning_rate_init', 0.001),
            max_iter=params.get('max_iter', 200),
            shuffle=params.get('shuffle', True),
            random_state=self.config.random_state,
            early_stopping=params.get('early_stopping', False),
            validation_fraction=params.get('validation_fraction', 0.1),
            n_iter_no_change=params.get('n_iter_no_change', 10),
            tol=params.get('tol', 1e-4)
        )
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize MLP hyperparameters."""
        
        # Scale data for optimization
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        optimizer = OptunaOptimizer(
            n_trials=self.config.n_trials,
            random_state=self.config.random_state
        )
        
        def objective(trial):
            # Suggest hidden layer sizes
            n_layers = trial.suggest_int('n_layers', 1, 3)
            hidden_layer_sizes = []
            for i in range(n_layers):
                size = trial.suggest_int(f'layer_{i}_size', 10, 200)
                hidden_layer_sizes.append(size)
            
            params = {
                'hidden_layer_sizes': tuple(hidden_layer_sizes),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
                'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs', 'sgd']),
                'alpha': trial.suggest_float('alpha', 1e-6, 1e-2, log=True),
                'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-5, 1e-2, log=True),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
                'early_stopping': trial.suggest_categorical('early_stopping', [True, False]),
                'validation_fraction': trial.suggest_float('validation_fraction', 0.1, 0.3),
                'n_iter_no_change': trial.suggest_int('n_iter_no_change', 5, 20),
                'tol': trial.suggest_float('tol', 1e-6, 1e-3, log=True)
            }
            
            model = self._build_model(**params)
            
            # Use cross-validation for proper evaluation
            n_folds = min(self.config.cv_folds, len(y) // 2)
            if n_folds < 2:
                n_folds = 2
            
            try:
                cv_scores = cross_val_score(
                    model, X_scaled, y_scaled,
                    cv=n_folds,
                    scoring='r2',
                    n_jobs=1
                )
                score = np.mean(cv_scores)
                if np.isnan(score) or np.isinf(score):
                    return -999.0
                return score
            except Exception as e:
                print(f"MLP CV failed: {e}")
                return -999.0
        
        try:
            best_params = optimizer.optimize(objective, direction='maximize')
            return best_params
        except Exception as e:
            print(f"Optimization failed: {str(e)}. Using default parameters.")
            return {
                'hidden_layer_sizes': (100,),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate': 'constant',
                'learning_rate_init': 0.001,
                'max_iter': 200,
                'early_stopping': False,
                'validation_fraction': 0.1,
                'n_iter_no_change': 10,
                'tol': 1e-4
            }
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> ModelResult:
        """Train MLP model."""
        start_time = time.time()
        
        # Optimize hyperparameters
        if self.config.verbose:
            print("Optimizing MLP hyperparameters...")
        
        optimal_params = self._optimize_hyperparameters(X, y)
        
        # Scale data for training
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Build and train final model
        self.model = self._build_model(**optimal_params)
        self.model.fit(X_scaled, y_scaled)
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        # Calculate metrics using original scale
        metrics = self.calculate_metrics(y, y_pred, X)
        
        # Cross-validation
        if self.config.cv_folds > 1:
            n_folds = min(self.config.cv_folds, len(y) // 2)
            if n_folds < 2:
                n_folds = 2
            
            try:
                cv_model = self._build_model(**optimal_params)
                cv_scores = cross_val_score(
                    cv_model, X_scaled, y_scaled,
                    cv=n_folds,
                    scoring='r2',
                    n_jobs=1
                )
                if not np.any(np.isnan(cv_scores)):
                    metrics.cv_scores = cv_scores.tolist()
                    metrics.cv_mean = float(np.mean(cv_scores))
                    metrics.cv_std = float(np.std(cv_scores))
                else:
                    metrics.cv_scores = None
                    metrics.cv_mean = None
                    metrics.cv_std = None
            except Exception:
                metrics.cv_scores = None
                metrics.cv_mean = None
                metrics.cv_std = None
        
        metrics.training_time = time.time() - start_time
        
        # Feature importance (not directly available for MLP, use coefficients if available)
        feature_importance = {}
        if hasattr(self.model, 'coefs_') and self.model.coefs_:
            # Use first layer weights as feature importance
            first_layer_weights = np.abs(self.model.coefs_[0])
            feature_importance = {
                f'feature_{i}': float(np.mean(first_layer_weights[:, i]))
                for i in range(first_layer_weights.shape[1])
            }
        
        self.is_fitted = True
        
        return ModelResult(
            model=self.model,
            metrics=metrics,
            config=self.config,
            predictions=y_pred,
            residuals=y - y_pred,
            feature_importance=feature_importance,
            hyperparameters=optimal_params
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale input data before prediction
        X_scaled = self.scaler.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        # Inverse transform predictions back to original scale
        return self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

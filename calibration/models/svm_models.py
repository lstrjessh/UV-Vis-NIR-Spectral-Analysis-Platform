"""
Support Vector Machine model implementations.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import time
import warnings
warnings.filterwarnings('ignore')

from ..core.base_model import BaseModel, ModelResult, ModelConfig
from ..utils.optimization import OptunaOptimizer, RandomSearchOptimizer


class SVRModel(BaseModel):
    """Support Vector Regression with optimized implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
    def _build_model(self, **params) -> SVR:
        """Build SVR model."""
        return SVR(
            kernel=params.get('kernel', 'rbf'),
            C=params.get('C', 1.0),
            epsilon=params.get('epsilon', 0.1),
            gamma=params.get('gamma', 'scale'),
            degree=params.get('degree', 3),
            coef0=params.get('coef0', 0.0),
            shrinking=params.get('shrinking', True),
            cache_size=200,
            max_iter=-1
        )
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize SVR hyperparameters."""
        
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        if self.config.optimization_method == 'random_search':
            optimizer = RandomSearchOptimizer(
                n_trials=self.config.n_trials,
                random_state=self.config.random_state
            )
            
            kernel_types = self.config.hyperparameters.get('kernel_types', ['rbf', 'linear'])
            
            search_space = {
                'kernel': kernel_types,
                'C': (0.01, 100),
                'epsilon': (0.001, 1.0),
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
            
            def objective(params):
                # Add kernel-specific parameters
                if params['kernel'] == 'poly':
                    params['degree'] = np.random.randint(2, 5)
                    params['coef0'] = np.random.uniform(0, 1)
                
                model = self._build_model(**params)
                
                try:
                    scores = cross_val_score(
                        model, X_scaled, y_scaled,
                        cv=min(self.config.cv_folds, len(y) // 2),
                        scoring='r2'
                    )
                    return np.mean(scores)
                except:
                    return -np.inf
            
            return optimizer.optimize(objective, search_space, direction='maximize')
        
        else:
            # Use Optuna
            optimizer = OptunaOptimizer(
                n_trials=self.config.n_trials,
                random_state=self.config.random_state
            )
            
            kernel_types = self.config.hyperparameters.get('kernel_types', ['rbf', 'linear'])
            
            def objective(trial):
                kernel = trial.suggest_categorical('kernel', kernel_types)
                
                params = {
                    'kernel': kernel,
                    'C': trial.suggest_float('C', 0.01, 100, log=True),
                    'epsilon': trial.suggest_float('epsilon', 0.001, 1.0, log=True)
                }
                
                # Kernel-specific parameters
                if kernel == 'rbf':
                    params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto']) \
                                    if trial.suggest_categorical('gamma_type', ['string', 'float']) == 'string' \
                                    else trial.suggest_float('gamma_value', 1e-4, 1, log=True)
                elif kernel == 'poly':
                    params['degree'] = trial.suggest_int('degree', 2, 5)
                    params['coef0'] = trial.suggest_float('coef0', 0, 1)
                    params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
                elif kernel == 'sigmoid':
                    params['coef0'] = trial.suggest_float('coef0', -1, 1)
                    params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
                
                model = self._build_model(**params)
                
                try:
                    scores = cross_val_score(
                        model, X_scaled, y_scaled,
                        cv=min(self.config.cv_folds, len(y) // 2),
                        scoring='r2'
                    )
                    return np.mean(scores)
                except:
                    return -np.inf
            
            return optimizer.optimize(objective, direction='maximize')
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> ModelResult:
        """Train SVR model."""
        start_time = time.time()
        
        # Scale features and targets
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Optimize hyperparameters
        if self.config.verbose:
            print(f"Optimizing SVR hyperparameters...")
        
        optimal_params = self._optimize_hyperparameters(X, y)
        
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
            cv_scores = cross_val_score(
                self.model, X_scaled, y_scaled,
                cv=min(self.config.cv_folds, len(y) // 2),
                scoring='r2'
            )
            metrics.cv_scores = cv_scores.tolist()
            metrics.cv_mean = float(np.mean(cv_scores))
            metrics.cv_std = float(np.std(cv_scores))
        
        metrics.training_time = time.time() - start_time
        
        # Support vectors info
        if hasattr(self.model, 'support_vectors_'):
            metrics.n_support_vectors = len(self.model.support_vectors_)
            metrics.support_vector_ratio = metrics.n_support_vectors / len(X)
        
        self.is_fitted = True
        
        if self.config.verbose:
            print(f"SVR with {optimal_params['kernel']} kernel: "
                  f"{metrics.n_support_vectors}/{len(X)} support vectors")
        
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
        y_pred_scaled = self.model.predict(X_scaled)
        # Inverse transform predictions back to original scale
        return self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

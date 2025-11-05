"""
Neural network model implementations with pipeline support to prevent data leakage.
"""

import numpy as np
from typing import Dict, Any
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from ..core.base_model import BaseModel, ModelResult, ModelConfig
from .pipeline_models import PipelineBaseModel


class MLPModel(PipelineBaseModel):
    """Multi-Layer Perceptron with pipeline to prevent data leakage."""
    
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
    
    def _create_pipeline(self, 
                        scaler_type: str = 'standard',
                        **model_params) -> TransformedTargetRegressor:
        """
        Create sklearn pipeline with proper scaling, wrapped in TransformedTargetRegressor
        for y-scaling to prevent data leakage.
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            **model_params: Parameters for the final model
            
        Returns:
            TransformedTargetRegressor wrapping the X-scaling pipeline
        """
        # Create base pipeline with X-scaling
        base_pipeline = super()._create_pipeline(scaler_type, **model_params)
        
        # Wrap in TransformedTargetRegressor for y-scaling
        return TransformedTargetRegressor(
            regressor=base_pipeline,
            transformer=StandardScaler(),
            check_inverse=True
        )
    
    def _get_param_grid(self):
        """Get parameter grid for MLP."""
        # Simple grid for grid search (if used)
        return [
            {'hidden_layer_sizes': (50,), 'activation': 'relu', 'solver': 'adam'},
            {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam'},
            {'hidden_layer_sizes': (100, 50), 'activation': 'relu', 'solver': 'adam'},
        ]
    
    def _suggest_hyperparameters(self, trial):
        """Suggest hyperparameters for MLP using Optuna."""
        # Suggest hidden layer sizes
        n_layers = trial.suggest_int('n_layers', 1, 3)
        hidden_layer_sizes = []
        for i in range(n_layers):
            size = trial.suggest_int(f'layer_{i}_size', 10, 200)
            hidden_layer_sizes.append(size)
        
        return {
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
    
    def _get_default_params(self):
        """Get default parameters for MLP."""
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
    
    def _validate_components(self, X: np.ndarray, n_components: int) -> int:
        """Validate components (MLP doesn't use components, but method exists for compatibility)."""
        return n_components
    
    def _extract_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Extract MLP feature importance using first layer weights."""
        # Access the wrapped pipeline's model
        if hasattr(self.pipeline, 'regressor_'):
            inner_pipeline = self.pipeline.regressor_
            if hasattr(inner_pipeline, 'named_steps') and 'model' in inner_pipeline.named_steps:
                model = inner_pipeline.named_steps['model']
                if hasattr(model, 'coefs_') and model.coefs_:
                    # Use first layer weights as feature importance
                    # Shape of coefs_[0]: (n_features, n_hidden_units)
                    first_layer_weights = np.abs(model.coefs_[0])
                    # Average across all hidden neurons for each input feature
                    return {
                        f'feature_{i}': float(np.mean(first_layer_weights[i, :]))
                        for i in range(first_layer_weights.shape[0])
                    }
        return {}

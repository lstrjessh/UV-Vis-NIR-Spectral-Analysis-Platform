"""
Support Vector Machine model implementations with pipeline support to prevent data leakage.
"""

import numpy as np
from typing import Dict, Any
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from ..core.base_model import BaseModel, ModelResult, ModelConfig
from .pipeline_models import PipelineBaseModel


class SVRModel(PipelineBaseModel):
    """Support Vector Regression with pipeline to prevent data leakage."""
    
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
        """Get parameter grid for SVR."""
        # Simple grid for grid search (if used)
        return [
            {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1},
            {'kernel': 'rbf', 'C': 10.0, 'epsilon': 0.1},
            {'kernel': 'linear', 'C': 1.0, 'epsilon': 0.1},
        ]
    
    def _suggest_hyperparameters(self, trial):
        """Suggest hyperparameters for SVR using Optuna."""
        kernel_types = self.config.hyperparameters.get('kernel_types', ['rbf', 'linear'])
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
        
        return params
    
    def _get_default_params(self):
        """Get default parameters for SVR."""
        return {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1,
            'gamma': 'scale'
        }
    
    def _validate_components(self, X: np.ndarray, n_components: int) -> int:
        """Validate components (SVR doesn't use components, but method exists for compatibility)."""
        return n_components
    
    def _extract_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Extract SVR feature importance (not directly available, return empty dict)."""
        # SVR doesn't provide direct feature importance
        # Could potentially use support vectors or coefficients for linear kernel
        if hasattr(self.pipeline, 'regressor_'):
            inner_pipeline = self.pipeline.regressor_
            if hasattr(inner_pipeline, 'named_steps') and 'model' in inner_pipeline.named_steps:
                model = inner_pipeline.named_steps['model']
                # For linear kernel, can use coefficients
                if hasattr(model, 'coef_') and model.kernel == 'linear':
                    coef = model.coef_[0]
                    return {
                        f'feature_{i}': float(np.abs(coef[i]))
                        for i in range(len(coef))
                    }
        return {}
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> ModelResult:
        """Train SVR pipeline model with support vector tracking."""
        # Call parent fit
        result = super().fit(X, y, **kwargs)
        
        # Add support vectors info if available
        if hasattr(self.pipeline, 'regressor_'):
            inner_pipeline = self.pipeline.regressor_
            if hasattr(inner_pipeline, 'named_steps') and 'model' in inner_pipeline.named_steps:
                model = inner_pipeline.named_steps['model']
                if hasattr(model, 'support_vectors_'):
                    result.metrics.n_support_vectors = len(model.support_vectors_)
                    result.metrics.support_vector_ratio = result.metrics.n_support_vectors / len(X)
                    
                    if self.config.verbose:
                        kernel_type = getattr(model, 'kernel', 'unknown')
                        print(f"SVR with {kernel_type} kernel: "
                              f"{result.metrics.n_support_vectors}/{len(X)} support vectors")
        
        return result

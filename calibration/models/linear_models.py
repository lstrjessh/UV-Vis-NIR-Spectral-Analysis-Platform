"""
Linear model implementations with pipeline support to prevent data leakage.
"""

import numpy as np
from typing import Dict, Any
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet

from ..core.base_model import BaseModel, ModelResult, ModelConfig
from .pipeline_models import PipelineBaseModel


class PLSRModel(PipelineBaseModel):
    """Partial Least Squares Regression with pipeline to prevent data leakage."""
    
    def _build_model(self, n_components: int = 2) -> PLSRegression:
        """Build PLSR model."""
        return PLSRegression(
            n_components=n_components,
            scale=False,  # Scaling handled by pipeline
            max_iter=500,
            tol=1e-6
        )
    
    def _get_param_grid(self):
        """Get parameter grid for PLSR."""
        max_comp = min(10, self.config.hyperparameters.get('max_components', 10))
        return [{'n_components': n} for n in range(1, max_comp + 1)]
    
    def _validate_components(self, X: np.ndarray, n_components: int) -> int:
        """Validate and adjust number of components based on data dimensions."""
        n_samples, n_features = X.shape
        # PLSR components cannot exceed min(n_samples, n_features)
        max_components = min(n_samples, n_features)
        return min(n_components, max_components)
    
    def _suggest_hyperparameters(self, trial):
        """Suggest hyperparameters for PLSR."""
        max_comp = min(10, self.config.hyperparameters.get('max_components', 10))
        return {
            'n_components': trial.suggest_int('n_components', 1, max_comp)
        }
    
    def _get_default_params(self):
        """Get default parameters for PLSR."""
        return {'n_components': 2}
    
    def _extract_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Extract PLSR loadings as feature importance."""
        if hasattr(self.pipeline.named_steps['model'], 'coef_'):
            coef = self.pipeline.named_steps['model'].coef_[0]
            return {
                f'feature_{i}': float(np.abs(coef[i]))
                for i in range(len(coef))
            }
        return {}


class RidgeModel(PipelineBaseModel):
    """Ridge regression with pipeline to prevent data leakage."""
    
    def _build_model(self, alpha: float = 1.0) -> Ridge:
        """Build Ridge model."""
        return Ridge(
            alpha=alpha,
            fit_intercept=True,
            max_iter=1000,
            tol=0.001,
            solver='auto',
            random_state=self.config.random_state
        )
    
    def _get_param_grid(self):
        """Get parameter grid for Ridge."""
        return [{'alpha': alpha} for alpha in np.logspace(-3, 3, 20)]
    
    def _suggest_hyperparameters(self, trial):
        """Suggest hyperparameters for Ridge."""
        return {
            'alpha': trial.suggest_float('alpha', 1e-3, 1000, log=True)
        }
    
    def _get_default_params(self):
        """Get default parameters for Ridge."""
        return {'alpha': 1.0}
    
    def _validate_components(self, X: np.ndarray, n_components: int) -> int:
        """Validate and adjust number of components based on data dimensions."""
        # Ridge doesn't use components, but we can add validation for other parameters
        return n_components
    
    def _extract_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Extract Ridge coefficients as feature importance."""
        if hasattr(self.pipeline.named_steps['model'], 'coef_'):
            coef = self.pipeline.named_steps['model'].coef_
            return {
                f'feature_{i}': float(np.abs(coef[i]))
                for i in range(len(coef))
            }
        return {}


class LassoModel(PipelineBaseModel):
    """Lasso regression with pipeline to prevent data leakage."""
    
    def _build_model(self, alpha: float = 1.0) -> Lasso:
        """Build Lasso model."""
        return Lasso(
            alpha=alpha,
            fit_intercept=True,
            max_iter=2000,
            tol=0.001,
            warm_start=False,
            selection='cyclic',
            random_state=self.config.random_state
        )
    
    def _get_param_grid(self):
        """Get parameter grid for Lasso."""
        return [{'alpha': alpha} for alpha in np.logspace(-4, 1, 20)]
    
    def _suggest_hyperparameters(self, trial):
        """Suggest hyperparameters for Lasso."""
        return {
            'alpha': trial.suggest_float('alpha', 1e-4, 10, log=True)
        }
    
    def _get_default_params(self):
        """Get default parameters for Lasso."""
        return {'alpha': 0.1}
    
    def _validate_components(self, X: np.ndarray, n_components: int) -> int:
        """Validate and adjust number of components based on data dimensions."""
        # Lasso doesn't use components, but we can add validation for other parameters
        return n_components
    
    def _extract_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Extract Lasso coefficients as feature importance."""
        if hasattr(self.pipeline.named_steps['model'], 'coef_'):
            coef = self.pipeline.named_steps['model'].coef_
            return {
                f'feature_{i}': float(np.abs(coef[i]))
                for i in range(len(coef))
                if coef[i] != 0
            }
        return {}


class ElasticNetModel(PipelineBaseModel):
    """Elastic Net with pipeline to prevent data leakage."""
    
    def _build_model(self, **params) -> ElasticNet:
        """Build Elastic Net model."""
        return ElasticNet(
            alpha=params.get('alpha', 1.0),
            l1_ratio=params.get('l1_ratio', 0.5),
            max_iter=params.get('max_iter', 1000),
            tol=params.get('tol', 0.001),
            random_state=self.config.random_state
        )
    
    def _get_param_grid(self):
        """Get parameter grid for Elastic Net."""
        return [
            {'alpha': alpha, 'l1_ratio': l1_ratio}
            for alpha in np.logspace(-3, 1, 10)
            for l1_ratio in np.linspace(0.1, 0.9, 5)
        ]
    
    def _suggest_hyperparameters(self, trial):
        """Suggest hyperparameters for Elastic Net."""
        return {
            'alpha': trial.suggest_float('alpha', 0.001, 10.0, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
            'max_iter': trial.suggest_int('max_iter', 1000, 5000),
            'tol': trial.suggest_float('tol', 1e-5, 1e-2, log=True)
        }
    
    def _get_default_params(self):
        """Get default parameters for Elastic Net."""
        return {
            'alpha': 1.0,
            'l1_ratio': 0.5,
            'max_iter': 1000,
            'tol': 0.001
        }
    
    def _validate_components(self, X: np.ndarray, n_components: int) -> int:
        """Validate and adjust number of components based on data dimensions."""
        # Elastic Net doesn't use components, but we can add validation for other parameters
        return n_components
    
    def _extract_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Extract Elastic Net coefficients as feature importance."""
        if hasattr(self.pipeline.named_steps['model'], 'coef_'):
            coef = self.pipeline.named_steps['model'].coef_
            return {
                f'feature_{i}': float(np.abs(coef[i]))
                for i in range(len(coef))
            }
        return {}

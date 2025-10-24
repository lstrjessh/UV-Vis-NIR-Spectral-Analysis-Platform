"""
Pipeline-based model implementations to prevent data leakage.

This module implements models using sklearn Pipeline to ensure proper
scaling and dimensionality reduction without data leakage.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score
import time

from ..core.base_model import BaseModel, ModelResult, ModelConfig
from ..utils.optimization import OptunaOptimizer, GridSearchOptimizer
from sklearn.base import BaseEstimator, TransformerMixin


class SNVTransformer(BaseEstimator, TransformerMixin):
    """Standard Normal Variate transformer for spectral data."""
    
    def fit(self, X, y=None):
        """Fit the transformer (no-op for SNV)."""
        return self
    
    def transform(self, X):
        """Apply SNV transformation to each sample."""
        X_transformed = np.zeros_like(X)
        for i in range(X.shape[0]):
            sample = X[i]
            mean = np.mean(sample)
            std = np.std(sample)
            if std > 0:
                X_transformed[i] = (sample - mean) / std
            else:
                X_transformed[i] = sample - mean
        return X_transformed


class PipelineBaseModel(BaseModel):
    """Base class for pipeline-based models that prevent data leakage."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.pipeline = None
    
    def _build_model(self, **kwargs):
        """Build the underlying model object."""
        raise NotImplementedError("Subclasses must implement _build_model")
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize model hyperparameters."""
        # Use the pipeline optimization instead
        return self._optimize_pipeline_hyperparameters(X, y)
        
    def _create_pipeline(self, 
                        scaler_type: str = 'standard',
                        **model_params) -> Pipeline:
        """
        Create sklearn pipeline with proper scaling.
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            **model_params: Parameters for the final model
            
        Returns:
            Configured sklearn Pipeline
        """
        steps = []
        
        # Add scaler
        if scaler_type == 'standard':
            steps.append(('scaler', StandardScaler()))
        elif scaler_type == 'minmax':
            steps.append(('scaler', MinMaxScaler()))
        elif scaler_type == 'robust':
            steps.append(('scaler', RobustScaler()))
        elif scaler_type == 'snv':
            steps.append(('snv', SNVTransformer()))
        
        # No UMAP - removed for simplicity
        
        # Add the final model
        model = self._build_model(**model_params)
        steps.append(('model', model))
        
        return Pipeline(steps)
    
    def _optimize_pipeline_hyperparameters(self, 
                                          X: np.ndarray, 
                                          y: np.ndarray,
                                          scaler_type: str = 'standard') -> Dict[str, Any]:
        """Optimize pipeline hyperparameters."""
        # Adjust CV folds based on data size
        n_folds = min(self.config.cv_folds, len(y) // 2)
        if n_folds < 2:
            n_folds = 2
        
        if self.config.optimization_method == 'grid_search':
            return self._grid_search_optimization(X, y, scaler_type, n_folds)
        else:
            return self._optuna_optimization(X, y, scaler_type, n_folds)
    
    def _grid_search_optimization(self, X, y, scaler_type, n_folds):
        """Grid search optimization for pipeline."""
        # This is a simplified version - can be expanded based on model needs
        best_score = -np.inf
        best_params = {}
        
        # Simple parameter grid (can be expanded)
        param_grid = self._get_param_grid()
        
        for params in param_grid:
            try:
                # Validate components if this is a PLSR model
                if hasattr(self, '_validate_components') and 'n_components' in params:
                    params['n_components'] = self._validate_components(X, params['n_components'])
                
                pipeline = self._create_pipeline(
                    scaler_type=scaler_type,
                    **params
                )
                
                scores = cross_val_score(
                    pipeline, X, y,
                    cv=n_folds,
                    scoring='r2'
                )
                score = np.mean(scores)
                
                if not np.isnan(score) and score > best_score:
                    best_score = score
                    best_params = params
            except Exception as e:
                print(f"Grid search failed for params {params}: {e}")
                continue
        
        return best_params
    
    def _optuna_optimization(self, X, y, scaler_type, n_folds):
        """Optuna optimization for pipeline."""
        optimizer = OptunaOptimizer(
            n_trials=self.config.n_trials,
            random_state=self.config.random_state
        )
        
        def objective(trial):
            params = self._suggest_hyperparameters(trial)
            
            try:
                # Validate components if this is a PLSR model
                if hasattr(self, '_validate_components') and 'n_components' in params:
                    params['n_components'] = self._validate_components(X, params['n_components'])
                
                pipeline = self._create_pipeline(
                    scaler_type=scaler_type,
                    **params
                )
                
                scores = cross_val_score(
                    pipeline, X, y,
                    cv=n_folds,
                    scoring='r2'
                )
                score = np.mean(scores)
                
                if np.isnan(score) or np.isinf(score):
                    return -999.0
                return score
            except Exception as e:
                print(f"Optuna trial failed for params {params}: {e}")
                return -999.0
        
        try:
            return optimizer.optimize(objective, direction='maximize')
        except Exception as e:
            print(f"Optimization failed: {str(e)}. Using default parameters.")
            return self._get_default_params()
    
    def _get_param_grid(self):
        """Get parameter grid for grid search. Override in subclasses."""
        return [{}]
    
    def _suggest_hyperparameters(self, trial):
        """Suggest hyperparameters for Optuna. Override in subclasses."""
        return {}
    
    def _get_default_params(self):
        """Get default parameters. Override in subclasses."""
        return {}
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> ModelResult:
        """Train pipeline model."""
        start_time = time.time()
        
        # Get optimization parameters
        scaler_type = kwargs.get('scaler_type', 'standard')
        
        # Optimize hyperparameters
        if self.config.verbose:
            print(f"Optimizing {self.__class__.__name__} pipeline hyperparameters...")
        
        optimal_params = self._optimize_pipeline_hyperparameters(
            X, y, scaler_type
        )
        
        # Validate components before creating final pipeline
        if hasattr(self, '_validate_components') and 'n_components' in optimal_params:
            optimal_params['n_components'] = self._validate_components(X, optimal_params['n_components'])
        
        # Create and train final pipeline
        self.pipeline = self._create_pipeline(
            scaler_type=scaler_type,
            **optimal_params
        )
        
        self.pipeline.fit(X, y)
        
        # No UMAP embeddings - removed for simplicity
        
        # Make predictions
        y_pred = self.pipeline.predict(X)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y, y_pred, X)
        
        # Cross-validation
        if self.config.cv_folds > 1:
            n_folds = min(self.config.cv_folds, len(y) // 2)
            if n_folds < 2:
                n_folds = 2
            
            try:
                cv_scores = cross_val_score(
                    self.pipeline, X, y,
                    cv=n_folds,
                    scoring='r2'
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
        
        # Feature importance (if available)
        feature_importance = self._extract_feature_importance(X)
        
        self.is_fitted = True
        
        return ModelResult(
            model=self.pipeline,
            metrics=metrics,
            config=self.config,
            predictions=y_pred,
            residuals=y - y_pred,
            feature_importance=feature_importance,
            hyperparameters=optimal_params
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using pipeline."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.pipeline.predict(X).ravel()
    
    def _extract_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Extract feature importance from pipeline. Override in subclasses."""
        return {}
    
    # UMAP methods removed for simplicity


class PipelinePLSRModel(PipelineBaseModel):
    """PLSR with pipeline to prevent data leakage."""
    
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


class PipelineRidgeModel(PipelineBaseModel):
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


class PipelineLassoModel(PipelineBaseModel):
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


class PipelineElasticNetModel(PipelineBaseModel):
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

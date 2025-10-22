"""
Optimized linear model implementations.
"""

import numpy as np
from typing import Dict, Any
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import time

from ..core.base_model import BaseModel, ModelResult, ModelConfig
from ..utils.optimization import OptunaOptimizer, GridSearchOptimizer


class PLSRModel(BaseModel):
    """Partial Least Squares Regression with optimized implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.max_components = config.hyperparameters.get('max_components', 10)
        
    def _build_model(self, n_components: int = 2) -> PLSRegression:
        """Build PLSR model."""
        return PLSRegression(
            n_components=n_components,
            scale=True,
            max_iter=500,
            tol=1e-6
        )
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize number of components."""
        # Limit components to data dimensions
        max_comp = min(self.max_components, X.shape[0] - 1, X.shape[1])
        
        # Ensure we have at least 1 component
        if max_comp < 1:
            max_comp = 1
        
        # Adjust CV folds based on data size
        n_folds = min(self.config.cv_folds, len(y) // 2)
        if n_folds < 2:
            n_folds = 2
        
        if self.config.optimization_method == 'grid_search':
            # Grid search for single parameter
            best_score = -np.inf
            best_n = min(2, max_comp)
            
            for n in range(1, max_comp + 1):
                try:
                    model = self._build_model(n_components=n)
                    scores = cross_val_score(
                        model, X, y, 
                        cv=n_folds,
                        scoring='r2'
                    )
                    score = np.mean(scores)
                    
                    # Check for NaN scores
                    if not np.isnan(score) and score > best_score:
                        best_score = score
                        best_n = n
                except Exception:
                    continue
            
            return {'n_components': best_n}
        
        else:
            # Use Optuna for more sophisticated optimization
            optimizer = OptunaOptimizer(
                n_trials=self.config.n_trials,
                random_state=self.config.random_state
            )
            
            def objective(trial):
                n_components = trial.suggest_int('n_components', 1, max_comp)
                model = self._build_model(n_components=n_components)
                
                try:
                    scores = cross_val_score(
                        model, X, y,
                        cv=n_folds,
                        scoring='r2'
                    )
                    score = np.mean(scores)
                    
                    # Check for NaN and return very poor score if found
                    if np.isnan(score) or np.isinf(score):
                        return -999.0
                    return score
                except Exception:
                    return -999.0
            
            try:
                best_params = optimizer.optimize(objective, direction='maximize')
                # Validate the returned parameters
                if 'n_components' not in best_params or best_params['n_components'] < 1:
                    best_params = {'n_components': min(2, max_comp)}
                return best_params
            except Exception as e:
                # Fallback to default parameters if optimization fails
                print(f"Optimization failed: {str(e)}. Using default parameters.")
                return {'n_components': min(2, max_comp)}
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> ModelResult:
        """Train PLSR model."""
        start_time = time.time()
        
        # Optimize hyperparameters
        if self.config.verbose:
            print("Optimizing PLSR hyperparameters...")
        
        optimal_params = self._optimize_hyperparameters(X, y)
        
        # Build and train final model
        self.model = self._build_model(**optimal_params)
        self.model.fit(X, y)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y, y_pred, X)
        
        # Cross-validation if enabled
        if self.config.cv_folds > 1:
            # Adjust CV folds based on data size
            n_folds = min(self.config.cv_folds, len(y) // 2)
            if n_folds < 2:
                n_folds = 2
            
            try:
                cv_scores = cross_val_score(
                    self.model, X, y,
                    cv=n_folds,
                    scoring='r2'
                )
                # Check for NaN scores
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
        
        # Training time
        metrics.training_time = time.time() - start_time
        
        # Feature importance (loadings)
        feature_importance = {
            f'feature_{i}': float(np.abs(self.model.coef_[0, i]))
            for i in range(X.shape[1])
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
        
        return self.model.predict(X).ravel()


class RidgeModel(BaseModel):
    """Ridge Regression with L2 regularization."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.scaler = StandardScaler()
    
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
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize alpha parameter."""
        # Adjust CV folds based on data size
        n_folds = min(self.config.cv_folds, len(y) // 2)
        if n_folds < 2:
            n_folds = 2
        
        # Scale data once
        X_scaled = self.scaler.fit_transform(X)
        
        if self.config.optimization_method == 'grid_search':
            optimizer = GridSearchOptimizer()
            search_space = {
                'alpha': np.logspace(-3, 3, 50)
            }
            
            def grid_objective(params):
                try:
                    model = self._build_model(**params)
                    scores = cross_val_score(
                        model, X_scaled, y,
                        cv=n_folds,
                        scoring='r2'
                    )
                    score = np.mean(scores)
                    if np.isnan(score) or np.isinf(score):
                        return -999.0
                    return score
                except Exception:
                    return -999.0
            
            return optimizer.optimize(grid_objective, search_space)
        else:
            optimizer = OptunaOptimizer(
                n_trials=self.config.n_trials,
                random_state=self.config.random_state
            )
            
            def objective(trial):
                alpha = trial.suggest_float('alpha', 1e-3, 1000, log=True)
                model = self._build_model(alpha=alpha)
                
                try:
                    scores = cross_val_score(
                        model, X_scaled, y,
                        cv=n_folds,
                        scoring='r2'
                    )
                    score = np.mean(scores)
                    
                    # Check for NaN/inf
                    if np.isnan(score) or np.isinf(score):
                        return -999.0
                    return score
                except Exception:
                    return -999.0
            
            try:
                return optimizer.optimize(objective, direction='maximize')
            except Exception as e:
                print(f"Optimization failed: {str(e)}. Using default parameters.")
                return {'alpha': 1.0}
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> ModelResult:
        """Train Ridge model."""
        start_time = time.time()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Optimize hyperparameters
        if self.config.verbose:
            print("Optimizing Ridge hyperparameters...")
        
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
            # Adjust CV folds based on data size
            n_folds = min(self.config.cv_folds, len(y) // 2)
            if n_folds < 2:
                n_folds = 2
            
            try:
                cv_scores = cross_val_score(
                    self.model, X_scaled, y,
                    cv=n_folds,
                    scoring='r2'
                )
                # Check for NaN scores
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
        
        # Feature importance
        feature_importance = {
            f'feature_{i}': float(np.abs(self.model.coef_[i]))
            for i in range(len(self.model.coef_))
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
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class LassoModel(BaseModel):
    """Lasso Regression with L1 regularization for feature selection."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.scaler = StandardScaler()
    
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
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize alpha parameter."""
        # Adjust CV folds based on data size
        n_folds = min(self.config.cv_folds, len(y) // 2)
        if n_folds < 2:
            n_folds = 2
        
        X_scaled = self.scaler.fit_transform(X)
        
        optimizer = OptunaOptimizer(
            n_trials=self.config.n_trials,
            random_state=self.config.random_state
        )
        
        def objective(trial):
            alpha = trial.suggest_float('alpha', 1e-4, 10, log=True)
            model = self._build_model(alpha=alpha)
            
            try:
                scores = cross_val_score(
                    model, X_scaled, y,
                    cv=n_folds,
                    scoring='r2'
                )
                score = np.mean(scores)
                
                # Check for NaN/inf
                if np.isnan(score) or np.isinf(score):
                    return -999.0
                return score
            except Exception:
                return -999.0
        
        try:
            return optimizer.optimize(objective, direction='maximize')
        except Exception as e:
            print(f"Optimization failed: {str(e)}. Using default parameters.")
            return {'alpha': 0.1}
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> ModelResult:
        """Train Lasso model."""
        start_time = time.time()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Optimize hyperparameters
        if self.config.verbose:
            print("Optimizing Lasso hyperparameters...")
        
        optimal_params = self._optimize_hyperparameters(X, y)
        
        # Build and train final model
        self.model = self._build_model(**optimal_params)
        self.model.fit(X_scaled, y)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y, y_pred, X)
        
        # Count selected features
        n_selected = np.sum(self.model.coef_ != 0)
        metrics.n_selected_features = n_selected
        
        # Cross-validation
        if self.config.cv_folds > 1:
            # Adjust CV folds based on data size
            n_folds = min(self.config.cv_folds, len(y) // 2)
            if n_folds < 2:
                n_folds = 2
            
            try:
                cv_scores = cross_val_score(
                    self.model, X_scaled, y,
                    cv=n_folds,
                    scoring='r2'
                )
                # Check for NaN scores
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
        
        # Feature importance (non-zero coefficients)
        feature_importance = {}
        for i, coef in enumerate(self.model.coef_):
            if coef != 0:
                feature_importance[f'feature_{i}'] = float(np.abs(coef))
        
        self.is_fitted = True
        
        if self.config.verbose:
            print(f"Lasso selected {n_selected}/{X.shape[1]} features")
        
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
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

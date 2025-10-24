"""
Optimized ensemble model implementations.
"""

import numpy as np
from typing import Dict, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

from ..core.base_model import BaseModel, ModelResult, ModelConfig
from ..utils.optimization import OptunaOptimizer, RandomSearchOptimizer


class RandomForestModel(BaseModel):
    """Optimized Random Forest implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
    def _build_model(self, **params) -> RandomForestRegressor:
        """Build Random Forest model."""
        return RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            max_features=params.get('max_features', 'sqrt'),
            bootstrap=params.get('bootstrap', True),
            n_jobs=-1,
            random_state=self.config.random_state
        )
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize Random Forest hyperparameters."""
        
        if self.config.optimization_method == 'random_search':
            optimizer = RandomSearchOptimizer(
                n_trials=self.config.n_trials,
                random_state=self.config.random_state
            )
            
            search_space = {
                'n_estimators': (50, 300),
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': ['sqrt', 'log2', 0.5, 0.8]
            }
            
            def objective(params):
                model = self._build_model(**params)
                scores = cross_val_score(
                    model, X, y,
                    cv=min(self.config.cv_folds, len(y) // 2),
                    scoring='r2',
                    n_jobs=-1
                )
                return np.mean(scores)
            
            return optimizer.optimize(objective, search_space, direction='maximize')
        
        else:
            # Use Optuna
            optimizer = OptunaOptimizer(
                n_trials=self.config.n_trials,
                random_state=self.config.random_state
            )
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20, 30, 50]),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8])
                }
                
                model = self._build_model(**params)
                
                # Adjust cv_folds
                n_folds = min(self.config.cv_folds, len(y) // 2)
                if n_folds < 2:
                    n_folds = 2
                
                try:
                    scores = cross_val_score(
                        model, X, y,
                        cv=n_folds,
                        scoring='r2',
                        n_jobs=1  # Avoid nested parallelism
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
                return {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt'
                }
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> ModelResult:
        """Train Random Forest model."""
        start_time = time.time()
        
        # Optimize hyperparameters
        if self.config.verbose:
            print("Optimizing Random Forest hyperparameters...")
        
        optimal_params = self._optimize_hyperparameters(X, y)
        
        # Build and train final model
        self.model = self._build_model(**optimal_params)
        self.model.fit(X, y)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
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
                    self.model, X, y,
                    cv=n_folds,
                    scoring='r2',
                    n_jobs=1
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
            f'feature_{i}': float(self.model.feature_importances_[i])
            for i in range(len(self.model.feature_importances_))
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
        
        return self.model.predict(X)


class GradientBoostingModel(BaseModel):
    """Optimized Gradient Boosting implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.scaler = StandardScaler()
        
    def _build_model(self, **params) -> GradientBoostingRegressor:
        """Build Gradient Boosting model."""
        return GradientBoostingRegressor(
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.1),
            max_depth=params.get('max_depth', 3),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            subsample=params.get('subsample', 1.0),
            max_features=params.get('max_features', None),
            random_state=self.config.random_state
        )
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize Gradient Boosting hyperparameters."""
        
        # Scale data for optimization
        X_scaled = self.scaler.fit_transform(X)
        
        optimizer = OptunaOptimizer(
            n_trials=self.config.n_trials,
            random_state=self.config.random_state
        )
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2', 0.5, 0.8])
            }
            
            model = self._build_model(**params)
            
            # Use cross-validation for proper evaluation
            from sklearn.model_selection import cross_val_score
            n_folds = min(self.config.cv_folds, len(y) // 2)
            if n_folds < 2:
                n_folds = 2
            
            try:
                cv_scores = cross_val_score(
                    model, X_scaled, y,
                    cv=n_folds,
                    scoring='r2',
                    n_jobs=1
                )
                score = np.mean(cv_scores)
                if np.isnan(score) or np.isinf(score):
                    return -999.0
                return score
            except Exception as e:
                print(f"Gradient Boosting CV failed: {e}")
                return -999.0
        
        try:
            best_params = optimizer.optimize(objective, direction='maximize')
            return best_params
        except Exception as e:
            print(f"Optimization failed: {str(e)}. Using default parameters.")
            return {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'subsample': 1.0,
                'max_features': None
            }
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> ModelResult:
        """Train Gradient Boosting model."""
        start_time = time.time()
        
        # Optimize hyperparameters
        if self.config.verbose:
            print("Optimizing Gradient Boosting hyperparameters...")
        
        optimal_params = self._optimize_hyperparameters(X, y)
        
        # Scale data for training
        X_scaled = self.scaler.fit_transform(X)
        
        # Build and train final model
        self.model = self._build_model(**optimal_params)
        self.model.fit(X_scaled, y)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y, y_pred, X)
        
        # Cross-validation
        if self.config.cv_folds > 1:
            n_folds = min(self.config.cv_folds, len(y) // 2)
            if n_folds < 2:
                n_folds = 2
            
            try:
                cv_model = self._build_model(**optimal_params)
                cv_scores = cross_val_score(
                    cv_model, X_scaled, y,
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
        
        # Feature importance
        feature_importance = {
            f'feature_{i}': float(self.model.feature_importances_[i])
            for i in range(len(self.model.feature_importances_))
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
        return self.model.predict(X_scaled)



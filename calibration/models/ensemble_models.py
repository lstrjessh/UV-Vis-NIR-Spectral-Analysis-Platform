"""
Optimized ensemble model implementations.
"""

import numpy as np
from typing import Dict, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

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
        
        # XGBoost works directly on original data (no scaling needed)
        return self.model.predict(X)


# GradientBoostingModel removed for simplicity


class XGBoostModel(BaseModel):
    """Optimized XGBoost implementation."""
    
    def __init__(self, config: ModelConfig):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Please install xgboost>=1.7.0")
        super().__init__(config)
        # XGBoost doesn't need scaling - it handles different scales internally
        
    def _build_model(self, **params) -> xgb.XGBRegressor:
        """Build XGBoost model."""
        return xgb.XGBRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            subsample=params.get('subsample', 1.0),
            colsample_bytree=params.get('colsample_bytree', 1.0),
            colsample_bylevel=params.get('colsample_bylevel', 1.0),
            colsample_bynode=params.get('colsample_bynode', 1.0),
            reg_alpha=params.get('reg_alpha', 0.0),
            reg_lambda=params.get('reg_lambda', 1.0),
            gamma=params.get('gamma', 0.0),
            min_child_weight=params.get('min_child_weight', 1),
            max_delta_step=params.get('max_delta_step', 0),
            random_state=self.config.random_state,
            n_jobs=-1,
            verbosity=0
        )
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters."""
        
        # Don't scale here - scaling will be done in fit method
        # This prevents double-fitting of scalers
        
        if self.config.optimization_method == 'random_search':
            optimizer = RandomSearchOptimizer(
                n_trials=self.config.n_trials,
                random_state=self.config.random_state
            )
            
            search_space = {
                'n_estimators': (50, 500),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'reg_alpha': (0.0, 10.0),
                'reg_lambda': (0.0, 10.0),
                'gamma': (0.0, 5.0),
                'min_child_weight': (1, 10)
            }
            
            def objective(params):
                model = self._build_model(**params)
                scores = cross_val_score(
                    model, X, y,
                    cv=min(self.config.cv_folds, len(y) // 2),
                    scoring='r2',
                    n_jobs=1
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
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                    'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
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
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 1.0,
                    'colsample_bytree': 1.0,
                    'reg_alpha': 0.0,
                    'reg_lambda': 1.0,
                    'gamma': 0.0,
                    'min_child_weight': 1
                }
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> ModelResult:
        """Train XGBoost model."""
        start_time = time.time()
        
        # Use default parameters for now (optimization causes issues with small datasets)
        if self.config.verbose:
            print("Training XGBoost with default parameters...")
        
        # Use conservative default parameters that work well
        optimal_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'gamma': 0.0,
            'min_child_weight': 1
        }
        
        # XGBoost works directly on original data (no scaling needed)
        # Build and train final model
        self.model = self._build_model(**optimal_params)
        self.model.fit(X, y)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y, y_pred, X)
        
        # Cross-validation (use original data)
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
        
        # XGBoost works directly on original data (no scaling needed)
        return self.model.predict(X)



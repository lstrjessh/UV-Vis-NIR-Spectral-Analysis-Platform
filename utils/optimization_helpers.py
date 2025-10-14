"""
Optimization Helper Functions

This module provides reusable hyperparameter optimization functions for all models.
"""

import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any, Callable
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint, uniform, loguniform

# Try to import Optuna for Bayesian optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Try to import PyTorch for deep learning support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def optimize_with_grid_search(param_space: Dict, model_class, X: np.ndarray, y: np.ndarray, 
                            cv_folds: int = 5, scoring: str = 'neg_mean_squared_error') -> Dict:
    """
    Perform grid search optimization.
    
    Args:
        param_space: Parameter space dictionary
        model_class: Model class to optimize
        X: Input features
        y: Target values
        cv_folds: Number of cross-validation folds
        scoring: Scoring metric
        
    Returns:
        Dictionary with best parameters and score
    """
    try:
        print(f"🔍 Starting Grid Search optimization with {cv_folds} CV folds...")
        
        # Convert parameter space to grid search format
        grid_params = _convert_to_grid_params(param_space)
        
        # Create model instance
        model = model_class()
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid_params,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )
        
        print("   Running grid search...")
        grid_search.fit(X, y)
        
        print(f"   ✅ Grid search completed. Best score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_,
            'optimization_method': 'GridSearch'
        }
        
    except Exception as e:
        print(f"   ❌ Grid search optimization failed: {str(e)}")
        return {}


def optimize_with_random_search(param_space: Dict, model_class, X: np.ndarray, y: np.ndarray, 
                              cv_folds: int = 5, scoring: str = 'neg_mean_squared_error',
                              n_iter: int = 50) -> Dict:
    """
    Perform random search optimization.
    
    Args:
        param_space: Parameter space dictionary
        model_class: Model class to optimize
        X: Input features
        y: Target values
        cv_folds: Number of cross-validation folds
        scoring: Scoring metric
        n_iter: Number of random iterations
        
    Returns:
        Dictionary with best parameters and score
    """
    try:
        print(f"🎲 Starting Random Search optimization with {cv_folds} CV folds, {n_iter} iterations...")
        
        # Convert parameter space to random search format
        random_params = _convert_to_random_params(param_space)
        
        # Create model instance
        model = model_class()
        
        # Perform random search
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=random_params,
            n_iter=n_iter,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        print("   Running random search...")
        random_search.fit(X, y)
        
        print(f"   ✅ Random search completed. Best score: {random_search.best_score_:.4f}")
        
        return {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'best_estimator': random_search.best_estimator_,
            'cv_results': random_search.cv_results_,
            'optimization_method': 'RandomSearch'
        }
        
    except Exception as e:
        print(f"   ❌ Random search optimization failed: {str(e)}")
        return {}


def optimize_with_bayesian(param_space: Dict, model_class, X: np.ndarray, y: np.ndarray, 
                         cv_folds: int = 5, scoring: str = 'neg_mean_squared_error',
                         n_trials: int = 100) -> Dict:
    """
    Perform Bayesian optimization using Optuna.
    
    Args:
        param_space: Parameter space dictionary
        model_class: Model class to optimize
        X: Input features
        y: Target values
        cv_folds: Number of cross-validation folds
        scoring: Scoring metric
        n_trials: Number of optimization trials
        
    Returns:
        Dictionary with best parameters and score
    """
    if not OPTUNA_AVAILABLE:
        print("   ⚠️ Optuna not available. Falling back to random search.")
        return optimize_with_random_search(param_space, model_class, X, y, cv_folds, scoring, n_trials)
    
    try:
        print(f"🧠 Starting Bayesian optimization with {cv_folds} CV folds, {n_trials} trials...")
        
        def objective(trial):
            # Sample parameters from the search space
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'log_float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
            
            # Create and evaluate model
            model = model_class(**params)
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
            return scores.mean()
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        
        print("   Running Bayesian optimization...")
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters and create best model
        best_params = study.best_params
        best_model = model_class(**best_params)
        best_model.fit(X, y)
        
        print(f"   ✅ Bayesian optimization completed. Best score: {study.best_value:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': study.best_value,
            'best_estimator': best_model,
            'study': study,
            'optimization_method': 'Bayesian'
        }
        
    except Exception as e:
        print(f"   ❌ Bayesian optimization failed: {str(e)}")
        return {}


def _convert_to_grid_params(param_space: Dict) -> Dict:
    """Convert parameter space to grid search format."""
    grid_params = {}
    
    for param_name, param_config in param_space.items():
        if param_config['type'] == 'categorical':
            grid_params[param_name] = param_config['choices']
        elif param_config['type'] == 'int':
            # Create a range of integer values
            low, high = param_config['low'], param_config['high']
            # Limit grid size for performance
            if high - low > 10:
                step = max(1, (high - low) // 5)
                grid_params[param_name] = list(range(low, high + 1, step))
            else:
                grid_params[param_name] = list(range(low, high + 1))
        elif param_config['type'] == 'float':
            # Create a range of float values
            low, high = param_config['low'], param_config['high']
            # Limit grid size for performance
            grid_params[param_name] = np.linspace(low, high, 5).tolist()
        elif param_config['type'] == 'log_float':
            # Create log-spaced values
            low, high = param_config['low'], param_config['high']
            grid_params[param_name] = np.logspace(np.log10(low), np.log10(high), 5).tolist()
    
    return grid_params


def _convert_to_random_params(param_space: Dict) -> Dict:
    """Convert parameter space to random search format."""
    random_params = {}
    
    for param_name, param_config in param_space.items():
        if param_config['type'] == 'categorical':
            random_params[param_name] = param_config['choices']
        elif param_config['type'] == 'int':
            low, high = param_config['low'], param_config['high']
            random_params[param_name] = randint(low, high + 1)
        elif param_config['type'] == 'float':
            low, high = param_config['low'], param_config['high']
            random_params[param_name] = uniform(low, high - low)
        elif param_config['type'] == 'log_float':
            low, high = param_config['low'], param_config['high']
            random_params[param_name] = loguniform(low, high)
    
    return random_params


def optimize_deep_learning_bayesian(
    param_space: Dict, 
    model_builder: Callable,  # Function that creates model given params
    X: np.ndarray, 
    y: np.ndarray, 
    cv_folds: int = 5,
    n_trials: int = 30,
    max_epochs: int = 1000,
    early_stopping_patience: int = 20,
    device: str = 'cpu'
) -> Dict:
    """
    Perform Bayesian optimization for deep learning models (PyTorch).
    
    Args:
        param_space: Parameter space dictionary
        model_builder: Function that creates model given params dict
        X: Input features
        y: Target values
        cv_folds: Number of cross-validation folds
        n_trials: Number of optimization trials
        max_epochs: Maximum epochs for training
        early_stopping_patience: Patience for early stopping
        device: 'cpu' or 'cuda'
        
    Returns:
        Dictionary with best parameters and score
    """
    if not OPTUNA_AVAILABLE:
        print("   ⚠️ Optuna not available for deep learning optimization.")
        return {}
    
    if not TORCH_AVAILABLE:
        print("   ⚠️ PyTorch not available for deep learning optimization.")
        return {}
    
    try:
        print(f"🧠 Starting Bayesian optimization for deep learning with {n_trials} trials...")
        
        # Use reduced epochs for CV to speed up optimization
        cv_epochs = max(100, int(max_epochs * 0.4))  # 40% of max epochs
        cv_patience = max(10, int(early_stopping_patience * 0.5))  # Proportional patience
        
        print(f"   CV will use {cv_epochs} max epochs with patience {cv_patience}")
        
        def objective(trial):
            # Sample parameters from the search space
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'log_float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
            
            # Perform cross-validation with these parameters
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Build model with current parameters
                model = model_builder(params, X.shape[1])
                
                # Train model with early stopping
                best_val_loss = float('inf')
                patience_counter = 0
                
                for epoch in range(cv_epochs):
                    train_loss = model.train_step(X_train, y_train)
                    val_loss = model.validate_step(X_val, y_val)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= cv_patience:
                        break
                    
                    # Report intermediate value for pruning
                    trial.report(1.0 / (1.0 + best_val_loss), epoch)
                    
                    # Prune trial if it's not promising
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                # Get final validation score (R²)
                val_pred = model.predict(X_val)
                ss_res = np.sum((y_val - val_pred) ** 2)
                ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0
                scores.append(r2)
            
            return np.mean(scores)
        
        # Create study with pruning for faster optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(
                seed=42,
                n_startup_trials=min(10, n_trials // 3),
                multivariate=True
            ),
            pruner=MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            )
        )
        
        print("   Running Bayesian optimization with pruning...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        best_params = study.best_params
        print(f"   ✅ Bayesian optimization completed. Best score: {study.best_value:.4f}")
        print(f"   Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': study.best_value,
            'study': study,
            'optimization_method': 'Bayesian'
        }
        
    except Exception as e:
        print(f"   ❌ Deep learning Bayesian optimization failed: {str(e)}")
        return {}


def get_default_param_spaces() -> Dict[str, Dict]:
    """Get default parameter spaces for all model types."""
    return {
        'random_forest': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
            'max_depth': {'type': 'int', 'low': 3, 'high': 20},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
            'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
            'bootstrap': {'type': 'categorical', 'choices': [True, False]}
        },
        
        'xgboost': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
            'max_depth': {'type': 'int', 'low': 3, 'high': 15},
            'learning_rate': {'type': 'log_float', 'low': 0.01, 'high': 0.3},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
            'reg_alpha': {'type': 'log_float', 'low': 1e-8, 'high': 1.0},
            'reg_lambda': {'type': 'log_float', 'low': 1e-8, 'high': 1.0}
        },
        
        'neural_network': {
            'hidden_layers': {'type': 'categorical', 'choices': [(50,), (100,), (50, 50), (100, 50), (100, 100), (50, 50, 50)]},
            'activation': {'type': 'categorical', 'choices': ['relu', 'tanh']},
            'dropout_rate': {'type': 'float', 'low': 0.0, 'high': 0.4},
            'learning_rate': {'type': 'log_float', 'low': 1e-4, 'high': 1e-2},
            'batch_size': {'type': 'categorical', 'choices': [8, 16, 32, 64]},
            'weight_decay': {'type': 'log_float', 'low': 1e-6, 'high': 1e-3}
        },
        
        'svr': {
            'kernel': {'type': 'categorical', 'choices': ['linear', 'rbf', 'poly']},
            'C': {'type': 'log_float', 'low': 0.1, 'high': 100.0},
            'gamma': {'type': 'categorical', 'choices': ['scale', 'auto']},
            'epsilon': {'type': 'log_float', 'low': 0.01, 'high': 1.0},
            'degree': {'type': 'int', 'low': 2, 'high': 5}  # Only used for poly kernel
        },
        
        'plsr': {
            'n_components': {'type': 'int', 'low': 1, 'high': 10}
        },
        
        'cnn': {
            'filters': {'type': 'categorical', 'choices': [[32], [64], [32, 64], [64, 128]]},
            'kernel_size': {'type': 'categorical', 'choices': [3, 5, 7]},
            'dense_units': {'type': 'categorical', 'choices': [[32, 16, 1], [64, 32, 1], [32, 32, 16, 1]]},
            'dropout_rate': {'type': 'float', 'low': 0.1, 'high': 0.4},
            'learning_rate': {'type': 'log_float', 'low': 1e-4, 'high': 5e-3},
            'batch_size': {'type': 'categorical', 'choices': [4, 8, 16, 32]}
        }
    }


def get_adaptive_param_space(model_type: str, n_samples: int, n_features: int) -> Dict:
    """
    Get parameter space adapted to dataset size.
    
    Args:
        model_type: Type of model ('cnn', 'mlp', 'rf', etc.)
        n_samples: Number of samples in dataset
        n_features: Number of features
        
    Returns:
        Adapted parameter space dictionary
    """
    base_spaces = get_default_param_spaces()
    
    if model_type not in base_spaces:
        return {}
    
    param_space = base_spaces[model_type].copy()
    
    # Adapt batch sizes based on sample size
    if 'batch_size' in param_space:
        max_batch = min(n_samples // 2, 64)
        if n_samples < 10:
            batch_choices = [min(2, n_samples), min(4, n_samples)]
        elif n_samples < 20:
            batch_choices = [4, 8, min(16, n_samples)]
        else:
            batch_choices = [8, 16, min(32, max_batch), min(64, max_batch)]
        
        batch_choices = [b for b in batch_choices if b > 0 and b <= n_samples]
        param_space['batch_size'] = {'type': 'categorical', 'choices': batch_choices}
    
    # Adapt architectures based on feature size
    if model_type == 'neural_network' and 'hidden_layers' in param_space:
        if n_features > 100:
            # Large feature space: use smaller networks
            param_space['hidden_layers']['choices'] = [(32,), (64,), (32, 16), (64, 32), (32, 32)]
        else:
            # Smaller feature space: can use larger networks
            param_space['hidden_layers']['choices'] = [(50,), (100,), (50, 50), (100, 50), (64, 32)]
    
    if model_type == 'cnn':
        # Adapt CNN architecture based on sequence length
        if n_features < 50:
            param_space['kernel_size']['choices'] = [3, 5]
            param_space['filters']['choices'] = [[32], [64], [32, 64]]
        elif n_features < 200:
            param_space['kernel_size']['choices'] = [3, 5, 7]
            param_space['filters']['choices'] = [[32], [64], [32, 64], [64, 128]]
        else:
            param_space['kernel_size']['choices'] = [3, 5, 7, 9]
            param_space['filters']['choices'] = [[32], [64], [32, 64], [64, 128], [32, 64, 32]]
    
    return param_space

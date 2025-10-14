"""
Support Vector Regression (SVR) Model Implementation

This module contains the SVRFitter class for performing Support Vector
Regression on spectroscopic data for concentration prediction.
"""

import numpy as np
import streamlit as st
import pickle
import base64
from typing import Dict, List
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import random

# Try to import Optuna for Bayesian optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def _grid_search_svr(x_data, y_data, param_space, cv_folds):
    """Perform grid search optimization for SVR."""
    print("Performing Grid Search optimization for SVR...")
    
    # Create parameter grid
    param_grid = {
        'kernel': param_space['kernel'],
        'C': param_space['C'],
        'gamma': param_space['gamma'],
        'epsilon': param_space['epsilon']
    }
    
    # Add degree parameter for polynomial kernel
    if 'poly' in param_space['kernel']:
        param_grid['degree'] = param_space['degree']
    
    # Perform grid search
    svr = SVR()
    grid_search = GridSearchCV(
        svr, param_grid, cv=cv_folds, scoring='r2', n_jobs=-1
    )
    grid_search.fit(x_data, y_data)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best parameters: {best_params}, Best CV score: {best_score:.4f}")
    return best_params


def _random_search_svr(x_data, y_data, param_space, cv_folds, n_trials=30):
    """Perform random search optimization for SVR."""
    print("Performing Random Search optimization for SVR...")
    
    best_score = -float('inf')
    best_params = None
    
    for trial in range(n_trials):
        # Sample random parameters
        params = {
            'kernel': random.choice(param_space['kernel']),
            'C': random.choice(param_space['C']),
            'gamma': random.choice(param_space['gamma']),
            'epsilon': random.choice(param_space['epsilon'])
        }
        
        # Add degree for polynomial kernel
        if params['kernel'] == 'poly':
            params['degree'] = random.choice(param_space['degree'])
        
        print(f"Trial {trial + 1}/{n_trials}: Testing params {params}")
        
        # Train and evaluate model
        score = _evaluate_svr_params(x_data, y_data, params, cv_folds)
        
        if score > best_score:
            best_score = score
            best_params = params
            print(f"New best score: {best_score:.4f}")
    
    print(f"Best parameters found: {best_params}")
    print(f"Best CV score: {best_score:.4f}")
    return best_params


def _bayesian_search_svr(x_data, y_data, param_space, cv_folds, n_trials=50):
    """Perform Bayesian optimization for SVR using Optuna."""
    if not OPTUNA_AVAILABLE:
        print("Optuna not available, falling back to Random Search...")
        return _random_search_svr(x_data, y_data, param_space, cv_folds, n_trials)
    
    print("Performing Bayesian optimization for SVR...")
    
    def objective(trial):
        # Sample parameters using param_space (not hardcoded values)
        kernel = trial.suggest_categorical('kernel', param_space['kernel'])
        
        # For C and epsilon, sample from param_space if categorical, else use log scale
        if isinstance(param_space['C'], list):
            C = trial.suggest_categorical('C', param_space['C'])
        else:
            C = trial.suggest_float('C', 0.1, 100.0, log=True)
        
        gamma = trial.suggest_categorical('gamma', param_space['gamma'])
        
        if isinstance(param_space['epsilon'], list):
            epsilon = trial.suggest_categorical('epsilon', param_space['epsilon'])
        else:
            epsilon = trial.suggest_float('epsilon', 0.01, 1.0, log=True)
        
        params = {
            'kernel': kernel,
            'C': C,
            'gamma': gamma,
            'epsilon': epsilon
        }
        
        # Add degree for polynomial kernel
        if kernel == 'poly':
            if isinstance(param_space['degree'], list):
                params['degree'] = trial.suggest_categorical('degree', param_space['degree'])
            else:
                params['degree'] = trial.suggest_int('degree', 2, 5)
        
        # Evaluate parameters
        score = _evaluate_svr_params(x_data, y_data, params, cv_folds)
        return score
    
    # Create study with better configuration and pruning
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=min(10, n_trials // 3)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    print(f"Best parameters found: {best_params}")
    print(f"Best CV score: {study.best_value:.4f}")
    
    return best_params


def _evaluate_svr_params(x_data, y_data, params, cv_folds):
    """Evaluate SVR parameters using cross-validation."""
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_data)):
        x_train, x_val = x_data[train_idx], x_data[val_idx]
        y_train, y_val = y_data[train_idx], y_data[val_idx]
        
        # Create and train model
        svr = SVR(**params)
        svr.fit(x_train, y_train)
        
        # Evaluate
        y_pred = svr.predict(x_val)
        r2 = r2_score(y_val, y_pred)
        scores.append(r2)
    
    return np.mean(scores)


class SVRFitter:
    """Support Vector Regression for concentration prediction using full spectrum."""
    
    @staticmethod
    def fit_svr(x_data_tuple: tuple, y_data_tuple: tuple,
                svr_configs: List[Dict] = None,
                optimization_method: str = "Random Search",
                cv_folds: int = 5,
                n_trials: int = 20,
                run_cv: bool = True) -> Dict[str, Dict]:
        """
        Fit SVR models with various kernels and parameters using full spectrum data.
        
        Args:
            x_data_tuple: Input spectral data as tuple for caching (n_samples, n_wavelengths)
            y_data_tuple: Output concentration data as tuple for caching (n_samples,)
            svr_configs: List of SVR configurations to try
            
        Returns:
            Dictionary with config_name -> fit_results mapping
        """
        # Convert tuples back to arrays for computation
        x_data = np.array(x_data_tuple)  # Full spectrum: (n_samples, n_wavelengths)
        y_data = np.array(y_data_tuple)  # Concentrations: (n_samples,)
        
        # Ensure x_data is 2D
        if x_data.ndim == 1:
            x_data = x_data.reshape(-1, 1)
        
        if len(x_data) < 3:
            st.warning("SVR requires at least 3 data points")
            return {}
        
        # Validate input data
        if np.any(~np.isfinite(x_data)) or np.any(~np.isfinite(y_data)):
            st.error("Data contains invalid values (NaN or Inf)")
            return {}
        
        # Standardize data for SVR (use all data - user will have separate test data later)
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        x_scaled = scaler_x.fit_transform(x_data)
        y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1)).ravel()
        
        # Use optimization to find best hyperparameters
        if svr_configs is None or (isinstance(svr_configs, dict) and 'kernel_types' in svr_configs):
            print("No SVR configurations provided. Using hyperparameter optimization...")
            
            # Extract kernel types if provided
            if isinstance(svr_configs, dict) and 'kernel_types' in svr_configs:
                kernel_types = svr_configs['kernel_types']
                print(f"   Testing kernel types: {kernel_types}")
            else:
                kernel_types = ['linear', 'rbf', 'poly']
            
            # Define parameter space for optimization
            param_space = {
                'kernel': kernel_types,
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                'epsilon': [0.01, 0.1, 0.2, 0.5],
                'degree': [2, 3, 4, 5]  # For polynomial kernel
            }
            
            # Perform optimization
            if optimization_method == "Grid Search":
                best_params = _grid_search_svr(x_scaled, y_scaled, param_space, cv_folds)
            elif optimization_method == "Random Search":
                best_params = _random_search_svr(x_scaled, y_scaled, param_space, cv_folds, n_trials)
            elif optimization_method == "Bayesian":
                best_params = _bayesian_search_svr(x_scaled, y_scaled, param_space, cv_folds, n_trials)
            
            # Create single optimized configuration
            svr_configs = [{
                'name': 'SVR_Optimized',
                **best_params
            }]
        
        fit_results = {}
        n = len(x_data)
        y_mean = np.mean(y_data)
        ss_tot = np.sum((y_data - y_mean) ** 2)
        
        for config in svr_configs:
            try:
                config_name = config['name']
                model_params = {k: v for k, v in config.items() if k != 'name'}
                
                # Fit SVR on all data
                svr = SVR(**model_params)
                import time
                training_start_time = time.time()
                svr.fit(x_scaled, y_scaled)
                training_time = time.time() - training_start_time
                
                # Make predictions on training data (for reference)
                y_pred_scaled = svr.predict(x_scaled)
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                
                # Calculate statistics on training data
                residuals = y_data - y_pred
                ss_res = np.sum(residuals ** 2)
                mse = ss_res / n
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(residuals))
                
                # Additional metrics
                mape = np.mean(np.abs((y_data - y_pred) / y_data)) * 100 if np.all(y_data != 0) else np.nan
                max_error = np.max(np.abs(residuals))
                
                # R-squared on training data (optimistic - for reference only)
                r_squared_train = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0
                
                # Adjusted R-squared (approximation)
                # SVR doesn't have a clear parameter count
                n_support = len(svr.support_)
                p_approx = n_support if n_support > 0 else 1
                if n > p_approx and r_squared_train < 0.999:
                    adj_r_squared = 1 - (1 - r_squared_train) * (n - 1) / (n - p_approx)
                else:
                    adj_r_squared = r_squared_train
                
                # Information criteria (approximation)
                if ss_res > 1e-12:
                    log_likelihood = -n/2 * np.log(2*np.pi) - n/2 * np.log(ss_res/n) - n/2
                    aic = 2 * p_approx - 2 * log_likelihood
                    bic = p_approx * np.log(n) - 2 * log_likelihood
                else:
                    aic = np.inf
                    bic = np.inf
                
                # Cross-validation score (MAIN PERFORMANCE METRIC)
                # This gives realistic estimate without holding out data
                cv_score = np.nan
                cv_rmse = np.nan
                cv_std = np.nan
                if run_cv:
                    try:
                        cv_folds = max(2, min(5, n - 1))
                        cv_scores = cross_val_score(
                            svr, x_scaled, y_scaled, cv=cv_folds,
                            scoring='r2'
                        )
                        cv_score = float(cv_scores.mean())
                        cv_std = float(cv_scores.std())
                        
                        # Calculate CV RMSE separately
                        cv_mse_scores = cross_val_score(
                            svr, x_scaled, y_scaled, cv=cv_folds,
                            scoring='neg_mean_squared_error'
                        )
                        cv_rmse = float(np.sqrt(-cv_mse_scores.mean()))
                    except Exception as e:
                        print(f"   ⚠️ SVR CV failed: {e}")
                        cv_score = np.nan
                        cv_rmse = np.nan
                        cv_std = np.nan
                
                # Use CV score as primary R² if available, otherwise use training R²
                r_squared = cv_score if not np.isnan(cv_score) else r_squared_train
                
                # Serialize model
                try:
                    serialization_data = {
                        'model': svr,
                        'scaler_x': scaler_x,
                        'scaler_y': scaler_y,
                        'config': config
                    }
                    model_pickle = pickle.dumps(serialization_data)
                    model_serialized = base64.b64encode(model_pickle).decode('utf-8')
                except Exception:
                    model_serialized = None
                
                fit_results[config_name] = {
                    'model': svr,
                    'scaler_x': scaler_x,
                    'scaler_y': scaler_y,
                    'predictions': y_pred,
                    'r_squared': max(0, min(1, r_squared)),  # CV score if available, else training
                    'r_squared_train': max(0, min(1, r_squared_train)),  # Training performance (reference)
                    'adj_r_squared': max(0, min(1, adj_r_squared)),
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'max_error': max_error,
                    'cv_score': cv_score,
                    'training_time': training_time,
                    'aic': aic,
                    'bic': bic,
                    'cv_rmse': cv_rmse,
                    'cv_std': cv_std,
                    'residuals': residuals,
                    'kernel': svr.kernel,
                    'n_support': n_support,
                    'model_serialized': model_serialized
                }
                
            except Exception as e:
                st.warning(f"Failed to fit SVR '{config_name}': {str(e)}")
                continue
        
        return fit_results
    
    @staticmethod
    def predict_with_svr(model_data: Dict, spectrum: np.ndarray) -> float:
        """
        Predict concentration using trained SVR model.
        
        Args:
            model_data: Dictionary with model, scalers, etc.
            spectrum: Full spectrum absorbance data (n_wavelengths,)
            
        Returns:
            Predicted concentration
        """
        try:
            model = model_data['model']
            scaler_x = model_data['scaler_x']
            scaler_y = model_data['scaler_y']
            
            # Ensure spectrum is 2D (1, n_wavelengths)
            if spectrum.ndim == 1:
                spectrum = spectrum.reshape(1, -1)
            
            # Scale input spectrum
            x_scaled = scaler_x.transform(spectrum)
            
            # Predict concentration
            y_pred_scaled = model.predict(x_scaled)
            
            # Inverse transform
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            return float(y_pred[0])
            
        except Exception as e:
            st.error(f"Error predicting with SVR: {str(e)}")
            return 0.0

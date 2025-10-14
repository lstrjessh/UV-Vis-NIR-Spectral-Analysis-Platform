"""
Partial Least Squares Regression (PLSR) Model Implementation

This module contains the PLSRFitter class for performing Partial Least Squares
Regression on spectroscopic data for concentration prediction.
"""

import numpy as np
import streamlit as st
import pickle
import base64
from typing import Dict, Tuple
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import random

# Try to import Optuna for Bayesian optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def _grid_search_plsr(x_data, y_data, param_space, cv_folds):
    """Perform grid search optimization for PLSR."""
    print("Performing Grid Search optimization for PLSR...")
    print(f"   Testing all {len(param_space['n_components'])} possible component values.")
    
    # Create parameter grid
    param_grid = {
        'n_components': param_space['n_components']
    }
    
    # Perform grid search
    plsr = PLSRegression()
    grid_search = GridSearchCV(
        plsr, param_grid, cv=cv_folds, scoring='r2', n_jobs=-1
    )
    grid_search.fit(x_data, y_data)
    
    best_n_components = grid_search.best_params_['n_components']
    best_score = grid_search.best_score_
    
    print(f"Best n_components: {best_n_components}, Best CV score: {best_score:.4f}")
    return best_n_components


def _random_search_plsr(x_data, y_data, param_space, cv_folds, n_trials=20):
    """Perform random search optimization for PLSR."""
    print("Performing Random Search optimization for PLSR...")
    
    best_score = -float('inf')
    best_n_components = param_space['n_components'][0]
    
    # PLSR only has n_components parameter, so trials are limited by unique values
    max_possible_trials = len(param_space['n_components'])
    effective_trials = min(n_trials, max_possible_trials)
    
    if n_trials > max_possible_trials:
        print(f"   ⚠️ Requested {n_trials} trials, but only {max_possible_trials} unique components available.")
        print(f"   Using all {effective_trials} possible values instead.")
    
    n_trials = effective_trials
    
    for trial in range(n_trials):
        n_components = random.choice(param_space['n_components'])
        
        # Evaluate this configuration
        plsr = PLSRegression(n_components=n_components)
        scores = cross_val_score(plsr, x_data, y_data, cv=cv_folds, scoring='r2')
        mean_score = np.mean(scores)
        
        print(f"Trial {trial + 1}/{n_trials}: n_components={n_components}, CV score={mean_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_n_components = n_components
    
    print(f"Best n_components: {best_n_components}, Best CV score: {best_score:.4f}")
    return best_n_components


def _bayesian_search_plsr(x_data, y_data, param_space, cv_folds, n_trials=30):
    """Perform Bayesian optimization for PLSR using Optuna."""
    if not OPTUNA_AVAILABLE:
        print("Optuna not available, falling back to Random Search...")
        return _random_search_plsr(x_data, y_data, param_space, cv_folds, n_trials)
    
    print("Performing Bayesian optimization for PLSR...")
    
    # PLSR only has n_components parameter, so trials are limited
    max_possible_trials = len(param_space['n_components'])
    effective_trials = min(n_trials, max_possible_trials)
    
    if n_trials > max_possible_trials:
        print(f"   ⚠️ Requested {n_trials} trials, but only {max_possible_trials} unique components available.")
        print(f"   Using all {effective_trials} possible values instead.")
    
    n_trials = effective_trials
    
    def objective(trial):
        n_components = trial.suggest_categorical('n_components', param_space['n_components'])
        
        # Evaluate this configuration
        plsr = PLSRegression(n_components=n_components)
        scores = cross_val_score(plsr, x_data, y_data, cv=cv_folds, scoring='r2')
        return np.mean(scores)
    
    # Create study with better sampler settings
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            seed=42, 
            n_startup_trials=min(5, n_trials // 2),
            multivariate=True
        )
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_n_components = study.best_params['n_components']
    best_score = study.best_value
    
    print(f"Best n_components: {best_n_components}, Best CV score: {best_score:.4f}")
    return best_n_components


class PLSRFitter:
    """Partial Least Squares Regression for concentration prediction using full spectrum."""
    
    @staticmethod
    def fit_plsr(x_data_tuple: tuple, y_data_tuple: tuple,
                 max_components: int = None, 
                 optimization_method: str = "Random Search",
                 cv_folds: int = 5,
                 n_trials: int = 20,
                 run_cv: bool = True) -> Dict[int, Dict]:
        """
        Fit PLSR models with various numbers of components using full spectrum data.
        
        Args:
            x_data_tuple: Input spectral data as tuple for caching (n_samples, n_wavelengths)
            y_data_tuple: Output concentration data as tuple for caching (n_samples,)
            max_components: Maximum number of components to try
            
        Returns:
            Dictionary with n_components -> fit_results mapping
        """
        # Convert tuples back to arrays for computation
        x_data = np.array(x_data_tuple)  # Now full spectrum: (n_samples, n_wavelengths)
        y_data = np.array(y_data_tuple)  # Concentrations: (n_samples,)
        
        if len(x_data) < 2:
            return {}
        
        # Ensure x_data is 2D
        if x_data.ndim == 1:
            x_data = x_data.reshape(-1, 1)
        
        # Validate input data
        if np.any(~np.isfinite(x_data)) or np.any(~np.isfinite(y_data)):
            st.error("Data contains invalid values (NaN or Inf)")
            return {}
        
        # Determine number of components
        # PLSR components cannot exceed min(n_samples-1, n_features)
        n_features = x_data.shape[1]  # Number of wavelengths
        n_samples = len(x_data)
        max_possible_components = min(n_samples - 1, n_features)
        
        if max_components is None:
            n_components = min(3, max_possible_components)  # Default to 3 components
        else:
            n_components = min(max_components, max_possible_components)
        
        if n_components < 1:
            st.warning("PLSR requires at least 2 samples. Cannot fit model.")
            return {}
        
        st.info(f"PLSR: {n_features} wavelengths, {n_components} components")
        
        fit_results = {}
        n = len(x_data)
        y_mean = np.mean(y_data)
        ss_tot = np.sum((y_data - y_mean) ** 2)
        
        # Standardize data
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        x_scaled = scaler_x.fit_transform(x_data)
        y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1)).ravel()
        
        # Perform hyperparameter optimization if requested
        optimized_n_components = n_components
        if optimization_method != "None":
            print(f"🔧 Optimizing PLSR hyperparameters using {optimization_method}...")
            
            # Define parameter space for PLSR
            param_space = {
                'n_components': list(range(1, min(max_possible_components + 1, 11)))  # 1 to 10 or max possible
            }
            
            # Perform optimization
            if optimization_method == "Grid Search":
                optimized_n_components = _grid_search_plsr(x_scaled, y_scaled, param_space, cv_folds)
            elif optimization_method == "Random Search":
                optimized_n_components = _random_search_plsr(x_scaled, y_scaled, param_space, cv_folds, n_trials)
            elif optimization_method == "Bayesian":
                optimized_n_components = _bayesian_search_plsr(x_scaled, y_scaled, param_space, cv_folds, n_trials)
            
            print(f"✅ PLSR optimization complete. Best n_components: {optimized_n_components}")
        
        # Train PLSR model with optimized parameters
        try:
            # Fit PLSR model
            plsr = PLSRegression(n_components=optimized_n_components)
            import time
            training_start_time = time.time()
            plsr.fit(x_scaled, y_scaled)
            training_time = time.time() - training_start_time
            
            # Make predictions
            y_pred_scaled = plsr.predict(x_scaled).ravel()
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            # Calculate statistics
            residuals = y_data - y_pred
            ss_res = np.sum(residuals ** 2)
            mse = ss_res / n
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(residuals))
            
            # R-squared
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0
            
            # Adjusted R-squared
            p = optimized_n_components + 1  # Number of parameters (approximation)
            if n > p and r_squared < 0.999:
                adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p)
            else:
                adj_r_squared = r_squared
            
            # Information criteria
            if ss_res > 1e-12:
                log_likelihood = -n/2 * np.log(2*np.pi) - n/2 * np.log(ss_res/n) - n/2
                aic = 2 * p - 2 * log_likelihood
                bic = p * np.log(n) - 2 * log_likelihood
            else:
                aic = np.inf
                bic = np.inf
            
            # Additional metrics
            mape = np.mean(np.abs((y_data - y_pred) / y_data)) * 100 if np.all(y_data != 0) else np.nan
            max_error = np.max(np.abs(residuals))
            
            # Cross-validation score (MAIN PERFORMANCE METRIC)
            # This gives realistic estimate without holding out data
            cv_score = np.nan
            cv_rmse = np.nan
            cv_std = np.nan
            r_squared_train = r_squared  # Store training R² for reference
            if run_cv:
                try:
                    cv_folds = max(2, min(5, n - 1))
                    cv_scores = cross_val_score(
                        plsr, x_scaled, y_scaled, cv=cv_folds,
                        scoring='r2'
                    )
                    cv_score = float(cv_scores.mean())
                    cv_std = float(cv_scores.std())
                    
                    # Calculate CV RMSE separately
                    cv_mse_scores = cross_val_score(
                        plsr, x_scaled, y_scaled, cv=cv_folds,
                        scoring='neg_mean_squared_error'
                    )
                    cv_rmse = float(np.sqrt(-cv_mse_scores.mean()))
                except Exception as e:
                    print(f"   ⚠️ PLSR CV failed: {e}")
                    cv_score = np.nan
                    cv_rmse = np.nan
                    cv_std = np.nan
            
            # Use CV score as primary R² if available, otherwise use training R²
            r_squared = cv_score if not np.isnan(cv_score) else r_squared_train
            
            # Serialize model
            try:
                serialization_data = {
                    'model': plsr,
                    'scaler_x': scaler_x,
                    'scaler_y': scaler_y,
                    'n_components': n_components
                }
                model_pickle = pickle.dumps(serialization_data)
                model_serialized = base64.b64encode(model_pickle).decode('utf-8')
            except Exception:
                model_serialized = None
            
            fit_results[optimized_n_components] = {
                'model': plsr,
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
                'n_components': optimized_n_components,
                'model_serialized': model_serialized
            }
            
        except Exception as e:
            st.warning(f"Failed to fit PLSR with {optimized_n_components} components: {str(e)}")
            return {}
        
        return fit_results
    
    @staticmethod
    def predict_with_plsr(model_data: Dict, spectrum: np.ndarray) -> float:
        """
        Predict concentration using trained PLSR model.
        
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
            y_pred_scaled = model.predict(x_scaled).ravel()
            
            # Inverse transform
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            return float(y_pred[0])
            
        except Exception as e:
            st.error(f"Error predicting with PLSR: {str(e)}")
            return 0.0


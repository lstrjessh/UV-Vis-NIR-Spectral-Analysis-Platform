"""
Random Forest Regression Model Implementation

This module contains the RandomForestFitter class for performing Random Forest
Regression on spectroscopic data for concentration prediction.
"""

import numpy as np
import streamlit as st
import pickle
import base64
from typing import Dict, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split

# Import optimization helpers
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.optimization_helpers import (
    optimize_with_grid_search, 
    optimize_with_random_search, 
    optimize_with_bayesian,
    get_default_param_spaces
)


class RandomForestFitter:
    """Random Forest Regression for concentration prediction using full spectrum."""
    
    @staticmethod
    def _fit_and_evaluate_random_forest(model_params: Dict, x_data: np.ndarray, y_data: np.ndarray, 
                                       ss_tot: float, run_cv: bool, config_name: str) -> Dict:
        """Private helper to fit, evaluate, and serialize a single Random Forest model."""
        import time
        
        # Fit Random Forest on ALL data (user will have separate test data later)
        rf = RandomForestRegressor(**model_params)
        training_start_time = time.time()
        rf.fit(x_data, y_data)
        training_time = time.time() - training_start_time
        
        # Make predictions on training data (for reference only)
        y_pred = rf.predict(x_data)
        
        # Calculate statistics on training data
        residuals = y_data - y_pred
        ss_res = np.sum(residuals ** 2)
        n = len(x_data)
        mse = ss_res / n
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        
        # Additional metrics
        mape = np.mean(np.abs((y_data - y_pred) / y_data)) * 100 if np.all(y_data != 0) else np.nan
        max_error = np.max(np.abs(residuals))
        
        # R-squared on training data (optimistic - for reference only)
        r_squared_train = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0
        
        # Adjusted R-squared (approximation)
        # Using n_estimators as a proxy for model complexity in tree-based models
        p_approx = min(rf.n_estimators, n // 2)
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
        
        # Feature importance
        feature_importance = rf.feature_importances_[0] if len(rf.feature_importances_) > 0 else 1.0
        
        # Cross-validation score (MAIN PERFORMANCE METRIC)
        # This gives realistic estimate without holding out data
        cv_score = np.nan
        cv_rmse = np.nan
        cv_std = np.nan
        if run_cv:
            try:
                cv_folds_calc = max(2, min(5, n - 1))
                cv_scores = cross_val_score(
                    rf, x_data, y_data, cv=cv_folds_calc,
                    scoring='r2'
                )
                cv_score = float(cv_scores.mean())
                cv_std = float(cv_scores.std())
                
                # Calculate CV RMSE separately
                cv_mse_scores = cross_val_score(
                    rf, x_data, y_data, cv=cv_folds_calc,
                    scoring='neg_mean_squared_error'
                )
                cv_rmse = float(np.sqrt(-cv_mse_scores.mean()))
            except Exception as e:
                print(f"   ⚠️ Random Forest CV failed: {e}")
                cv_score = np.nan
                cv_rmse = np.nan
                cv_std = np.nan
        
        # Use CV score as primary R² if available, otherwise use training R²
        r_squared = cv_score if not np.isnan(cv_score) else r_squared_train
        
        # Serialize model
        try:
            serialization_data = {
                'model': rf,
                'config': model_params
            }
            model_pickle = pickle.dumps(serialization_data)
            model_serialized = base64.b64encode(model_pickle).decode('utf-8')
        except Exception:
            model_serialized = None
        
        return {
            'model': rf,
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
            'feature_importance': feature_importance,
            'n_estimators': rf.n_estimators,
            'max_depth': rf.max_depth,
            'model_serialized': model_serialized
        }
    
    @staticmethod
    def fit_random_forest(x_data_tuple: tuple, y_data_tuple: tuple,
                          rf_configs: List[Dict] = None,
                          optimization_method: str = "Random Search",
                          cv_folds: int = 5,
                          n_trials: int = 20,
                          run_cv: bool = True) -> Dict[str, Dict]:
        """
        Fit Random Forest models with various configurations using full spectrum data.
        
        Args:
            x_data_tuple: Input spectral data as tuple for caching (n_samples, n_wavelengths)
            y_data_tuple: Output concentration data as tuple for caching (n_samples,)
            rf_configs: List of RF configurations to try
            
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
            st.warning("Random Forest requires at least 3 data points")
            return {}
        
        # Validate input data
        if np.any(~np.isfinite(x_data)) or np.any(~np.isfinite(y_data)):
            st.error("Data contains invalid values (NaN or Inf)")
            return {}
        
        fit_results = {}
        n = len(x_data)
        y_mean = np.mean(y_data)
        ss_tot = np.sum((y_data - y_mean) ** 2)
        
        # Perform hyperparameter optimization if requested
        if optimization_method != "None":
            print(f"🔧 Optimizing Random Forest hyperparameters using {optimization_method}...")
            
            # Get parameter space for Random Forest
            param_spaces = get_default_param_spaces()
            param_space = param_spaces['random_forest']
            
            # Perform optimization
            if optimization_method == "Grid Search":
                opt_results = optimize_with_grid_search(param_space, RandomForestRegressor, x_data, y_data, cv_folds)
            elif optimization_method == "Random Search":
                opt_results = optimize_with_random_search(param_space, RandomForestRegressor, x_data, y_data, cv_folds, n_iter=n_trials)
            elif optimization_method == "Bayesian":
                opt_results = optimize_with_bayesian(param_space, RandomForestRegressor, x_data, y_data, cv_folds, n_trials=n_trials)
            else:
                opt_results = {}
            
            if opt_results and 'best_params' in opt_results:
                best_params = opt_results['best_params']
                print(f"   ✅ Random Forest optimization completed. Best params: {best_params}")
                
                # Use optimized parameters
                try:
                    config_name = 'RF_Optimized'
                    model_params = best_params.copy()
                    model_params['random_state'] = 42  # Add random state for reproducibility
                    
                    # Use helper method to fit and evaluate
                    fit_results[config_name] = RandomForestFitter._fit_and_evaluate_random_forest(
                        model_params, x_data, y_data, ss_tot, run_cv, config_name
                    )
                    
                except Exception as e:
                    st.warning(f"Failed to fit optimized Random Forest: {str(e)}")
            else:
                print(f"   ⚠️ Random Forest optimization failed, using default parameters")
                # Fallback to default configuration
                default_config = {
                    'name': 'RF_Default',
                    'n_estimators': 100,
                    'max_depth': 5,
                    'min_samples_split': 2,
                    'random_state': 42
                }
                
                try:
                    config_name = default_config['name']
                    model_params = {k: v for k, v in default_config.items() if k != 'name'}
                    
                    # Use helper method to fit and evaluate
                    fit_results[config_name] = RandomForestFitter._fit_and_evaluate_random_forest(
                        model_params, x_data, y_data, ss_tot, run_cv, config_name
                    )
                    
                except Exception as e:
                    st.warning(f"Failed to fit default Random Forest: {str(e)}")
        else:
            # No optimization - use default configuration
            default_config = {
                'name': 'RF_Default',
                'n_estimators': 100,
                'max_depth': 5,
                'min_samples_split': 2,
                'random_state': 42
            }
            
            try:
                config_name = default_config['name']
                model_params = {k: v for k, v in default_config.items() if k != 'name'}
                
                # Use helper method to fit and evaluate
                fit_results[config_name] = RandomForestFitter._fit_and_evaluate_random_forest(
                    model_params, x_data, y_data, ss_tot, run_cv, config_name
                )
                
            except Exception as e:
                st.warning(f"Failed to fit default Random Forest: {str(e)}")
        
        return fit_results
    
    @staticmethod
    def predict_with_rf(model_data: Dict, spectrum: np.ndarray) -> float:
        """
        Predict concentration using trained Random Forest model.
        
        Args:
            model_data: Dictionary with model, etc.
            spectrum: Full spectrum absorbance data (n_wavelengths,)
            
        Returns:
            Predicted concentration
        """
        try:
            model = model_data['model']
            # Ensure spectrum is 2D (1, n_wavelengths)
            if spectrum.ndim == 1:
                spectrum = spectrum.reshape(1, -1)
            return float(model.predict(spectrum)[0])
        except Exception as e:
            st.error(f"Error predicting with Random Forest: {str(e)}")
            return 0.0


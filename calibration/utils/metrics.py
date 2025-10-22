"""
Advanced metrics and model comparison utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, max_error
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def calculate_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive set of regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        X: Feature matrix (for adjusted metrics)
        sample_weight: Sample weights
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['r2'] = r2_score(y_true, y_pred, sample_weight=sample_weight)
    metrics['mse'] = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
    
    # Additional metrics
    metrics['max_error'] = max_error(y_true, y_pred)
    
    # MAPE (handle zeros)
    if np.all(y_true != 0):
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
    else:
        mask = y_true != 0
        if np.any(mask):
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['mape'] = np.nan
    
    # Normalized RMSE
    y_range = y_true.max() - y_true.min()
    if y_range > 0:
        metrics['nrmse'] = metrics['rmse'] / y_range
    else:
        metrics['nrmse'] = np.nan
    
    # Adjusted R²
    if X is not None and len(y_true) > X.shape[1] + 1:
        n = len(y_true)
        p = X.shape[1]
        metrics['adj_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)
        
        # AIC and BIC
        log_likelihood = -0.5 * n * (np.log(2 * np.pi * metrics['mse']) + 1)
        metrics['aic'] = 2 * p - 2 * log_likelihood
        metrics['bic'] = np.log(n) * p - 2 * log_likelihood
    
    # Residual statistics
    residuals = y_true - y_pred
    metrics['residual_mean'] = np.mean(residuals)
    metrics['residual_std'] = np.std(residuals)
    metrics['residual_skewness'] = stats.skew(residuals)
    metrics['residual_kurtosis'] = stats.kurtosis(residuals)
    
    # Durbin-Watson statistic (autocorrelation test)
    if len(residuals) > 1:
        diff = residuals[1:] - residuals[:-1]
        metrics['durbin_watson'] = np.sum(diff**2) / np.sum(residuals**2)
    
    # Relative error metrics
    rel_errors = np.abs(residuals) / (np.abs(y_true) + 1e-10)
    metrics['mean_rel_error'] = np.mean(rel_errors)
    metrics['median_rel_error'] = np.median(rel_errors)
    metrics['p95_rel_error'] = np.percentile(rel_errors, 95)
    
    # Correlation coefficient
    metrics['pearson_r'], metrics['pearson_p'] = stats.pearsonr(y_true, y_pred)
    metrics['spearman_r'], metrics['spearman_p'] = stats.spearmanr(y_true, y_pred)
    
    # Explained variance
    metrics['explained_variance'] = 1 - np.var(residuals) / np.var(y_true)
    
    return metrics


def compare_models(
    model_results: Dict[str, Any],
    test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    criteria: List[str] = ['r2', 'rmse', 'mae']
) -> pd.DataFrame:
    """
    Compare multiple models based on various criteria.
    
    Args:
        model_results: Dictionary of model_name -> ModelResult
        test_data: Optional (X_test, y_test) for test evaluation
        criteria: Metrics to compare
        
    Returns:
        DataFrame with comparison results
    """
    comparison_data = []
    
    for name, result in model_results.items():
        row = {'Model': name}
        
        # Training metrics
        for metric in criteria:
            if hasattr(result.metrics, metric):
                value = getattr(result.metrics, metric)
                row[f'Train_{metric}'] = value
        
        # Test metrics if available
        if hasattr(result, 'test_metrics'):
            for metric in criteria:
                if hasattr(result.test_metrics, metric):
                    value = getattr(result.test_metrics, metric)
                    row[f'Test_{metric}'] = value
        
        # Additional metrics
        if hasattr(result.metrics, 'cv_mean'):
            row['CV_Mean'] = result.metrics.cv_mean
            row['CV_Std'] = result.metrics.cv_std
        
        if hasattr(result.metrics, 'training_time'):
            row['Training_Time'] = result.metrics.training_time
        
        if hasattr(result.metrics, 'n_iterations'):
            row['Iterations'] = result.metrics.n_iterations
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Calculate rankings for each metric
    for col in df.columns:
        if col != 'Model' and 'Train_' in col or 'Test_' in col:
            if 'r2' in col.lower():
                # Higher is better
                df[f'{col}_Rank'] = df[col].rank(ascending=False)
            else:
                # Lower is better
                df[f'{col}_Rank'] = df[col].rank(ascending=True)
    
    # Calculate average rank
    rank_cols = [col for col in df.columns if '_Rank' in col]
    if rank_cols:
        df['Average_Rank'] = df[rank_cols].mean(axis=1)
        df = df.sort_values('Average_Rank')
    
    return df


def calculate_prediction_intervals(
    model,
    X: np.ndarray,
    confidence: float = 0.95,
    method: str = 'bootstrap',
    n_iterations: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction intervals for model predictions.
    
    Args:
        model: Trained model with predict method
        X: Input features
        confidence: Confidence level
        method: Method for interval calculation ('bootstrap' or 'quantile')
        n_iterations: Number of bootstrap iterations
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    alpha = 1 - confidence
    
    if method == 'bootstrap':
        # Bootstrap prediction intervals
        predictions = []
        n_samples = len(X)
        
        for _ in range(n_iterations):
            # Sample with replacement
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[idx]
            
            try:
                y_pred = model.predict(X_boot)
                predictions.append(y_pred)
            except:
                continue
        
        if predictions:
            predictions = np.array(predictions)
            lower = np.percentile(predictions, alpha/2 * 100, axis=0)
            upper = np.percentile(predictions, (1 - alpha/2) * 100, axis=0)
        else:
            # Fallback to point predictions
            y_pred = model.predict(X)
            lower = upper = y_pred
            
    elif method == 'quantile':
        # Quantile regression (if model supports it)
        try:
            # Try to use model's quantile prediction if available
            lower = model.predict(X, quantile=alpha/2)
            upper = model.predict(X, quantile=1-alpha/2)
        except:
            # Fallback to normal approximation
            y_pred = model.predict(X)
            # Estimate prediction std from residuals
            try:
                residuals = model.residuals_ if hasattr(model, 'residuals_') else np.zeros_like(y_pred)
                std_pred = np.std(residuals)
                z_score = stats.norm.ppf(1 - alpha/2)
                lower = y_pred - z_score * std_pred
                upper = y_pred + z_score * std_pred
            except:
                lower = upper = y_pred
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return lower, upper


def calculate_feature_stability(
    model_results: List[Any],
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate stability of feature importance across multiple model runs.
    
    Args:
        model_results: List of model results with feature_importance
        feature_names: Optional feature names
        
    Returns:
        DataFrame with feature stability metrics
    """
    # Collect all feature importances
    importance_matrix = []
    
    for result in model_results:
        if hasattr(result, 'feature_importance') and result.feature_importance:
            importances = list(result.feature_importance.values())
            importance_matrix.append(importances)
    
    if not importance_matrix:
        return pd.DataFrame()
    
    importance_matrix = np.array(importance_matrix)
    
    # Calculate statistics
    stability_data = []
    n_features = importance_matrix.shape[1]
    
    for i in range(n_features):
        feature_name = feature_names[i] if feature_names else f'Feature_{i}'
        importances = importance_matrix[:, i]
        
        stability_data.append({
            'Feature': feature_name,
            'Mean_Importance': np.mean(importances),
            'Std_Importance': np.std(importances),
            'CV_Importance': np.std(importances) / (np.mean(importances) + 1e-10),
            'Min_Importance': np.min(importances),
            'Max_Importance': np.max(importances),
            'Median_Importance': np.median(importances)
        })
    
    df = pd.DataFrame(stability_data)
    df = df.sort_values('Mean_Importance', ascending=False)
    
    return df


def performance_summary_table(
    model_results: Dict[str, Any],
    metrics_to_show: Optional[List[str]] = None
) -> str:
    """
    Create a formatted performance summary table.
    
    Args:
        model_results: Dictionary of model results
        metrics_to_show: List of metrics to display
        
    Returns:
        Formatted table string
    """
    if metrics_to_show is None:
        metrics_to_show = ['r2', 'rmse', 'mae', 'cv_mean']
    
    # Create comparison DataFrame
    df = compare_models(model_results, criteria=metrics_to_show)
    
    # Format for display
    format_dict = {
        col: '{:.4f}' if any(metric in col.lower() for metric in ['r2', 'rmse', 'mae', 'cv'])
        else '{:.2f}' if 'time' in col.lower()
        else '{:.0f}'
        for col in df.columns if col != 'Model'
    }
    
    formatted_df = df.to_string(formatters=format_dict, index=False)
    
    return formatted_df

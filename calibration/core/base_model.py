"""
Base model classes for calibration modeling.

Provides abstract base classes and common functionality for all models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
from datetime import datetime
import hashlib
import json
import pickle


@dataclass
class ModelConfig:
    """Configuration for model training and optimization."""
    
    name: str
    optimization_method: str = "random_search"
    cv_folds: int = 5
    n_trials: int = 50
    random_state: int = 42
    early_stopping: bool = True
    early_stopping_patience: int = 50
    max_epochs: Optional[int] = None
    verbose: bool = False
    enable_cache: bool = True
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_hash(self) -> str:
        """Generate unique hash for configuration."""
        config_str = json.dumps(self.__dict__, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if self.n_trials < 1:
            raise ValueError("n_trials must be at least 1")
        if self.max_epochs is not None and self.max_epochs < 1:
            raise ValueError("max_epochs must be positive")


@dataclass
class ModelMetrics:
    """Comprehensive model performance metrics."""
    
    # Basic metrics
    r2: float
    rmse: float
    mae: float
    mse: float
    
    # Advanced metrics
    adj_r2: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    mape: Optional[float] = None
    max_error: Optional[float] = None
    
    # Cross-validation metrics
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    
    # Training metrics
    training_time: Optional[float] = None
    n_iterations: Optional[int] = None
    
    def __post_init__(self):
        """Calculate derived metrics if not provided."""
        if self.cv_scores and self.cv_mean is None:
            self.cv_mean = np.mean(self.cv_scores)
            self.cv_std = np.std(self.cv_scores)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    def summary_string(self) -> str:
        """Generate summary string of key metrics."""
        summary = f"R²={self.r2:.4f}, RMSE={self.rmse:.4f}, MAE={self.mae:.4f}"
        if self.cv_mean is not None:
            summary += f", CV={self.cv_mean:.4f}±{self.cv_std:.4f}"
        return summary


@dataclass
class ModelResult:
    """Container for model training results."""
    
    model: Any  # Trained model object
    metrics: ModelMetrics
    config: ModelConfig
    predictions: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __repr__(self) -> str:
        return f"ModelResult({self.config.name}, {self.metrics.summary_string()})"
    
    def save(self, filepath: str) -> None:
        """Save model result to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelResult':
        """Load model result from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class BaseModel(ABC):
    """Abstract base class for all calibration models."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize base model.
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self._cache = {} if config.enable_cache else None
        
        # Validate configuration
        self.config.validate()
    
    @abstractmethod
    def _build_model(self, **kwargs) -> Any:
        """
        Build the underlying model object.
        
        Returns:
            Model object ready for training
        """
        pass
    
    @abstractmethod
    def _optimize_hyperparameters(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Optimize model hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Optimal hyperparameters
        """
        pass
    
    @abstractmethod
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        **kwargs
    ) -> ModelResult:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional training arguments
            
        Returns:
            ModelResult object with trained model and metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        pass
    
    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        X: Optional[np.ndarray] = None
    ) -> ModelMetrics:
        """
        Calculate comprehensive model metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            X: Feature matrix (for adjusted metrics)
            
        Returns:
            ModelMetrics object
        """
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        # Basic metrics
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Advanced metrics
        n = len(y_true)
        adj_r2 = None
        aic = None
        bic = None
        
        if X is not None:
            p = X.shape[1]  # Number of features
            if n > p + 1:
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                
                # AIC and BIC
                log_likelihood = -0.5 * n * (np.log(2 * np.pi * mse) + 1)
                aic = 2 * p - 2 * log_likelihood
                bic = np.log(n) * p - 2 * log_likelihood
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else None
        max_error = np.max(np.abs(y_true - y_pred))
        
        return ModelMetrics(
            r2=r2,
            rmse=rmse,
            mae=mae,
            mse=mse,
            adj_r2=adj_r2,
            aic=aic,
            bic=bic,
            mape=mape,
            max_error=max_error
        )
    
    def cross_validate(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[float, float, List[float]]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Tuple of (mean_score, std_score, all_scores)
        """
        from sklearn.model_selection import cross_val_score
        
        if self.model is None:
            raise ValueError("Model not initialized. Call fit() first.")
        
        scores = cross_val_score(
            self.model, X, y, 
            cv=self.config.cv_folds,
            scoring='r2'
        )
        
        return float(np.mean(scores)), float(np.std(scores)), scores.tolist()
    
    def get_cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        key_str = f"{self.config.to_hash()}_{args}"
        return hashlib.md5(str(key_str).encode()).hexdigest()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config.name})"

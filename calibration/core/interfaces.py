"""
Interfaces for the calibration modeling system.

Defines contracts for pluggable components.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from .data_structures import SpectralData, CalibrationDataset
try:
    import plotly.graph_objects as go
except ImportError:  # Optional for non-web builds
    from types import SimpleNamespace
    from typing import Any
    go = SimpleNamespace(Figure=Any)


class IDataLoader(ABC):
    """Interface for data loading implementations."""
    
    @abstractmethod
    def load_file(self, filepath: str) -> SpectralData:
        """Load a single spectral file."""
        pass
    
    @abstractmethod
    def load_multiple(self, filepaths: List[str]) -> CalibrationDataset:
        """Load multiple spectral files."""
        pass
    
    @abstractmethod
    def validate_format(self, filepath: str) -> bool:
        """Validate file format."""
        pass


class IPreprocessor(ABC):
    """Interface for spectral preprocessing methods."""
    
    @abstractmethod
    def preprocess(self, data: SpectralData) -> SpectralData:
        """Apply preprocessing to spectral data."""
        pass
    
    @abstractmethod
    def preprocess_dataset(self, dataset: CalibrationDataset) -> CalibrationDataset:
        """Apply preprocessing to entire dataset."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get preprocessing parameters."""
        pass


class IOptimizer(ABC):
    """Interface for hyperparameter optimization strategies."""
    
    @abstractmethod
    def optimize(
        self,
        objective_func: callable,
        search_space: Dict[str, Any],
        n_trials: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters.
        
        Args:
            objective_func: Function to minimize/maximize
            search_space: Hyperparameter search space
            n_trials: Number of optimization trials
            
        Returns:
            Optimal hyperparameters
        """
        pass
    
    @abstractmethod
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        pass
    
    @abstractmethod
    def get_best_trial(self) -> Dict[str, Any]:
        """Get details of best trial."""
        pass


class IVisualizer(ABC):
    """Interface for visualization components."""
    
    @abstractmethod
    def plot_spectra(
        self,
        dataset: CalibrationDataset,
        **kwargs
    ) -> go.Figure:
        """Create spectral plot."""
        pass
    
    @abstractmethod
    def plot_calibration(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs
    ) -> go.Figure:
        """Create calibration plot."""
        pass
    
    @abstractmethod
    def plot_residuals(
        self,
        residuals: np.ndarray,
        **kwargs
    ) -> go.Figure:
        """Create residual plot."""
        pass
    
    @abstractmethod
    def create_dashboard(
        self,
        results: Dict[str, Any],
        **kwargs
    ) -> go.Figure:
        """Create comprehensive dashboard."""
        pass

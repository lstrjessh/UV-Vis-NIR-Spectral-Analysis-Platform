"""
Core module for the calibration modeling system.

This module provides base classes, interfaces, and core functionality
for the spectroscopic calibration and machine learning platform.
"""

from .base_model import BaseModel, ModelResult, ModelConfig
from .data_structures import SpectralData, CalibrationDataset, ModelMetrics
from .exceptions import CalibrationError, DataValidationError, ModelTrainingError
from .interfaces import IDataLoader, IPreprocessor, IOptimizer, IVisualizer

__all__ = [
    'BaseModel',
    'ModelResult', 
    'ModelConfig',
    'SpectralData',
    'CalibrationDataset',
    'ModelMetrics',
    'CalibrationError',
    'DataValidationError',
    'ModelTrainingError',
    'IDataLoader',
    'IPreprocessor',
    'IOptimizer',
    'IVisualizer'
]

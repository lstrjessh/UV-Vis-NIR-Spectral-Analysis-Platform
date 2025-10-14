"""
Custom exceptions for the calibration modeling system.
"""


class CalibrationError(Exception):
    """Base exception for calibration-related errors."""
    pass


class DataValidationError(CalibrationError):
    """Raised when data validation fails."""
    pass


class ModelTrainingError(CalibrationError):
    """Raised when model training encounters an error."""
    pass


class OptimizationError(CalibrationError):
    """Raised when hyperparameter optimization fails."""
    pass


class FileProcessingError(CalibrationError):
    """Raised when file processing encounters an error."""
    pass


class ConfigurationError(CalibrationError):
    """Raised when configuration is invalid."""
    pass

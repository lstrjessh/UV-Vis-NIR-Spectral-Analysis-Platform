"""
Core data structures for spectral data handling.

Provides efficient data containers with validation and preprocessing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime


@dataclass
class SpectralData:
    """Container for spectral data with metadata."""
    
    wavelengths: np.ndarray
    absorbance: np.ndarray
    filename: str
    concentration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate data after initialization."""
        self._validate()
        self._ensure_arrays()
    
    def _validate(self):
        """Validate spectral data."""
        if len(self.wavelengths) != len(self.absorbance):
            raise ValueError("Wavelengths and absorbance must have same length")
        
        if len(self.wavelengths) < 2:
            raise ValueError("Need at least 2 data points")
        
        # Check for reasonable ranges
        if np.any(self.wavelengths < 100) or np.any(self.wavelengths > 3000):
            self.metadata['warning'] = "Unusual wavelength range detected"
        
        if np.any(self.absorbance < -1) or np.any(self.absorbance > 10):
            self.metadata['warning'] = "Unusual absorbance values detected"
    
    def _ensure_arrays(self):
        """Ensure data is in numpy array format."""
        self.wavelengths = np.asarray(self.wavelengths, dtype=np.float64)
        self.absorbance = np.asarray(self.absorbance, dtype=np.float64)
    
    def interpolate(self, new_wavelengths: np.ndarray) -> np.ndarray:
        """
        Interpolate absorbance to new wavelength grid.
        
        Args:
            new_wavelengths: Target wavelength grid
            
        Returns:
            Interpolated absorbance values
        """
        return np.interp(new_wavelengths, self.wavelengths, self.absorbance)
    
    def smooth(self, window_size: int = 5, polyorder: int = 2) -> 'SpectralData':
        """
        Apply Savitzky-Golay smoothing.
        
        Args:
            window_size: Size of smoothing window
            polyorder: Polynomial order
            
        Returns:
            New SpectralData with smoothed absorbance
        """
        from scipy.signal import savgol_filter
        
        if len(self.absorbance) < window_size:
            return self  # Can't smooth if too few points
        
        smoothed_abs = savgol_filter(self.absorbance, window_size, polyorder)
        
        return SpectralData(
            wavelengths=self.wavelengths.copy(),
            absorbance=smoothed_abs,
            filename=self.filename,
            concentration=self.concentration,
            metadata={**self.metadata, 'smoothed': True}
        )
    
    def derivative(self, order: int = 1) -> 'SpectralData':
        """
        Calculate spectral derivative.
        
        Args:
            order: Derivative order (1 or 2)
            
        Returns:
            New SpectralData with derivative
        """
        if order == 1:
            deriv = np.gradient(self.absorbance, self.wavelengths)
        elif order == 2:
            deriv = np.gradient(np.gradient(self.absorbance, self.wavelengths), self.wavelengths)
        else:
            raise ValueError("Only 1st and 2nd order derivatives supported")
        
        return SpectralData(
            wavelengths=self.wavelengths.copy(),
            absorbance=deriv,
            filename=self.filename,
            concentration=self.concentration,
            metadata={**self.metadata, f'derivative_order': order}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'wavelengths': self.wavelengths.tolist(),
            'absorbance': self.absorbance.tolist(),
            'filename': self.filename,
            'concentration': self.concentration,
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        return (f"SpectralData(file={self.filename}, "
                f"points={len(self.wavelengths)}, "
                f"range=[{self.wavelengths[0]:.1f}-{self.wavelengths[-1]:.1f}nm], "
                f"conc={self.concentration})")


@dataclass
class CalibrationDataset:
    """Container for complete calibration dataset."""
    
    spectra: List[SpectralData]
    name: str = "Calibration Dataset"
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived attributes."""
        self._common_wavelengths = None
        self._matrix_cache = {}
        self._validate()
    
    def _validate(self):
        """Validate dataset."""
        if not self.spectra:
            raise ValueError("Dataset must contain at least one spectrum")
        
        # Check concentration availability
        self.has_concentrations = all(s.concentration is not None for s in self.spectra)
        
        if self.has_concentrations:
            concentrations = [s.concentration for s in self.spectra]
            if len(set(concentrations)) < len(concentrations):
                self.metadata['warning'] = "Duplicate concentrations detected"
    
    @property
    def common_wavelengths(self) -> np.ndarray:
        """Get common wavelength grid across all spectra."""
        if self._common_wavelengths is None:
            # Find common wavelength range
            min_wl = max(s.wavelengths.min() for s in self.spectra)
            max_wl = min(s.wavelengths.max() for s in self.spectra)
            
            if min_wl >= max_wl:
                raise ValueError("No common wavelength range found")
            
            # Create common grid with 1nm resolution
            self._common_wavelengths = np.arange(min_wl, max_wl + 1, 1.0)
        
        return self._common_wavelengths
    
    def to_matrix(self, 
                  interpolate: bool = True,
                  derivative: Optional[int] = None,
                  smooth: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert dataset to matrix format for ML.
        
        Args:
            interpolate: Whether to interpolate to common grid
            derivative: Derivative order (None, 1, or 2)
            smooth: Whether to apply smoothing
            
        Returns:
            Tuple of (X_matrix, y_vector, wavelengths)
        """
        # Check cache
        cache_key = (interpolate, derivative, smooth)
        if cache_key in self._matrix_cache:
            return self._matrix_cache[cache_key]
        
        # Process spectra
        processed_spectra = self.spectra.copy()
        
        if smooth:
            processed_spectra = [s.smooth() for s in processed_spectra]
        
        if derivative is not None:
            processed_spectra = [s.derivative(derivative) for s in processed_spectra]
        
        # Build matrix
        if interpolate:
            wavelengths = self.common_wavelengths
            X = []
            for s in processed_spectra:
                # Ensure we have valid data for interpolation
                if len(s.wavelengths) >= 2:
                    interpolated = s.interpolate(wavelengths)
                else:
                    # If not enough points, use constant value
                    interpolated = np.full(len(wavelengths), s.absorbance[0] if len(s.absorbance) > 0 else 0.0)
                X.append(interpolated)
            X = np.array(X)
        else:
            # Use original wavelengths (assuming all same)
            wavelengths = processed_spectra[0].wavelengths
            X = np.array([s.absorbance for s in processed_spectra])
        
        # Extract concentrations
        if self.has_concentrations:
            y = np.array([s.concentration for s in self.spectra])
        else:
            y = np.array([])
        
        # Validate data for NaN and inf values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            # Replace NaN and inf with 0
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            self.metadata['warning'] = "NaN or inf values detected and replaced with 0"
        
        if len(y) > 0 and (np.any(np.isnan(y)) or np.any(np.isinf(y))):
            raise ValueError("Concentration values contain NaN or inf - please clean your data")
        
        # Cache result
        self._matrix_cache[cache_key] = (X, y, wavelengths)
        
        return X, y, wavelengths
    
    def split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple['CalibrationDataset', 'CalibrationDataset']:
        """
        Split dataset into train and test sets.
        
        Args:
            test_size: Fraction for test set
            random_state: Random seed
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        from sklearn.model_selection import train_test_split
        
        indices = np.arange(len(self.spectra))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        
        train_spectra = [self.spectra[i] for i in train_idx]
        test_spectra = [self.spectra[i] for i in test_idx]
        
        train_dataset = CalibrationDataset(
            spectra=train_spectra,
            name=f"{self.name} - Train",
            metadata={**self.metadata, 'split': 'train'}
        )
        
        test_dataset = CalibrationDataset(
            spectra=test_spectra,
            name=f"{self.name} - Test",
            metadata={**self.metadata, 'split': 'test'}
        )
        
        return train_dataset, test_dataset
    
    def filter_by_concentration(self, min_conc: float = 0, max_conc: float = float('inf')) -> 'CalibrationDataset':
        """Filter dataset by concentration range."""
        filtered = [s for s in self.spectra 
                   if s.concentration is not None and min_conc <= s.concentration <= max_conc]
        
        return CalibrationDataset(
            spectra=filtered,
            name=f"{self.name} - Filtered",
            metadata={**self.metadata, 'filtered': True, 'conc_range': (min_conc, max_conc)}
        )
    
    def summary(self) -> Dict[str, Any]:
        """Get dataset summary statistics."""
        summary = {
            'n_spectra': len(self.spectra),
            'has_concentrations': self.has_concentrations,
            'wavelength_range': (self.common_wavelengths[0], self.common_wavelengths[-1]),
            'n_wavelengths': len(self.common_wavelengths)
        }
        
        if self.has_concentrations:
            concentrations = [s.concentration for s in self.spectra]
            summary.update({
                'concentration_range': (min(concentrations), max(concentrations)),
                'concentration_mean': np.mean(concentrations),
                'concentration_std': np.std(concentrations)
            })
        
        return summary
    
    def __len__(self) -> int:
        return len(self.spectra)
    
    def __repr__(self) -> str:
        return f"CalibrationDataset(name='{self.name}', n_spectra={len(self.spectra)})"


# Re-export ModelMetrics from base_model to avoid circular import
from .base_model import ModelMetrics
__all__ = ['SpectralData', 'CalibrationDataset', 'ModelMetrics']

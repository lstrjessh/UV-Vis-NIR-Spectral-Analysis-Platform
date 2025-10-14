"""
Preprocessing implementations for spectral data.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from ..core.interfaces import IPreprocessor
from ..core.data_structures import SpectralData, CalibrationDataset
import copy


class StandardPreprocessor(IPreprocessor):
    """Standard preprocessing pipeline for spectral data."""
    
    def __init__(self,
                 smoothing: bool = False,
                 smoothing_window: int = 5,
                 smoothing_polyorder: int = 2,
                 derivative: Optional[int] = None,
                 normalization: Optional[str] = None,
                 baseline_correction: bool = False):
        """
        Initialize standard preprocessor.
        
        Args:
            smoothing: Apply Savitzky-Golay smoothing
            smoothing_window: Window size for smoothing
            smoothing_polyorder: Polynomial order for smoothing
            derivative: Derivative order (None, 1, or 2)
            normalization: Normalization method ('standard', 'minmax', 'robust', 'snv')
            baseline_correction: Apply baseline correction
        """
        self.smoothing = smoothing
        self.smoothing_window = smoothing_window
        self.smoothing_polyorder = smoothing_polyorder
        self.derivative = derivative
        self.normalization = normalization
        self.baseline_correction = baseline_correction
        
        # Initialize scalers if needed
        self._init_scalers()
    
    def _init_scalers(self):
        """Initialize normalization scalers."""
        self.scaler = None
        if self.normalization == 'standard':
            self.scaler = StandardScaler()
        elif self.normalization == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.normalization == 'robust':
            self.scaler = RobustScaler()
    
    def preprocess(self, data: SpectralData) -> SpectralData:
        """
        Apply preprocessing to single spectrum.
        
        Args:
            data: Input spectral data
            
        Returns:
            Preprocessed spectral data
        """
        # Create copy to avoid modifying original
        processed = copy.deepcopy(data)
        
        # Apply baseline correction
        if self.baseline_correction:
            processed.absorbance = self._correct_baseline(
                processed.wavelengths, 
                processed.absorbance
            )
        
        # Apply smoothing
        if self.smoothing and len(processed.absorbance) > self.smoothing_window:
            processed.absorbance = signal.savgol_filter(
                processed.absorbance,
                self.smoothing_window,
                self.smoothing_polyorder
            )
        
        # Apply derivative
        if self.derivative is not None:
            processed.absorbance = self._calculate_derivative(
                processed.wavelengths,
                processed.absorbance,
                self.derivative
            )
        
        # Apply SNV normalization
        if self.normalization == 'snv':
            processed.absorbance = self._apply_snv(processed.absorbance)
        
        # Update metadata
        processed.metadata['preprocessed'] = True
        processed.metadata['preprocessing'] = self.get_parameters()
        
        return processed
    
    def preprocess_dataset(self, dataset: CalibrationDataset) -> CalibrationDataset:
        """
        Apply preprocessing to entire dataset.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Preprocessed dataset
        """
        # Preprocess each spectrum
        processed_spectra = [self.preprocess(s) for s in dataset.spectra]
        
        # Apply matrix-level normalization if needed
        if self.normalization in ['standard', 'minmax', 'robust']:
            # First get the matrix without preprocessing to get original shape
            X_original, y, wavelengths = CalibrationDataset(
                spectra=processed_spectra,
                name=dataset.name
            ).to_matrix(interpolate=True)
            
            # Fit and transform
            if self.scaler is not None:
                X_scaled = self.scaler.fit_transform(X_original.T).T  # Scale features
                
                # Update spectra with scaled values, maintaining original wavelengths
                for i, spectrum in enumerate(processed_spectra):
                    # Interpolate scaled values back to original wavelengths if needed
                    if len(wavelengths) == len(spectrum.wavelengths) and np.allclose(wavelengths, spectrum.wavelengths):
                        spectrum.absorbance = X_scaled[i]
                    else:
                        # Interpolate back to original wavelength grid
                        spectrum.absorbance = np.interp(spectrum.wavelengths, wavelengths, X_scaled[i])
        
        # Create new dataset
        processed_dataset = CalibrationDataset(
            spectra=processed_spectra,
            name=f"{dataset.name} (Preprocessed)",
            metadata={
                **dataset.metadata,
                'preprocessed': True,
                'preprocessing': self.get_parameters()
            }
        )
        
        return processed_dataset
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get preprocessing parameters."""
        return {
            'smoothing': self.smoothing,
            'smoothing_window': self.smoothing_window if self.smoothing else None,
            'smoothing_polyorder': self.smoothing_polyorder if self.smoothing else None,
            'derivative': self.derivative,
            'normalization': self.normalization,
            'baseline_correction': self.baseline_correction
        }
    
    def _correct_baseline(self, wavelengths: np.ndarray, absorbance: np.ndarray) -> np.ndarray:
        """Apply baseline correction using asymmetric least squares."""
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
        
        def baseline_als(y, lam=1e6, p=0.01, niter=10):
            """Asymmetric Least Squares baseline correction."""
            L = len(y)
            D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
            w = np.ones(L)
            for i in range(niter):
                W = sparse.spdiags(w, 0, L, L)
                Z = W + lam * D.dot(D.transpose())
                z = spsolve(Z, w * y)
                w = p * (y > z) + (1 - p) * (y < z)
            return z
        
        try:
            baseline = baseline_als(absorbance)
            return absorbance - baseline
        except:
            # Fallback to simple linear baseline
            coeffs = np.polyfit([wavelengths[0], wavelengths[-1]], 
                               [absorbance[0], absorbance[-1]], 1)
            baseline = np.polyval(coeffs, wavelengths)
            return absorbance - baseline
    
    def _calculate_derivative(self, wavelengths: np.ndarray, absorbance: np.ndarray, order: int) -> np.ndarray:
        """Calculate spectral derivative."""
        if order == 1:
            return np.gradient(absorbance, wavelengths)
        elif order == 2:
            first_deriv = np.gradient(absorbance, wavelengths)
            return np.gradient(first_deriv, wavelengths)
        else:
            raise ValueError(f"Unsupported derivative order: {order}")
    
    def _apply_snv(self, absorbance: np.ndarray) -> np.ndarray:
        """Apply Standard Normal Variate transformation."""
        mean = np.mean(absorbance)
        std = np.std(absorbance)
        if std > 0:
            return (absorbance - mean) / std
        return absorbance - mean


class AdvancedPreprocessor(StandardPreprocessor):
    """Advanced preprocessing with additional techniques."""
    
    def __init__(self,
                 smoothing: bool = False,
                 smoothing_window: int = 5,
                 smoothing_polyorder: int = 2,
                 derivative: Optional[int] = None,
                 normalization: Optional[str] = None,
                 baseline_correction: bool = False,
                 msc: bool = False,
                 detrend: bool = False,
                 wavelet_denoise: bool = False,
                 spike_removal: bool = False):
        """
        Initialize advanced preprocessor.
        
        Args:
            msc: Apply Multiplicative Scatter Correction
            detrend: Apply detrending
            wavelet_denoise: Apply wavelet denoising
            spike_removal: Remove spikes/outliers
        """
        super().__init__(smoothing, smoothing_window, smoothing_polyorder,
                        derivative, normalization, baseline_correction)
        
        self.msc = msc
        self.detrend = detrend
        self.wavelet_denoise = wavelet_denoise
        self.spike_removal = spike_removal
        self.reference_spectrum = None
    
    def preprocess(self, data: SpectralData) -> SpectralData:
        """Apply advanced preprocessing to single spectrum."""
        # Apply standard preprocessing first
        processed = super().preprocess(data)
        
        # Remove spikes
        if self.spike_removal:
            processed.absorbance = self._remove_spikes(processed.absorbance)
        
        # Apply wavelet denoising
        if self.wavelet_denoise:
            processed.absorbance = self._wavelet_denoise(processed.absorbance)
        
        # Apply detrending
        if self.detrend:
            processed.absorbance = signal.detrend(processed.absorbance)
        
        return processed
    
    def preprocess_dataset(self, dataset: CalibrationDataset) -> CalibrationDataset:
        """Apply advanced preprocessing to dataset."""
        # For MSC, we need the reference spectrum
        if self.msc:
            X, _, _ = dataset.to_matrix()
            self.reference_spectrum = np.mean(X, axis=0)
        
        # Process each spectrum
        processed_spectra = []
        for spectrum in dataset.spectra:
            processed = self.preprocess(spectrum)
            
            # Apply MSC if enabled
            if self.msc and self.reference_spectrum is not None:
                processed.absorbance = self._apply_msc(
                    processed.absorbance,
                    self.reference_spectrum
                )
            
            processed_spectra.append(processed)
        
        # Apply remaining matrix-level normalizations
        processed_dataset = CalibrationDataset(
            spectra=processed_spectra,
            name=f"{dataset.name} (Advanced Preprocessing)",
            metadata={
                **dataset.metadata,
                'preprocessed': True,
                'preprocessing': self.get_parameters()
            }
        )
        
        # Apply final normalization if needed
        if self.normalization in ['standard', 'minmax', 'robust']:
            return super().preprocess_dataset(processed_dataset)
        
        return processed_dataset
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get all preprocessing parameters."""
        params = super().get_parameters()
        params.update({
            'msc': self.msc,
            'detrend': self.detrend,
            'wavelet_denoise': self.wavelet_denoise,
            'spike_removal': self.spike_removal
        })
        return params
    
    def _remove_spikes(self, absorbance: np.ndarray, threshold: float = 3) -> np.ndarray:
        """Remove spikes using median absolute deviation."""
        median = np.median(absorbance)
        mad = np.median(np.abs(absorbance - median))
        
        if mad > 0:
            modified_z_scores = 0.6745 * (absorbance - median) / mad
            mask = np.abs(modified_z_scores) < threshold
            
            # Interpolate spike positions
            if np.any(~mask):
                indices = np.arange(len(absorbance))
                absorbance[~mask] = np.interp(indices[~mask], indices[mask], absorbance[mask])
        
        return absorbance
    
    def _wavelet_denoise(self, absorbance: np.ndarray) -> np.ndarray:
        """Apply wavelet denoising."""
        try:
            import pywt
            
            # Decompose
            coeffs = pywt.wavedec(absorbance, 'db4', level=4)
            
            # Estimate noise level
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            
            # Threshold
            threshold = sigma * np.sqrt(2 * np.log(len(absorbance)))
            coeffs_thresh = list(coeffs)
            coeffs_thresh[1:] = [pywt.threshold(c, threshold, 'soft') for c in coeffs_thresh[1:]]
            
            # Reconstruct
            return pywt.waverec(coeffs_thresh, 'db4', mode='per')[:len(absorbance)]
            
        except ImportError:
            # Fallback to simple smoothing
            return signal.savgol_filter(absorbance, 5, 2)
    
    def _apply_msc(self, absorbance: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Apply Multiplicative Scatter Correction."""
        # Fit linear model: absorbance = a + b * reference
        fit = np.polyfit(reference, absorbance, 1)
        
        # Correct spectrum
        return (absorbance - fit[1]) / fit[0]

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# Check if XGBoost is available
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    # Don't show warning at module level - will show in sidebar instead
from functools import lru_cache
import io
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import warnings
import hashlib
import pickle
import base64

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import modularized models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models import (
    PLSRFitter,
    NeuralNetworkFitter,
    RandomForestFitter,
    SVRFitter,
    CNN1DFitter,
    XGBoostFitter
)

# Import shared utilities
from utils.shared_utils import (
    extract_concentration_from_filename,
    SUPPORTED_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    COLOR_PALETTE
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Page Configuration ---
st.set_page_config(
    page_title="Mathematical Modelling",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---

# --- Session State Initialization ---
def initialize_session_state():
    """Initialize session state variables for the machine learning analysis."""
    defaults = {
        'last_file_hash': None,
        'cached_data': None,
        'model_results_cache': {},
        'concentration_data': {},
        'analysis_completed': False,
        'analysis_requested': False,
        'analysis_stop_requested': False,
        'show_concentration_inputs': True,
        'analysis_data': {},
        'show_mlp_cal': True,
        'show_plsr_cal': True,
        'show_rf_cal': True,
        'show_svr_cal': True,
        'show_cnn_cal': True,
        'show_xgb_cal': True,
        'cnn_model': 'CNN1D_Simple',
        'run_cv': True,  # Enable cross-validation by default
        'enable_early_stopping': True,  # Enable early stopping by default
        'early_stopping_patience': 25,  # Consistent with updated defaults
        'n_trials': 30,  # Default number of optimization trials
        'optimization_method': 'Random Search'  # Default optimization method
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- Enhanced Data Processing Classes ---
class FileProcessor:
    """Enhanced file processing with caching and validation."""
    
    
    @staticmethod
    def _compute_file_hash(uploaded_files: List) -> str:
        """Compute hash of uploaded files for caching."""
        if not uploaded_files:
            return ""
        
        hasher = hashlib.md5()
        for file in uploaded_files:
            hasher.update(file.name.encode())
            hasher.update(str(file.size).encode())
        return hasher.hexdigest()
    
    @staticmethod
    @st.cache_data(ttl=3600, max_entries=10)
    def read_absorbance_file(file_content: bytes, filename: str) -> Optional[pd.DataFrame]:
        """
        Read and validate absorbance CSV file with enhanced error handling.
        
        Args:
            file_content: Raw file content as bytes
            filename: Name of the file for error reporting
            
        Returns:
            Validated DataFrame or None if invalid
        """
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    content_str = file_content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                st.error(f"Could not decode file '{filename}' with any encoding")
                return None
                
            content_io = io.StringIO(content_str)
            
            # Try different separators
            for sep in [',', ';', '\t']:
                try:
                    content_io.seek(0)
                    df = pd.read_csv(content_io, sep=sep)
                    if len(df.columns) >= 2:
                        break
                except:
                    continue
            else:
                st.error(f"Could not parse CSV file '{filename}'")
                return None
            
            # Validate required columns (case-insensitive)
            df.columns = df.columns.str.strip()
            col_mapping = {}
            # --- Start of Replacement ---
            absorbance_candidates = []
            for col in df.columns:
                col_lower = col.lower()
                # Find Nanometers column
                if 'nanometer' in col_lower or 'wavelength' in col_lower or 'nm' in col_lower:
                    col_mapping['Nanometers'] = col
                # Find all possible Absorbance columns
                elif 'absorb' in col_lower or 'abs' in col_lower:
                    absorbance_candidates.append(col)

            # Select the best absorbance candidate
            if absorbance_candidates:
                # 1. Look for an exact (case-insensitive) match
                exact_matches = [c for c in absorbance_candidates if c.lower() == 'absorbance']
                if exact_matches:
                    col_mapping['Absorbance'] = exact_matches[0]
                else:
                    # 2. If no exact match, sort by length (longest first) to prefer more descriptive names
                    absorbance_candidates.sort(key=len, reverse=True)
                    col_mapping['Absorbance'] = absorbance_candidates[0]
            # --- End of Replacement ---
            
            if len(col_mapping) < 2:
                st.error(f"File '{filename}' missing required columns. Need wavelength and absorbance columns.")
                return None
            
            # Rename columns
            df = df.rename(columns=col_mapping)
            
            # Convert to numeric and validate
            for col in ['Nanometers', 'Absorbance']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove invalid rows
            initial_rows = len(df)
            df = df.dropna(subset=['Nanometers', 'Absorbance'])
            
            if len(df) == 0:
                st.error(f"File '{filename}' contains no valid numeric data.")
                return None
            
            # Check for reasonable data ranges
            if df['Nanometers'].min() < 100 or df['Nanometers'].max() > 3000:
                st.warning(f"File '{filename}': Unusual wavelength range detected")
            
            if df['Absorbance'].min() < -1 or df['Absorbance'].max() > 10:
                st.warning(f"File '{filename}': Unusual absorbance range detected")
            
            if len(df) < initial_rows * 0.8:
                st.warning(f"File '{filename}': Removed {initial_rows - len(df)} invalid rows.")
            
            # Sort by wavelength and remove duplicates
            df = df.sort_values('Nanometers').drop_duplicates(subset=['Nanometers']).reset_index(drop=True)
            
            return df
            
        except Exception as e:
            st.error(f"Error reading file '{filename}': {str(e)}")
            return None
    
    @staticmethod
    def process_uploaded_files(uploaded_files: List) -> Dict[str, pd.DataFrame]:
        """Process multiple uploaded files with progress tracking and caching."""
        if not uploaded_files:
            return {}
        
        # Check cache
        file_hash = FileProcessor._compute_file_hash(uploaded_files)
        if (st.session_state.last_file_hash == file_hash and 
            st.session_state.cached_data is not None):
            return st.session_state.cached_data
        
        valid_data = {}
        progress_bar = st.progress(0, text="Processing uploaded files...")
        
        for i, file in enumerate(uploaded_files):
            try:
                # Check file size
                if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    st.warning(f"File '{file.name}' too large (>{MAX_FILE_SIZE_MB}MB)")
                    continue
                
                if file.size == 0:
                    st.warning(f"File '{file.name}' is empty")
                    continue
                
                # Read and validate file
                content = file.read()
                file.seek(0)
                
                df = FileProcessor.read_absorbance_file(content, file.name)
                if df is not None and len(df) > 1:  # Need at least 2 points
                    valid_data[file.name] = df
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            except Exception as e:
                st.error(f"Error processing '{file.name}': {str(e)}")
        
        progress_bar.empty()
        
        # Cache results
        st.session_state.last_file_hash = file_hash
        st.session_state.cached_data = valid_data
        
        return valid_data

class PeakAnalyzer:
    """Analysis methods for wavelength selection and linearity."""
    
    
    @staticmethod
    def extract_full_spectrum_data(data_dict: Dict[str, pd.DataFrame], 
                                   concentrations: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts full spectrum data from multiple datasets.
        
        Args:
            data_dict: Dictionary of filename -> DataFrame with Nanometers and Absorbance columns
            concentrations: Dictionary of filename -> concentration value
            
        Returns:
            Tuple of (concentrations_array, spectra_matrix, wavelengths_array)
            - concentrations_array: 1D array of shape (n_samples,)
            - spectra_matrix: 2D array of shape (n_samples, n_wavelengths)
            - wavelengths_array: 1D array of shape (n_wavelengths,)
        """
        if not data_dict or not concentrations:
            return np.array([]), np.array([[]]), np.array([])
        
        # Filter valid files with positive concentrations
        valid_files = [(filename, df) for filename, df in data_dict.items() 
                      if filename in concentrations and concentrations[filename] > 0]
        
        if len(valid_files) < 2:
            st.warning("⚠️ Need at least 2 files with valid concentrations for analysis")
            return np.array([]), np.array([[]]), np.array([])
        
        # Find common wavelength range across all spectra
        min_wl = max(df['Nanometers'].min() for _, df in valid_files)
        max_wl = min(df['Nanometers'].max() for _, df in valid_files)
        
        if min_wl >= max_wl:
            st.error("❌ No common wavelength range found across all files")
            return np.array([]), np.array([[]]), np.array([])
        
        # Create common wavelength grid (1nm resolution)
        wavelengths = np.arange(min_wl, max_wl + 1, 1.0)
        n_wavelengths = len(wavelengths)
        n_samples = len(valid_files)
        
        # Initialize arrays
        concentrations_array = np.zeros(n_samples)
        spectra_matrix = np.zeros((n_samples, n_wavelengths))
        
        # Extract and interpolate spectra
        for i, (filename, df) in enumerate(valid_files):
            try:
                concentrations_array[i] = concentrations[filename]
                
                # Interpolate spectrum onto common wavelength grid
                if len(df) >= 2:
                    spectra_matrix[i, :] = np.interp(wavelengths, 
                                                     df['Nanometers'].values,
                                                     df['Absorbance'].values)
                elif len(df) == 1:
                    # For single-point data, use that value for all wavelengths
                    spectra_matrix[i, :] = df['Absorbance'].iloc[0]
            except Exception as e:
                st.warning(f"⚠️ Error extracting spectrum from {filename}: {str(e)}")
                concentrations_array[i] = concentrations.get(filename, np.nan)
                spectra_matrix[i, :] = 0.0
        
        return concentrations_array, spectra_matrix, wavelengths


# --- Model Training Infrastructure ---


@dataclass
class TrainerConfig:
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = None
    cache_tokens: Tuple[Any, ...] = ()

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.cache_tokens is None:
            self.cache_tokens = ()


@dataclass
class ModelTrainer:
    result_key: str
    label: str
    state_key: str
    available: bool
    caption: str
    warn_message: str
    error_message: str
    train_fn: Callable[..., Dict[str, Dict]]
    config_builder: Callable[[np.ndarray, np.ndarray], TrainerConfig]
    success_formatter: Callable[[Dict[str, Dict]], str]

    def run(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Dict]:
        if not self.available or not st.session_state.get(self.state_key, False):
            return {}

        try:
            config = self.config_builder(x_data, y_data)
        except Exception as exc:
            st.error(f"{self.error_message}: {exc}")
            return {}

        cache_key = (self.result_key, config.cache_tokens)
        previous_result = st.session_state.model_results_cache.get(cache_key)
        if previous_result is not None:
            return previous_result

        try:
            st.caption(self.caption)
            results = self.train_fn(*config.args, **config.kwargs)

            if results:
                st.caption(self.success_formatter(results))
                st.session_state.model_results_cache[cache_key] = results
            else:
                st.warning(self.warn_message)

            return results or {}

        except Exception as exc:
            st.error(f"{self.error_message}: {exc}")
            return {}

        if not self.available or not st.session_state.get(self.state_key, False):
            return {}

        try:
            config = self.config_builder(x_data, y_data)
        except Exception as exc:
            st.error(f"{self.error_message}: {exc}")
            return {}

        cache_key = (self.result_key, config.cache_tokens)
        previous_result = st.session_state.model_results_cache.get(cache_key)
        if previous_result is not None:
            return previous_result

        try:
            st.caption(self.caption)
            results = self.train_fn(*config.args, **config.kwargs)

            if results:
                st.caption(self.success_formatter(results))
                st.session_state.model_results_cache[cache_key] = results
            else:
                st.warning(self.warn_message)

            return results or {}

        except Exception as exc:
            st.error(f"{self.error_message}: {exc}")
            return {}


def build_model_trainers() -> List[ModelTrainer]:
    def build_mlp_config(x_data: np.ndarray, y_data: np.ndarray) -> TrainerConfig:
        optimization_method = st.session_state.get('mlp_optimization_method', 'Random Search')
        cv_folds = st.session_state.get('cv_folds', 5)
        max_epochs = st.session_state.get('mlp_max_epochs', 1000)
        enable_early_stopping = st.session_state.get('mlp_enable_early_stopping', True)
        early_stopping_patience = st.session_state.get('mlp_early_stopping_patience', 50)
        n_trials = st.session_state.get('mlp_n_trials', 20)
        run_cv = st.session_state.get('run_cv', True)

        cache_tokens = (
            optimization_method,
            cv_folds,
            max_epochs,
            enable_early_stopping,
            early_stopping_patience,
            n_trials,
            run_cv,
            tuple(tuple(row) for row in np.round(x_data, 8)),
            tuple(float(val) for val in np.round(y_data, 8)),
        )

        return TrainerConfig(
            args=(
                tuple(tuple(row) for row in x_data),
                tuple(float(val) for val in y_data),
                None,
                optimization_method,
                cv_folds,
                max_epochs,
                enable_early_stopping,
                early_stopping_patience,
                n_trials,
                run_cv,
            ),
            cache_tokens=cache_tokens
        )

    def build_plsr_config(x_data: np.ndarray, y_data: np.ndarray) -> TrainerConfig:
        optimization_method = st.session_state.get('plsr_optimization_method', 'Bayesian')
        cv_folds = st.session_state.get('cv_folds', 5)
        max_components = st.session_state.get('plsr_max_components', 10)
        run_cv = st.session_state.get('run_cv', True)

        cache_tokens = (
            optimization_method,
            cv_folds,
            max_components,
            run_cv,
            tuple(tuple(row) for row in np.round(x_data, 8)),
            tuple(float(val) for val in np.round(y_data, 8)),
        )

        return TrainerConfig(
            args=(
                tuple(tuple(row) for row in x_data),
                tuple(float(val) for val in y_data),
                max_components,
                optimization_method,
                cv_folds,
                20,
                run_cv,
            ),
            cache_tokens=cache_tokens
        )

    def build_rf_config(x_data: np.ndarray, y_data: np.ndarray) -> TrainerConfig:
        optimization_method = st.session_state.get('rf_optimization_method', 'Random Search')
        cv_folds = st.session_state.get('cv_folds', 5)
        n_trials = st.session_state.get('rf_n_trials', 30)
        run_cv = st.session_state.get('run_cv', True)

        cache_tokens = (
            optimization_method,
            cv_folds,
            n_trials,
            run_cv,
            tuple(tuple(row) for row in np.round(x_data, 8)),
            tuple(float(val) for val in np.round(y_data, 8)),
        )

        return TrainerConfig(
            args=(
                tuple(tuple(row) for row in x_data),
                tuple(float(val) for val in y_data),
                None,
                optimization_method,
                cv_folds,
                n_trials,
                run_cv,
            ),
            cache_tokens=cache_tokens
        )

    def build_svr_config(x_data: np.ndarray, y_data: np.ndarray) -> TrainerConfig:
        optimization_method = st.session_state.get('svr_optimization_method', 'Random Search')
        kernel_types = st.session_state.get('svr_kernel_types', ['rbf', 'linear'])
        cv_folds = st.session_state.get('cv_folds', 5)
        n_trials = st.session_state.get('svr_n_trials', 30)
        run_cv = st.session_state.get('run_cv', True)

        svr_config = {'kernel_types': kernel_types}

        cache_tokens = (
            optimization_method,
            cv_folds,
            n_trials,
            run_cv,
            tuple(sorted(kernel_types)),
            tuple(tuple(row) for row in np.round(x_data, 8)),
            tuple(float(val) for val in np.round(y_data, 8)),
        )

        return TrainerConfig(
            args=(
                tuple(tuple(row) for row in x_data),
                tuple(float(val) for val in y_data),
                svr_config,
                optimization_method,
                cv_folds,
                n_trials,
                run_cv,
            ),
            cache_tokens=cache_tokens
        )

    def build_cnn_config(x_data: np.ndarray, y_data: np.ndarray) -> TrainerConfig:
        optimization_method = st.session_state.get('cnn_optimization_method', 'Random Search')
        cv_folds = st.session_state.get('cv_folds', 5)
        max_epochs = st.session_state.get('cnn_max_epochs', 1000)
        enable_early_stopping = st.session_state.get('cnn_enable_early_stopping', True)
        early_stopping_patience = st.session_state.get('cnn_early_stopping_patience', 50)
        n_trials = st.session_state.get('cnn_n_trials', 20)
        run_cv = st.session_state.get('run_cv', True)

        cache_tokens = (
            optimization_method,
            cv_folds,
            max_epochs,
            enable_early_stopping,
            early_stopping_patience,
            n_trials,
            run_cv,
            tuple(tuple(row) for row in np.round(x_data, 8)),
            tuple(float(val) for val in np.round(y_data, 8)),
        )

        return TrainerConfig(
            args=(
                tuple(tuple(row) for row in x_data),
                tuple(float(val) for val in y_data),
                None,
                optimization_method,
                cv_folds,
                max_epochs,
                enable_early_stopping,
                early_stopping_patience,
                n_trials,
                run_cv,
            ),
            cache_tokens=cache_tokens
        )

    def build_xgb_config(x_data: np.ndarray, y_data: np.ndarray) -> TrainerConfig:
        optimization_method = st.session_state.get('xgb_optimization_method', 'Random Search')
        cv_folds = st.session_state.get('cv_folds', 5)
        n_trials = st.session_state.get('xgb_n_trials', 30)
        run_cv = st.session_state.get('run_cv', True)

        cache_tokens = (
            optimization_method,
            cv_folds,
            n_trials,
            run_cv,
            tuple(tuple(row) for row in np.round(x_data, 8)),
            tuple(float(val) for val in np.round(y_data, 8)),
        )

        return TrainerConfig(
            args=(
                tuple(tuple(row) for row in x_data),
                tuple(float(val) for val in y_data),
                None,
                optimization_method,
                cv_folds,
                n_trials,
                run_cv,
            ),
            cache_tokens=cache_tokens
        )

    trainers: List[ModelTrainer] = [
        ModelTrainer(
            result_key='nn',
            label='MLP',
            state_key='enable_mlp',
            available=TORCH_AVAILABLE,
            caption="Training MLP models...",
            warn_message="⚠️ MLP training failed",
            error_message="❌ MLP fitting failed",
            train_fn=NeuralNetworkFitter.fit_neural_networks,
            config_builder=build_mlp_config,
            success_formatter=lambda res: f"✅ Trained {len(res)} MLP model(s)",
        ),
        ModelTrainer(
            result_key='plsr',
            label='PLSR',
            state_key='enable_plsr',
            available=True,
            caption="Training PLSR models...",
            warn_message="⚠️ PLSR training failed",
            error_message="❌ PLSR fitting failed",
            train_fn=PLSRFitter.fit_plsr,
            config_builder=build_plsr_config,
            success_formatter=lambda res: f"✅ Trained {len(res)} PLSR model(s)",
        ),
        ModelTrainer(
            result_key='rf',
            label='Random Forest',
            state_key='enable_rf',
            available=True,
            caption="Training Random Forest models...",
            warn_message="⚠️ Random Forest training failed",
            error_message="❌ Random Forest fitting failed",
            train_fn=RandomForestFitter.fit_random_forest,
            config_builder=build_rf_config,
            success_formatter=lambda res: f"✅ Trained {len(res)} Random Forest model(s)",
        ),
        ModelTrainer(
            result_key='svr',
            label='SVR',
            state_key='enable_svr',
            available=True,
            caption="Training SVR models...",
            warn_message="⚠️ SVR training failed",
            error_message="❌ SVR fitting failed",
            train_fn=SVRFitter.fit_svr,
            config_builder=build_svr_config,
            success_formatter=lambda res: f"✅ Trained {len(res)} SVR model(s)",
        ),
        ModelTrainer(
            result_key='cnn',
            label='1D-CNN',
            state_key='enable_cnn',
            available=TORCH_AVAILABLE,
            caption="Training 1D-CNN models...",
            warn_message="⚠️ 1D-CNN training failed",
            error_message="❌ 1D-CNN fitting failed",
            train_fn=CNN1DFitter.fit_cnn1d,
            config_builder=build_cnn_config,
            success_formatter=lambda res: "✅ Trained 1D-CNN model(s)",
        ),
        ModelTrainer(
            result_key='xgb',
            label='XGBoost',
            state_key='enable_xgb',
            available=XGB_AVAILABLE,
            caption="Training XGBoost models...",
            warn_message="⚠️ XGBoost training failed",
            error_message="❌ XGBoost fitting failed",
            train_fn=XGBoostFitter.fit_xgboost,
            config_builder=build_xgb_config,
            success_formatter=lambda res: f"✅ Trained {len(res)} XGBoost model(s)",
        ),
    ]

    return trainers


def run_model_trainers(x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Dict[str, Dict]]:
    trainer_results: Dict[str, Dict[str, Dict]] = {}

    for trainer in build_model_trainers():
        if st.session_state.get('analysis_stop_requested'):
            st.warning("Analysis halted by user request.")
            break
        trainer_results[trainer.result_key] = trainer.run(x_data, y_data)

    return trainer_results


# Model classes have been moved to separate modules in the models/ directory
class PlotManager:
    """Advanced plotting utilities for spectral analysis."""
    
    
    
    @staticmethod
    def create_concentration_absorbance_plot(data_dict: Dict[str, pd.DataFrame], 
                                           concentrations: Dict[str, float], 
                                           target_wavelength: float) -> go.Figure:
        """Create concentration vs absorbance plot at specific wavelength."""
        fig = go.Figure()
        
        # Extract data points
        conc_abs_pairs = []
        filenames = []
        
        for filename, df in data_dict.items():
            if filename in concentrations and concentrations[filename] > 0:
                concentration = concentrations[filename]
                
                # Interpolate absorbance at target wavelength
                if len(df) >= 2:
                    absorbance = np.interp(target_wavelength, df['Nanometers'].values, df['Absorbance'].values)
                elif len(df) == 1:
                    absorbance = df['Absorbance'].iloc[0]
                else:
                    continue
                
                conc_abs_pairs.append((concentration, absorbance))
                filenames.append(filename)
        
        if not conc_abs_pairs:
            fig.add_annotation(
                text="No valid data points found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Sort by concentration
        sorted_data = sorted(zip(conc_abs_pairs, filenames), key=lambda x: x[0][0])
        concentrations_sorted = [pair[0] for pair, _ in sorted_data]
        absorbances_sorted = [pair[1] for pair, _ in sorted_data]
        filenames_sorted = [filename for _, filename in sorted_data]
        
        # Plot data points
        fig.add_trace(go.Scatter(
            x=concentrations_sorted,
            y=absorbances_sorted,
            mode='markers+lines',
            name='Data Points',
            marker=dict(size=8, color='blue'),
            line=dict(color='blue', width=1, dash='dot'),
            text=filenames_sorted,
            hovertemplate='File: %{text}<br>Concentration: %{x:.3f}<br>Absorbance: %{y:.4f}<extra></extra>'
        ))
        
        # Calculate and plot linear regression
        if len(concentrations_sorted) >= 2:
            coefficients = np.polyfit(concentrations_sorted, absorbances_sorted, 1)
            polynomial = np.poly1d(coefficients)
            
            # Generate smooth line
            conc_range = np.linspace(min(concentrations_sorted), max(concentrations_sorted), 100)
            abs_fit = polynomial(conc_range)
            
            fig.add_trace(go.Scatter(
                x=conc_range,
                y=abs_fit,
                mode='lines',
                name=f'Linear Fit (slope = {coefficients[0]:.4f})',
                line=dict(color='red', width=2),
                hovertemplate='Concentration: %{x:.3f}<br>Fitted Absorbance: %{y:.4f}<extra></extra>'
            ))
            
            # Calculate R²
            predicted_abs = polynomial(concentrations_sorted)
            ss_res = np.sum((absorbances_sorted - predicted_abs) ** 2)
            ss_tot = np.sum((absorbances_sorted - np.mean(absorbances_sorted)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0
            
            # Add equation and R² to plot
            equation_text = f"A = {coefficients[0]:.4f} × C + {coefficients[1]:.4f}<br>R² = {r_squared:.4f}"
            fig.add_annotation(
                text=equation_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1
            )
        
        fig.update_layout(
            title=f'Concentration vs Absorbance at {target_wavelength:.1f} nm',
            xaxis_title='Concentration',
            yaxis_title='Absorbance',
            template='plotly_white',
            hovermode='closest',
            height=400,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_spectral_overview(data_dict: Dict[str, pd.DataFrame], 
                                concentrations: Dict[str, float], 
                                target_wavelength: float) -> go.Figure:
        """Create comprehensive spectral overview showing all spectra with concentration info."""
        fig = go.Figure()
        
        if not data_dict:
            fig.add_annotation(
                text="No spectral data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Sort files by concentration for consistent coloring
        sorted_files = []
        for filename, df in data_dict.items():
            concentration = concentrations.get(filename, 0)
            sorted_files.append((concentration, filename, df))
        
        sorted_files.sort(key=lambda x: x[0])
        
        # Plot each spectrum
        for i, (concentration, filename, df) in enumerate(sorted_files):
            # Choose color based on concentration
            color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
            
            # Create display name
            display_name = f"{filename} (C={concentration:.3f})" if concentration > 0 else filename
            
            # Plot spectrum
            fig.add_trace(go.Scatter(
                x=df['Nanometers'],
                y=df['Absorbance'],
                mode='lines',
                name=display_name,
                line=dict(color=color, width=2),
                hovertemplate=f'<b>{filename}</b><br>Concentration: {concentration:.3f}<br>Wavelength: %{{x:.1f}} nm<br>Absorbance: %{{y:.4f}}<extra></extra>'
            ))
        
        # Highlight target wavelength
        if target_wavelength:
            fig.add_vline(
                x=target_wavelength,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Analysis Wavelength: {target_wavelength:.1f} nm",
                annotation_position="top left"
            )
        
        fig.update_layout(
            title='Spectral Overview - All Uploaded Files',
            xaxis_title='Wavelength (nm)',
            yaxis_title='Absorbance',
            template='plotly_white',
            hovermode='closest',
            height=600,
            showlegend=True
        )
        
        return fig

# --- Main Application ---
def main():
    """Enhanced main application with better organization."""
    
    # Header
    st.title("Spectroscopic Data Analysis & Machine Learning Platform")
    # Count available models
    available_models = ["PLSR", "MLP", "Random Forest", "SVR"]
    if TORCH_AVAILABLE:
        available_models.append("1D-CNN")
    if XGB_AVAILABLE:
        available_models.append("XGBoost")
    
    model_count = len(available_models)
    model_list = ", ".join(available_models)
    
    st.markdown(f"""
    ### Comprehensive Spectral Analysis & Predictive Modeling
    
    Advanced machine learning platform for spectroscopic data analysis, featuring {model_count} state-of-the-art algorithms: **{model_list}**.
    
    *Full-spectrum multivariate analysis with automated model selection, performance evaluation, and export capabilities.*
    """)
    
    # Sidebar configuration
    setup_sidebar()
    
    # Main content
    with st.container():
        # File upload section
        st.subheader("📁 Data Upload")
        uploaded_files = st.file_uploader(
            "Upload Absorbance CSV Files",
            type=SUPPORTED_EXTENSIONS,
            accept_multiple_files=True,
            help="Upload CSV files with 'Nanometers' and 'Absorbance' columns"
        )
        
        if uploaded_files:
            process_and_analyze(uploaded_files)
        else:
            display_welcome_message()

def setup_sidebar():
    """Setup sidebar with enhanced controls and better state management."""
    with st.sidebar:
        st.header("⚙️ Analysis Configuration")
        
        # Model selection criterion
        model_criterion = st.selectbox(
            "Model Selection Criterion",
            ["r_squared", "adj_r_squared", "aic", "bic"],
            index=0,
            help="Criterion for selecting best model",
            key="model_criterion_select"
        )
        st.session_state.model_selection_criterion = model_criterion
        
        st.markdown("---")
        st.markdown("### 🔬 Model Configuration")
        
        # === NEURAL NETWORKS ===
        st.markdown("### 🧠 Neural Networks")
        
        # Initialize default values
        enable_mlp = getattr(st.session_state, 'enable_mlp', True)
        mlp_optimization = "Bayesian"  # Best for expensive PyTorch training
        mlp_max_epochs = 1000
        mlp_early_stopping = True
        mlp_patience = 50
        
        # MLP Settings
        with st.expander("⚙️ Multi-Layer Perceptron (MLP)", expanded=False):
            enable_mlp = st.checkbox(
                "Enable MLP",
                value=enable_mlp,
                help="Feed-forward neural network",
                key="enable_mlp_check"
            )
            
            if enable_mlp:
                mlp_optimization = st.selectbox(
                    "Optimization Method",
                    ["Bayesian", "Random Search", "Grid Search"],
                    index=0,
                    help="Bayesian is best for expensive neural network training",
                    key="mlp_optimization_select"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    mlp_max_epochs = st.slider(
                        "Maximum Epochs",
                        min_value=100,
                        max_value=5000,
                        value=1000,
                        step=100,
                        help="Maximum training epochs for final model",
                        key="mlp_max_epochs_slider"
                    )
                with col2:
                    mlp_n_trials = st.slider(
                        "Hyperparameter Trials",
                        min_value=10,
                        max_value=50,
                        value=20,
                        step=5,
                        help="Number of hyperparameter configurations to test",
                        key="mlp_n_trials_slider"
                    )
                
                mlp_early_stopping = st.checkbox(
                    "Enable Early Stopping",
                    value=True,
                    help="Stop if validation loss doesn't improve",
                    key="mlp_early_stopping_check"
                )
                
                if mlp_early_stopping:
                    mlp_patience = st.slider(
                        "Early Stopping Patience",
                        min_value=10,
                        max_value=100,
                        value=50,
                        step=5,
                        help="Epochs to wait before stopping",
                        key="mlp_patience_slider"
                    )
                else:
                    mlp_patience = 50
            else:
                mlp_n_trials = 20  # Default when MLP disabled
        
        # Initialize CNN defaults
        enable_cnn = getattr(st.session_state, 'enable_cnn', True) if TORCH_AVAILABLE else False
        cnn_optimization = "Bayesian"  # Best for expensive PyTorch training
        cnn_max_epochs = 1000
        cnn_early_stopping = True
        cnn_patience = 50
        
        # CNN Settings
        if TORCH_AVAILABLE:
            with st.expander("⚙️ 1D-CNN (Convolutional)", expanded=False):
                enable_cnn = st.checkbox(
                    "Enable 1D-CNN",
                    value=enable_cnn,
                    help="1D Convolutional Neural Network",
                    key="enable_cnn_check"
                )
                
                if enable_cnn:
                    cnn_optimization = st.selectbox(
                        "Optimization Method",
                        ["Bayesian", "Random Search", "Grid Search"],
                        index=0,
                        help="Bayesian is best for expensive neural network training",
                        key="cnn_optimization_select"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        cnn_max_epochs = st.slider(
                            "Maximum Epochs",
                            min_value=100,
                            max_value=5000,
                            value=1000,
                            step=100,
                            help="Maximum training epochs for final model",
                            key="cnn_max_epochs_slider"
                        )
                    with col2:
                        cnn_n_trials = st.slider(
                            "Hyperparameter Trials",
                            min_value=10,
                            max_value=50,
                            value=20,
                            step=5,
                            help="Number of hyperparameter configurations to test",
                            key="cnn_n_trials_slider"
                        )
                    
                    cnn_early_stopping = st.checkbox(
                        "Enable Early Stopping",
                        value=True,
                        help="Stop if validation loss doesn't improve",
                        key="cnn_early_stopping_check"
                    )
                    
                    if cnn_early_stopping:
                        cnn_patience = st.slider(
                            "Early Stopping Patience",
                            min_value=10,
                            max_value=100,
                            value=50,
                            step=5,
                            help="Epochs to wait before stopping",
                            key="cnn_patience_slider"
                        )
                    else:
                        cnn_patience = 50
                else:
                    cnn_n_trials = 20  # Default when CNN disabled
        else:
            enable_cnn = False
            st.info("ℹ️ 1D-CNN requires PyTorch installation")
        
        st.markdown("---")
        
        # === CLASSICAL ML MODELS ===
        st.markdown("### 📊 Classical ML Models")
        
        # Initialize PLSR defaults
        enable_plsr = getattr(st.session_state, 'enable_plsr', True)
        plsr_optimization = "Grid Search"  # Fast, only 1 parameter (n_components)
        plsr_max_components = 10
        
        # PLSR Settings
        with st.expander("⚙️ Partial Least Squares Regression (PLSR)", expanded=False):
            # Handle quick select
            if st.session_state.get('quick_select_all'):
                checkbox_value = True
            elif st.session_state.get('quick_select_none'):
                checkbox_value = False
            else:
                checkbox_value = enable_plsr
            
            enable_plsr = st.checkbox(
                "Enable PLSR",
                value=checkbox_value,
                help="Linear dimensionality reduction + regression",
                key="enable_plsr_check"
            )
            
            if enable_plsr:
                plsr_optimization = st.selectbox(
                    "Optimization Method",
                    ["Grid Search", "Random Search", "Bayesian"],
                    index=0,
                    help="Method to find optimal number of components",
                    key="plsr_optimization_select"
                )
                
                plsr_max_components = st.slider(
                    "Maximum Components",
                    min_value=1,
                    max_value=20,
                    value=10,
                    help="Maximum number of PLS components to test",
                    key="plsr_max_components_slider"
                )
        
        # Initialize SVR defaults
        enable_svr = getattr(st.session_state, 'enable_svr', True)
        svr_optimization = "Random Search"  # Better for many hyperparameters
        svr_kernel_types = ["rbf", "linear"]
        svr_n_trials = 50  # Increased for better exploration
        
        # SVR Settings
        with st.expander("⚙️ Support Vector Regression (SVR)", expanded=False):
            # Handle quick select
            if st.session_state.get('quick_select_all'):
                checkbox_value = True
            elif st.session_state.get('quick_select_none'):
                checkbox_value = False
            else:
                checkbox_value = enable_svr
            
            enable_svr = st.checkbox(
                "Enable SVR",
                value=checkbox_value,
                help="Kernel-based regression",
                key="enable_svr_check"
            )
            
            if enable_svr:
                svr_optimization = st.selectbox(
                    "Optimization Method",
                    ["Random Search", "Bayesian", "Grid Search"],
                    index=0,
                    help="Random Search is efficient for SVR's many hyperparameters",
                    key="svr_optimization_select"
                )
                
                svr_kernel_types = st.multiselect(
                    "Kernel Types to Test",
                    ["rbf", "linear", "poly"],
                    default=["rbf", "linear"],
                    help="SVR kernel functions to evaluate",
                    key="svr_kernel_multiselect"
                )
                
                svr_n_trials = st.slider(
                    "Optimization Trials",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10,
                    help="Number of hyperparameter combinations to try (more is better)",
                    key="svr_n_trials_slider"
                )
        
        # Initialize Random Forest defaults
        enable_rf = getattr(st.session_state, 'enable_rf', True)
        rf_optimization = "Random Search"  # Better for many hyperparameters
        rf_n_trials = 50  # Increased for better exploration
        
        # Random Forest Settings
        with st.expander("⚙️ Random Forest", expanded=False):
            # Handle quick select
            if st.session_state.get('quick_select_all'):
                checkbox_value = True
            elif st.session_state.get('quick_select_none'):
                checkbox_value = False
            else:
                checkbox_value = enable_rf
            
            enable_rf = st.checkbox(
                "Enable Random Forest",
                value=checkbox_value,
                help="Ensemble of decision trees",
                key="enable_rf_check"
            )
            
            if enable_rf:
                rf_optimization = st.selectbox(
                    "Optimization Method",
                    ["Random Search", "Bayesian", "Grid Search"],
                    index=0,
                    help="Random Search is efficient for Random Forest's many hyperparameters",
                    key="rf_optimization_select"
                )
                
                rf_n_trials = st.slider(
                    "Optimization Trials",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10,
                    help="Number of hyperparameter combinations to try (more is better)",
                    key="rf_n_trials_slider"
                )
        
        # Initialize XGBoost defaults
        enable_xgb = getattr(st.session_state, 'enable_xgb', True) if XGB_AVAILABLE else False
        xgb_optimization = "Random Search"  # Better for many hyperparameters
        xgb_n_trials = 100  # Increased - XGBoost has many hyperparameters
        
        # XGBoost Settings
        if XGB_AVAILABLE:
            with st.expander("⚙️ XGBoost", expanded=False):
                # Handle quick select
                if st.session_state.get('quick_select_all'):
                    checkbox_value = True
                elif st.session_state.get('quick_select_none'):
                    checkbox_value = False
                else:
                    checkbox_value = enable_xgb
                
                enable_xgb = st.checkbox(
                    "Enable XGBoost",
                    value=checkbox_value,
                    help="Gradient boosting regression",
                    key="enable_xgb_check"
                )
                
                if enable_xgb:
                    xgb_optimization = st.selectbox(
                        "Optimization Method",
                        ["Random Search", "Bayesian", "Grid Search"],
                        index=0,
                        help="Random Search is efficient for XGBoost's many hyperparameters",
                        key="xgb_optimization_select"
                    )
                    
                    xgb_n_trials = st.slider(
                        "Optimization Trials",
                        min_value=20,
                        max_value=300,
                        value=100,
                        step=20,
                        help="Number of hyperparameter combinations to try (XGBoost needs more)",
                        key="xgb_n_trials_slider"
                    )
        else:
            enable_xgb = False
            st.info("ℹ️ XGBoost requires installation: `pip install xgboost`")
        
        st.markdown("---")
        
        # === GLOBAL SETTINGS ===
        st.markdown("### ⚙️ Global Settings")
        
        cv_folds = st.slider(
            "Cross-Validation Folds",
            min_value=2,
            max_value=10,
            value=5,
            help="K-folds for cross-validation (applies to all models)",
            key="cv_folds_slider"
        )
        
        run_cv = st.checkbox(
            "Enable Full Cross-Validation",
            value=True,
            help="Run CV on final models (slower but more robust evaluation)",
            key="run_cv_check"
        )
        
        # Update session state with model-specific settings
        st.session_state.enable_mlp = enable_mlp
        st.session_state.enable_nn = enable_mlp  # Backward compatibility
        st.session_state.enable_plsr = enable_plsr
        st.session_state.enable_rf = enable_rf
        st.session_state.enable_svr = enable_svr
        st.session_state.enable_cnn = enable_cnn and TORCH_AVAILABLE
        st.session_state.enable_xgb = enable_xgb and XGB_AVAILABLE
        st.session_state.analysis_stop_requested = False
        
        # Global settings
        st.session_state.cv_folds = cv_folds
        st.session_state.run_cv = run_cv
        
        # MLP-specific settings
        if enable_mlp:
            st.session_state.mlp_optimization_method = mlp_optimization
            st.session_state.mlp_max_epochs = mlp_max_epochs
            st.session_state.mlp_enable_early_stopping = mlp_early_stopping
            st.session_state.mlp_early_stopping_patience = mlp_patience
            st.session_state.mlp_n_trials = mlp_n_trials
        else:
            # Defaults for backward compatibility
            st.session_state.mlp_optimization_method = "Bayesian"
            st.session_state.mlp_max_epochs = 1000
            st.session_state.mlp_enable_early_stopping = True
            st.session_state.mlp_early_stopping_patience = 50
            st.session_state.mlp_n_trials = 20
        
        # CNN-specific settings
        if enable_cnn:
            st.session_state.cnn_optimization_method = cnn_optimization
            st.session_state.cnn_max_epochs = cnn_max_epochs
            st.session_state.cnn_enable_early_stopping = cnn_early_stopping
            st.session_state.cnn_early_stopping_patience = cnn_patience
            st.session_state.cnn_n_trials = cnn_n_trials
        else:
            st.session_state.cnn_optimization_method = "Bayesian"
            st.session_state.cnn_max_epochs = 1000
            st.session_state.cnn_enable_early_stopping = True
            st.session_state.cnn_early_stopping_patience = 50
            st.session_state.cnn_n_trials = 20
        
        # PLSR-specific settings
        if enable_plsr:
            st.session_state.plsr_optimization_method = plsr_optimization
            st.session_state.plsr_max_components = plsr_max_components
        else:
            st.session_state.plsr_optimization_method = "Grid Search"
            st.session_state.plsr_max_components = 10
        
        # SVR-specific settings
        if enable_svr:
            st.session_state.svr_optimization_method = svr_optimization
            st.session_state.svr_kernel_types = svr_kernel_types if svr_kernel_types else ["rbf"]
            st.session_state.svr_n_trials = svr_n_trials
        else:
            st.session_state.svr_optimization_method = "Random Search"
            st.session_state.svr_kernel_types = ["rbf", "linear"]
            st.session_state.svr_n_trials = 50
        
        # RF-specific settings
        if enable_rf:
            st.session_state.rf_optimization_method = rf_optimization
            st.session_state.rf_n_trials = rf_n_trials
        else:
            st.session_state.rf_optimization_method = "Random Search"
            st.session_state.rf_n_trials = 50
        
        # XGBoost-specific settings
        if enable_xgb:
            st.session_state.xgb_optimization_method = xgb_optimization
            st.session_state.xgb_n_trials = xgb_n_trials
        else:
            st.session_state.xgb_optimization_method = "Random Search"
            st.session_state.xgb_n_trials = 100
        
        # Backward compatibility - use MLP settings as defaults
        st.session_state.optimization_method = st.session_state.get('mlp_optimization_method', 'Bayesian')
        st.session_state.max_epochs = st.session_state.get('mlp_max_epochs', 1000)
        st.session_state.enable_early_stopping = st.session_state.get('mlp_enable_early_stopping', True)
        st.session_state.early_stopping_patience = st.session_state.get('mlp_early_stopping_patience', 50)
        st.session_state.n_trials = st.session_state.get('rf_n_trials', 30)
        
        # Display settings
        with st.expander("🎨 Display Options", expanded=False):
            pass  # No polynomial equations to show anymore
            
            show_statistics = st.checkbox(
                "Show Detailed Statistics", True,
                key="show_statistics_check"
            )
            
            plot_confidence = st.checkbox(
                "Plot Confidence Intervals", False,
                key="plot_confidence_check"
            )
            
            # Update session state
            st.session_state.show_statistics = show_statistics
            st.session_state.plot_confidence_intervals = plot_confidence
            
            # Calibration Smoothing
            smooth_calibration_data = st.checkbox(
                "Smooth Calibration Data", False,
                key="smooth_calibration_data_check"
            )
            st.session_state.smooth_calibration_data = smooth_calibration_data

def process_and_analyze(uploaded_files):
    """Process uploaded files and perform full spectrum analysis."""
    
    # Check if files have changed (reset analysis if they have)
    current_file_hash = FileProcessor._compute_file_hash(uploaded_files)
    last_file_hash = st.session_state.get('last_file_hash')
    
    # Only reset analysis if files actually changed AND we have a previous hash
    if current_file_hash != last_file_hash and last_file_hash is not None:
        # Files changed, reset analysis state
        st.session_state.analysis_completed = False
        st.session_state.analysis_requested = False
        st.session_state.analysis_data = {}
    
    # Always update the file hash
    st.session_state.last_file_hash = current_file_hash
    
    # Process files
    with st.spinner("🔄 Processing uploaded files..."):
        start_time = time.time()
        all_data = FileProcessor.process_uploaded_files(uploaded_files)
        processing_time = time.time() - start_time
    
    if not all_data:
        st.error("❌ No valid files could be processed.")
        return
    
    st.caption(f"✅ Successfully processed {len(all_data)} file(s) in {processing_time:.2f} seconds")
    
    # Concentration input
    analysis_in_progress = st.session_state.get('analysis_requested', False) and not st.session_state.get('analysis_completed', False)

    if analysis_in_progress:
        st.subheader("📊 Concentration Data")
        st.info("⏳ **Analysis in Progress** - Training models with your data. Use 'Stop Analysis' to cancel.")
        concentrations = dict(st.session_state.concentration_data)
        analysis_ready = False
    else:
        st.subheader("📊 Concentration Data")
        concentrations, analysis_ready = setup_concentration_input(all_data)

    controls_col1, controls_col2, controls_col3 = st.columns([1, 1, 1])
    with controls_col1:
        analysis_ready_button = st.button(
            "🚀 Start Analysis",
            key="start_analysis_button",
            use_container_width=True,
            disabled=analysis_in_progress,
            help="Run wavelength selection and modelling with the current concentrations"
        )
    with controls_col2:
        stop_pressed = st.button(
            "🛑 Stop Analysis",
            key="stop_analysis_button",
            use_container_width=True,
            disabled=not analysis_in_progress,
            help="Cancel ongoing model training"
        )
        if stop_pressed:
            st.session_state.analysis_stop_requested = True
            st.toast("Analysis stop requested", icon="🛑")
    with controls_col3:
        if st.button("🔄 Clear Cached Results", key="clear_results_button", use_container_width=True,
                     help="Forget prior model results and retrain on next run"):
            st.session_state.model_results_cache = {}
            st.success("Cached model results cleared.")
    
    # Only trigger analysis when button is explicitly clicked (not auto-start)
    if analysis_ready_button:
        st.session_state.analysis_data = {
            'concentrations': concentrations,
            'all_data': all_data
        }
        st.session_state.analysis_requested = True
        # Reset analysis completed flag to allow retraining
        st.session_state.analysis_completed = False
        st.session_state.analysis_stop_requested = False
    
    # Perform analysis if requested and not already completed
    if st.session_state.get('analysis_requested', False) and not st.session_state.get('analysis_completed', False):
        if st.session_state.get('analysis_stop_requested'):
            st.warning("Analysis stopped. Adjust settings and start again when ready.")
            st.session_state.analysis_requested = False
            st.session_state.analysis_stop_requested = False
            return
        analysis_data = st.session_state.get('analysis_data', {})
        if analysis_data:
            # Extract full spectrum data (no wavelength selection needed)
            st.subheader("🌈 Full Spectrum Analysis")
            st.info("Using full spectral data for all models")
            
            concentrations_array, spectra_matrix, wavelengths = PeakAnalyzer.extract_full_spectrum_data(
                analysis_data['all_data'], analysis_data['concentrations']
            )
            
            if len(concentrations_array) >= 2:
                # Perform machine learning analysis with full spectrum data
                fit_analysis(concentrations_array, spectra_matrix, wavelengths, analysis_data['all_data'], analysis_data['concentrations'])
            # Mark analysis as completed
            st.session_state.analysis_completed = True
        else:
                st.warning("Need at least 2 data points for analysis.")
    
    # If analysis is completed, show the results
    elif st.session_state.get('analysis_completed', False):
        analysis_data = st.session_state.get('analysis_data', {})
        if analysis_data:
            # Extract full spectrum data for display
            concentrations_array, spectra_matrix, wavelengths = PeakAnalyzer.extract_full_spectrum_data(
                analysis_data['all_data'], analysis_data['concentrations']
            )
            
            if len(concentrations_array) >= 2:
                # Show the analysis results without re-running
                st.subheader("🌈 Full Spectrum Analysis")
                st.info("Using full spectral data for all models")
                fit_analysis(concentrations_array, spectra_matrix, wavelengths, analysis_data['all_data'], analysis_data['concentrations'])
    
    # Show appropriate messages (less intrusive)
    if not st.session_state.get('analysis_completed', False) and not analysis_in_progress:
        if analysis_ready and not all(conc > 0 for conc in concentrations.values()):
            st.warning("⚠️ Please ensure all concentrations are greater than 0.")
        elif not analysis_ready:
            st.caption("💡 **Tip:** Enter concentrations for all files, then click 'Start Analysis'")

def setup_concentration_input(data_dict: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, float], bool]:
    """
    Setup concentration input interface with better validation.
    
    Args:
        data_dict: Dictionary of filename -> DataFrame with spectral data
        
    Returns:
        Tuple of (concentrations_dict, all_valid_flag)
        - concentrations_dict: Dictionary of filename -> concentration value
        - all_valid_flag: Boolean indicating if all concentrations are > 0
    """
    st.write("Enter the analyte concentration for each uploaded file:")
    
    # Show info about filename auto-detection
    with st.expander("💡 Filename Auto-Detection", expanded=False):
        st.markdown("""
        Concentrations can be automatically extracted from filenames. Recognized patterns:
        - **With units**: `0.1mL.csv`, `2.5ppm.csv`, `10ppb.csv`, `1.5mg_L.csv`, `3mg/L.csv`
        - **Element names**: `Cu_0.1ppm.csv`, `Pb_2.5mg_L.csv`, `Zn-1.0-ppb.csv`
        - **Underscore/dash separated**: `sample_0.5_mM.csv`, `test-1.2-ppm.csv`
        - **Concentration prefix**: `concentration_0.5.csv`, `conc-1.2.csv`
        - **Sample prefix**: `sample_0.1.csv`, `sample-2.5.csv`
        - **Just numbers**: `0.1.csv`, `2.5.csv`
        
        Supported units: mL, ppm, ppb, mg, mg/L, μg, mM, μM, nM, M, g, g/L, L
        """)
    
    if not data_dict:
        st.warning("No valid data files to configure concentrations for.")
        return {}, False
    
    concentrations = {}
    num_files = len(data_dict)
    num_cols = min(num_files, 4)
    
    # Handle case with many files
    if num_files > 12:
        st.info(f"You have {num_files} files. Consider using batch upload for large datasets.")
    
    cols = st.columns(num_cols)
    
    # Count auto-detected concentrations
    auto_detected_count = 0
    
    for i, filename in enumerate(sorted(data_dict.keys())):  # Sort for consistent order
        with cols[i % num_cols]:
            # Try to extract concentration from filename first
            extracted_conc = extract_concentration_from_filename(filename)
            
            # Use extracted concentration if available and not already set
            if extracted_conc is not None and filename not in st.session_state.concentration_data:
                default_value = extracted_conc
                auto_detected_count += 1
            else:
                default_value = st.session_state.concentration_data.get(filename, 1.0)
            
            concentrations[filename] = st.number_input(
                f"Concentration",
                min_value=0.0,
                max_value=1e6,  # Reasonable upper limit
                value=default_value,
                step=0.1,
                format="%.3f",
                key=f"conc_{i}_{filename[:10]}",  # Stable key with file truncation
                help=f"Concentration for {filename}"
            )
            
            # Show abbreviated filename if too long
            display_name = filename if len(filename) <= 20 else filename[:17] + "..."
            
            # Add indicator if concentration was auto-detected
            if extracted_conc is not None and filename not in st.session_state.concentration_data:
                st.caption(f"{display_name} ✨", help=f"Concentration auto-detected: {extracted_conc}")
            else:
                st.caption(display_name)
    
    # Validate concentrations
    invalid_files = [f for f, c in concentrations.items() if c <= 0]
    
    # Store in session state
    st.session_state.concentration_data.update(concentrations)
    
    # Show auto-detection summary if any were detected
    if auto_detected_count > 0:
        st.success(f"Auto-detected {auto_detected_count} concentration{'s' if auto_detected_count > 1 else ''}")
    
    # Show validation results
    col1, col2 = st.columns([3, 2])
    
    with col1:
        if invalid_files:
            st.warning(f"Files with invalid concentrations (≤0): {', '.join(invalid_files[:3])}" + 
                       (f" and {len(invalid_files)-3} more" if len(invalid_files) > 3 else ""))
        else:
            st.caption(f"✅ All {len(concentrations)} concentration values are valid")
    
    all_valid = len(invalid_files) == 0

    with col2:
        if all_valid:
            st.success("Ready to start analysis")
        else:
            st.error("❌ Fix errors before running analysis")
    
    # Show summary of concentrations
    if concentrations:
        # Check if analysis was completed before
        has_previous_analysis = st.session_state.get('analysis_completed', False)
        expander_text = "📋 Concentration Summary" + (" (Previous analysis available)" if has_previous_analysis else "")
        
        with st.expander(expander_text, expanded=False):
            summary_data = []
            for filename, conc in sorted(concentrations.items()):
                summary_data.append({
                    'File': filename,
                    'Concentration': conc,
                    'Status': '✅ Valid' if conc > 0 else '❌ Invalid'
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            if has_previous_analysis:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info("Previous analysis results are available. Click 'Start Analysis' to run with current concentrations.")
                with col2:
                    if st.button("🔄 Reset Analysis", help="Clear previous analysis results"):
                        st.session_state.analysis_completed = False
                        st.session_state.analysis_requested = False
                        st.session_state.analysis_data = {}
                        st.rerun()
    
    if all_valid:
        st.info("💡 All concentrations are set! Click 'Start Analysis' above to proceed.")

    return concentrations, all_valid


def fit_analysis(concentrations_array: np.ndarray, spectra_matrix: np.ndarray, wavelengths: np.ndarray,
                 all_data: Dict[str, pd.DataFrame], concentrations_dict: Dict[str, float]):
    """Perform machine learning analysis with comprehensive results using full spectrum data."""
    
    st.subheader("🤖 Machine Learning Analysis Results")
    st.info(f"Full spectrum: {len(concentrations_array)} samples × {len(wavelengths)} wavelengths")
    
    with st.spinner("Performing analysis..."):
        start_time = time.time()
        
        # --- Smooth calibration data if enabled (apply to each wavelength) ---
        if st.session_state.get('smooth_calibration_data', False):
            window_length = 5
            polyorder = 2
            
            if len(concentrations_array) > window_length:
                try:
                    # Sort by concentration
                    sort_indices = np.argsort(concentrations_array)
                    concentrations_sorted = concentrations_array[sort_indices]
                    spectra_sorted = spectra_matrix[sort_indices, :]
                    
                    # Apply smoothing to each wavelength (column-wise)
                    spectra_smoothed = np.zeros_like(spectra_sorted)
                    for i in range(spectra_sorted.shape[1]):
                        spectra_smoothed[:, i] = savgol_filter(
                            spectra_sorted[:, i],
                        window_length=window_length,
                        polyorder=polyorder
                    )
                    
                    # Use smoothed data
                    x_data = spectra_smoothed  # Full spectrum features
                    y_data = concentrations_sorted  # Concentrations as targets
                    st.toast("✅ Calibration data smoothed across all wavelengths.", icon="✨")
                except ValueError as e:
                    st.warning(f"Could not apply smoothing: {e}. Using raw data.")
                    x_data = spectra_matrix
                    y_data = concentrations_array
            else:
                st.warning("Not enough data points to apply smoothing. Using raw data.")
                x_data = spectra_matrix
                y_data = concentrations_array
        else:
            x_data = spectra_matrix  # Full spectrum: (n_samples, n_wavelengths)
            y_data = concentrations_array  # Concentrations: (n_samples,)
        
        # Store wavelengths for later use
        st.session_state.wavelengths = wavelengths
        
        # No polynomial fitting needed - using full spectrum ML models only
        
        trainer_outputs = run_model_trainers(x_data, y_data)

        nn_results = trainer_outputs.get('nn', {})
        plsr_results = trainer_outputs.get('plsr', {})
        rf_results = trainer_outputs.get('rf', {})
        svr_results = trainer_outputs.get('svr', {})
        cnn_results = trainer_outputs.get('cnn', {})
        xgb_results = trainer_outputs.get('xgb', {})
        
        fit_time = time.time() - start_time
        st.session_state.last_fit_time = fit_time
    
    if not nn_results and not plsr_results and not rf_results and not svr_results and not cnn_results and not xgb_results:
        st.error("❌ All fitting methods failed.")
        return
    
    # Success message
    success_msg = f"✅ Fitting completed in {fit_time:.3f} seconds"
    models_summary = []
    if nn_results:
        models_summary.append(f"MLP: {len(nn_results)}")
    if plsr_results:
        models_summary.append(f"PLSR: {len(plsr_results)}")
    if rf_results:
        models_summary.append(f"Random Forest: {len(rf_results)}")
    if svr_results:
        models_summary.append(f"SVR: {len(svr_results)}")
    if cnn_results:
        models_summary.append(f"1D-CNN: {len(cnn_results)}")
    if xgb_results:
        models_summary.append(f"XGBoost: {len(xgb_results)}")
    
    if models_summary:
        success_msg += f" ({', '.join(models_summary)})"
    
    st.caption(success_msg)

    # Store data in session state for plotting
    st.session_state.concentrations_array = y_data  # Store concentrations
    st.session_state.spectra_matrix = x_data  # Store full spectrum data

    # Get calibration range
    calib_range = {
        'min_conc': y_data.min(),
        'max_conc': y_data.max(),
        'min_spec': x_data.min(),
        'max_spec': x_data.max()
    }
    
    # Display results in tabs
    display_fitting_results(y_data, x_data, wavelengths, nn_results, plsr_results,
                            rf_results, svr_results, cnn_results, xgb_results, all_data, 
                            concentrations_dict, calib_range)

def display_fitting_results(concentrations_array: np.ndarray, spectra_matrix: np.ndarray, wavelengths: np.ndarray,
                            nn_results: Dict[str, Dict], plsr_results: Dict[int, Dict], rf_results: Dict[str, Dict], 
                            svr_results: Dict[str, Dict], cnn_results: Dict[str, Dict], xgb_results: Dict[str, Dict], 
                            all_data: Dict[str, pd.DataFrame], concentrations_dict: Dict[str, float], calib_range: Dict):
    """Display comprehensive fitting results in organized tabs using full spectrum data."""
    
    # Create simplified tabs
    tabs = ["📊 Model Comparison", "📈 Calibration Curves", "🔍 Spectral Overview", "📋 Data Export"]
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(tabs)
    
    # Model Comparison tab (combines all statistics)
    with tab1:
        display_statistics_tab(nn_results, plsr_results, rf_results, svr_results, cnn_results, xgb_results)
    
    # Calibration Curves tab (shows all model predictions)
    with tab2:
        display_combined_calibration_tab(concentrations_array, spectra_matrix, wavelengths,
                                         nn_results, plsr_results, 
                                         rf_results, svr_results, cnn_results, xgb_results)
    
    # Spectral Overview tab
    with tab3:
        display_spectral_tab(all_data, concentrations_dict, wavelengths)
    
    # Data Export tab
    with tab4:
        display_export_tab(concentrations_array, spectra_matrix, wavelengths, 
                          nn_results, plsr_results, rf_results, svr_results, cnn_results, xgb_results, calib_range)

def display_statistics_tab(nn_results: Dict[str, Dict] = None,
                          plsr_results: Dict[int, Dict] = None, rf_results: Dict[str, Dict] = None,
                          svr_results: Dict[str, Dict] = None, cnn_results: Dict[str, Dict] = None,
                          xgb_results: Dict[str, Dict] = None):
    """Display detailed statistical analysis."""
    st.markdown("#### 📊 Model Comparison")
    
    # Create statistics DataFrame for all models
    stats_data = []
    
    # No polynomial results to process - using ML models only
    
    # Add neural network results if available
    if nn_results:
        for name, result in nn_results.items():
            stats_data.append({
                'Model': 'MLP',
                'Type': 'MLP',
                'R²': result['r_squared'],
                'Adj. R²': result['adj_r_squared'],
                'RMSE': result['rmse'],
                'MAE': result.get('mae', 'N/A'),
                'MAPE': result.get('mape', 'N/A'),
                'Max Error': result.get('max_error', 'N/A'),
                'CV Score': result.get('cv_score', 'N/A'),
                'Training Time (s)': result.get('training_time', 'N/A'),
                'AIC': result['aic'],
                'BIC': result['bic']
            })
    
    # Add PLSR results if available
    if plsr_results:
        for n_comp, result in plsr_results.items():
            stats_data.append({
                'Model': 'PLSR',
                'Type': 'PLSR',
                'R²': result['r_squared'],
                'Adj. R²': result['adj_r_squared'],
                'RMSE': result['rmse'],
                'MAE': result.get('mae', 'N/A'),
                'MAPE': result.get('mape', 'N/A'),
                'Max Error': result.get('max_error', 'N/A'),
                'CV Score': result.get('cv_score', 'N/A'),
                'Training Time (s)': result.get('training_time', 'N/A'),
                'AIC': result['aic'],
                'BIC': result['bic']
            })
    
    # Add Random Forest results if available
    if rf_results:
        for name, result in rf_results.items():
            stats_data.append({
                'Model': 'Random Forest',
                'Type': 'Random Forest',
                'R²': result['r_squared'],
                'Adj. R²': result['adj_r_squared'],
                'RMSE': result['rmse'],
                'MAE': result.get('mae', 'N/A'),
                'MAPE': result.get('mape', 'N/A'),
                'Max Error': result.get('max_error', 'N/A'),
                'CV Score': result.get('cv_score', 'N/A'),
                'Training Time (s)': result.get('training_time', 'N/A'),
                'AIC': result['aic'],
                'BIC': result['bic']
            })
    
    # Add SVR results if available
    if svr_results:
        for name, result in svr_results.items():
            stats_data.append({
                'Model': 'SVR',
                'Type': 'SVR',
                'R²': result['r_squared'],
                'Adj. R²': result['adj_r_squared'],
                'RMSE': result['rmse'],
                'MAE': result.get('mae', 'N/A'),
                'MAPE': result.get('mape', 'N/A'),
                'Max Error': result.get('max_error', 'N/A'),
                'CV Score': result.get('cv_score', 'N/A'),
                'Training Time (s)': result.get('training_time', 'N/A'),
                'AIC': result['aic'],
                'BIC': result['bic']
            })
    
    # Add 1D-CNN results if available
    if cnn_results:
        for name, result in cnn_results.items():
            stats_data.append({
                'Model': '1D-CNN',
                'Type': '1D-CNN',
                'R²': result['r_squared'],
                'Adj. R²': result['adj_r_squared'],
                'RMSE': result['rmse'],
                'MAE': result.get('mae', 'N/A'),
                'MAPE': result.get('mape', 'N/A'),
                'Max Error': result.get('max_error', 'N/A'),
                'CV Score': result.get('cv_score', 'N/A'),
                'Training Time (s)': result.get('training_time', 'N/A'),
                'AIC': result['aic'],
                'BIC': result['bic']
            })
    
    # Add XGBoost results if available
    if xgb_results:
        for name, result in xgb_results.items():
            stats_data.append({
                'Model': 'XGBoost',
                'Type': 'XGBoost',
                'R²': result['r_squared'],
                'Adj. R²': result.get('adj_r_squared', 'N/A'),
                'RMSE': result['rmse'],
                'MAE': result.get('mae', 'N/A'),
                'MAPE': result.get('mape', 'N/A'),
                'Max Error': result.get('max_error', 'N/A'),
                'CV Score': result.get('cv_score', 'N/A'),
                'Training Time (s)': result.get('training_time', 'N/A'),
                'AIC': result.get('aic', 'N/A'),
                'BIC': result.get('bic', 'N/A')
            })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Convert numeric columns to proper types, handling 'N/A' values
    numeric_columns = ['R²', 'Adj. R²', 'RMSE', 'MAE', 'MAPE', 'Max Error', 'CV Score', 'Training Time (s)', 'AIC', 'BIC']
    for col in numeric_columns:
        if col in stats_df.columns:
            # Replace 'N/A' with NaN and convert to numeric
            stats_df[col] = stats_df[col].replace('N/A', np.nan)
            stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')
    
    # Display with highlighting (only if we have numeric data)
    try:
        # Check if we have any numeric data to highlight
        numeric_data_available = any(stats_df[col].notna().any() for col in ['R²', 'Adj. R²', 'RMSE', 'AIC', 'BIC'] if col in stats_df.columns)
        
        if numeric_data_available:
            st.dataframe(
                stats_df.style.highlight_max(subset=['R²', 'Adj. R²'])
                           .highlight_min(subset=['RMSE', 'AIC', 'BIC'])
                           .format({
                               'R²': '{:.6f}',
                               'Adj. R²': '{:.6f}',
                               'RMSE': '{:.6f}',
                               'AIC': '{:.2f}',
                               'BIC': '{:.2f}',
                               'F-statistic': lambda x: f'{x:.3f}' if isinstance(x, (int, float)) and not pd.isna(x) else str(x),
                               'F p-value': lambda x: f'{x:.3e}' if isinstance(x, (int, float)) and not pd.isna(x) else str(x)
                           }),
                use_container_width=True
            )
        else:
            # Fallback without highlighting
            st.dataframe(stats_df, use_container_width=True)
    except Exception as e:
        # Fallback without highlighting if there's any error
        st.dataframe(stats_df, use_container_width=True)
    
    # Best model selection
    all_models = {}
    
    # No polynomial models to process - using ML models only
    
    # Add best neural network
    if nn_results:
        best_nn_name = NeuralNetworkFitter.select_best_neural_network(
            nn_results, st.session_state.model_selection_criterion
        )
        if best_nn_name:
            all_models[f'MLP ({best_nn_name})'] = nn_results[best_nn_name]
    
    # Add best PLSR
    if plsr_results:
        criterion = st.session_state.model_selection_criterion
        if criterion == "r_squared":
            best_plsr = max(plsr_results.keys(), key=lambda k: plsr_results[k].get('r_squared', -np.inf))
        elif criterion == "adj_r_squared":
            best_plsr = max(plsr_results.keys(), key=lambda k: plsr_results[k].get('adj_r_squared', -np.inf))
        elif criterion == "aic":
            best_plsr = min(plsr_results.keys(), key=lambda k: plsr_results[k].get('aic', np.inf))
        elif criterion == "bic":
            best_plsr = min(plsr_results.keys(), key=lambda k: plsr_results[k].get('bic', np.inf))
        else:  # default to r_squared
            best_plsr = max(plsr_results.keys(), key=lambda k: plsr_results[k].get('r_squared', -np.inf))
        all_models['PLSR'] = plsr_results[best_plsr]
    
    # Add best Random Forest
    if rf_results:
        criterion = st.session_state.model_selection_criterion
        if criterion == "r_squared":
            best_rf = max(rf_results.keys(), key=lambda k: rf_results[k].get('r_squared', -np.inf))
        elif criterion == "adj_r_squared":
            best_rf = max(rf_results.keys(), key=lambda k: rf_results[k].get('adj_r_squared', -np.inf))
        elif criterion == "aic":
            best_rf = min(rf_results.keys(), key=lambda k: rf_results[k].get('aic', np.inf))
        elif criterion == "bic":
            best_rf = min(rf_results.keys(), key=lambda k: rf_results[k].get('bic', np.inf))
        else:  # default to r_squared
            best_rf = max(rf_results.keys(), key=lambda k: rf_results[k].get('r_squared', -np.inf))
        all_models[f'Random Forest ({best_rf})'] = rf_results[best_rf]
    
    # Add best SVR
    if svr_results:
        criterion = st.session_state.model_selection_criterion
        if criterion == "r_squared":
            best_svr = max(svr_results.keys(), key=lambda k: svr_results[k].get('r_squared', -np.inf))
        elif criterion == "adj_r_squared":
            best_svr = max(svr_results.keys(), key=lambda k: svr_results[k].get('adj_r_squared', -np.inf))
        elif criterion == "aic":
            best_svr = min(svr_results.keys(), key=lambda k: svr_results[k].get('aic', np.inf))
        elif criterion == "bic":
            best_svr = min(svr_results.keys(), key=lambda k: svr_results[k].get('bic', np.inf))
        else:  # default to r_squared
            best_svr = max(svr_results.keys(), key=lambda k: svr_results[k].get('r_squared', -np.inf))
        all_models[f'SVR ({best_svr})'] = svr_results[best_svr]
    
    # Add best 1D-CNN
    if cnn_results:
        criterion = st.session_state.model_selection_criterion
        if criterion == "r_squared":
            best_cnn = max(cnn_results.keys(), key=lambda k: cnn_results[k].get('r_squared', -np.inf))
        elif criterion == "adj_r_squared":
            best_cnn = max(cnn_results.keys(), key=lambda k: cnn_results[k].get('adj_r_squared', -np.inf))
        elif criterion == "aic":
            best_cnn = min(cnn_results.keys(), key=lambda k: cnn_results[k].get('aic', np.inf))
        elif criterion == "bic":
            best_cnn = min(cnn_results.keys(), key=lambda k: cnn_results[k].get('bic', np.inf))
        else:  # default to r_squared
            best_cnn = max(cnn_results.keys(), key=lambda k: cnn_results[k].get('r_squared', -np.inf))
        all_models[f'1D-CNN ({best_cnn})'] = cnn_results[best_cnn]
    
    # Add best XGBoost
    if xgb_results:
        criterion = st.session_state.model_selection_criterion
        if criterion == "r_squared":
            best_xgb = max(xgb_results.keys(), key=lambda k: xgb_results[k]['r_squared'])
        elif criterion == "rmse":
            best_xgb = min(xgb_results.keys(), key=lambda k: xgb_results[k]['rmse'])
        else:  # default to r_squared
            best_xgb = max(xgb_results.keys(), key=lambda k: xgb_results[k]['r_squared'])
        all_models[f'XGBoost ({best_xgb})'] = xgb_results[best_xgb]
    
    # Find overall best model
    if all_models:
        criterion = st.session_state.model_selection_criterion
        
        # Helper function to safely get metric values
        def get_metric_value(model_data, metric):
            value = model_data.get(metric, None)
            if value is None or value == 'N/A' or pd.isna(value):
                return -np.inf if metric in ['r_squared', 'adj_r_squared'] else np.inf
            return float(value)
        
        if criterion == "r_squared":
            best_model_name = max(all_models.keys(), key=lambda k: get_metric_value(all_models[k], 'r_squared'))
        elif criterion == "adj_r_squared":
            best_model_name = max(all_models.keys(), key=lambda k: get_metric_value(all_models[k], 'adj_r_squared'))
        elif criterion == "rmse":
            best_model_name = min(all_models.keys(), key=lambda k: get_metric_value(all_models[k], 'rmse'))
        elif criterion == "aic":
            best_model_name = min(all_models.keys(), key=lambda k: get_metric_value(all_models[k], 'aic'))
        elif criterion == "bic":
            best_model_name = min(all_models.keys(), key=lambda k: get_metric_value(all_models[k], 'bic'))
        else:  # default to r_squared
            best_model_name = max(all_models.keys(), key=lambda k: get_metric_value(all_models[k], 'r_squared'))
        
        st.markdown(f"#### 🏆 Overall Best Model: {best_model_name}")
        
        best_result = all_models[best_model_name]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R²", f"{best_result['r_squared']:.6f}")
        with col2:
            st.metric("RMSE", f"{best_result['rmse']:.6f}")
        with col3:
            aic_value = best_result.get('aic', 'N/A')
            if aic_value != 'N/A' and np.isfinite(aic_value):
                st.metric("AIC", f"{aic_value:.2f}")
            else:
                st.metric("AIC", "N/A")
    
    # Optimized Parameters Section - Organized by Category
    st.markdown("---")
    st.markdown("#### ⚙️ Training Details")
    
    # Neural Network Models
    st.markdown("##### 🧠 Neural Network Models")
    nn_param_rows = []
    
    def add_param_row(rows_list: List[Dict], model_label: str, values: Dict[str, Any]) -> None:
        if not values:
            return
        clean_values = {k: ("N/A" if v in (None, 'N/A') else v) for k, v in values.items()}
        rows_list.append({"Model": model_label, **clean_values})

    if nn_results:
        name, result = next(iter(nn_results.items()))
        model_info = result.get('model_info', {})
        train_losses = result.get('train_losses', [])
        epochs_used = len(train_losses) if train_losses else 'N/A'
        add_param_row(
            nn_param_rows,
            "MLP",
            {
                "Architecture": model_info.get('architecture', 'N/A'),
                "Activation": model_info.get('activation', 'N/A'),
                "Dropout": model_info.get('dropout_rate', 'N/A'),
                "Learning Rate": model_info.get('learning_rate', 'N/A'),
                "Epochs": epochs_used,
                "Parameters": f"{model_info.get('n_parameters', 'N/A'):,}",
                "Training Time (s)": f"{result.get('training_time', 'N/A')}"
            }
        )

    if cnn_results:
        name, result = next(iter(cnn_results.items()))
        model_info = result.get('model_info', {})
        config = result.get('config', {})
        train_losses = result.get('train_losses', [])
        epochs_used = len(train_losses) if train_losses else config.get('epochs', 'N/A')
        add_param_row(
            nn_param_rows,
            "1D-CNN",
            {
                "Architecture": model_info.get('architecture', 'N/A'),
                "Epochs": epochs_used,
                "Parameters": f"{result.get('n_parameters', 'N/A'):,}",
                "Training Time (s)": f"{result.get('training_time', 'N/A')}"
            }
        )

    if nn_param_rows:
        nn_df = pd.DataFrame(nn_param_rows)
        st.dataframe(nn_df, use_container_width=True, hide_index=True)
    else:
        st.info("No neural network models trained.")
    
    # Ensemble Models
    st.markdown("##### 🌳 Ensemble Models")
    ensemble_param_rows = []
    
    if rf_results:
        name, result = next(iter(rf_results.items()))
        add_param_row(
            ensemble_param_rows,
            "Random Forest",
            {
                "Estimators": result.get('n_estimators', 'N/A'),
                "Max Depth": result.get('max_depth', 'N/A'),
                "Training Time (s)": f"{result.get('training_time', 'N/A')}"
            }
        )

    if xgb_results:
        name, result = next(iter(xgb_results.items()))
        add_param_row(
            ensemble_param_rows,
            "XGBoost",
            {
                "Estimators": result.get('n_estimators', 'N/A'),
                "Max Depth": result.get('max_depth', 'N/A'),
                "Learning Rate": result.get('learning_rate', 'N/A'),
                "Training Time (s)": f"{result.get('training_time', 'N/A')}"
            }
        )

    if ensemble_param_rows:
        ensemble_df = pd.DataFrame(ensemble_param_rows)
        st.dataframe(ensemble_df, use_container_width=True, hide_index=True)
    else:
        st.info("No ensemble models trained.")
    
    # Nonlinear Models
    st.markdown("##### 📐 Nonlinear Models")
    nonlinear_param_rows = []
    
    if plsr_results:
        n_comp, result = next(iter(plsr_results.items()))
        add_param_row(
            nonlinear_param_rows,
            "PLSR",
            {
                "Components": n_comp,
                "Training Time (s)": f"{result.get('training_time', 'N/A')}"
            }
        )

    if svr_results:
        name, result = next(iter(svr_results.items()))
        add_param_row(
            nonlinear_param_rows,
            "SVR",
            {
                "Kernel": result.get('kernel', 'N/A'),
                "Support Vectors": result.get('n_support', 'N/A'),
                "Training Time (s)": f"{result.get('training_time', 'N/A')}"
            }
        )

    if nonlinear_param_rows:
        nonlinear_df = pd.DataFrame(nonlinear_param_rows)
        st.dataframe(nonlinear_df, use_container_width=True, hide_index=True)
    else:
        st.info("No nonlinear models trained.")
    
    # Optimization Method Used
    st.markdown("---")
    optimization_method = st.session_state.get('optimization_method', 'Random Search')
    cv_folds = st.session_state.get('cv_folds', 5)
    st.info(f"🔧 **Optimization Method:** {optimization_method} | **CV Folds:** {cv_folds}")
    
    # Individual Model Export Section
    st.markdown("---")
    st.markdown("#### 🎯 Individual Model Export")
    st.markdown("Download trained models in their optimal formats for external use.")
    
    # Create clean download buttons organized by category
    st.markdown("##### 🧠 Neural Network Models")
    col_nn1, col_nn2 = st.columns(2)
    
    with col_nn1:
        if nn_results:
            name, result = next(iter(nn_results.items()))
            serialized = result.get('model_serialized')
            if serialized and serialized not in ['N/A', 'TF_SERIALIZATION_NOT_SUPPORTED', 'TF_SERIALIZATION_FAILED']:
                st.download_button(
                    "📥 Download MLP Model",
                    data=serialized,
                    file_name=f"mlp_{name}.pkl",
                    mime="application/octet-stream",
                    key=f"export_mlp_{name}",
                    use_container_width=True
                )
            else:
                st.button("MLP - Not Available", disabled=True, use_container_width=True)
        else:
            st.button("MLP - Not Available", disabled=True, use_container_width=True)
    
    with col_nn2:
        if cnn_results:
            name, result = next(iter(cnn_results.items()))
            serialized = result.get('model_serialized')
            if serialized and serialized not in ['N/A', 'TF_SERIALIZATION_NOT_SUPPORTED', 'TF_SERIALIZATION_FAILED']:
                st.download_button(
                    "📥 Download 1D-CNN Model",
                    data=serialized,
                    file_name=f"cnn_{name}.pkl",
                    mime="application/octet-stream",
                    key=f"export_cnn_{name}",
                    use_container_width=True
                )
            else:
                st.button("1D-CNN - Not Available", disabled=True, use_container_width=True)
        else:
            st.button("1D-CNN - Not Available", disabled=True, use_container_width=True)
    
    st.markdown("##### 🌳 Ensemble Models")
    col_ens1, col_ens2 = st.columns(2)
    
    with col_ens1:
        if rf_results:
            name, result = next(iter(rf_results.items()))
            serialized = result.get('model_serialized')
            if serialized and serialized not in ['N/A']:
                st.download_button(
                    "📥 Download Random Forest",
                    data=serialized,
                    file_name=f"rf_{name}.pkl",
                    mime="application/octet-stream",
                    key=f"export_rf_{name}",
                    use_container_width=True
                )
            else:
                st.button("Random Forest - Not Available", disabled=True, use_container_width=True)
        else:
            st.button("Random Forest - Not Available", disabled=True, use_container_width=True)
    
    with col_ens2:
        if xgb_results:
            name, result = next(iter(xgb_results.items()))
            serialized = result.get('model_serialized')
            if serialized and serialized not in ['N/A']:
                st.download_button(
                    "📥 Download XGBoost",
                    data=serialized,
                    file_name=f"xgb_{name}.pkl",
                    mime="application/octet-stream",
                    key=f"export_xgb_{name}",
                    use_container_width=True
                )
            else:
                st.button("XGBoost - Not Available", disabled=True, use_container_width=True)
        else:
            st.button("XGBoost - Not Available", disabled=True, use_container_width=True)
    
    st.markdown("##### 📐 Nonlinear Models")
    col_nl1, col_nl2 = st.columns(2)
    
    with col_nl1:
        if plsr_results:
            n_comp, result = next(iter(plsr_results.items()))
            serialized = result.get('model_serialized')
            if serialized and serialized not in ['N/A']:
                st.download_button(
                    "📥 Download PLSR",
                    data=serialized,
                    file_name=f"plsr_{n_comp}.pkl",
                    mime="application/octet-stream",
                    key=f"export_plsr_{n_comp}",
                    use_container_width=True
                )
            else:
                st.button("PLSR - Not Available", disabled=True, use_container_width=True)
        else:
            st.button("PLSR - Not Available", disabled=True, use_container_width=True)
    
    with col_nl2:
        if svr_results:
            name, result = next(iter(svr_results.items()))
            serialized = result.get('model_serialized')
            if serialized and serialized not in ['N/A']:
                st.download_button(
                    "📥 Download SVR",
                    data=serialized,
                    file_name=f"svr_{name}.pkl",
                    mime="application/octet-stream",
                    key=f"export_svr_{name}",
                    use_container_width=True
                )
            else:
                st.button("SVR - Not Available", disabled=True, use_container_width=True)
        else:
            st.button("SVR - Not Available", disabled=True, use_container_width=True)
    
    # Model Usage Instructions
    st.markdown("---")
    with st.expander("📖 Model Usage Instructions", expanded=False):
        st.markdown("""
        **Downloaded Model Files:**
        
        **PLSR, Random Forest, SVR, MLP, XGBoost Models (.pkl files):**
        ```python
        import pickle
        import base64
        
        # Load the model
        with open('model_name.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract components
        model = model_data['model']
        scaler_x = model_data.get('scaler_x')  # If available
        scaler_y = model_data.get('scaler_y')  # If available
        
        # Make predictions
        if scaler_x and scaler_y:
            x_scaled = scaler_x.transform(spectrum_data)
            y_pred_scaled = model.predict(x_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
        else:
            y_pred = model.predict(spectrum_data)
        ```
        
        **CNN Models (.pkl files):**
        ```python
        import torch
        import torch.nn as nn
        import pickle
        import numpy as np
        from models.cnn_model import CNN1DModel
        
        # Load the CNN model (PyTorch)
        # model = CNN1DModel(**config)
        # model.load_state_dict(torch.load('model_name.pth'))
        
        # Load model with scalers (all in one .pkl file)
        with open('model_name.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        scaler_x = model_data['scaler_x']
        scaler_y = model_data['scaler_y']
        
        # Make predictions
        # 1. Scale input spectrum
        x_scaled = scaler_x.transform(spectrum_data)
        
        # 2. Reshape for CNN: (n_samples, n_wavelengths, 1)
        if x_scaled.ndim == 2:
            x_cnn = x_scaled.reshape(x_scaled.shape[0], x_scaled.shape[1], 1)
        elif x_scaled.ndim == 1:
            x_cnn = x_scaled.reshape(1, x_scaled.shape[0], 1)
        
        # 3. Convert to PyTorch tensor and predict
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        x_tensor = torch.FloatTensor(x_cnn).to(device)
        
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(x_tensor).cpu().numpy().ravel()
        
        # 4. Inverse transform to get actual concentrations
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        ```
        """)
    
    # Export Summary
    total_models = (len(plsr_results) + len(rf_results) + len(svr_results) + 
                   len(nn_results) + len(xgb_results) + len(cnn_results))
    
    st.info(f"📊 **Export Summary:** {total_models} models available for download")



def display_combined_calibration_tab(concentrations_array: np.ndarray, spectra_matrix: np.ndarray, wavelengths: np.ndarray,
                                    nn_results: Dict[str, Dict], plsr_results: Dict[int, Dict],
                                    rf_results: Dict[str, Dict], svr_results: Dict[str, Dict],
                                    cnn_results: Dict[str, Dict], xgb_results: Dict[str, Dict]):
    """Display combined calibration curves for all models using full spectrum data."""
    st.info("Calibration curves: predictions vs. actual concentrations")
    st.markdown("#### 📈 Model Calibration Curves")
    
    # Create selection options for models to display
    st.markdown("##### Select Models to Display")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # No polynomial model checkboxes needed - using ML models only
        pass
    
    with col2:
        # ML models - Streamlit automatically manages session state with keys
        show_mlp = st.checkbox("MLP", 
                              value=st.session_state.get('show_mlp_cal', bool(nn_results)), 
                              key="show_mlp_cal")
        
        show_plsr = st.checkbox("PLSR", 
                               value=st.session_state.get('show_plsr_cal', bool(plsr_results)), 
                               key="show_plsr_cal")
    
    with col3:
        # Advanced models - Streamlit automatically manages session state with keys
        show_rf = st.checkbox("Random Forest", 
                             value=st.session_state.get('show_rf_cal', bool(rf_results)), 
                             key="show_rf_cal")
        
        show_svr = st.checkbox("SVR", 
                              value=st.session_state.get('show_svr_cal', bool(svr_results)), 
                              key="show_svr_cal")
        
        show_cnn = st.checkbox("1D-CNN", 
                              value=st.session_state.get('show_cnn_cal', bool(cnn_results)), 
                              key="show_cnn_cal")
        
        show_xgb = st.checkbox("XGBoost", 
                              value=st.session_state.get('show_xgb_cal', bool(xgb_results)), 
                              key="show_xgb_cal")
    
    # Create the combined plot
    fig = go.Figure()
    
    # Note: For full spectrum models, we're plotting actual concentrations vs predicted concentrations
    # (not absorbance vs concentration, since models take full spectrum as input)
    st.info("📊 Plotting actual vs. predicted concentrations for each model")
    
    # Store actual concentrations for later use
    actual_concentrations = concentrations_array
    
    # Add ideal line (y=x)
    conc_min, conc_max = actual_concentrations.min(), actual_concentrations.max()
    conc_range = np.linspace(conc_min * 0.9, conc_max * 1.1, 100)
    
    fig.add_trace(go.Scatter(
        x=conc_range, y=conc_range,
        mode='lines',
        name='Ideal (y=x)',
        line=dict(color='gray', width=2, dash='dash'),
        hovertemplate='Ideal<br>Actual: %{x}<br>Predicted: %{y}<extra></extra>'
    ))
    
    # Color palette for models
    colors = {
        'mlp': '#2ca02c',
        'plsr': '#d62728',
        'rf': '#9467bd',
        'svr': '#8c564b',
        'cnn': '#e377c2',
        'xgb': '#bcbd22'
    }
    
    # Plot MLP models (predictions already stored)
    if show_mlp and nn_results:
        best_mlp = max(nn_results.keys(), key=lambda k: nn_results[k].get('r_squared', 0))
        result = nn_results[best_mlp]
        fig.add_trace(go.Scatter(
            x=actual_concentrations, y=result['predictions'],
            mode='markers',
            name=f'MLP {best_mlp} (R²={result["r_squared"]:.4f})',
            marker=dict(size=8, color=colors['mlp']),
            hovertemplate='MLP<br>Actual: %{x}<br>Predicted: %{y}<extra></extra>'
        ))
    
    # Plot PLSR (predictions already stored)
    if show_plsr and plsr_results:
        best_plsr = max(plsr_results.keys(), key=lambda k: plsr_results[k].get('r_squared', 0))
        result = plsr_results[best_plsr]
        fig.add_trace(go.Scatter(
            x=actual_concentrations, y=result['predictions'],
            mode='markers',
            name=f'PLSR {best_plsr}comp (R²={result["r_squared"]:.4f})',
            marker=dict(size=8, color=colors['plsr'], symbol='square'),
            hovertemplate='PLSR<br>Actual: %{x}<br>Predicted: %{y}<extra></extra>'
        ))
    
    # Plot Random Forest (predictions already stored)
    if show_rf and rf_results:
        best_rf = max(rf_results.keys(), key=lambda k: rf_results[k].get('r_squared', 0))
        result = rf_results[best_rf]
        fig.add_trace(go.Scatter(
            x=actual_concentrations, y=result['predictions'],
            mode='markers',
            name=f'RF {best_rf} (R²={result["r_squared"]:.4f})',
            marker=dict(size=8, color=colors['rf'], symbol='diamond'),
            hovertemplate='RF<br>Actual: %{x}<br>Predicted: %{y}<extra></extra>'
        ))
    
    # Plot SVR (predictions already stored)
    if show_svr and svr_results:
        best_svr = max(svr_results.keys(), key=lambda k: svr_results[k].get('r_squared', 0))
        result = svr_results[best_svr]
        fig.add_trace(go.Scatter(
            x=actual_concentrations, y=result['predictions'],
            mode='markers',
            name=f'SVR {best_svr} (R²={result["r_squared"]:.4f})',
            marker=dict(size=8, color=colors['svr'], symbol='triangle-up'),
            hovertemplate='SVR<br>Actual: %{x}<br>Predicted: %{y}<extra></extra>'
        ))
    
    # Plot CNN (predictions already stored)
    if show_cnn and cnn_results:
        best_cnn = max(cnn_results.keys(), key=lambda k: cnn_results[k].get('r_squared', 0))
        result = cnn_results[best_cnn]
        fig.add_trace(go.Scatter(
            x=actual_concentrations, y=result['predictions'],
            mode='markers',
            name=f'CNN {best_cnn} (R²={result["r_squared"]:.4f})',
            marker=dict(size=8, color=colors['cnn'], symbol='hexagon'),
            hovertemplate='CNN<br>Actual: %{x}<br>Predicted: %{y}<extra></extra>'
        ))
    
    # Plot XGBoost (predictions already stored)
    if show_xgb and xgb_results:
        best_xgb = max(xgb_results.keys(), key=lambda k: xgb_results[k].get('r_squared', 0))
        result = xgb_results[best_xgb]
        fig.add_trace(go.Scatter(
            x=actual_concentrations, y=result['predictions'],
            mode='markers',
            name=f'XGB {best_xgb} (R²={result["r_squared"]:.4f})',
            marker=dict(size=8, color=colors['xgb'], symbol='star'),
            hovertemplate='XGBoost<br>Actual: %{x}<br>Predicted: %{y}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title='Actual vs. Predicted Concentrations',
        xaxis_title='Actual Concentration',
        yaxis_title='Predicted Concentration',
        hovermode='closest',
        showlegend=True,
        height=600,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show residuals analysis if requested
    if st.checkbox("Show Residuals Analysis", key="show_residuals_combined"):
        display_combined_residuals_analysis(concentrations_array, spectra_matrix, wavelengths,
                                          nn_results, plsr_results,
                                          rf_results, svr_results, cnn_results, xgb_results)

def display_combined_residuals_analysis(concentrations_array: np.ndarray, spectra_matrix: np.ndarray, wavelengths: np.ndarray,
                                       nn_results: Dict[str, Dict], plsr_results: Dict[int, Dict],
                                       rf_results: Dict[str, Dict], svr_results: Dict[str, Dict],
                                       cnn_results: Dict[str, Dict], xgb_results: Dict[str, Dict]):
    """Display residuals analysis for all selected models using full spectrum data."""
    # Use actual concentrations
    x_data = concentrations_array
    
    # Collect all models and their residuals
    all_models = []
    
    # Polynomial models not used with full spectrum approach
    # (residuals already stored in each model)
    
    # Add other models
    if nn_results:
        best_nn = max(nn_results.keys(), key=lambda k: nn_results[k]['r_squared'])
        all_models.append((f'MLP {best_nn}', nn_results[best_nn]['residuals']))
    
    if plsr_results:
        best_plsr = max(plsr_results.keys(), key=lambda k: plsr_results[k]['r_squared'])
        all_models.append(('PLSR', plsr_results[best_plsr]['residuals']))
    
    if rf_results:
        best_rf = max(rf_results.keys(), key=lambda k: rf_results[k]['r_squared'])
        all_models.append((f'RF {best_rf}', rf_results[best_rf]['residuals']))
    
    if svr_results:
        best_svr = max(svr_results.keys(), key=lambda k: svr_results[k]['r_squared'])
        all_models.append((f'SVR {best_svr}', svr_results[best_svr]['residuals']))
    
    if cnn_results:
        best_cnn = max(cnn_results.keys(), key=lambda k: cnn_results[k]['r_squared'])
        all_models.append((f'1D-CNN {best_cnn}', cnn_results[best_cnn]['residuals']))
    
    if xgb_results:
        best_xgb = max(xgb_results.keys(), key=lambda k: xgb_results[k]['r_squared'])
        all_models.append((f'XGBoost {best_xgb}', xgb_results[best_xgb]['residuals']))
    
    # Create subplots
    n_models = len(all_models)
    if n_models == 0:
        st.warning("No models available for residuals analysis")
        return
    
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[name for name, _ in all_models],
        vertical_spacing=0.15
    )
    
    for i, (name, residuals) in enumerate(all_models):
        row = i // cols + 1
        col = i % cols + 1
        
        fig.add_trace(
            go.Scatter(
                x=x_data, y=residuals,
                mode='markers',
                name=name,
                showlegend=False,
                marker=dict(size=6)
            ),
            row=row, col=col
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=row, col=col)
    
    fig.update_layout(
        height=300 * rows,
        title="Residuals Analysis",
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Concentration")
    fig.update_yaxes(title_text="Residuals")
    
    st.plotly_chart(fig, use_container_width=True)



def display_spectral_tab(all_data: Dict[str, pd.DataFrame], concentrations: Dict[str, float],
                         wavelengths: np.ndarray):
    """Display spectral overview using full spectrum data."""
    st.markdown("#### 🔍 Spectral Data Overview")
    
    # Display all spectra overlaid
    spectral_fig = PlotManager.create_spectral_overview(
        all_data, concentrations, None  # No specific wavelength highlighting needed
    )
    st.plotly_chart(spectral_fig, use_container_width=True)

    st.info(f"📊 Full spectrum range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm ({len(wavelengths)} wavelengths)")

def display_export_tab(concentrations_array: np.ndarray, spectra_matrix: np.ndarray, wavelengths: np.ndarray,
                       nn_results: Dict[str, Dict], plsr_results: Dict[int, Dict], rf_results: Dict[str, Dict], 
                       svr_results: Dict[str, Dict], cnn_results: Dict[str, Dict], xgb_results: Dict[str, Dict], 
                       calib_range: Dict):
    """Display data export options for full spectrum models."""
    st.markdown("#### 📋 Data Export")
    
    # Spectrum data export
    st.markdown("**Full Spectrum Data Summary:**")
    spectrum_info = pd.DataFrame({
        'Concentration': concentrations_array,
        'N_Wavelengths': [spectra_matrix.shape[1]] * len(concentrations_array),
        'Wavelength_Min': [wavelengths[0]] * len(concentrations_array),
        'Wavelength_Max': [wavelengths[-1]] * len(concentrations_array),
        'Mean_Absorbance': np.mean(spectra_matrix, axis=1),
        'Max_Absorbance': np.max(spectra_matrix, axis=1)
    })
    st.dataframe(spectrum_info, use_container_width=True)
    
    peak_csv = spectrum_info.to_csv(index=False)
    st.download_button(
        "📥 Download Peak Data",
        data=peak_csv,
        file_name="peak_analysis_data.csv",
        mime="text/csv"
    )

    # No polynomial regression results to export - using ML models only
    
    # MLP results export
    if nn_results:
        st.markdown("**Multi-Layer Perceptron (MLP) Results:**")
        
        # Create neural network results DataFrame
        nn_summary = []
        for name, result in nn_results.items():
            model_info = result.get('model_info', {})
            nn_summary.append({
                'Model_Type': 'MLP',
                'Model_Name': name,
                'Network_Type': model_info.get('type', 'N/A'),
                'Architecture': model_info.get('architecture', 'N/A'),
                'Parameters': model_info.get('n_parameters', 'N/A'),
                'R_squared': result['r_squared'],
                'Adj_R_squared': result['adj_r_squared'],
                'RMSE': result['rmse'],
                'AIC': result['aic'],
                'BIC': result['bic'],
                'CV_RMSE': result.get('cv_rmse', 'N/A'),
                'Min_Concentration': calib_range['min_conc'],
                'Max_Concentration': calib_range['max_conc'],
                'Min_Spectrum': calib_range['min_spec'],
                'Max_Spectrum': calib_range['max_spec'],
            })
        
        nn_df = pd.DataFrame(nn_summary)
        st.dataframe(nn_df, use_container_width=True)
        
        nn_csv = nn_df.to_csv(index=False)
        st.download_button(
            "📥 Download MLP Results",
            data=nn_csv,
            file_name="neural_network_results.csv",
            mime="text/csv"
        )
    
    # PLSR results export
    if plsr_results:
        st.markdown("**PLSR Results:**")
        
        plsr_summary = []
        for n_comp, result in plsr_results.items():
            plsr_summary.append({
                'Model_Type': 'PLSR',
                'N_Components': n_comp,
                'R_squared': result['r_squared'],
                'Adj_R_squared': result['adj_r_squared'],
                'RMSE': result['rmse'],
                'AIC': result['aic'],
                'BIC': result['bic'],
                'CV_RMSE': result.get('cv_rmse', 'N/A'),
            })
        
        plsr_df = pd.DataFrame(plsr_summary)
        st.dataframe(plsr_df, use_container_width=True)
        
        plsr_csv = plsr_df.to_csv(index=False)
        st.download_button(
            "📥 Download PLSR Results",
            data=plsr_csv,
            file_name="plsr_results.csv",
            mime="text/csv"
        )
    
    # Random Forest results export
    if rf_results:
        st.markdown("**Random Forest Results:**")
        
        rf_summary = []
        for name, result in rf_results.items():
            rf_summary.append({
                'Model_Type': 'Random_Forest',
                'Model_Name': name,
                'N_Estimators': result['n_estimators'],
                'Max_Depth': result['max_depth'],
                'R_squared': result['r_squared'],
                'Adj_R_squared': result['adj_r_squared'],
                'RMSE': result['rmse'],
                'AIC': result['aic'],
                'BIC': result['bic'],
                'CV_RMSE': result.get('cv_rmse', 'N/A'),
                'Feature_Importance': result['feature_importance'],
            })
        
        rf_df = pd.DataFrame(rf_summary)
        st.dataframe(rf_df, use_container_width=True)
        
        rf_csv = rf_df.to_csv(index=False)
        st.download_button(
            "📥 Download Random Forest Results",
            data=rf_csv,
            file_name="random_forest_results.csv",
            mime="text/csv"
        )
    
    # SVR results export
    if svr_results:
        st.markdown("**SVR Results:**")
        
        svr_summary = []
        for name, result in svr_results.items():
            svr_summary.append({
                'Model_Type': 'SVR',
                'Model_Name': name,
                'Kernel': result['kernel'],
                'N_Support_Vectors': result['n_support'],
                'R_squared': result['r_squared'],
                'Adj_R_squared': result['adj_r_squared'],
                'RMSE': result['rmse'],
                'AIC': result['aic'],
                'BIC': result['bic'],
                'CV_RMSE': result.get('cv_rmse', 'N/A'),
            })
        
        svr_df = pd.DataFrame(svr_summary)
        st.dataframe(svr_df, use_container_width=True)
        
        svr_csv = svr_df.to_csv(index=False)
        st.download_button(
            "📥 Download SVR Results",
            data=svr_csv,
            file_name="svr_results.csv",
            mime="text/csv"
        )
    
    # 1D-CNN results export
    if cnn_results:
        st.markdown("**1D-CNN Results:**")
        
        cnn_summary = []
        for name, result in cnn_results.items():
            model_info = result.get('model_info', {})
            cnn_summary.append({
                'Model_Type': '1D-CNN',
                'Model_Name': name,
                'Architecture': model_info.get('architecture', 'N/A'),
                'N_Parameters': model_info.get('n_parameters', 'N/A'),
                'R_squared': result['r_squared'],
                'Adj_R_squared': result['adj_r_squared'],
                'RMSE': result['rmse'],
                'AIC': result['aic'],
                'BIC': result['bic'],
            })
        
        cnn_df = pd.DataFrame(cnn_summary)
        st.dataframe(cnn_df, use_container_width=True)
        
        cnn_csv = cnn_df.to_csv(index=False)
        st.download_button(
            "📥 Download 1D-CNN Results",
            data=cnn_csv,
            file_name="cnn_results.csv",
            mime="text/csv"
        )
    
    # XGBoost results export
    if xgb_results:
        st.markdown("**XGBoost Results:**")
        
        xgb_summary = []
        for name, result in xgb_results.items():
            config = result.get('config', {})
            xgb_summary.append({
                'Model_Type': 'XGBoost',
                'Model_Name': name,
                'N_Estimators': config.get('n_estimators', 'N/A'),
                'Max_Depth': config.get('max_depth', 'N/A'),
                'Learning_Rate': config.get('learning_rate', 'N/A'),
                'Subsample': config.get('subsample', 'N/A'),
                'Colsample_ByTree': config.get('colsample_bytree', 'N/A'),
                'R_squared': result['r_squared'],
                'RMSE': result['rmse'],
                'MAE': result['mae'],
                'CV_RMSE': result.get('cv_rmse', 'N/A'),
                'CV_Std': result.get('cv_std', 'N/A'),
            })
        
        xgb_df = pd.DataFrame(xgb_summary)
        st.dataframe(xgb_df, use_container_width=True)
        
        xgb_csv = xgb_df.to_csv(index=False)
        st.download_button(
            "📥 Download XGBoost Results",
            data=xgb_csv,
            file_name="xgboost_results.csv",
            mime="text/csv"
        )
    
    # Comparative analysis results export
    if nn_results or plsr_results or rf_results or svr_results or cnn_results or xgb_results:
        st.markdown("**Comparative Analysis Results:**")
        
        combined_summary = []
        
        # No polynomial models to add - using ML models only
        
        # Add neural network models
        for name, result in nn_results.items():
            model_info = result.get('model_info', {})
            
            combined_summary.append({
                'Model_Type': 'MLP',
                'Model_Name': name,
                'Degree': 'N/A',
                'R_squared': result['r_squared'],
                'Adj_R_squared': result['adj_r_squared'],
                'RMSE': result['rmse'],
                'AIC': result['aic'],
                'BIC': result['bic'],
                'Equation': f"MLP ({model_info.get('architecture', 'N/A')})",
                'Min_Concentration': calib_range['min_conc'],
                'Max_Concentration': calib_range['max_conc'],
                'Min_Spectrum': calib_range['min_spec'],
                'Max_Spectrum': calib_range['max_spec']
            })
        
        # Add PLSR models
        for n_comp, result in plsr_results.items():
            combined_summary.append({
                'Model_Type': 'PLSR',
                'Model_Name': 'PLSR' if n_comp == 1 else f'PLSR_{n_comp}_comp',
                'Degree': n_comp,
                'R_squared': result['r_squared'],
                'Adj_R_squared': result['adj_r_squared'],
                'RMSE': result['rmse'],
                'AIC': result['aic'],
                'BIC': result['bic'],
                'Equation': "PLSR (1 component)" if n_comp == 1 else f"PLSR with {n_comp} components",
                'Min_Concentration': calib_range['min_conc'],
                'Max_Concentration': calib_range['max_conc'],
                'Min_Spectrum': calib_range['min_spec'],
                'Max_Spectrum': calib_range['max_spec']
            })
        
        # Add Random Forest models
        for name, result in rf_results.items():
            
            combined_summary.append({
                'Model_Type': 'Random_Forest',
                'Model_Name': name,
                'Degree': 'N/A',
                'R_squared': result['r_squared'],
                'Adj_R_squared': result['adj_r_squared'],
                'RMSE': result['rmse'],
                'AIC': result['aic'],
                'BIC': result['bic'],
                'Equation': f"Random Forest ({result['n_estimators']} trees, max_depth={result['max_depth']})",
                'Min_Concentration': calib_range['min_conc'],
                'Max_Concentration': calib_range['max_conc'],
                'Min_Spectrum': calib_range['min_spec'],
                'Max_Spectrum': calib_range['max_spec'],
            })
        
        # Add SVR models
        for name, result in svr_results.items():
            
            combined_summary.append({
                'Model_Type': 'SVR',
                'Model_Name': name,
                'Degree': 'N/A',
                'R_squared': result['r_squared'],
                'Adj_R_squared': result['adj_r_squared'],
                'RMSE': result['rmse'],
                'AIC': result['aic'],
                'BIC': result['bic'],
                'Equation': f"SVR ({result['kernel']} kernel)",
                'Min_Concentration': calib_range['min_conc'],
                'Max_Concentration': calib_range['max_conc'],
                'Min_Spectrum': calib_range['min_spec'],
                'Max_Spectrum': calib_range['max_spec'],
            })
        
        # Add 1D-CNN models
        for name, result in cnn_results.items():
            model_info = result.get('model_info', {})
            model_serialized = result.get('model_serialized', 'N/A')
            
            if model_serialized == 'TF_SERIALIZATION_NOT_SUPPORTED':
                st.info(f"ℹ️ 1D-CNN model '{name}' cannot be serialized - available for analysis only")
            
            combined_summary.append({
                'Model_Type': '1D-CNN',
                'Model_Name': name,
                'Degree': 'N/A',
                'R_squared': result['r_squared'],
                'Adj_R_squared': result['adj_r_squared'],
                'RMSE': result['rmse'],
                'AIC': result['aic'],
                'BIC': result['bic'],
                'Equation': f"1D-CNN ({model_info.get('architecture', 'N/A')})",
                'Min_Concentration': calib_range['min_conc'],
                'Max_Concentration': calib_range['max_conc'],
                'Min_Spectrum': calib_range['min_spec'],
                'Max_Spectrum': calib_range['max_spec'],
            })
        
        # Add XGBoost models
        for name, result in xgb_results.items():
            config = result.get('config', {})
            combined_summary.append({
                'Model_Type': 'XGBoost',
                'Model_Name': name,
                'Degree': 'N/A',
                'R_squared': result['r_squared'],
                'Adj_R_squared': 'N/A',
                'RMSE': result['rmse'],
                'AIC': 'N/A',
                'BIC': 'N/A',
                'Equation': f"XGBoost (n_est={config.get('n_estimators', 'N/A')}, max_depth={config.get('max_depth', 'N/A')})",
                'Min_Concentration': calib_range['min_conc'],
                'Max_Concentration': calib_range['max_conc'],
                'Min_Spectrum': calib_range['min_spec'],
                'Max_Spectrum': calib_range['max_spec'],
            })
        
        combined_df = pd.DataFrame(combined_summary)
        
        # Convert numeric columns to proper types for export
        numeric_export_columns = ['Degree', 'R_squared', 'Adj_R_squared', 'RMSE', 'AIC', 'BIC']
        for col in numeric_export_columns:
            if col in combined_df.columns:
                # Replace 'N/A' with NaN and convert to numeric
                combined_df[col] = combined_df[col].replace('N/A', np.nan)
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        
        # Display the combined results
        st.dataframe(combined_df, use_container_width=True)
        
        combined_csv = combined_df.to_csv(index=False)
        st.download_button(
            "📥 Download Comparative Results",
            data=combined_csv,
            file_name="model_comparative_analysis.csv",
            mime="text/csv"
        )
    
    # Model Export Section
    st.markdown("---")
    st.markdown("#### 🤖 Export Trained Models (.pkl)")
    
    # Create a container for model export buttons
    export_col1, export_col2, export_col3 = st.columns(3)
    
    # MLP Model Export
    if nn_results:
        with export_col1:
            for name, result in nn_results.items():
                if 'model_serialized' in result and result['model_serialized'] != 'N/A':
                    st.download_button(
                        "📥 Download MLP Model",
                        data=result['model_serialized'],
                        file_name="mlp_model.pkl",
                        mime="application/octet-stream"
                    )
                    break
    
    # PLSR Model Export
    if plsr_results:
        with export_col1:
            for n_comp, result in plsr_results.items():
                if 'model_serialized' in result and result['model_serialized'] != 'N/A':
                    st.download_button(
                        "📥 Download PLSR Model",
                        data=result['model_serialized'],
                        file_name="plsr_model.pkl",
                        mime="application/octet-stream"
                    )
                    break
    
    # Random Forest Model Export
    if rf_results:
        with export_col2:
            for name, result in rf_results.items():
                if 'model_serialized' in result and result['model_serialized'] != 'N/A':
                    st.download_button(
                        "📥 Download Random Forest Model",
                        data=result['model_serialized'],
                        file_name="random_forest_model.pkl",
                        mime="application/octet-stream"
                    )
                    break
    
    # SVR Model Export
    if svr_results:
        with export_col2:
            for name, result in svr_results.items():
                if 'model_serialized' in result and result['model_serialized'] != 'N/A':
                    st.download_button(
                        "📥 Download SVR Model",
                        data=result['model_serialized'],
                        file_name="svr_model.pkl",
                        mime="application/octet-stream"
                    )
                    break
    
    # CNN Model Export
    if cnn_results:
        with export_col3:
            for name, result in cnn_results.items():
                if 'model_serialized' in result and result['model_serialized'] != 'N/A':
                    st.download_button(
                        "📥 Download CNN Model",
                        data=result['model_serialized'],
                        file_name="cnn_model.pkl",
                        mime="application/octet-stream"
                    )
                    break
                elif 'model_serialized' in result and result['model_serialized'] == 'TF_SERIALIZATION_NOT_SUPPORTED':
                    st.info("ℹ️ CNN model cannot be exported as .pkl")
                    break
    
    # XGBoost Model Export
    if xgb_results:
        with export_col3:
            for name, result in xgb_results.items():
                if 'model_serialized' in result and result['model_serialized'] != 'N/A':
                    st.download_button(
                        "📥 Download XGBoost Model",
                        data=result['model_serialized'],
                        file_name="xgboost_model.pkl",
                        mime="application/octet-stream"
                    )
                    break
    

def display_welcome_message():
    """Display welcome message and instructions."""
    st.info("Upload CSV files with absorbance data to begin analysis.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.expander("🚀 Quick Start", expanded=True):
            st.markdown("""
            **1. Upload CSV files** - wavelength and absorbance columns
            
            **2. Enter concentrations** - for each file, click 'Start Analysis'
            
            **3. Build models** - 6 ML models: PLSR, MLP, Random Forest, SVR, 1D-CNN, XGBoost
            
            **4. Export results** - models and statistics
            """)

    with col2:
        with st.expander("⚠️ Troubleshooting", expanded=False):
            st.markdown("""
            **Common Issues:**
            - File errors: Check column names and encoding
            - MLP/CNN errors: Install PyTorch
            - Analysis not starting: Click 'Start Analysis' after entering concentrations
            
            **Requirements:**   
            - Minimum 2 files for modeling
            - Wavelength range: 200-1000 nm
            - Consistent concentration units
            
            **Tips:**
            - Ensure concentration range spans analyte levels
            - Check for outliers in plots
            """)
    
    # Session info
    if hasattr(st.session_state, 'last_file_hash') and st.session_state.last_file_hash:
        with st.expander("📊 Session Info", expanded=False):
            cache_info = st.cache_data.clear.__doc__  # Get cache status
            st.write(f"Files processed: {len(st.session_state.cached_data) if st.session_state.cached_data else 0}")
            if st.button("Clear Cache", help="Clear cached data to free memory"):
                st.cache_data.clear()
                st.session_state.last_file_hash = None
                st.session_state.cached_data = None
                st.success("Cache cleared")

if __name__ == "__main__":
    main()

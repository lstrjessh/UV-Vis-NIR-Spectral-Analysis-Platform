import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize_scalar, brentq
import io
import time
from typing import Dict, List, Tuple, Optional, Union
import warnings
import hashlib
import re
import pickle
import base64
import binascii
from sklearn.preprocessing import StandardScaler

# Try to import neural network dependencies
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import shared utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.shared_utils import (
    extract_concentration_from_filename,
    SUPPORTED_EXTENSIONS,
    MAX_FILE_SIZE_MB
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Page Configuration ---
st.set_page_config(
    page_title="Concentration Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
MAX_FILE_SIZE_MB = 50
SUPPORTED_EXTENSIONS = ["csv"]
COLOR_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# --- Session State Initialization ---
def initialize_session_state():
    """Initialize session state variables for the concentration predictor."""
    defaults = {
        'loaded_model': None,
        'model_metadata': None,
        'prediction_results': [],
        'last_spectrum_hash': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- Model Processing Classes ---
class FileProcessor:
    """File processing utilities."""
    

class ModelLoader:
    """Load and validate exported polynomial models."""
    
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def load_model_file(file_content: bytes, filename: str) -> Optional[Dict]:
        """
        Load and validate model file from exported fitting results.
        
        Args:
            file_content: Raw file content as bytes
            filename: Name of the file for error reporting
            
        Returns:
            Dictionary with model information or None if invalid
        """
        try:
            # Decode file content
            content_str = file_content.decode('utf-8')
            content_io = io.StringIO(content_str)
            
            # Try to read as CSV
            df = pd.read_csv(content_io)
            
            # Check if this is a combined model file (with both polynomial and neural network models)
            has_model_type = 'Model_Type' in df.columns
            
            if has_model_type:
                # New format with both polynomial and neural network models
                models = {}
                
                for _, row in df.iterrows():
                    model_type = row['Model_Type']
                    model_name = row['Model_Name']
                    
                    if model_type == 'Polynomial':
                        # Parse polynomial model
                        degree = int(row['Degree'])
                        equation = row['Equation']
                        
                        coefficients, parsed_degree = ModelLoader.parse_polynomial_equation(equation)
                        
                        if len(coefficients) > 0 and parsed_degree == degree:
                            model_data = {
                                'type': 'polynomial',
                                'degree': degree,
                                'coefficients': coefficients,
                                'equation': equation,
                                'polynomial': np.poly1d(coefficients)
                            }
                            
                            # Add optional metrics if available
                            for metric in ['R_squared', 'Adj_R_squared', 'RMSE', 'AIC', 'BIC']:
                                if metric in row and pd.notna(row[metric]):
                                    model_data[metric.lower()] = float(row[metric])
                            
                            models[model_name] = model_data
                    
                    elif model_type == 'Neural_Network':
                        # Parse neural network model
                        model_serialized = row.get('Model_Serialized', 'N/A')
                        
                        # Check if model_serialized is valid
                        if (model_serialized != 'N/A' and 
                            model_serialized != 'SERIALIZATION_FAILED' and
                            model_serialized != 'PYTORCH_SERIALIZATION_FAILED' and
                            pd.notna(model_serialized) and 
                            isinstance(model_serialized, str) and 
                            len(model_serialized.strip()) > 0):
                            try:
                                # Clean the string (remove any whitespace)
                                model_serialized = model_serialized.strip()
                                
                                # Verify it looks like base64 (basic check)
                                if not model_serialized.replace('+', '').replace('/', '').replace('=', '').isalnum():
                                    st.warning(f"Neural network model '{model_name}' has invalid serialized data format")
                                    continue
                                
                                # Deserialize the model
                                serialization_data = pickle.loads(base64.b64decode(model_serialized))
                                
                                # Check if this is a PyTorch model (has state_dict and config)
                                if 'state_dict' in serialization_data and 'config' in serialization_data:
                                    # PyTorch model
                                    if TORCH_AVAILABLE:
                                        # Import the MLP model class
                                        from models.neural_network_model import MLPModel
                                        
                                        # Reconstruct the model
                                        model_config = serialization_data['config']
                                        model = MLPModel(**model_config)
                                        model.load_state_dict(serialization_data['state_dict'])
                                        model.eval()
                                        
                                        model_data = {
                                            'type': 'neural_network',
                                            'name': model_name,
                                            'model': model,
                                            'scaler_x': serialization_data['scaler_x'],
                                            'scaler_y': serialization_data['scaler_y'],
                                            'equation': row['Equation']
                                        }
                                        
                                        # Add optional metrics if available
                                        for metric in ['R_squared', 'Adj_R_squared', 'RMSE', 'AIC', 'BIC']:
                                            if metric in row and pd.notna(row[metric]):
                                                model_data[metric.lower()] = float(row[metric])
                                        
                                        models[model_name] = model_data
                                    else:
                                        st.warning(f"PyTorch not available - cannot load neural network model '{model_name}'")
                                        continue
                                else:
                                    # Legacy sklearn model format
                                    required_keys = ['model', 'scaler_x', 'scaler_y']
                                    if not all(key in serialization_data for key in required_keys):
                                        missing_keys = [key for key in required_keys if key not in serialization_data]
                                        st.warning(f"Neural network model '{model_name}' missing required components: {missing_keys}")
                                        continue
                                    
                                    # Add metadata
                                    serialization_data['type'] = 'neural_network'
                                    serialization_data['name'] = model_name
                                    serialization_data['equation'] = row['Equation']
                                    
                                    # Add optional metrics if available
                                    for metric in ['R_squared', 'Adj_R_squared', 'RMSE', 'AIC', 'BIC']:
                                        if metric in row and pd.notna(row[metric]):
                                            serialization_data[metric.lower()] = float(row[metric])
                                    
                                    models[model_name] = serialization_data
                                
                            except (binascii.Error, pickle.PickleError) as decode_error:
                                st.warning(f"Could not decode neural network model '{model_name}': Invalid data format")
                                continue
                            except Exception as e:
                                st.warning(f"Could not deserialize neural network model '{model_name}': {str(e)}")
                                continue
                        else:
                            if model_serialized == 'SERIALIZATION_FAILED':
                                st.warning(f"Neural network model '{model_name}' failed to serialize during export - cannot be used for prediction")
                            elif model_serialized == 'PYTORCH_SERIALIZATION_FAILED':
                                st.warning(f"PyTorch model '{model_name}' serialization failed - cannot be used for prediction")
                            else:
                                st.warning(f"Neural network model '{model_name}' has no valid serialized data")
                            continue
                    
                    elif model_type == 'MLP':
                        # MLP model (same as Neural_Network)
                        model_serialized = row.get('Model_Serialized', 'N/A')
                        
                        if (model_serialized != 'N/A' and 
                            model_serialized != 'SERIALIZATION_FAILED' and
                            model_serialized != 'PYTORCH_SERIALIZATION_FAILED' and
                            pd.notna(model_serialized) and 
                            isinstance(model_serialized, str) and 
                            len(model_serialized.strip()) > 0):
                            try:
                                model_serialized = model_serialized.strip()
                                serialization_data = pickle.loads(base64.b64decode(model_serialized))
                                
                                # Check if this is a PyTorch model (has state_dict and config)
                                if 'state_dict' in serialization_data and 'config' in serialization_data:
                                    # PyTorch model
                                    if TORCH_AVAILABLE:
                                        # Import the MLP model class
                                        from models.neural_network_model import MLPModel
                                        
                                        # Reconstruct the model
                                        model_config = serialization_data['config']
                                        model = MLPModel(**model_config)
                                        model.load_state_dict(serialization_data['state_dict'])
                                        model.eval()
                                        
                                        model_data = {
                                            'type': 'neural_network',
                                            'name': model_name,
                                            'model': model,
                                            'scaler_x': serialization_data['scaler_x'],
                                            'scaler_y': serialization_data['scaler_y'],
                                            'equation': row['Equation']
                                        }
                                        
                                        for metric in ['R_squared', 'Adj_R_squared', 'RMSE', 'AIC', 'BIC']:
                                            if metric in row and pd.notna(row[metric]):
                                                model_data[metric.lower()] = float(row[metric])
                                        
                                        models[model_name] = model_data
                                    else:
                                        st.warning(f"PyTorch not available - cannot load MLP model '{model_name}'")
                                        continue
                                else:
                                    # Legacy sklearn model format
                                    required_keys = ['model', 'scaler_x', 'scaler_y']
                                    if not all(key in serialization_data for key in required_keys):
                                        st.warning(f"MLP model '{model_name}' missing required components")
                                        continue
                                    
                                    serialization_data['type'] = 'neural_network'
                                    serialization_data['name'] = model_name
                                    serialization_data['equation'] = row['Equation']
                                    
                                    for metric in ['R_squared', 'Adj_R_squared', 'RMSE', 'AIC', 'BIC']:
                                        if metric in row and pd.notna(row[metric]):
                                            serialization_data[metric.lower()] = float(row[metric])
                                    
                                    models[model_name] = serialization_data
                                
                            except Exception as e:
                                st.warning(f"Could not deserialize MLP model '{model_name}': {str(e)}")
                                continue
                        else:
                            st.warning(f"MLP model '{model_name}' has no valid serialized data")
                        continue
                    
                    elif model_type == 'PLSR':
                        # PLSR model
                        model_serialized = row.get('Model_Serialized', 'N/A')
                        
                        if (model_serialized != 'N/A' and 
                            model_serialized != 'SERIALIZATION_FAILED' and
                            pd.notna(model_serialized) and 
                            isinstance(model_serialized, str) and 
                            len(model_serialized.strip()) > 0):
                            try:
                                model_data = pickle.loads(base64.b64decode(model_serialized.strip()))
                                
                                required_keys = ['model', 'scaler_x', 'scaler_y']
                                if not all(key in model_data for key in required_keys):
                                    st.warning(f"PLSR model '{model_name}' missing required components")
                                    continue
                                
                                model_data['type'] = 'plsr'
                                model_data['name'] = model_name
                                model_data['n_components'] = model_data.get('n_components', 1)
                                model_data['equation'] = row.get('Equation', 'PLSR')
                                
                                for metric in ['R_squared', 'Adj_R_squared', 'RMSE', 'AIC', 'BIC']:
                                    if metric in row and pd.notna(row[metric]):
                                        model_data[metric.lower()] = float(row[metric])
                                
                                models[model_name] = model_data
                                
                            except Exception as e:
                                st.warning(f"PLSR model '{model_name}' loading error: {str(e)}")
                                continue
                        else:
                            st.warning(f"PLSR model '{model_name}' has no valid serialized data")
                        continue
                    
                    elif model_type == 'Random_Forest':
                        # Random Forest model
                        model_serialized = row.get('Model_Serialized', 'N/A')
                        
                        if (model_serialized != 'N/A' and 
                            model_serialized != 'SERIALIZATION_FAILED' and
                            pd.notna(model_serialized) and 
                            isinstance(model_serialized, str) and 
                            len(model_serialized.strip()) > 0):
                            try:
                                serialization_data = pickle.loads(base64.b64decode(model_serialized.strip()))
                                
                                model_data = {
                                    'type': 'random_forest',
                                    'name': model_name,
                                    'model': serialization_data['model'],
                                    'config': serialization_data.get('config', {}),
                                    'equation': row.get('Equation', 'Random Forest')
                                }
                                
                                for metric in ['R_squared', 'Adj_R_squared', 'RMSE', 'AIC', 'BIC']:
                                    if metric in row and pd.notna(row[metric]):
                                        model_data[metric.lower()] = float(row[metric])
                                
                                models[model_name] = model_data
                                
                            except Exception as e:
                                st.warning(f"Random Forest model '{model_name}' loading error: {str(e)}")
                                continue
                        else:
                            st.warning(f"Random Forest model '{model_name}' has no valid serialized data")
                        continue
                    
                    elif model_type == 'SVR':
                        # SVR model
                        model_serialized = row.get('Model_Serialized', 'N/A')
                        
                        if (model_serialized != 'N/A' and 
                            model_serialized != 'SERIALIZATION_FAILED' and
                            pd.notna(model_serialized) and 
                            isinstance(model_serialized, str) and 
                            len(model_serialized.strip()) > 0):
                            try:
                                model_data = pickle.loads(base64.b64decode(model_serialized.strip()))
                                
                                required_keys = ['model', 'scaler_x', 'scaler_y']
                                if not all(key in model_data for key in required_keys):
                                    st.warning(f"SVR model '{model_name}' missing required components")
                                    continue
                                
                                model_data['type'] = 'svr'
                                model_data['name'] = model_name
                                model_data['kernel'] = model_data.get('kernel', 'rbf')
                                model_data['equation'] = row.get('Equation', 'SVR')
                                
                                for metric in ['R_squared', 'Adj_R_squared', 'RMSE', 'AIC', 'BIC']:
                                    if metric in row and pd.notna(row[metric]):
                                        model_data[metric.lower()] = float(row[metric])
                                
                                models[model_name] = model_data
                                
                            except Exception as e:
                                st.warning(f"SVR model '{model_name}' loading error: {str(e)}")
                                continue
                        else:
                            st.warning(f"SVR model '{model_name}' has no valid serialized data")
                        continue
                    
                    elif model_type == '1D-CNN':
                        # Try to load CNN model with PyTorch serialization
                        model_serialized = row.get('Model_Serialized', 'N/A')
                        scalers_serialized = row.get('Scalers_Serialized', 'N/A')
                        
                        if (model_serialized != 'N/A' and 
                            model_serialized != 'SERIALIZATION_FAILED' and
                            model_serialized != 'PYTORCH_SERIALIZATION_FAILED' and
                            pd.notna(model_serialized) and 
                            isinstance(model_serialized, str) and 
                            len(model_serialized.strip()) > 0):
                            try:
                                # Load the CNN model (PyTorch pickle format)
                                if TORCH_AVAILABLE:
                                    # Import the CNN model class
                                    from models.cnn_model import CNN1DModel
                                    
                                    # Decode and deserialize the model
                                    model_pickle = base64.b64decode(model_serialized.strip())
                                    serialization_data = pickle.loads(model_pickle)
                                    
                                    # Reconstruct the model
                                    model_config = serialization_data['config']
                                    model = CNN1DModel(**model_config)
                                    model.load_state_dict(serialization_data['state_dict'])
                                    
                                    # Set to evaluation mode
                                    model.eval()
                                    
                                    model_data = {
                                        'type': 'cnn',
                                        'name': model_name,
                                        'model': model,
                                        'scaler_x': serialization_data['scaler_x'],
                                        'scaler_y': serialization_data['scaler_y'],
                                        'config': serialization_data.get('model_config', {}),
                                        'equation': row.get('Equation', '1D-CNN')
                                    }
                                    
                                    for metric in ['R_squared', 'Adj_R_squared', 'RMSE', 'AIC', 'BIC']:
                                        if metric in row and pd.notna(row[metric]):
                                            model_data[metric.lower()] = float(row[metric])
                                    
                                    models[model_name] = model_data
                                    
                                else:
                                    st.warning(f"PyTorch not available - cannot load CNN model '{model_name}'")
                                    continue
                                
                            except Exception as e:
                                st.warning(f"CNN model '{model_name}' loading error: {str(e)}")
                                continue
                        else:
                            st.info(f"ℹ️ 1D-CNN model '{model_name}' cannot be loaded for prediction (missing model data)")
                        continue
                    
                    elif model_type == 'XGBoost':
                        # XGBoost model
                        model_serialized = row.get('Model_Serialized', 'N/A')
                        
                        if (model_serialized != 'N/A' and 
                            model_serialized != 'SERIALIZATION_FAILED' and
                            pd.notna(model_serialized) and 
                            isinstance(model_serialized, str) and 
                            len(model_serialized.strip()) > 0):
                            try:
                                model_data = pickle.loads(base64.b64decode(model_serialized.strip()))
                                
                                required_keys = ['model', 'scaler_x', 'scaler_y']
                                if not all(key in model_data for key in required_keys):
                                    st.warning(f"XGBoost model '{model_name}' missing required components")
                                    continue
                                
                                model_data['type'] = 'xgboost'
                                model_data['name'] = model_name
                                model_data['equation'] = row.get('Equation', 'XGBoost')
                                
                                for metric in ['R_squared', 'Adj_R_squared', 'RMSE', 'AIC', 'BIC']:
                                    if metric in row and pd.notna(row[metric]):
                                        model_data[metric.lower()] = float(row[metric])
                                
                                models[model_name] = model_data
                                
                            except Exception as e:
                                st.warning(f"XGBoost model '{model_name}' loading error: {str(e)}")
                                continue
                        else:
                            st.warning(f"XGBoost model '{model_name}' has no valid serialized data")
                        continue
                
                if not models:
                    st.error("No valid models found in file")
                    return None
            
            else:
                # Legacy format - only polynomial models
                required_cols = ['Degree', 'Equation', 'Min_Concentration', 'Max_Concentration', 'Min_Absorbance', 'Max_Absorbance']
                if not all(col in df.columns for col in required_cols):
                    st.error(f"Model file missing required columns: {required_cols}")
                    return None
                
                # Parse all models
                models = {}
                for _, row in df.iterrows():
                    degree = int(row['Degree'])
                    equation = row['Equation']
                    
                    coefficients, parsed_degree = ModelLoader.parse_polynomial_equation(equation)
                    
                    if len(coefficients) > 0 and parsed_degree == degree:
                        model_data = {
                            'type': 'polynomial',
                            'degree': degree,
                            'coefficients': coefficients,
                            'equation': equation,
                            'polynomial': np.poly1d(coefficients)
                        }
                        
                        # Add optional metrics if available
                        for metric in ['R_squared', 'Adj_R_squared', 'RMSE', 'AIC', 'BIC']:
                            if metric in row and pd.notna(row[metric]):
                                model_data[metric.lower()] = float(row[metric])
                        
                        models[degree] = model_data
                
                if not models:
                    st.error("No valid polynomial models found in file")
                    return None
            
            # Create metadata
            metadata = {
                'filename': filename,
                'models': models,
                'upload_time': time.time(),
                'best_model': None,
                'min_conc': float(df.loc[0, 'Min_Concentration']),
                'max_conc': float(df.loc[0, 'Max_Concentration']),
                'min_abs': float(df.loc[0, 'Min_Absorbance']),
                'max_abs': float(df.loc[0, 'Max_Absorbance'])
            }
            
            # Select best model (highest R² or lowest AIC)
            if any('r_squared' in model for model in models.values()):
                best_degree = max(models.keys(), 
                                key=lambda d: models[d].get('r_squared', -np.inf))
            else:
                best_degree = min(models.keys())  # Default to lowest degree
            
            metadata['best_model'] = best_degree
            
            return metadata
            
        except Exception as e:
            st.error(f"Error loading model file '{filename}': {str(e)}")
            return None

    @staticmethod
    def load_pickle_model(file_content: bytes, filename: str) -> Optional[Dict]:
        """
        Load a standalone .pkl exported model.
        Supports either:
          - a dict with keys: model, scaler_x (optional), scaler_y (optional), name, type/model_type
          - a bundle dict: { 'models': {name: {model,...}}, 'best_model': name, ... }
        Returns a structure compatible with loaded_model used by this page.
        """
        try:
            data = pickle.loads(file_content)
            def _normalize_type(t: Optional[str]) -> str:
                if not t:
                    return 'neural_network'
                t_low = str(t).strip().lower()
                if t_low in ('plsr', 'partial_least_squares', 'partial least squares'):
                    return 'plsr'
                if t_low in ('svr', 'support_vector_regression', 'support vector regression'):
                    return 'svr'
                if t_low in ('random_forest', 'rf', 'randomforest'):
                    return 'random_forest'
                if t_low in ('xgboost', 'xgb'):
                    return 'xgboost'
                if t_low in ('mlp', 'neural_network', 'nn'):
                    return 'neural_network'
                if t_low in ('cnn', '1d-cnn', '1d_cnn'):
                    return 'cnn'
                return t_low
            def _infer_from_model_obj(model_obj) -> str:
                cls = type(model_obj).__name__.lower()
                if 'pls' in cls:
                    return 'plsr'
                if 'svr' in cls or 'svm' in cls:
                    return 'svr'
                if 'forest' in cls:
                    return 'random_forest'
                if 'xgb' in cls or 'xgboost' in cls:
                    return 'xgboost'
                if 'mlp' in cls or 'regressor' in cls:
                    return 'neural_network'
                if 'sequential' in cls or 'functional' in cls or 'keras' in cls:
                    return 'cnn'
                return 'neural_network'
            # If already in bundle format, normalize and return
            if isinstance(data, dict) and 'models' in data and isinstance(data['models'], dict):
                # Normalize types in each model
                for k, v in data['models'].items():
                    norm = _normalize_type(v.get('Model_Type') or v.get('type') or v.get('model_type'))
                    # If normalized type claims MLP but model object is clearly another algo, override
                    inferred = _infer_from_model_obj(v.get('model')) if 'model' in v else norm
                    v['type'] = inferred or norm
                # Ensure best_model exists
                best = data.get('best_model') or next(iter(data['models'].keys()))
                return {
                    'models': data['models'],
                    'best_model': best,
                    'min_conc': data.get('min_conc', 0.0),
                    'max_conc': data.get('max_conc', 1.0),
                    'min_abs': data.get('min_abs', 0.0),
                    'max_abs': data.get('max_abs', 1.0)
                }

            # Otherwise treat as single model entry
            if not isinstance(data, dict):
                return None
            model_entry = data
            # Skip pure scaler bundles (no model present)
            if 'model' not in model_entry:
                return None
            model_type = _normalize_type(model_entry.get('Model_Type') or model_entry.get('type') or model_entry.get('model_type'))
            name = model_entry.get('name', filename.rsplit('.', 1)[0])
            # If type is missing or generic, infer from the actual model object
            if 'model' in model_entry:
                inferred = _infer_from_model_obj(model_entry['model'])
                model_entry['type'] = inferred or model_type
            else:
                model_entry['type'] = model_type
            # Allow missing scalers; downstream will handle identity transform
            models = {name: model_entry}
            return {
                'models': models,
                'best_model': name,
                'min_conc': model_entry.get('min_conc', 0.0),
                'max_conc': model_entry.get('max_conc', 1.0),
                'min_abs': model_entry.get('min_abs', 0.0),
                'max_abs': model_entry.get('max_abs', 1.0)
            }
        except Exception:
            st.error(f"Error loading pickle model '{filename}'")
            return None

    @staticmethod
    def load_cnn_pkl_model(model_bytes: bytes, scalers_bytes: Optional[bytes], filename: str, scalers_name: Optional[str]) -> Optional[Dict]:
        """
        Load CNN .pkl model produced by Mathematical_Modelling.
        """
        try:
            if not TORCH_AVAILABLE:
                st.error("PyTorch is required to load .pkl CNN models")
                return None
            
            # Import the CNN model class
            from models.cnn_model import CNN1DModel
            
            # Deserialize the model
            serialization_data = pickle.loads(model_bytes)
            
            # Reconstruct the model
            model_config = serialization_data['config']
            model = CNN1DModel(**model_config)
            model.load_state_dict(serialization_data['state_dict'])
            model.eval()
            
            name = filename.rsplit('.', 1)[0]
            models = {
                name: {
                    'type': 'cnn',
                    'name': name,
                    'model': model,
                    'scaler_x': serialization_data['scaler_x'],
                    'scaler_y': serialization_data['scaler_y']
                }
            }
            return {
                'models': models,
                'best_model': name,
                'min_conc': 0.0,
                'max_conc': 1.0,
                'min_abs': 0.0,
                'max_abs': 1.0
            }
        except Exception as e:
            st.error(f"Error loading CNN model: {str(e)}")
            return None

class SpectrumProcessor:
    """Process absorbance spectra for concentration prediction."""
    
    @staticmethod
    @st.cache_data(ttl=1800)
    def read_spectrum_file(file_content: bytes, filename: str) -> Optional[pd.DataFrame]:
        """
        Read and validate absorbance spectrum file.
        
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
            
            # Validate and map columns (same logic as in Polynomial_Fitter.py)
            df.columns = df.columns.str.strip()
            col_mapping = {}
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
                if 'Absorbance' in absorbance_candidates:
                    col_mapping['Absorbance'] = 'Absorbance'
                else:
                    absorbance_candidates.sort(key=len)
                    col_mapping['Absorbance'] = absorbance_candidates[0]
            
            if len(col_mapping) < 2:
                st.error(f"File '{filename}' missing required columns. Need wavelength and absorbance columns.")
                return None
            
            # Rename columns
            df = df.rename(columns=col_mapping)
            
            # Convert to numeric and validate
            for col in ['Nanometers', 'Absorbance']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove invalid rows
            df = df.dropna(subset=['Nanometers', 'Absorbance'])
            
            if len(df) == 0:
                st.error(f"File '{filename}' contains no valid numeric data.")
                return None
            
            # Sort by wavelength and remove duplicates
            df = df.sort_values('Nanometers').drop_duplicates(subset=['Nanometers']).reset_index(drop=True)
            
            return df
            
        except Exception as e:
            st.error(f"Error reading spectrum file '{filename}': {str(e)}")
            return None
    
    @staticmethod
    def resample_spectrum_to_features(df: pd.DataFrame, n_features: int) -> np.ndarray:
        """
        Interpolate spectrum to a fixed number of features matching model input size.
        Returns a shape (1, n_features) array for prediction.
        """
        if df.empty or n_features <= 0:
            return np.zeros((1, n_features))
        wl = df['Nanometers'].values
        ab = df['Absorbance'].values
        wl_min, wl_max = float(wl.min()), float(wl.max())
        grid = np.linspace(wl_min, wl_max, num=n_features)
        ab_interp = np.interp(grid, wl, ab)
        return ab_interp.reshape(1, -1)

class ConcentrationPredictor:
    """Predict concentrations using loaded polynomial models."""
    
    @staticmethod
    def predict_concentration(model_data: Dict, absorbance: float, calib_range: Dict, settings: Dict = None) -> Dict:
        """
        Predict concentration from absorbance using polynomial or neural network model.
        
        Args:
            model_data: Dictionary containing model information
            absorbance: Measured absorbance value
            calib_range: Dictionary with min/max concentration of calibration
            
        Returns:
            Dictionary with prediction results
        """
        try:
            model_type = model_data.get('type', 'polynomial')
            
            if model_type == 'neural_network':
                return ConcentrationPredictor.predict_concentration_neural_network(
                    model_data, absorbance, calib_range, settings
                )
            elif model_type == 'plsr':
                return ConcentrationPredictor.predict_concentration_plsr(
                    model_data, absorbance, calib_range, settings
                )
            elif model_type == 'random_forest':
                return ConcentrationPredictor.predict_concentration_random_forest(
                    model_data, absorbance, calib_range, settings
                )
            elif model_type == 'svr':
                return ConcentrationPredictor.predict_concentration_svr(
                    model_data, absorbance, calib_range, settings
                )
            elif model_type == 'xgboost':
                return ConcentrationPredictor.predict_concentration_xgboost(
                    model_data, absorbance, calib_range, settings
                )
            elif model_type == 'cnn':
                return ConcentrationPredictor.predict_concentration_cnn(
                    model_data, absorbance, calib_range, settings
                )
            else:
                return {
                    'concentration': None,
                    'error': 'Unsupported model type',
                    'method': 'unknown',
                    'debug_info': {'error': f'Model type {model_type} not supported'}
                }
        except Exception as e:
            return {
                'success': False,
                'error': f"Prediction error: {str(e)}"
            }
    @staticmethod
    def predict_from_spectrum(model_data: Dict, df: pd.DataFrame, calib_range: Dict, settings: Dict = None) -> Dict:
        """
        Predict concentration directly from a full spectrum using exported models.
        Supports MLP/PLSR/RF/SVR/XGBoost with scaler_x expecting spectral vectors,
        and CNN (.h5) expecting (n_features, 1) input.
        Falls back to polynomial using target wavelength if necessary.
        """
        try:
            model_type = model_data.get('type', 'polynomial')
            if model_type in ('neural_network', 'plsr', 'svr', 'xgboost'):
                scaler_x = model_data.get('scaler_x')
                scaler_y = model_data.get('scaler_y')
                model = model_data['model']
                # Determine feature count from scaler_x
                n_features = None
                if scaler_x is not None:
                    n_features = getattr(scaler_x, 'n_features_in_', None)
                    if n_features is None and hasattr(scaler_x, 'scale_'):
                        n_features = len(scaler_x.scale_)
                if n_features is None and hasattr(model, 'n_features_in_'):
                    n_features = int(model.n_features_in_)
                if not n_features:
                    return {
                        'success': False,
                        'error': 'Cannot infer model input size'
                    }
                x_vec = SpectrumProcessor.resample_spectrum_to_features(df, int(n_features))
                # Apply scaling if available; otherwise use identity
                x_scaled = scaler_x.transform(x_vec) if scaler_x is not None else x_vec
                if hasattr(model, 'predict'):
                    y_pred_scaled = model.predict(x_scaled)
                else:
                    y_pred_scaled = model.predict(x_scaled, verbose=0).ravel()
                # Inverse scale if available; else identity
                if scaler_y is not None:
                    y_pred = scaler_y.inverse_transform(np.array(y_pred_scaled).reshape(-1, 1)).ravel()[0]
                else:
                    y_pred = float(np.ravel(y_pred_scaled)[0])
                # y_pred here is concentration
                return {
                    'success': True,
                    'concentration': float(y_pred),
                    'prediction_error': 0.0,
                    'model_type': model_type,
                    'model_name': model_data.get('name', 'Model'),
                    'confidence': 'N/A',
                    'is_extrapolation': False,
                    'solution_method': 'direct_full_spectrum'
                }
            elif model_type == 'random_forest' and hasattr(model_data.get('model', None), 'predict'):
                model = model_data['model']
                # Try to infer expected feature size from model
                n_features = getattr(model, 'n_features_in_', None)
                if not n_features:
                    return {
                        'success': False,
                        'error': 'Cannot infer RF input size'
                    }
                x_vec = SpectrumProcessor.resample_spectrum_to_features(df, int(n_features))
                conc = float(model.predict(x_vec)[0])
                return {
                    'success': True,
                    'concentration': conc,
                    'prediction_error': 0.0,
                    'model_type': model_type,
                    'model_name': model_data.get('name', 'Random Forest'),
                    'confidence': 'N/A',
                    'is_extrapolation': False,
                    'solution_method': 'direct_full_spectrum'
                }
            elif model_type == 'cnn' and TORCH_AVAILABLE:
                model = model_data.get('model')
                scaler_x = model_data.get('scaler_x')
                scaler_y = model_data.get('scaler_y')
                if model is None or scaler_x is None or scaler_y is None:
                    return {
                        'success': False,
                        'error': 'CNN model/scalers missing'
                    }
                n_features = getattr(scaler_x, 'n_features_in_', None)
                if n_features is None and hasattr(scaler_x, 'scale_'):
                    n_features = len(scaler_x.scale_)
                if not n_features:
                    return {
                        'success': False,
                        'error': 'Cannot infer CNN input size'
                    }
                x_vec = SpectrumProcessor.resample_spectrum_to_features(df, int(n_features))
                x_scaled = scaler_x.transform(x_vec)
                x_cnn = x_scaled.reshape(x_scaled.shape[0], x_scaled.shape[1], 1)
                
                # Convert to PyTorch tensor and predict
                import torch
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                x_tensor = torch.FloatTensor(x_cnn).to(device)
                
                model.eval()
                with torch.no_grad():
                    y_scaled = model(x_tensor).cpu().numpy().ravel()
                
                conc = float(scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()[0])
                return {
                    'success': True,
                    'concentration': conc,
                    'prediction_error': 0.0,
                    'model_type': 'cnn',
                    'model_name': model_data.get('name', 'CNN'),
                    'confidence': 'N/A',
                    'is_extrapolation': False,
                    'solution_method': 'direct_full_spectrum'
                }
            elif model_type in ('neural_network', 'plsr', 'svr', 'xgboost'):
                # Model requires scalers but they are missing; inform user clearly
                return {
                    'success': False,
                    'error': "Selected model requires scaler_x/scaler_y for full-spectrum prediction. Please load the combined CSV export or a .pkl that includes scalers."
                }
            else:
                # Fallback: use polynomial with target wavelength
                wl = st.session_state.get('target_wavelength', 400.0)
                if len(df) >= 1:
                    if len(df) >= 2:
                        absorbance = float(np.interp(wl, df['Nanometers'].values, df['Absorbance'].values))
                    else:
                        absorbance = float(df['Absorbance'].iloc[0])
                    return ConcentrationPredictor.predict_concentration(model_data, absorbance, calib_range, settings)
                return {
                    'success': False,
                    'error': 'Spectrum has no data'
                }
        except Exception as e:
            return {
                'success': False,
                'error': f'Full-spectrum prediction failed: {str(e)}'
            }
                
        except Exception as e:
            error_result = {
                'success': False,
                'error': f"Prediction error: {str(e)}"
            }
            if settings and settings.get('debug_mode', False):
                error_result['debug_info'] = {'error': 'Debug info not available'}
            return error_result
    
    @staticmethod
    def predict_concentration_neural_network(model_data: Dict, absorbance: float, calib_range: Dict, settings: Dict = None) -> Dict:
        """
        Predict concentration from absorbance using neural network model.
        """
        try:
            if not SKLEARN_AVAILABLE and not TORCH_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Neural network dependencies not available'
                }
            
            model = model_data['model']
            scaler_x = model_data['scaler_x']
            scaler_y = model_data['scaler_y']
            
            min_cal_conc = calib_range['min_conc']
            max_cal_conc = calib_range['max_conc']
            min_cal_abs = calib_range['min_abs']
            max_cal_abs = calib_range['max_abs']
            
            # Get settings or use defaults
            if settings is None:
                settings = {
                    'extrapolation_method': 'Conservative',
                    'max_extrapolation_percent': 50,
                    'show_extrapolation_warnings': True,
                    'debug_mode': False
                }
            
            # Use numerical optimization to find concentration
            def objective(concentration):
                """Objective function: |predicted_absorbance - target_absorbance|"""
                try:
                    # Scale concentration
                    x_scaled = scaler_x.transform([[concentration]])
                    
                    # Predict absorbance
                    if hasattr(model, 'state_dict') and TORCH_AVAILABLE:
                        # PyTorch model
                        import torch
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model_eval = model.to(device)
                        x_tensor = torch.FloatTensor(x_scaled).to(device)
                        
                        model_eval.eval()
                        with torch.no_grad():
                            y_pred_scaled = model_eval(x_tensor).cpu().numpy().ravel()
                    elif hasattr(model, 'predict'):
                        # sklearn model
                        y_pred_scaled = model.predict(x_scaled)
                    else:
                        # Legacy TensorFlow model
                        y_pred_scaled = model.predict(x_scaled, verbose=0).ravel()
                    
                    # Inverse transform
                    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                    
                    return abs(y_pred[0] - absorbance)
                    
                except Exception as e:
                    return float('inf')
            
            # Search for optimal concentration
            try:
                # First, try within calibration range
                bounds = (max(0, min_cal_conc), max_cal_conc)
                result = minimize_scalar(objective, bounds=bounds, method='bounded')
                
                if result.success and result.fun < 1e-6:
                    predicted_concentration = result.x
                    solution_method = "optimization_in_range"
                else:
                    # Try extended range
                    conc_range = max_cal_conc - min_cal_conc
                    extended_factor = settings.get('max_extrapolation_percent', 50) / 100
                    extended_min = max(0, min_cal_conc - conc_range * extended_factor)
                    extended_max = max_cal_conc + conc_range * extended_factor
                    
                    bounds = (extended_min, extended_max)
                    result = minimize_scalar(objective, bounds=bounds, method='bounded')
                    
                    if result.success:
                        predicted_concentration = result.x
                        solution_method = "optimization_extended"
                    else:
                        return {
                            'success': False,
                            'error': f'Could not find valid concentration for absorbance {absorbance:.4f}'
                        }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Optimization failed: {str(e)}'
                }
            
            # Calculate predicted absorbance for validation
            try:
                x_scaled = scaler_x.transform([[predicted_concentration]])
                if hasattr(model, 'state_dict') and TORCH_AVAILABLE:
                    # PyTorch model
                    import torch
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model_eval = model.to(device)
                    x_tensor = torch.FloatTensor(x_scaled).to(device)
                    
                    model_eval.eval()
                    with torch.no_grad():
                        y_pred_scaled = model_eval(x_tensor).cpu().numpy().ravel()
                elif hasattr(model, 'predict'):
                    y_pred_scaled = model.predict(x_scaled)
                else:
                    y_pred_scaled = model.predict(x_scaled, verbose=0).ravel()
                
                predicted_abs = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()[0]
                error = abs(predicted_abs - absorbance)
                
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Validation failed: {str(e)}'
                }
            
            # Check if extrapolation is required
            is_extrapolation = not (min_cal_conc <= predicted_concentration <= max_cal_conc)
            
            # Calculate prediction confidence
            confidence = "N/A"
            confidence_score = None
            
            if 'r_squared' in model_data:
                r_squared = model_data['r_squared']
                
                # Base confidence on R² and prediction error
                if is_extrapolation:
                    # Reduce confidence for extrapolation
                    if r_squared >= 0.99 and error < 0.01:
                        confidence = "High"
                        confidence_score = 0.75
                    elif r_squared >= 0.95 and error < 0.05:
                        confidence = "Medium"
                        confidence_score = 0.60
                    elif r_squared >= 0.90 and error < 0.1:
                        confidence = "Low"
                        confidence_score = 0.45
                    else:
                        confidence = "Very Low"
                        confidence_score = 0.25
                else:
                    # Original confidence for interpolation
                    if r_squared >= 0.99 and error < 0.01:
                        confidence = "Very High"
                        confidence_score = 0.95
                    elif r_squared >= 0.95 and error < 0.05:
                        confidence = "High"
                        confidence_score = 0.85
                    elif r_squared >= 0.90 and error < 0.1:
                        confidence = "Medium"
                        confidence_score = 0.70
                    else:
                        confidence = "Low"
                        confidence_score = 0.50
            
            result = {
                'success': True,
                'concentration': predicted_concentration,
                'absorbance_input': absorbance,
                'absorbance_predicted': predicted_abs,
                'prediction_error': error,
                'model_type': 'neural_network',
                'model_name': model_data.get('name', 'N/A'),
                'r_squared': model_data.get('r_squared', None),
                'confidence': confidence,
                'confidence_score': confidence_score,
                'is_extrapolation': is_extrapolation,
                'solution_method': solution_method
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Neural network prediction error: {str(e)}"
            }
    
    @staticmethod
    def predict_concentration_plsr(model_data: Dict, absorbance: float, calib_range: Dict, settings: Dict = None) -> Dict:
        """
        Predict concentration from absorbance using PLSR model.
        """
        try:
            model = model_data['model']
            scaler_x = model_data['scaler_x']
            scaler_y = model_data['scaler_y']
            
            min_cal_conc = calib_range['min_conc']
            max_cal_conc = calib_range['max_conc']
            
            # Use optimization to find concentration that gives the target absorbance
            def objective(concentration):
                x_scaled = scaler_x.transform([[concentration]])
                y_pred_scaled = model.predict(x_scaled)
                # PLSR returns 1D array, need to reshape for scaler
                y_pred_scaled = y_pred_scaled.reshape(-1, 1)
                predicted_abs = scaler_y.inverse_transform(y_pred_scaled)[0, 0]
                return abs(predicted_abs - absorbance)
            
            # Try optimization in calibration range first
            bounds = (max(0, min_cal_conc), max_cal_conc)
            result = minimize_scalar(objective, bounds=bounds, method='bounded')
            
            if result.success and result.fun < 1e-6:
                predicted_concentration = result.x
                is_extrapolation = False
            else:
                # Try extended range
                conc_range = max_cal_conc - min_cal_conc
                extended_factor = settings.get('max_extrapolation_percent', 50) / 100 if settings else 0.5
                extended_min = max(0, min_cal_conc - conc_range * extended_factor)
                extended_max = max_cal_conc + conc_range * extended_factor
                
                bounds = (extended_min, extended_max)
                result = minimize_scalar(objective, bounds=bounds, method='bounded')
                
                if result.success:
                    predicted_concentration = result.x
                    is_extrapolation = not (min_cal_conc <= predicted_concentration <= max_cal_conc)
                else:
                    return {
                        'success': False,
                        'error': f'Could not find valid concentration for absorbance {absorbance:.4f}'
                    }
            
            # Validate prediction
            x_scaled = scaler_x.transform([[predicted_concentration]])
            y_pred_scaled = model.predict(x_scaled)
            # PLSR returns 1D array, need to reshape for scaler
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
            predicted_abs = scaler_y.inverse_transform(y_pred_scaled)[0, 0]
            
            error = abs(predicted_abs - absorbance)
            
            # Determine confidence
            if is_extrapolation:
                confidence = "Low (Extrapolated)"
                confidence_score = 0.50
            else:
                if error < 0.001:
                    confidence = "High"
                    confidence_score = 0.95
                elif error < 0.01:
                    confidence = "Medium"
                    confidence_score = 0.75
                else:
                    confidence = "Low"
                    confidence_score = 0.50
            
            return {
                'success': True,
                'concentration': predicted_concentration,
                'absorbance_input': absorbance,
                'absorbance_predicted': predicted_abs,
                'prediction_error': error,
                'model_type': 'plsr',
                'model_name': model_data.get('name', 'PLSR'),
                'confidence': confidence,
                'confidence_score': confidence_score,
                'is_extrapolation': is_extrapolation,
                'solution_method': 'optimization'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"PLSR prediction error: {str(e)}"
            }
    
    @staticmethod
    def predict_concentration_random_forest(model_data: Dict, absorbance: float, calib_range: Dict, settings: Dict = None) -> Dict:
        """
        Predict concentration from absorbance using Random Forest model.
        """
        try:
            model = model_data['model']
            
            min_cal_conc = calib_range['min_conc']
            max_cal_conc = calib_range['max_conc']
            
            # Since Random Forest is a direct mapping model, we need to use optimization
            def objective(concentration):
                predicted_abs = model.predict([[concentration]])[0]
                return abs(predicted_abs - absorbance)
            
            # Try optimization in calibration range first
            bounds = (max(0, min_cal_conc), max_cal_conc)
            result = minimize_scalar(objective, bounds=bounds, method='bounded')
            
            if result.success and result.fun < 1e-6:
                predicted_concentration = result.x
                is_extrapolation = False
            else:
                # Try extended range
                conc_range = max_cal_conc - min_cal_conc
                extended_factor = settings.get('max_extrapolation_percent', 50) / 100 if settings else 0.5
                extended_min = max(0, min_cal_conc - conc_range * extended_factor)
                extended_max = max_cal_conc + conc_range * extended_factor
                
                bounds = (extended_min, extended_max)
                result = minimize_scalar(objective, bounds=bounds, method='bounded')
                
                if result.success:
                    predicted_concentration = result.x
                    is_extrapolation = not (min_cal_conc <= predicted_concentration <= max_cal_conc)
                else:
                    return {
                        'success': False,
                        'error': f'Could not find valid concentration for absorbance {absorbance:.4f}'
                    }
            
            # Validate prediction
            predicted_abs = model.predict([[predicted_concentration]])[0]
            error = abs(predicted_abs - absorbance)
            
            # Determine confidence
            if is_extrapolation:
                confidence = "Low (Extrapolated)"
                confidence_score = 0.50
            else:
                if error < 0.001:
                    confidence = "High"
                    confidence_score = 0.95
                elif error < 0.01:
                    confidence = "Medium"
                    confidence_score = 0.75
                else:
                    confidence = "Low"
                    confidence_score = 0.50
            
            return {
                'success': True,
                'concentration': predicted_concentration,
                'absorbance_input': absorbance,
                'absorbance_predicted': predicted_abs,
                'prediction_error': error,
                'model_type': 'random_forest',
                'model_name': model_data.get('name', 'Random Forest'),
                'confidence': confidence,
                'confidence_score': confidence_score,
                'is_extrapolation': is_extrapolation,
                'solution_method': 'optimization'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Random Forest prediction error: {str(e)}"
            }
    
    @staticmethod
    def predict_concentration_svr(model_data: Dict, absorbance: float, calib_range: Dict, settings: Dict = None) -> Dict:
        """
        Predict concentration from absorbance using SVR model.
        """
        try:
            model = model_data['model']
            scaler_x = model_data['scaler_x']
            scaler_y = model_data['scaler_y']
            
            min_cal_conc = calib_range['min_conc']
            max_cal_conc = calib_range['max_conc']
            
            # Use optimization to find concentration that gives the target absorbance
            def objective(concentration):
                x_scaled = scaler_x.transform([[concentration]])
                y_pred_scaled = model.predict(x_scaled)
                predicted_abs = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]
                return abs(predicted_abs - absorbance)
            
            # Try optimization in calibration range first
            bounds = (max(0, min_cal_conc), max_cal_conc)
            result = minimize_scalar(objective, bounds=bounds, method='bounded')
            
            if result.success and result.fun < 1e-6:
                predicted_concentration = result.x
                is_extrapolation = False
            else:
                # Try extended range
                conc_range = max_cal_conc - min_cal_conc
                extended_factor = settings.get('max_extrapolation_percent', 50) / 100 if settings else 0.5
                extended_min = max(0, min_cal_conc - conc_range * extended_factor)
                extended_max = max_cal_conc + conc_range * extended_factor
                
                bounds = (extended_min, extended_max)
                result = minimize_scalar(objective, bounds=bounds, method='bounded')
                
                if result.success:
                    predicted_concentration = result.x
                    is_extrapolation = not (min_cal_conc <= predicted_concentration <= max_cal_conc)
                else:
                    return {
                        'success': False,
                        'error': f'Could not find valid concentration for absorbance {absorbance:.4f}'
                    }
            
            # Validate prediction
            x_scaled = scaler_x.transform([[predicted_concentration]])
            y_pred_scaled = model.predict(x_scaled)
            predicted_abs = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]
            
            error = abs(predicted_abs - absorbance)
            
            # Determine confidence
            if is_extrapolation:
                confidence = "Low (Extrapolated)"
                confidence_score = 0.50
            else:
                if error < 0.001:
                    confidence = "High"
                    confidence_score = 0.95
                elif error < 0.01:
                    confidence = "Medium"
                    confidence_score = 0.75
                else:
                    confidence = "Low"
                    confidence_score = 0.50
            
            return {
                'success': True,
                'concentration': predicted_concentration,
                'absorbance_input': absorbance,
                'absorbance_predicted': predicted_abs,
                'prediction_error': error,
                'model_type': 'svr',
                'model_name': model_data.get('name', 'SVR'),
                'confidence': confidence,
                'confidence_score': confidence_score,
                'is_extrapolation': is_extrapolation,
                'solution_method': 'optimization'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"SVR prediction error: {str(e)}"
            }
    
    @staticmethod
    def predict_concentration_polynomial(model_data: Dict, absorbance: float, calib_range: Dict, settings: Dict = None) -> Dict:
        """
        Predict concentration from absorbance using polynomial model.
        """
        try:
            polynomial = model_data['polynomial']
            degree = model_data['degree']
            
            min_cal_conc = calib_range['min_conc']
            max_cal_conc = calib_range['max_conc']
            min_cal_abs = calib_range['min_abs']
            max_cal_abs = calib_range['max_abs']
            
            # Get settings or use defaults
            if settings is None:
                settings = {
                    'extrapolation_method': 'Conservative',
                    'max_extrapolation_percent': 50,
                    'show_extrapolation_warnings': True,
                    'debug_mode': False
                }
            
            # Debug information
            debug_info = {
                'methods_tried': [],
                'polynomial_info': {
                    'degree': degree,
                    'coefficients': polynomial.coefficients.tolist(),
                    'calibration_range': [min_cal_conc, max_cal_conc],
                    'absorbance_range': [min_cal_abs, max_cal_abs]
                },
                'target_absorbance': absorbance
            }
            
            # Calculate extrapolation limits
            conc_range = max_cal_conc - min_cal_conc
            max_extrapolation_distance = conc_range * (settings.get('max_extrapolation_percent', 50) / 100)
            
            # Method 1: Try numerical optimization approach
            def objective(c):
                """Objective function to minimize: |polynomial(c) - absorbance|"""
                return abs(polynomial(c) - absorbance)
            
            # First, try to find solution within calibration range
            try:
                # Use golden section search within calibration range
                bounds = (max(0, min_cal_conc), max_cal_conc)
                result = minimize_scalar(
                    objective,
                    bounds=bounds,
                    method='bounded'
                )
                
                debug_info['methods_tried'].append({
                    'method': 'optimization_in_range',
                    'bounds': bounds,
                    'success': result.success,
                    'fun': result.fun,
                    'x': result.x if result.success else None
                })
                
                if result.success and result.fun < 1e-6:  # Very small error
                    predicted_concentration = result.x
                    solution_method = "optimization_in_range"
                else:
                    # If optimization didn't converge well, try root-finding methods
                    raise ValueError("Optimization didn't converge")
                    
            except (ValueError, RuntimeError):
                # Method 2: Try root-finding approach with extended range
                try:
                    # Define function for root finding: f(c) = polynomial(c) - absorbance
                    def poly_minus_abs(c):
                        return polynomial(c) - absorbance
                    
                    # Try to find bracket where function changes sign
                    bracket_found = False
                    predicted_concentration = None
                    
                    # Search for sign change in extended range
                    extended_min = max(0, min_cal_conc - (max_cal_conc - min_cal_conc) * 0.5)
                    extended_max = max_cal_conc + (max_cal_conc - min_cal_conc) * 0.5
                    
                    # Check multiple intervals
                    search_points = np.linspace(extended_min, extended_max, 100)
                    sign_changes = []
                    
                    for i in range(len(search_points) - 1):
                        c1, c2 = search_points[i], search_points[i + 1]
                        f1, f2 = poly_minus_abs(c1), poly_minus_abs(c2)
                        
                        # Check if there's a sign change (root exists)
                        if f1 * f2 < 0:
                            sign_changes.append((c1, c2, f1, f2))
                            try:
                                root = brentq(poly_minus_abs, c1, c2)
                                if root >= 0:  # Only accept positive concentrations
                                    predicted_concentration = root
                                    bracket_found = True
                                    break
                            except ValueError:
                                continue
                    
                    debug_info['methods_tried'].append({
                        'method': 'root_finding',
                        'search_range': [extended_min, extended_max],
                        'sign_changes_found': len(sign_changes),
                        'bracket_found': bracket_found,
                        'root': predicted_concentration if bracket_found else None
                    })
                    
                    if not bracket_found:
                        # Method 3: Use polynomial roots as backup
                        coeffs = polynomial.coefficients.copy()
                        coeffs[-1] -= absorbance
                        roots = np.roots(coeffs)
                        
                        # Filter for real, positive roots
                        real_roots = [np.real(r) for r in roots 
                                    if np.isreal(r) and np.real(r) >= 0]
                        
                        debug_info['methods_tried'].append({
                            'method': 'polynomial_roots',
                            'all_roots': [complex(r) for r in roots],
                            'real_positive_roots': real_roots,
                            'num_valid_roots': len(real_roots)
                        })
                        
                        if real_roots:
                            # Choose the root closest to the calibration range
                            if any(min_cal_conc <= r <= max_cal_conc for r in real_roots):
                                # Prefer roots within calibration range
                                in_range_roots = [r for r in real_roots 
                                                if min_cal_conc <= r <= max_cal_conc]
                                predicted_concentration = min(in_range_roots)
                                solution_method = "roots_in_range"
                            else:
                                # For extrapolation, be more selective about which root to choose
                                extrapolation_method = settings.get('extrapolation_method', 'Conservative')
                                
                                if extrapolation_method == 'Polynomial Only':
                                    # Use only polynomial root finding
                                    distances = [min(abs(r - min_cal_conc), abs(r - max_cal_conc)) 
                                               for r in real_roots]
                                    predicted_concentration = real_roots[np.argmin(distances)]
                                    solution_method = "roots_extrapolation"
                                else:
                                    # First, try optimization in extended range to guide root selection
                                    try:
                                        if extrapolation_method == 'Conservative':
                                            # Limited extrapolation range
                                            ext_factor = max_extrapolation_distance / conc_range
                                        elif extrapolation_method == 'Aggressive':
                                            # Wider extrapolation range
                                            ext_factor = max_extrapolation_distance / conc_range * 2
                                        else:
                                            ext_factor = 0.5  # Default
                                        
                                        extended_min = max(0, min_cal_conc - conc_range * ext_factor)
                                        extended_max = max_cal_conc + conc_range * ext_factor
                                        
                                        opt_result = minimize_scalar(
                                            objective,
                                            bounds=(extended_min, extended_max),
                                            method='bounded'
                                        )
                                        
                                        if opt_result.success and opt_result.fun < 1e-4:
                                            # Check if result is within allowed extrapolation distance
                                            result_distance = min(abs(opt_result.x - min_cal_conc), 
                                                                abs(opt_result.x - max_cal_conc))
                                            
                                            if (min_cal_conc <= opt_result.x <= max_cal_conc or 
                                                result_distance <= max_extrapolation_distance):
                                                predicted_concentration = opt_result.x
                                                solution_method = "optimization_extrapolation"
                                            else:
                                                # Limit to maximum allowed extrapolation
                                                if opt_result.x < min_cal_conc:
                                                    predicted_concentration = min_cal_conc - max_extrapolation_distance
                                                else:
                                                    predicted_concentration = max_cal_conc + max_extrapolation_distance
                                                solution_method = "optimization_limited"
                                        else:
                                            # Fall back to closest root logic
                                            distances = [min(abs(r - min_cal_conc), abs(r - max_cal_conc)) 
                                                       for r in real_roots]
                                            predicted_concentration = real_roots[np.argmin(distances)]
                                            solution_method = "roots_extrapolation"
                                    except:
                                        # Fall back to closest root logic
                                        distances = [min(abs(r - min_cal_conc), abs(r - max_cal_conc)) 
                                                   for r in real_roots]
                                        predicted_concentration = real_roots[np.argmin(distances)]
                                        solution_method = "roots_extrapolation"
                        else:
                            # Last resort: try a simple grid search
                            try:
                                # Create a wide concentration range for grid search
                                c_test = np.linspace(0, max_cal_conc * 3, 1000)
                                abs_test = polynomial(c_test)
                                
                                # Find the closest match
                                errors = np.abs(abs_test - absorbance)
                                min_error_idx = np.argmin(errors)
                                
                                debug_info['methods_tried'].append({
                                    'method': 'grid_search',
                                    'search_range': [0, max_cal_conc * 3],
                                    'min_error': float(errors[min_error_idx]),
                                    'best_concentration': float(c_test[min_error_idx]),
                                    'predicted_absorbance': float(abs_test[min_error_idx])
                                })
                                
                                if errors[min_error_idx] < 0.1:  # Reasonable error threshold
                                    predicted_concentration = c_test[min_error_idx]
                                    solution_method = "grid_search"
                                else:
                                    error_result = {
                                        'success': False,
                                        'error': f'No valid solution found for absorbance {absorbance:.4f}. Tried all methods: optimization, root-finding, polynomial roots, and grid search. The polynomial model may not be suitable for this measurement.'
                                    }
                                    if settings.get('debug_mode', False):
                                        error_result['debug_info'] = debug_info
                                    return error_result
                            except:
                                error_result = {
                                    'success': False,
                                    'error': f'No valid solution found for absorbance {absorbance:.4f}. The polynomial model may not be suitable for this measurement.'
                                }
                                if settings.get('debug_mode', False):
                                    error_result['debug_info'] = debug_info
                                return error_result
                    else:
                        solution_method = "root_finding"
                        
                except Exception as e:
                    return {
                        'success': False,
                        'error': f'Numerical solution failed: {str(e)}'
                    }
            
            # Validate the solution
            predicted_abs = polynomial(predicted_concentration)
            error = abs(predicted_abs - absorbance)
            
            # Check if extrapolation is required
            is_extrapolation = not (min_cal_conc <= predicted_concentration <= max_cal_conc)
            
            # Calculate prediction confidence
            confidence = "N/A"
            confidence_score = None
            
            if 'r_squared' in model_data:
                r_squared = model_data['r_squared']
                
                # Base confidence on R² and prediction error
                if is_extrapolation:
                    # Reduce confidence for extrapolation
                    if r_squared >= 0.99 and error < 0.01:
                        confidence = "High"  # Reduced from Very High
                        confidence_score = 0.75
                    elif r_squared >= 0.95 and error < 0.05:
                        confidence = "Medium"  # Reduced from High
                        confidence_score = 0.60
                    elif r_squared >= 0.90 and error < 0.1:
                        confidence = "Low"  # Reduced from Medium
                        confidence_score = 0.45
                    else:
                        confidence = "Very Low"
                        confidence_score = 0.25
                else:
                    # Original confidence for interpolation
                    if r_squared >= 0.99 and error < 0.01:
                        confidence = "Very High"
                        confidence_score = 0.95
                    elif r_squared >= 0.95 and error < 0.05:
                        confidence = "High"
                        confidence_score = 0.85
                    elif r_squared >= 0.90 and error < 0.1:
                        confidence = "Medium"
                        confidence_score = 0.70
                    else:
                        confidence = "Low"
                        confidence_score = 0.50
            
            result = {
                'success': True,
                'concentration': predicted_concentration,
                'absorbance_input': absorbance,
                'absorbance_predicted': predicted_abs,
                'prediction_error': error,
                'model_type': 'polynomial',
                'model_degree': degree,
                'r_squared': model_data.get('r_squared', None),
                'confidence': confidence,
                'confidence_score': confidence_score,
                'is_extrapolation': is_extrapolation,
                'solution_method': solution_method if 'solution_method' in locals() else "optimization_in_range"
            }
            
            # Add debug information if requested
            if settings.get('debug_mode', False):
                result['debug_info'] = debug_info
            
            return result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': f"Prediction error: {str(e)}"
            }
            if settings and settings.get('debug_mode', False):
                error_result['debug_info'] = debug_info if 'debug_info' in locals() else {'error': 'Debug info not available'}
            return error_result


# --- Color Utilities (for wavelength-colored spectra) ---
@st.cache_data
def wavelength_to_rgb(wavelength: float) -> Tuple[float, float, float]:
    """Convert wavelength (nm) to an approximate RGB triple (0-1 range)."""
    wl = max(300, min(850, float(wavelength)))
    if wl < 380:  # UV
        ratio = (wl - 300) / 80
        return (0.3 + 0.3 * ratio, 0.0, 0.6 + 0.4 * ratio)
    elif wl < 440:  # Violet
        ratio = (wl - 380) / 60
        return (0.6 - 0.6 * ratio, 0.0, 1.0)
    elif wl < 490:  # Blue
        ratio = (wl - 440) / 50
        return (0.0, ratio, 1.0)
    elif wl < 510:  # Cyan
        ratio = (wl - 490) / 20
        return (0.0, 1.0, 1.0 - ratio)
    elif wl < 580:  # Green
        ratio = (wl - 510) / 70
        return (ratio, 1.0, 0.0)
    elif wl < 645:  # Yellow to Orange
        ratio = (wl - 580) / 65
        return (1.0, 1.0 - 0.5 * ratio, 0.0)
    elif wl <= 780:  # Red
        return (1.0, 0.0, 0.0)
    else:  # Near IR
        ratio = (wl - 780) / 70
        return (1.0 - 0.2 * ratio, 0.0, 0.0)

class PlotManager:
    """Create visualization plots for concentration prediction."""
    
    @staticmethod
    def create_spectrum_plot(df: pd.DataFrame, target_wavelength: float = None) -> go.Figure:
        """Create spectrum plot with wavelength-colored line. (No peak highlighting)"""
        fig = go.Figure()
        
        wl = df['Nanometers'].values
        ab = df['Absorbance'].values
        
        # Draw gradient line by segments colored by wavelength
        segments = min(120, max(2, len(wl) - 1))
        for i in range(segments):
            idx = int(i * len(wl) / segments)
            next_idx = int((i + 1) * len(wl) / segments)
            seg_wl = wl[idx:next_idx+1]
            seg_ab = ab[idx:next_idx+1]
            if len(seg_wl) < 2:
                continue
            r, g, b = wavelength_to_rgb(seg_wl[len(seg_wl)//2])
            fig.add_trace(go.Scatter(
                x=seg_wl, y=seg_ab, mode='lines',
                line=dict(color=f'rgb({int(r*255)},{int(g*255)},{int(b*255)})', width=3),
                name='Spectrum', showlegend=False,
                hovertemplate='λ: %{x:.1f} nm<br>A: %{y:.4f}<extra></extra>'
            ))
        
        # Mark target wavelength if provided
        if target_wavelength is not None:
            fig.add_vline(
                x=target_wavelength,
                line_dash="dash", line_color="red",
                annotation_text=f"Target: {float(target_wavelength):.1f} nm"
            )
        
        fig.update_layout(
            title='Absorbance Spectrum',
            xaxis_title='Wavelength (nm)',
            yaxis_title='Absorbance',
            template='plotly_white',
            hovermode='closest',
            height=400
        )
        
        return fig
    
    @staticmethod
    def predict_concentration_xgboost(model_data: Dict, absorbance: float, calib_range: Dict, settings: Dict = None) -> Dict:
        """
        Predict concentration from absorbance using XGBoost model.
        """
        try:
            if not XGB_AVAILABLE:
                return {
                    'concentration': None,
                    'confidence': 0.0,
                    'method': 'xgboost',
                    'error': 'XGBoost not available',
                    'extrapolation_warning': False,
                    'debug_info': {'error': 'XGBoost library not installed'}
                }
            
            model = model_data['model']
            scaler_x = model_data['scaler_x']
            scaler_y = model_data['scaler_y']
            
            min_cal_conc = calib_range['min_conc']
            max_cal_conc = calib_range['max_conc']
            min_cal_abs = calib_range['min_abs']
            max_cal_abs = calib_range['max_abs']
            
            # Get settings or use defaults
            if settings is None:
                settings = {
                    'extrapolation_method': 'Conservative',
                    'max_extrapolation_percent': 50,
                    'show_extrapolation_warnings': True,
                    'debug_mode': False
                }
            
            # Check if absorbance is within calibration range
            within_range = min_cal_abs <= absorbance <= max_cal_abs
            
            # Prepare input data (single absorbance value)
            X = np.array([[absorbance]])
            
            # Scale the input
            X_scaled = scaler_x.transform(X)
            
            # Make prediction
            y_pred_scaled = model.predict(X_scaled)
            
            # Inverse transform to get actual concentration
            concentration = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]
            
            # Calculate confidence based on how close the absorbance is to calibration range
            if within_range:
                confidence = 0.95
                extrapolation_warning = False
            else:
                # Calculate how far outside the range
                if absorbance < min_cal_abs:
                    distance = (min_cal_abs - absorbance) / (max_cal_abs - min_cal_abs)
                else:
                    distance = (absorbance - max_cal_abs) / (max_cal_abs - min_cal_abs)
                
                # Reduce confidence based on distance
                confidence = max(0.1, 0.95 - distance * 0.5)
                extrapolation_warning = True
            
            # Ensure concentration is positive
            concentration = max(0, concentration)
            
            result = {
                'concentration': concentration,
                'confidence': confidence,
                'method': 'xgboost',
                'extrapolation_warning': extrapolation_warning,
                'within_calibration_range': within_range,
                'debug_info': {
                    'model_name': model_data.get('name', 'XGBoost'),
                    'absorbance_input': absorbance,
                    'scaled_input': X_scaled[0, 0],
                    'scaled_prediction': y_pred_scaled[0],
                    'final_concentration': concentration,
                    'calibration_range': [min_cal_conc, max_cal_conc],
                    'absorbance_range': [min_cal_abs, max_cal_abs]
                }
            }
            
            return result
            
        except Exception as e:
            return {
                'concentration': None,
                'confidence': 0.0,
                'method': 'xgboost',
                'error': str(e),
                'extrapolation_warning': False,
                'debug_info': {'error': str(e)}
            }

    @staticmethod
    def predict_concentration_cnn(model_data: Dict, absorbance: float, calib_range: Dict, settings: Dict = None) -> Dict:
        """
        Predict concentration from absorbance using CNN model.
        """
        try:
            if not TORCH_AVAILABLE:
                return {
                    'concentration': None,
                    'confidence': 0.0,
                    'method': 'cnn',
                    'error': 'PyTorch not available',
                    'extrapolation_warning': False,
                    'debug_info': {'error': 'PyTorch library not installed'}
                }
            
            model = model_data['model']
            scaler_x = model_data['scaler_x']
            scaler_y = model_data['scaler_y']
            
            min_cal_conc = calib_range['min_conc']
            max_cal_conc = calib_range['max_conc']
            min_cal_abs = calib_range['min_abs']
            max_cal_abs = calib_range['max_abs']
            
            # Get settings or use defaults
            if settings is None:
                settings = {
                    'extrapolation_method': 'Conservative',
                    'max_extrapolation_percent': 50,
                    'show_extrapolation_warnings': True,
                    'debug_mode': False
                }
            
            # Check if absorbance is within calibration range
            within_range = min_cal_abs <= absorbance <= max_cal_abs
            
            # Prepare input data (single absorbance value)
            X = np.array([[absorbance]])
            
            # Scale the input
            X_scaled = scaler_x.transform(X)
            
            # Convert to PyTorch tensor and make prediction
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Reshape for CNN: (1, 1) -> (1, 1, 1)
            X_cnn = X_scaled.reshape(-1, 1, 1)
            X_tensor = torch.FloatTensor(X_cnn).to(device)
            
            model.eval()
            with torch.no_grad():
                y_pred_scaled = model(X_tensor).cpu().numpy()
            
            # Inverse transform to get actual concentration
            concentration = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]
            
            # Calculate confidence based on how close the absorbance is to calibration range
            if within_range:
                confidence = 0.95
                extrapolation_warning = False
            else:
                # Calculate how far outside the range
                if absorbance < min_cal_abs:
                    distance = (min_cal_abs - absorbance) / (max_cal_abs - min_cal_abs)
                else:
                    distance = (absorbance - max_cal_abs) / (max_cal_abs - min_cal_abs)
                
                # Reduce confidence based on distance
                confidence = max(0.1, 0.95 - distance * 0.5)
                extrapolation_warning = True
            
            # Ensure concentration is positive
            concentration = max(0, concentration)
            
            result = {
                'concentration': concentration,
                'confidence': confidence,
                'method': 'cnn',
                'extrapolation_warning': extrapolation_warning,
                'within_calibration_range': within_range,
                'debug_info': {
                    'model_name': model_data.get('name', '1D-CNN'),
                    'absorbance_input': absorbance,
                    'scaled_input': X_scaled[0, 0],
                    'scaled_prediction': y_pred_scaled[0, 0],
                    'final_concentration': concentration,
                    'calibration_range': [min_cal_conc, max_cal_conc],
                    'absorbance_range': [min_cal_abs, max_cal_abs]
                }
            }
            
            return result
            
        except Exception as e:
            return {
                'concentration': None,
                'confidence': 0.0,
                'method': 'cnn',
                'error': str(e),
                'extrapolation_warning': False,
                'debug_info': {'error': str(e)}
            }

    @staticmethod
    def create_prediction_plot(model_data: Dict, prediction_result: Dict,
                             calib_range: Dict,
                             absorbance_range: Tuple[float, float] = None,
                             show_band: bool = True,
                             show_uncertainty: bool = True) -> go.Figure:
        """Create calibration curve with prediction visualization."""
        fig = go.Figure()
        
        model_type = model_data.get('type', 'polynomial')
        
        # Generate concentration range for curve
        if absorbance_range:
            min_abs, max_abs = absorbance_range
            # Extend range slightly
            abs_range = max_abs - min_abs
            min_abs = max(0, min_abs - abs_range * 0.1)
            max_abs = max_abs + abs_range * 0.1
        else:
            min_abs, max_abs = 0, 2.0
        
        # Create concentration points for curve
        c_range = np.linspace(0, 10, 1000)  # Adjust range as needed
        
        if model_type == 'polynomial':
            # Polynomial model
            polynomial = model_data['polynomial']
            degree = model_data['degree']
            a_curve = polynomial(c_range)
            model_name = f'Polynomial (Degree {degree})'
            
        elif model_type == 'neural_network':
            # Neural network model
            model_name = f"Neural Network ({model_data.get('name', 'N/A')})"
            a_curve = []
            
            # Generate predictions using neural network
            for c in c_range:
                try:
                    # Scale concentration
                    scaler_x = model_data['scaler_x']
                    scaler_y = model_data['scaler_y']
                    model = model_data['model']
                    
                    x_scaled = scaler_x.transform([[c]])
                    
                    # Predict absorbance
                    if hasattr(model, 'predict'):
                        # sklearn model
                        y_pred_scaled = model.predict(x_scaled)
                    else:
                        # TensorFlow model
                        y_pred_scaled = model.predict(x_scaled, verbose=0).ravel()
                    
                    # Inverse transform
                    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                    a_curve.append(y_pred[0])
                    
                except Exception as e:
                    a_curve.append(np.nan)
            
            a_curve = np.array(a_curve)
            
        elif model_type == 'xgboost':
            # XGBoost model
            model_name = f"XGBoost ({model_data.get('name', 'N/A')})"
            a_curve = []
            
            # Generate predictions using XGBoost
            for c in c_range:
                try:
                    # Scale concentration
                    scaler_x = model_data['scaler_x']
                    scaler_y = model_data['scaler_y']
                    model = model_data['model']
                    
                    x_scaled = scaler_x.transform([[c]])
                    
                    # Predict absorbance
                    y_pred_scaled = model.predict(x_scaled)
                    
                    # Inverse transform
                    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                    a_curve.append(y_pred[0])
                    
                except Exception as e:
                    a_curve.append(np.nan)
            
            a_curve = np.array(a_curve)
            
        elif model_type == 'cnn':
            # CNN model
            model_name = f"1D-CNN ({model_data.get('name', 'N/A')})"
            a_curve = []
            
            # Generate predictions using CNN
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model_data['model'].to(device)
            model.eval()
            
            for c in c_range:
                try:
                    # Scale concentration
                    scaler_x = model_data['scaler_x']
                    scaler_y = model_data['scaler_y']
                    
                    x_scaled = scaler_x.transform([[c]])
                    
                    # Convert to PyTorch tensor
                    x_cnn = x_scaled.reshape(-1, 1, 1)
                    x_tensor = torch.FloatTensor(x_cnn).to(device)
                    
                    # Predict absorbance
                    with torch.no_grad():
                        y_pred_scaled = model(x_tensor).cpu().numpy()
                    
                    # Inverse transform
                    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                    a_curve.append(y_pred[0])
                    
                except Exception as e:
                    a_curve.append(np.nan)
            
            a_curve = np.array(a_curve)
            
        else:
            # Unknown model type
            a_curve = np.full_like(c_range, np.nan)
            model_name = f"Unknown Model ({model_type})"
        
        # Filter to reasonable absorbance range
        mask = (a_curve >= min_abs) & (a_curve <= max_abs) & (c_range >= 0) & np.isfinite(a_curve)
        c_filtered = c_range[mask]
        a_filtered = a_curve[mask]
        
        # Plot calibration curve
        fig.add_trace(go.Scatter(
            x=c_filtered, y=a_filtered,
            mode='lines', name=f'Calibration Curve ({model_name})',
            line=dict(color='blue', width=2),
            hovertemplate='C: %{x:.3f}<br>A: %{y:.4f}<extra></extra>'
        ))
        
        # Calibration range band
        if show_band and calib_range is not None:
            try:
                min_c, max_c = calib_range['min_conc'], calib_range['max_conc']
                min_a, max_a = calib_range['min_abs'], calib_range['max_abs']
                fig.add_shape(
                    type="rect",
                    x0=min_c, x1=max_c, y0=min_a, y1=max_a,
                    fillcolor="rgba(0, 200, 0, 0.08)", line=dict(color="rgba(0,0,0,0)"),
                    layer="below"
                )
                fig.add_annotation(
                    x=(min_c+max_c)/2, y=max_a, yshift=10,
                    text="Calibration Range",
                    showarrow=False, font=dict(size=10, color="#2ca02c")
                )
            except Exception:
                pass

        # Mark prediction point if successful
        if prediction_result.get('success', False):
            pred_conc = prediction_result.get('concentration')
            pred_abs = prediction_result.get('absorbance_input', prediction_result.get('absorbance_predicted', None))
            errorbar_x = None
            if show_uncertainty and pred_abs is not None and np.isfinite(pred_abs):
                # Estimate concentration uncertainty from RMSE and local slope
                rmse = model_data.get('rmse', None)
                if rmse is not None and np.isfinite(rmse):
                    try:
                        if model_type == 'polynomial':
                            deriv = model_data['polynomial'].deriv()(pred_conc)
                        else:
                            # Numerical derivative for NN/CNN
                            eps = max(1e-6, 1e-3 * max(pred_conc, 1.0))
                            # compute A(c+eps) and A(c-eps)
                            def nn_abs(c):
                                scaler_x = model_data['scaler_x']
                                scaler_y = model_data['scaler_y']
                                model = model_data['model']
                                x_scaled = scaler_x.transform([[c]])
                                
                                if model_type == 'cnn' and TORCH_AVAILABLE:
                                    # PyTorch CNN prediction
                                    import torch
                                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                    model = model.to(device)
                                    x_cnn = x_scaled.reshape(-1, 1, 1)
                                    x_tensor = torch.FloatTensor(x_cnn).to(device)
                                    model.eval()
                                    with torch.no_grad():
                                        y_pred_scaled = model(x_tensor).cpu().numpy()
                                elif hasattr(model, 'predict'):
                                    y_pred_scaled = model.predict(x_scaled)
                                else:
                                    y_pred_scaled = model.predict(x_scaled, verbose=0).ravel()
                                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()[0]
                                return y_pred
                            deriv = (nn_abs(pred_conc + eps) - nn_abs(pred_conc - eps)) / (2*eps)
                        if abs(deriv) > 1e-8:
                            sigma_c = float(min( (rmse/abs(deriv)), max(pred_conc, 1.0)*0.5 ))
                            errorbar_x = sigma_c
                    except Exception:
                        errorbar_x = None
            if pred_abs is not None and np.isfinite(pred_abs):
                fig.add_trace(go.Scatter(
                    x=[pred_conc], y=[pred_abs],
                    mode='markers', name='Prediction',
                    marker=dict(size=15, color='red', symbol='diamond'),
                    error_x=dict(type='constant', value=errorbar_x, visible=bool(errorbar_x)) if errorbar_x else None,
                    hovertemplate=f'<b>Prediction</b><br>C: {pred_conc:.4f}<br>A: {pred_abs:.4f}<extra></extra>'
                ))
                # Add horizontal and vertical lines to show prediction
                fig.add_hline(
                    y=pred_abs, line_dash="dash", line_color="red", opacity=0.7,
                    annotation_text=f"A = {pred_abs:.4f}"
                )
                fig.add_vline(
                    x=pred_conc, line_dash="dash", line_color="red", opacity=0.7,
                    annotation_text=f"C = {pred_conc:.4f}"
                )
        
        fig.update_layout(
            title='Calibration Curve and Concentration Prediction',
            xaxis_title='Concentration',
            yaxis_title='Absorbance',
            template='plotly_white',
            hovermode='closest',
            height=400
        )
        
        return fig

    @staticmethod
    def create_sensitivity_plot(df: pd.DataFrame, model_data: Dict, calib_range: Dict,
                               target_wavelength: float, window_nm: float,
                               settings: Dict) -> Optional[go.Figure]:
        """Plot concentration vs wavelength in a ±window around target."""
        try:
            wl = df['Nanometers'].values
            ab = df['Absorbance'].values
            wl_min = max(min(wl), target_wavelength - window_nm)
            wl_max = min(max(wl), target_wavelength + window_nm)
            if wl_max <= wl_min:
                return None
            wl_grid = np.linspace(wl_min, wl_max, 60)
            abs_grid = np.interp(wl_grid, wl, ab)
            concs = []
            for a in abs_grid:
                res = ConcentrationPredictor.predict_concentration(model_data, float(a), calib_range, settings)
                concs.append(res['concentration'] if res.get('success', False) else np.nan)
            concs = np.array(concs)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=wl_grid, y=concs, mode='lines', name='C(λ)',
                                     line=dict(color='#1f77b4')))
            fig.add_vline(x=target_wavelength, line_dash='dash', line_color='red',
                          annotation_text=f"Target {target_wavelength:.1f} nm")
            # Calibration band as horizontal band
            if calib_range:
                min_c, max_c = calib_range['min_conc'], calib_range['max_conc']
                fig.add_shape(type='rect', x0=wl_min, x1=wl_max, y0=min_c, y1=max_c,
                              fillcolor='rgba(0,200,0,0.08)', line=dict(color='rgba(0,0,0,0)'), layer='below')
            fig.update_layout(
                title='Sensitivity: Predicted Concentration vs Wavelength',
                xaxis_title='Wavelength (nm)', yaxis_title='Predicted Concentration',
                template='plotly_white', height=350
            )
            return fig
        except Exception:
            return None

    @staticmethod
    def create_overview_plot(model_data: Dict, predictions: List[Dict], calib_range: Dict) -> Optional[go.Figure]:
        """Overlay all predictions on the calibration curve."""
        try:
            # Build a synthetic absorbance range from calib_range
            absorbance_range = (calib_range['min_abs'], calib_range['max_abs']) if calib_range else None
            base = PlotManager.create_prediction_plot(model_data, {'success': False}, calib_range, absorbance_range, True, False)
            # Add all points
            xs, ys, texts = [], [], []
            for p in predictions:
                pred = p['prediction']
                if pred.get('success', False):
                    xs.append(pred['concentration'])
                    ys.append(pred['absorbance_input'])
                    texts.append(p['filename'])
            if xs:
                base.add_trace(go.Scatter(x=xs, y=ys, mode='markers', name='All Predictions',
                                          marker=dict(size=10, color='#d62728', symbol='diamond-open'),
                                          text=texts, hovertemplate='%{text}<br>C: %{x:.4f}<br>A: %{y:.4f}<extra></extra>'))
            base.update_layout(title='Overview: All Predictions on Calibration Curve', height=420)
            return base
        except Exception:
            return None

# --- Main Application ---
def main():
    """Main application for concentration prediction."""
    
    # Header
    st.title("Concentration Prediction & Model Deployment")
    with st.container():
        st.markdown(
            """
            <div class="card">
              <div style="display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;">
                <div>
                  <div class="muted">Spectroscopic ML Platform</div>
                  <div style="font-weight:600;margin-top:2px;">Upload trained models and predict concentrations from spectra.</div>
                </div>
                <div class="muted">Supported: MLP, PLSR, Random Forest, SVR, XGBoost, 1D-CNN</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # Sidebar configuration
    setup_sidebar()
    
    # Main content in tabs
    tab1, tab2, tab3 = st.tabs(["📋 Model & Spectrum", "🔮 Predictions", "📊 Batch Analysis"])
    
    with tab1:
        model_spectrum_tab()
    
    with tab2:
        prediction_tab()
    
    with tab3:
        batch_analysis_tab()
    
    # Use target wavelength from modeling session if available
    if not hasattr(st.session_state, 'target_wavelength'):
        st.session_state.target_wavelength = 400.0  # Default wavelength

def setup_sidebar():
    """Setup sidebar with prediction settings."""
    with st.sidebar:
        st.header("⚙️ Prediction Settings")
        # (Extrapolation settings removed – full-spectrum models predict directly)
        
        with st.expander("📊 Display Options", expanded=False):
            show_confidence = st.checkbox(
                "Show Confidence Estimates", True,
                key="show_confidence"
            )
            
            show_model_info = st.checkbox(
                "Show Model Information", True,
                key="show_model_info"
            )
            
            auto_refresh = st.checkbox(
                "Auto-refresh Predictions", False,
                key="auto_refresh"
            )

            st.markdown("---")
            st.markdown("**Visualization Enhancements**")
            st.session_state.setdefault('show_calibration_band', True)
            st.session_state.setdefault('show_uncertainty', True)
            st.checkbox(
                "Show Calibration Range Band", value=st.session_state.show_calibration_band,
                key="show_calibration_band"
            )
            st.checkbox(
                "Show Uncertainty Bar (uses RMSE)", value=st.session_state.show_uncertainty,
                key="show_uncertainty"
            )

def model_spectrum_tab():
    """Tab for loading model and spectrum files."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Load Model(s)")
        model_files = st.file_uploader(
            "Upload Model File(s) (.pkl)",
            type=["pkl"],
            accept_multiple_files=True,
            help="Upload one or more .pkl models (MLP/PLSR/RF/SVR/XGB/CNN). All models are now serialized as .pkl files.",
            key="model_uploader"
        )
        
        def _ensure_loaded_model():
            if 'loaded_model' not in st.session_state or not st.session_state.loaded_model:
                st.session_state.loaded_model = {
                    'models': {},
                    'best_model': None,
                    'min_conc': 0.0,
                    'max_conc': 1.0,
                    'min_abs': 0.0,
                    'max_abs': 1.0
                }
        
        if model_files:
            with st.spinner("Loading model(s)..."):
                _ensure_loaded_model()
                added = 0
                for mf in model_files:
                    content = mf.read()
                    filename = mf.name
                    model_data = None
                    if filename.lower().endswith('.pkl'):
                        # Try loading as general pickle model first, then as CNN-specific
                        model_data = ModelLoader.load_pickle_model(content, filename)
                        if not model_data:
                            # Try as CNN-specific model
                            model_data = ModelLoader.load_cnn_pkl_model(content, None, filename, None)
                    mf.seek(0)
                    if not model_data:
                        continue
                    # Merge models
                    for key, model in model_data['models'].items():
                        st.session_state.loaded_model['models'][key] = model
                        if not st.session_state.loaded_model['best_model']:
                            st.session_state.loaded_model['best_model'] = key
                        added += 1
                    # Merge ranges if provided
                    for rng_key in ('min_conc','max_conc','min_abs','max_abs'):
                        if rng_key in model_data:
                            st.session_state.loaded_model[rng_key] = model_data[rng_key]
                if added:
                    types = {m.get('type','unknown') for m in st.session_state.loaded_model['models'].values()}
                    st.markdown(
                        f"""
                        <div class="card" style="margin-top:8px;">
                          <b>Loaded {added} model(s)</b> · Types: {', '.join(sorted(types))} · Total: {len(st.session_state.loaded_model['models'])}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.session_state.get('show_model_info', True):
                        display_model_info(st.session_state.loaded_model)
                    
                    # Model manager: remove one-by-one
                    with st.expander("Manage Loaded Models", expanded=False):
                        all_keys = list(st.session_state.loaded_model['models'].keys())
                        if not all_keys:
                            st.caption("No models loaded.")
                        else:
                            sel = st.multiselect("Select models to remove", options=all_keys, default=[])
                            col_rm1, col_rm2 = st.columns([1,1])
                            with col_rm1:
                                if st.button("Remove Selected", key="remove_selected_models"):
                                    for key in sel:
                                        st.session_state.loaded_model['models'].pop(key, None)
                                    # reset best_model if needed
                                    if st.session_state.loaded_model.get('best_model') not in st.session_state.loaded_model['models']:
                                        st.session_state.loaded_model['best_model'] = next(iter(st.session_state.loaded_model['models']), None)
                                    st.success(f"Removed {len(sel)} model(s)")
                                    st.rerun()
                            with col_rm2:
                                if st.button("Remove All Models", key="remove_all_models"):
                                    st.session_state.loaded_model['models'].clear()
                                    st.session_state.loaded_model['best_model'] = None
                                    st.success("All models removed")
                                    st.rerun()
    
    with col2:
        st.subheader("📈 Load Spectrum")
        
        # Show info about filename auto-detection
        with st.expander("💡 Filename Auto-Detection", expanded=False):
            st.markdown("""
            Expected concentrations can be automatically extracted from filenames. Recognized patterns:
            - **With units**: `0.1mL.csv`, `2.5ppm.csv`, `10ppb.csv`, `1.5mg_L.csv`, `3mg/L.csv`
            - **Element names**: `Cu_0.1ppm.csv`, `Pb_2.5mg_L.csv`, `Zn-1.0-ppb.csv`
            - **Underscore/dash separated**: `sample_0.5_mM.csv`, `test-1.2-ppm.csv`
            - **Concentration prefix**: `concentration_0.5.csv`, `conc-1.2.csv`
            - **Sample prefix**: `sample_0.1.csv`, `sample-2.5.csv`
            - **Just numbers**: `0.1.csv`, `2.5.csv`
            
            Supported units: mL, ppm, ppb, mg, mg/L, μg, mM, μM, nM, M, g, g/L, L
            """)
        
        spectrum_files = st.file_uploader(
            "Upload Spectrum File(s) (CSV)",
            type=["csv"],
            accept_multiple_files=True,
            help="Upload absorbance spectrum CSV files for prediction",
            key="spectrum_uploader"
        )
        
        if spectrum_files and st.session_state.loaded_model:
            process_spectra(spectrum_files)

def display_model_info(model_data: Dict):
    """Display information about the loaded model."""
    st.markdown("**Model Information:**")
    
    models = model_data['models']
    best_model = model_data['best_model']
    
    # Create summary table
    summary_data = []
    for key, model in models.items():
        model_type = (model.get('type') or model.get('model_type') or '').lower()
        if model_type not in ('polynomial', 'neural_network', 'plsr', 'random_forest', 'svr', 'xgboost', 'cnn'):
            # Heuristic fallback: inspect model object/class name
            cls_name = type(model.get('model', object()) ).__name__.lower()
            if 'plsr' in cls_name or 'pls' in cls_name:
                model_type = 'plsr'
            elif 'svr' in cls_name or 'svm' in cls_name:
                model_type = 'svr'
            elif 'forest' in cls_name:
                model_type = 'random_forest'
            elif 'xgb' in cls_name or 'xgboost' in cls_name:
                model_type = 'xgboost'
            elif 'mlp' in cls_name or 'neural' in cls_name:
                model_type = 'neural_network'
            elif 'cnn' in cls_name:
                model_type = 'cnn'
            else:
                model_type = 'neural_network'
        
        if model_type == 'polynomial':
            degree = model.get('degree', 'N/A')
            if degree == 1:
                model_name = "Linear Regression"
            else:
                model_name = f"Polynomial Degree {degree}"
        elif model_type == 'neural_network':
            model_name = f"MLP ({model.get('name', key)})"
        elif model_type == 'plsr':
            n_comp = model.get('n_components', 1)
            model_name = f"PLSR ({n_comp} component{'s' if n_comp > 1 else ''})"
        elif model_type == 'random_forest':
            model_name = f"Random Forest ({model.get('name', key)})"
        elif model_type == 'svr':
            model_name = f"SVR ({model.get('name', key)})"
        else:
            model_name = str(key)
        
        summary_data.append({
            'Model': model_name,
            'Type': ('Polynomial' if model_type=='polynomial' else
                     'MLP' if model_type=='neural_network' else
                     'PLSR' if model_type=='plsr' else
                     'Random Forest' if model_type=='random_forest' else
                     'SVR' if model_type=='svr' else
                     'XGBoost' if model_type=='xgboost' else
                     '1D-CNN' if model_type=='cnn' else model_type.title()),
            'R²': model.get('r_squared', 'N/A'),
            'RMSE': model.get('rmse', 'N/A'),
            'AIC': model.get('aic', 'N/A'),
            'Best': '✅' if key == best_model else ''
        })
    
    summary_df = pd.DataFrame(summary_data)
    if summary_df.empty:
        st.warning("No model metadata available to display.")
        # Show available keys for debugging visibility
        try:
            st.caption(f"Loaded models: {', '.join(models.keys())}")
        except Exception:
            pass
    else:
        st.dataframe(summary_df, width='stretch', hide_index=True)
    
    # Show best model equation
    if best_model in models:
        best_model_data = models[best_model]
        model_type = best_model_data.get('type', 'polynomial')
        
        if model_type == 'polynomial':
            degree = best_model_data.get('degree', 'N/A')
            if degree == 1:
                st.markdown("**Best Model (Linear Regression):**")
            else:
                st.markdown(f"**Best Model (Polynomial Degree {degree}):**")
            eq = best_model_data.get('equation')
            if eq:
                st.code(eq)
            else:
                st.caption("No equation available")
        elif model_type == 'neural_network':
            st.markdown(f"**Best Model (MLP {best_model_data.get('name', best_model)}):**")
            eq = best_model_data.get('equation')
            if eq:
                st.code(eq)
            else:
                st.caption("No equation available")
        elif model_type == 'plsr':
            n_comp = best_model_data.get('n_components', 1)
            st.markdown(f"**Best Model (PLSR - {n_comp} component{'s' if n_comp > 1 else ''}):**")
            eq = best_model_data.get('equation')
            if eq:
                st.code(eq)
            else:
                st.caption("No equation available")
        elif model_type == 'random_forest':
            st.markdown(f"**Best Model (Random Forest {best_model_data.get('name', best_model)}):**")
            eq = best_model_data.get('equation')
            if eq:
                st.code(eq)
            else:
                st.caption("No equation available")
        elif model_type == 'svr':
            st.markdown(f"**Best Model (SVR {best_model_data.get('name', best_model)}):**")
            eq = best_model_data.get('equation')
            if eq:
                st.code(eq)
            else:
                st.caption("No equation available")
        elif model_type == 'xgboost':
            st.markdown(f"**Best Model (XGBoost {best_model_data.get('name', best_model)}):**")
            eq = best_model_data.get('equation')
            if eq:
                st.code(eq)
            else:
                st.caption("No equation available")
        elif model_type == 'cnn':
            st.markdown(f"**Best Model (1D-CNN {best_model_data.get('name', best_model)}):**")
            st.caption("Model loaded for prediction (no text equation)")

def process_spectra(spectrum_files):
    """Process uploaded spectrum files."""
    results = []
    auto_detected_count = 0
    
    for i, file in enumerate(spectrum_files):
        with st.spinner(f"Processing {file.name}..."):
            content = file.read()
            file.seek(0)
            
            df = SpectrumProcessor.read_spectrum_file(content, file.name)
            
            if df is not None:
                # Extract expected concentration from filename
                expected_concentration = extract_concentration_from_filename(file.name)
                if expected_concentration is not None:
                    auto_detected_count += 1
                
                results.append({
                    'filename': file.name,
                    'dataframe': df,
                    'expected_concentration': expected_concentration
                })
    
    if results:
        st.session_state.spectrum_results = results
        st.caption(f"✅ Processed {len(results)} spectrum file(s)")
        if auto_detected_count > 0:
            st.success(f"✨ Auto-detected {auto_detected_count} expected concentration{'s' if auto_detected_count > 1 else ''} from filename{'s' if auto_detected_count > 1 else ''}")

def prediction_tab():
    """Tab for displaying predictions."""
    if not st.session_state.loaded_model:
        st.info("📁 Please load a model file in the 'Model & Spectrum' tab first.")
        return
    
    if not hasattr(st.session_state, 'spectrum_results'):
        st.info("📈 Please load spectrum file(s) in the 'Model & Spectrum' tab first.")
        return
    
    st.subheader("🔮 Concentration Predictions")
    
    # Model selection
    models = st.session_state.loaded_model['models']
    
    # Create model options based on model type
    model_options = []
    model_keys = []
    
    for key, model in models.items():
        model_type = model.get('type', 'polynomial')
        r_squared = model.get('r_squared', 'N/A')
        
        if model_type == 'polynomial':
            degree = model.get('degree', 'N/A')
            if degree == 1:
                option_text = f"Linear Regression (R²={r_squared})"
            else:
                option_text = f"Polynomial Degree {degree} (R²={r_squared})"
        elif model_type == 'neural_network':
            name = model.get('name', key)
            option_text = f"MLP {name} (R²={r_squared})"
        elif model_type == 'plsr':
            n_comp = model.get('n_components', 1)
            option_text = f"PLSR ({n_comp} component{'s' if n_comp > 1 else ''}) (R²={r_squared})"
        elif model_type == 'random_forest':
            name = model.get('name', key)
            option_text = f"Random Forest {name} (R²={r_squared})"
        elif model_type == 'svr':
            name = model.get('name', key)
            kernel = model.get('kernel', 'rbf')
            option_text = f"SVR {name} ({kernel} kernel) (R²={r_squared})"
        elif model_type == 'xgboost':
            name = model.get('name', key)
            option_text = f"XGBoost {name} (R²={r_squared})"
        elif model_type == 'cnn':
            name = model.get('name', key)
            option_text = f"1D-CNN {name} (R²={r_squared})"
        else:
            option_text = f"Model {key} (R²={r_squared})"
        
        model_options.append(option_text)
        model_keys.append(key)
    
    # Default selection (best model)
    best_model_key = st.session_state.loaded_model['best_model']
    try:
        default_index = model_keys.index(best_model_key)
    except ValueError:
        default_index = 0
    
    selected_model_str = st.selectbox(
        "Select Model for Prediction",
        options=model_options,
        index=default_index
    )
    
    selected_model_key = model_keys[model_options.index(selected_model_str)]
    selected_model = models[selected_model_key]
    calib_range = {
        'min_conc': st.session_state.loaded_model['min_conc'],
        'max_conc': st.session_state.loaded_model['max_conc'],
        'min_abs': st.session_state.loaded_model['min_abs'],
        'max_abs': st.session_state.loaded_model['max_abs']
    }
    
    # Process predictions (full-spectrum based)
    predictions = []
    
    for result in st.session_state.spectrum_results:
        filename = result['filename']
        df = result['dataframe']
        expected_concentration = result.get('expected_concentration')
        
        # Predict directly from full spectrum if model supports it; otherwise fallback to polynomial using target wavelength
        prediction = ConcentrationPredictor.predict_from_spectrum(selected_model, df, calib_range, None)
        
        predictions.append({
            'filename': filename,
            'prediction': prediction,
            'dataframe': df,
            'expected_concentration': expected_concentration
        })
    
    # Display results
    display_predictions(predictions, selected_model)

def display_predictions(predictions: List[Dict], model_data: Dict):
    """Display prediction results with visualizations."""
    if not predictions:
        st.warning("No valid predictions to display.")
        return
    
    # Summary table
    st.markdown("#### 📊 Prediction Summary")
    
    summary_data = []
    has_expected_values = any(pred.get('expected_concentration') is not None for pred in predictions)
    
    for pred in predictions:
        filename = pred['filename']
        prediction = pred['prediction']
        expected_concentration = pred.get('expected_concentration')
        
        if prediction.get('success', False):
            # Format concentration
            conc_str = f"{prediction['concentration']:.4f}"
            
            # Calculate actual error if expected concentration is available
            if expected_concentration is not None:
                actual_error = abs(prediction['concentration'] - expected_concentration)
                percent_error = (actual_error / expected_concentration * 100) if expected_concentration != 0 else None
                expected_str = f"{expected_concentration:.4f}"
                error_str = f"{actual_error:.4f}"
                if percent_error is not None:
                    error_str += f" ({percent_error:.1f}%)"
            else:
                expected_str = "N/A"
                error_str = "N/A"
            
            row_data = {
                'Filename': filename,
                'Predicted Concentration': conc_str,
                'Confidence': prediction.get('confidence', 'N/A'),
                'Method': prediction.get('solution_method', 'N/A')
            }
            
            if has_expected_values:
                row_data['Expected Concentration'] = expected_str
                row_data['Actual Error'] = error_str
            
            summary_data.append(row_data)
        else:
            row_data = {
                'Filename': filename,
                'Predicted Concentration': prediction.get('error', 'Unknown error'),
                'Confidence': 'N/A',
                'Method': 'N/A'
            }
            
            if has_expected_values:
                row_data['Expected Concentration'] = f"{expected_concentration:.4f}" if expected_concentration is not None else "N/A"
                row_data['Actual Error'] = 'N/A'
            
            summary_data.append(row_data)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Show info about auto-detected values
    if has_expected_values:
        st.info("✨ indicates expected concentration was auto-detected from filename")
    
    st.dataframe(summary_df, width='stretch', hide_index=True)
    
    # Export predictions
    if summary_data:
        csv_data = summary_df.to_csv(index=False)
        st.download_button(
            "📥 Download Predictions",
            data=csv_data,
            file_name="concentration_predictions.csv",
            mime="text/csv"
        )
    
    # Validation metrics (using expected concentrations when available)
    y_true = []
    y_pred = []
    for pred in predictions:
        exp_c = pred.get('expected_concentration')
        res = pred.get('prediction', {})
        if exp_c is not None and res.get('success', False):
            y_true.append(float(exp_c))
            y_pred.append(float(res['concentration']))
    if len(y_true) >= 2:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        rpd = float(np.std(y_true, ddof=1) / rmse) if rmse > 0 else np.nan
        colm1, colm2, colm3, colm4 = st.columns(4)
        with colm1:
            st.metric("R²", f"{r2:.4f}")
        with colm2:
            st.metric("MAE", f"{mae:.6f}")
        with colm3:
            st.metric("RMSE", f"{rmse:.6f}")
        with colm4:
            st.metric("RPD", f"{rpd:.4f}")
        # Actual vs Predicted plot
        fig_ap = go.Figure()
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        fig_ap.add_trace(go.Scatter(x=[lo - pad, hi + pad], y=[lo - pad, hi + pad],
                                    mode='lines', name='Ideal (y=x)',
                                    line=dict(color='gray', dash='dash')))
        fig_ap.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', name='Samples',
                                    marker=dict(size=9, color='#1f77b4'),
                                    hovertemplate='Actual: %{x:.6f}<br>Predicted: %{y:.6f}<extra></extra>'))
        fig_ap.update_layout(title='Actual vs Predicted Concentration',
                             xaxis_title='Actual', yaxis_title='Predicted',
                             template='plotly_white', height=420)
        st.plotly_chart(fig_ap, use_container_width=True)

def batch_analysis_tab():
    """Tab for batch analysis and comparison."""
    st.subheader("📊 Batch Analysis")
    
    if not st.session_state.loaded_model:
        st.info("📁 Please load a model file first.")
        return
    
    if not hasattr(st.session_state, 'spectrum_results') or not st.session_state.spectrum_results:
        st.info("📈 Please load spectrum file(s) first.")
        return
    
    # Batch prediction with all models
    st.markdown("#### 🔄 Compare All Models")
    
    if st.button("Run Batch Prediction", type="primary"):
        run_batch_prediction()
    
    # Display batch results if available
    if hasattr(st.session_state, 'batch_results'):
        display_batch_results()

def run_batch_prediction():
    """Run prediction with all available models."""
    models = st.session_state.loaded_model['models']
    spectrum_results = st.session_state.spectrum_results
    calib_range = {
        'min_conc': st.session_state.loaded_model['min_conc'],
        'max_conc': st.session_state.loaded_model['max_conc'],
        'min_abs': st.session_state.loaded_model['min_abs'],
        'max_abs': st.session_state.loaded_model['max_abs']
    }
    
    batch_results = {}
    
    progress_bar = st.progress(0, text="Running batch predictions...")
    total_combinations = len(models) * len(spectrum_results)
    current = 0
    
    for model_key, model_data in models.items():
        batch_results[model_key] = []
        
        for result in spectrum_results:
            filename = result['filename']
            df = result['dataframe']
            expected_concentration = result.get('expected_concentration')
            
            # Predict using full spectrum
            prediction = ConcentrationPredictor.predict_from_spectrum(model_data, df, calib_range, None)
            
            batch_results[model_key].append({
                'filename': filename,
                'prediction': prediction,
                'expected_concentration': expected_concentration
            })
            
            current += 1
            progress_bar.progress(current / total_combinations)
    
    progress_bar.empty()
    st.session_state.batch_results = batch_results
    st.caption("✅ Batch prediction completed!")

def display_batch_results():
    """Display batch prediction results."""
    batch_results = st.session_state.batch_results
    
    # Check if any files have expected concentrations
    has_expected_values = any(r.get('expected_concentration') is not None for r in st.session_state.spectrum_results)
    
    # Create comparison table
    comparison_data = []
    
    for result in st.session_state.spectrum_results:
        filename = result['filename']
        expected_concentration = result.get('expected_concentration')
        
        row = {'Filename': filename}
        
        # Add expected concentration if available
        if has_expected_values:
            if expected_concentration is not None:
                row['Expected Conc.'] = f"{expected_concentration:.4f}"
            else:
                row['Expected Conc.'] = "N/A"
        
        for model_key in sorted(batch_results.keys()):
            # Find prediction for this file and model
            model_results = batch_results[model_key]
            file_result = next((r for r in model_results if r['filename'] == filename), None)
            
            # Get model info for column name
            model_data = st.session_state.loaded_model['models'][model_key]
            model_type = model_data.get('type', 'polynomial')
            
            if model_type == 'polynomial':
                degree = model_data.get('degree', 'N/A')
                if degree == 1:
                    col_name = "Linear Regression"
                else:
                    col_name = f"Polynomial Deg {degree}"
            elif model_type == 'neural_network':
                col_name = f"MLP ({model_data.get('name', model_key)})"
            elif model_type == 'plsr':
                n_comp = model_data.get('n_components', 1)
                col_name = f"PLSR ({n_comp} comp)"
            elif model_type == 'random_forest':
                col_name = f"Random Forest ({model_data.get('name', model_key)})"
            elif model_type == 'svr':
                col_name = f"SVR ({model_data.get('name', model_key)})"
            elif model_type == 'xgboost':
                col_name = f"XGBoost ({model_data.get('name', model_key)})"
            elif model_type == 'cnn':
                col_name = f"1D-CNN ({model_data.get('name', model_key)})"
            else:
                col_name = str(model_key)
            
            if file_result and file_result['prediction'].get('success', False):
                concentration = file_result['prediction']['concentration']
                
                # Calculate error if expected concentration is available
                if expected_concentration is not None:
                    error = abs(concentration - expected_concentration)
                    percent_error = (error / expected_concentration * 100) if expected_concentration != 0 else None
                    if percent_error is not None:
                        row[col_name] = f"{concentration:.4f} ({percent_error:.1f}%)"
                    else:
                        row[col_name] = f"{concentration:.4f}"
                else:
                    row[col_name] = f"{concentration:.4f}"
            else:
                row[col_name] = "Error"
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    st.markdown("#### 📊 Model Comparison")
    
    # Show info about auto-detected values
    if has_expected_values:
        st.info("✨ indicates expected concentration was auto-detected from filename | Percentages show error relative to expected value")
    
    st.dataframe(comparison_df, width='stretch', hide_index=True)
    
    # Export batch results
    csv_data = comparison_df.to_csv(index=False)
    st.download_button(
        "📥 Download Batch Results",
        data=csv_data,
        file_name="batch_concentration_predictions.csv",
        mime="text/csv"
    )
    
    # Model metrics comparison (R², MAE, RMSE, RPD)
    metrics_rows = []
    for model_key in sorted(batch_results.keys()):
        model_results = batch_results[model_key]
        y_true, y_pred = [], []
        for r in model_results:
            exp_c = r.get('expected_concentration')
            pred = r.get('prediction', {})
            if exp_c is not None and pred.get('success', False):
                y_true.append(float(exp_c))
                y_pred.append(float(pred['concentration']))
        if len(y_true) >= 2:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            mae = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            rpd = float(np.std(y_true, ddof=1) / rmse) if rmse > 0 else np.nan
        else:
            r2 = mae = rmse = rpd = np.nan
        # human-readable model name
        model_data = st.session_state.loaded_model['models'][model_key]
        mtype = model_data.get('type', 'model')
        if mtype == 'polynomial':
            degree = model_data.get('degree', 'N/A')
            name = 'Linear Regression' if degree == 1 else f'Polynomial Deg {degree}'
        elif mtype == 'neural_network':
            name = f"MLP ({model_data.get('name', model_key)})"
        elif mtype == 'plsr':
            n_comp = model_data.get('n_components', 1)
            name = f"PLSR ({n_comp} comp)"
        elif mtype == 'random_forest':
            name = f"Random Forest ({model_data.get('name', model_key)})"
        elif mtype == 'svr':
            name = f"SVR ({model_data.get('name', model_key)})"
        elif mtype == 'xgboost':
            name = f"XGBoost ({model_data.get('name', model_key)})"
        elif mtype == 'cnn':
            name = f"1D-CNN ({model_data.get('name', model_key)})"
        else:
            name = str(model_key)
        metrics_rows.append({
            'Model': name,
            'R²': r2,
            'MAE': mae,
            'RMSE': rmse,
            'RPD': rpd,
            'N': len(y_true) if isinstance(y_true, np.ndarray) else 0
        })
    metrics_df = pd.DataFrame(metrics_rows)
    st.markdown("#### 📏 Model Metrics Comparison")
    if not metrics_df.empty:
        st.dataframe(metrics_df, width='stretch', hide_index=True)
        # Quick bar charts for R² and RMSE
        try:
            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df['R²'], marker_color='#2ca02c'))
            fig_r2.update_layout(title='R² by Model', xaxis_title='Model', yaxis_title='R²', template='plotly_white', height=360)
            st.plotly_chart(fig_r2, use_container_width=True)
            fig_rmse = go.Figure()
            fig_rmse.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df['RMSE'], marker_color='#d62728'))
            fig_rmse.update_layout(title='RMSE by Model', xaxis_title='Model', yaxis_title='RMSE', template='plotly_white', height=360)
            st.plotly_chart(fig_rmse, use_container_width=True)
        except Exception:
            pass
        # Download metrics
        st.download_button(
            "📥 Download Model Metrics",
            data=metrics_df.to_csv(index=False),
            file_name="model_metrics_comparison.csv",
            mime="text/csv"
        )
    else:
        st.info("Provide expected concentrations (via filenames) to compute metrics.")
    
    # Visualization of prediction differences
    if len(comparison_df) > 1:
        create_batch_comparison_plot(comparison_df)

def create_batch_comparison_plot(comparison_df: pd.DataFrame):
    """Create plot comparing predictions across models."""
    st.markdown("#### 📈 Prediction Comparison")
    
    # Get numeric columns (exclude filename)
    model_cols = [col for col in comparison_df.columns if col != 'Filename']
    
    # Create scatter plot
    fig = go.Figure()
    
    for i, col in enumerate(model_cols):
        # Extract numeric values (skip "Error" entries)
        numeric_data = []
        filenames = []
        
        for _, row in comparison_df.iterrows():
            if row[col] != "Error":
                try:
                    numeric_data.append(float(row[col]))
                    filenames.append(row['Filename'])
                except ValueError:
                    continue
        
        if numeric_data:
            fig.add_trace(go.Scatter(
                x=list(range(len(numeric_data))),
                y=numeric_data,
                mode='markers+lines',
                name=col,
                text=filenames,
                hovertemplate=f'{col}<br>%{{text}}<br>Concentration: %{{y:.4f}}<extra></extra>'
            ))
    
    fig.update_layout(
        title='Concentration Predictions by Model',
        xaxis_title='Sample Index',
        yaxis_title='Predicted Concentration',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 
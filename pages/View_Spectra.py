"""
Spectrum Viewer and Overlay Application

Advanced spectrum visualization tool with overlay capabilities, peak detection,
and interactive range controls for spectroscopic data analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import io
import time
from typing import List, Dict, Tuple, Optional, Any
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Spectrum Viewer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DEFAULT_WAVELENGTH_MIN = 350
DEFAULT_WAVELENGTH_MAX = 800
MAX_FILES = 10
MAX_FILE_SIZE_MB = 50
SUPPORTED_EXTENSIONS = ["txt", "csv", "dat"]

# Color palette for multiple spectra
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        'uploaded_spectra': [],
        'spectrum_names': [],
        'wavelength_min': DEFAULT_WAVELENGTH_MIN,
        'wavelength_max': DEFAULT_WAVELENGTH_MAX,
        'show_peaks': True,
        'peak_height': 0.01,
        'peak_distance': 5.0,
        'peak_prominence': 0.01,
        'smooth_spectra': False,
        'smoothing_sigma': 1.0,
        'normalize_spectra': False,
        'selected_spectra': [],
        'plot_height': 600
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def parse_spectrum_file(file_content: bytes, filename: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Parse spectrum file and return wavelength and intensity arrays."""
    try:
        content = file_content.decode('utf-8')
    except UnicodeDecodeError:
        content = file_content.decode('latin-1')

    # Try with header first
    try:
        df = pd.read_csv(io.StringIO(content))
        if df.shape[1] >= 2:
            # Check for wavelength/intensity columns
            cols = [str(c).lower() for c in df.columns]
            wl_col = None
            int_col = None
            
            for i, col in enumerate(cols):
                if any(x in col for x in ['wavelength', 'wl', 'lambda', 'nm']):
                    wl_col = df.columns[i]
                elif any(x in col for x in ['counts', 'intensity', 'absorbance', 'signal']):
                    int_col = df.columns[i]
            
            if wl_col and int_col:
                wl = pd.to_numeric(df[wl_col], errors='coerce').dropna()
                intensity = pd.to_numeric(df[int_col], errors='coerce').dropna()
                if len(wl) > 0 and len(intensity) > 0:
                    return wl.values, intensity.values
    except:
        pass

    # Try without header
    for sep in [',', '\t', ';', ' ']:
        try:
            df = pd.read_csv(io.StringIO(content), sep=sep, header=None)
            if df.shape[1] >= 2:
                wl = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
                intensity = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna()
                if len(wl) > 0 and len(intensity) > 0:
                    return wl.values, intensity.values
        except:
            continue

    st.error(f"Could not parse {filename}")
    return None

def detect_peaks(intensity: np.ndarray, wavelength: np.ndarray, 
                height: float, distance: float, prominence: float) -> Tuple[np.ndarray, np.ndarray]:
    """Detect peaks in the spectrum."""
    try:
        peaks, properties = find_peaks(
            intensity,
            height=height,
            distance=distance,
            prominence=prominence
        )
        
        peak_wavelengths = wavelength[peaks]
        peak_intensities = intensity[peaks]
        
        return peak_wavelengths, peak_intensities
    except Exception as e:
        st.warning(f"Peak detection failed: {str(e)}")
        return np.array([]), np.array([])

def smooth_spectrum(intensity: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing to spectrum."""
    try:
        return gaussian_filter1d(intensity, sigma=sigma)
    except Exception as e:
        st.warning(f"Smoothing failed: {str(e)}")
        return intensity

def normalize_spectrum(intensity: np.ndarray) -> np.ndarray:
    """Normalize spectrum to 0-1 range."""
    try:
        min_val = np.min(intensity)
        max_val = np.max(intensity)
        if max_val - min_val > 0:
            return (intensity - min_val) / (max_val - min_val)
        else:
            return intensity
    except Exception as e:
        st.warning(f"Normalization failed: {str(e)}")
        return intensity

def create_overlay_plot(spectra_data: List[Dict], wavelength_min: float, wavelength_max: float,
                       show_peaks: bool, peak_params: Dict, normalize: bool, smooth: bool, 
                       smoothing_sigma: float, selected_spectra: List[int]) -> go.Figure:
    """Create interactive overlay plot of multiple spectra."""
    
    fig = go.Figure()
    
    # Filter spectra based on wavelength range
    filtered_spectra = []
    for i, spectrum in enumerate(spectra_data):
        if i in selected_spectra or len(selected_spectra) == 0:
            wavelength, intensity = spectrum['wavelength'], spectrum['intensity']
            
            # Apply wavelength filter
            mask = (wavelength >= wavelength_min) & (wavelength <= wavelength_max)
            wl_filtered = wavelength[mask]
            int_filtered = intensity[mask]
            
            if len(wl_filtered) > 0:
                # Apply smoothing if requested
                if smooth:
                    int_filtered = smooth_spectrum(int_filtered, smoothing_sigma)
                
                # Apply normalization if requested
                if normalize:
                    int_filtered = normalize_spectrum(int_filtered)
                
                filtered_spectra.append({
                    'wavelength': wl_filtered,
                    'intensity': int_filtered,
                    'name': spectrum['name'],
                    'color': spectrum['color']
                })
    
    # Add spectrum traces
    for spectrum in filtered_spectra:
        fig.add_trace(go.Scatter(
            x=spectrum['wavelength'],
            y=spectrum['intensity'],
            mode='lines',
            name=spectrum['name'],
            line=dict(color=spectrum['color'], width=2),
            hovertemplate=f"<b>{spectrum['name']}</b><br>" +
                         "Wavelength: %{x:.2f} nm<br>" +
                         "Intensity: %{y:.4f}<br>" +
                         "<extra></extra>"
        ))
        
        # Add peaks if requested
        if show_peaks:
            peak_wavelengths, peak_intensities = detect_peaks(
                spectrum['intensity'], spectrum['wavelength'],
                peak_params['height'], peak_params['distance'], peak_params['prominence']
            )
            
            if len(peak_wavelengths) > 0:
                # Add peak markers
                fig.add_trace(go.Scatter(
                    x=peak_wavelengths,
                    y=peak_intensities,
                    mode='markers',
                    name=f"{spectrum['name']} Peaks",
                    marker=dict(
                        color=spectrum['color'],
                        size=8,
                        symbol='diamond',
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate=f"<b>{spectrum['name']} Peak</b><br>" +
                                 "Wavelength: %{x:.2f} nm<br>" +
                                 "Intensity: %{y:.4f}<br>" +
                                 "<extra></extra>",
                    showlegend=False
                ))
                
                # Add peak labels
                for i, (wl, intensity) in enumerate(zip(peak_wavelengths, peak_intensities)):
                    fig.add_annotation(
                        x=wl,
                        y=intensity,
                        text=f"{wl:.0f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=spectrum['color'],
                        ax=0,
                        ay=-30,
                        font=dict(size=10, color=spectrum['color']),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor=spectrum['color'],
                        borderwidth=1
                    )
    
    # Update layout
    fig.update_layout(
        title="Spectrum Overlay Viewer",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity",
        height=st.session_state.plot_height,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        margin=dict(r=200)
    )
    
    # Update x-axis
    fig.update_layout(
        xaxis=dict(type="linear")
    )
    
    return fig

# Initialize session state
initialize_session_state()

# Main page
st.title("Spectrum Viewer")

# Sidebar controls
with st.sidebar:
    st.header("File Upload")
    
    uploaded_files = st.file_uploader(
        "Upload spectrum files",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if len(uploaded_files) > MAX_FILES:
            st.warning(f"Too many files! Please upload maximum {MAX_FILES} files.")
            uploaded_files = uploaded_files[:MAX_FILES]
        
        # Process uploaded files
        new_spectra = []
        new_names = []
        
        for file in uploaded_files:
            if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.warning(f"File {file.name} is too large. Maximum size: {MAX_FILE_SIZE_MB}MB")
                continue
            
            spectrum_data = parse_spectrum_file(file.read(), file.name)
            if spectrum_data is not None:
                wavelength, intensity = spectrum_data
                new_spectra.append({
                    'wavelength': wavelength,
                    'intensity': intensity,
                    'name': file.name,
                    'color': COLORS[len(new_spectra) % len(COLORS)]
                })
                new_names.append(file.name)
        
        if new_spectra:
            st.session_state.uploaded_spectra = new_spectra
            st.session_state.spectrum_names = new_names
            st.success(f"Loaded {len(new_spectra)} files")
    
    st.header("Display Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        wavelength_min = st.number_input("Min (nm)", min_value=200.0, max_value=2000.0, 
                                      value=float(st.session_state.wavelength_min), step=10.0, key="wl_min")
    with col2:
        wavelength_max = st.number_input("Max (nm)", min_value=200.0, max_value=2000.0, 
                                      value=float(st.session_state.wavelength_max), step=10.0, key="wl_max")
    
    if wavelength_min >= wavelength_max:
        st.error("Minimum wavelength must be less than maximum wavelength!")
    else:
        st.session_state.wavelength_min = wavelength_min
        st.session_state.wavelength_max = wavelength_max
    
    if st.session_state.spectrum_names:
        selected_indices = st.multiselect("Select spectra", 
                                        options=list(range(len(st.session_state.spectrum_names))),
                                        format_func=lambda x: st.session_state.spectrum_names[x],
                                        default=list(range(len(st.session_state.spectrum_names))))
        st.session_state.selected_spectra = selected_indices
    
    normalize = st.checkbox("Normalize", value=st.session_state.normalize_spectra)
    st.session_state.normalize_spectra = normalize
    
    smooth = st.checkbox("Smooth", value=st.session_state.smooth_spectra)
    st.session_state.smooth_spectra = smooth
    
    if smooth:
        st.session_state.smoothing_sigma = st.slider("Smoothing", 0.1, 5.0, 
                                                    st.session_state.smoothing_sigma, 0.1)
    
    show_peaks = st.checkbox("Show peaks", value=st.session_state.show_peaks)
    st.session_state.show_peaks = show_peaks
    
    if show_peaks:
        st.session_state.peak_height = st.slider("Peak height", 0.001, 0.5, 
                                                st.session_state.peak_height, 0.001)
        st.session_state.peak_distance = st.slider("Peak distance", 1.0, 30.0, 
                                                  st.session_state.peak_distance, 0.5)
        st.session_state.peak_prominence = st.slider("Peak prominence", 0.001, 0.5, 
                                                   st.session_state.peak_prominence, 0.001)
    
    if st.button("Clear", type="secondary"):
        st.session_state.uploaded_spectra = []
        st.session_state.spectrum_names = []
        st.session_state.selected_spectra = []
        st.rerun()

# Main content area
if st.session_state.uploaded_spectra:
    
    # Create and display plot
    peak_params = {
        'height': st.session_state.peak_height,
        'distance': st.session_state.peak_distance,
        'prominence': st.session_state.peak_prominence
    }
    
    fig = create_overlay_plot(
        st.session_state.uploaded_spectra,
        st.session_state.wavelength_min,
        st.session_state.wavelength_max,
        st.session_state.show_peaks,
        peak_params,
        st.session_state.normalize_spectra,
        st.session_state.smooth_spectra,
        st.session_state.smoothing_sigma,
        st.session_state.selected_spectra
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Peak comparison table
    if st.session_state.show_peaks and st.session_state.uploaded_spectra:
        st.markdown("### Peak Comparison")
        
        # Collect all peaks from all spectra
        all_peaks_data = []
        for spectrum in st.session_state.uploaded_spectra:
            peak_wavelengths, peak_intensities = detect_peaks(
                spectrum['intensity'], spectrum['wavelength'],
                peak_params['height'], peak_params['distance'], peak_params['prominence']
            )
            
            for wl, intensity in zip(peak_wavelengths, peak_intensities):
                all_peaks_data.append({
                    'Spectrum': spectrum['name'],
                    'Wavelength (nm)': f"{wl:.1f}",
                    'Intensity': f"{intensity:.4f}"
                })
        
        if all_peaks_data:
            peaks_df = pd.DataFrame(all_peaks_data)
            peaks_df = peaks_df.sort_values('Wavelength (nm)')
            st.dataframe(peaks_df, width='stretch', hide_index=True)
        else:
            st.info("No peaks detected in any spectrum.")

else:
    st.info("Upload spectrum files using the sidebar. Supports CSV, TXT, DAT formats.")

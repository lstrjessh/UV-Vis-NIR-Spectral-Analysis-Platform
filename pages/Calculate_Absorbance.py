import streamlit as st
import numpy as np
import pandas as pd
import io
import time
from typing import Optional, Tuple, List, Dict, Union
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Application Configuration ---
st.set_page_config(
    page_title="Absorbance Calculator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
DEFAULT_SPECTRUM_MIN = 350
DEFAULT_SPECTRUM_MAX = 800
EPSILON = 1e-10
MAX_FILE_SIZE_MB = 50

SUPPORTED_EXTENSIONS = ["txt", "csv", "dat"]

# --- Session State Initialization ---
def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        'analysis_min_wl': float(DEFAULT_SPECTRUM_MIN),
        'analysis_max_wl': float(DEFAULT_SPECTRUM_MAX),
        'peak_source': 'Raw Absorbance',
        'plot_range': None,
        'apply_smoothing': False,
        'sg_window': 11,
        'sg_poly': 2,
        'peak_height': 0.1,
        'peak_distance': 15.0,
        'peak_prominence': 0.05,
        'last_calculation_time': None,
        'processing_cache': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- Enhanced Caching Functions ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def read_spectral_file(file_content: bytes, filename: str) -> Optional[pd.DataFrame]:
    """
    Optimized spectral file reader with enhanced error handling and validation.

    Args:
        file_content: Raw file content as bytes
        filename: Name of the file for error reporting

    Returns:
        DataFrame with validated spectral data or None if invalid
    """
    try:
        # Convert bytes to StringIO for pandas
        content_str = file_content.decode('utf-8')
        content_io = io.StringIO(content_str)
        
        # Try different delimiters
        delimiters = [None, '\t', ' ', ',', ';']
        df = None
        
        for delimiter in delimiters:
            content_io.seek(0)
            try:
                if delimiter is None:
                    df = pd.read_csv(content_io, delim_whitespace=True)
                else:
                    df = pd.read_csv(content_io, delimiter=delimiter)
                
                # Check if we have the required columns
                if len(df.columns) >= 2:
                    # Rename columns to standard format
                    df.columns = ['Nanometers', 'Counts'] + list(df.columns[2:])
                    break
            except Exception:
                continue
        
        if df is None or df.empty:
            st.error(f"Could not parse file '{filename}'. Please check the format.")
            return None

        # Validate and clean data
        df = validate_and_clean_data(df, filename)
        return df

    except UnicodeDecodeError:
        st.error(f"File '{filename}' contains invalid characters. Please ensure it's a text file.")
        return None
    except Exception as e:
        st.error(f"Unexpected error reading '{filename}': {str(e)}")
        return None

def validate_and_clean_data(df: pd.DataFrame, filename: str) -> Optional[pd.DataFrame]:
    """Validate and clean spectral data."""
    try:
        # Ensure we have required columns
        if len(df.columns) < 2:
            st.error(f"File '{filename}' must have at least 2 columns (wavelength, intensity).")
            return None
        
        # Convert to numeric
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        
        # Remove invalid rows
        initial_rows = len(df)
        df = df.dropna(subset=df.columns[:2])
        
        if len(df) == 0:
            st.error(f"File '{filename}' contains no valid numeric data.")
            return None
        
        if len(df) < initial_rows * 0.8:  # If we lost more than 20% of data
            st.warning(f"File '{filename}': Removed {initial_rows - len(df)} invalid rows.")
        
        # Sort by wavelength
        df = df.sort_values(by=df.columns[0]).reset_index(drop=True)
        
        # Rename columns to standard format
        df.columns = ['Nanometers', 'Counts'] + list(df.columns[2:])
        
        return df
        
    except Exception as e:
        st.error(f"Error validating data in '{filename}': {str(e)}")
        return None

@st.cache_data
def wavelength_to_rgb(wavelength: float) -> Tuple[float, float, float]:
    """Optimized wavelength to RGB conversion with smooth transitions."""
    # Clamp wavelength to reasonable range
    wl = max(300, min(850, wavelength))
    
    # Color mapping with smooth transitions
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

@st.cache_data
def get_color_name(wavelength: float) -> str:
    """Get color name for wavelength."""
    if wavelength < 380: return "Near UV"
    elif wavelength < 450: return "Violet"
    elif wavelength < 485: return "Blue"
    elif wavelength < 500: return "Cyan"
    elif wavelength < 565: return "Green"
    elif wavelength < 590: return "Yellow"
    elif wavelength < 625: return "Orange"
    elif wavelength <= 780: return "Red"
    else: return "Near IR"

# --- Enhanced Data Processing ---
class SpectralProcessor:
    """Class to handle spectral data processing with better organization."""
    
    @staticmethod
    def average_dataframes(dataframes: List[pd.DataFrame], file_type: str) -> Optional[pd.DataFrame]:
        """Average multiple spectral dataframes with enhanced validation."""
        if not dataframes:
            return None

        if len(dataframes) == 1:
            return dataframes[0].copy()
        
        # Use first dataframe as reference
        reference = dataframes[0]
        reference_wl = reference['Nanometers'].values
        
        # Validate compatibility
        for i, df in enumerate(dataframes[1:], 1):
            if not np.allclose(df['Nanometers'].values, reference_wl, rtol=1e-3):
                st.error(f"Wavelength mismatch in {file_type} file {i+1}. "
                        "All files must have identical wavelength points.")
                return None

        # Calculate average
        result = reference.copy()
        counts_matrix = np.column_stack([df['Counts'].values for df in dataframes])
        result['Counts'] = np.mean(counts_matrix, axis=1)
        
        return result
    
    @staticmethod
    def calculate_absorbance(df_ref: pd.DataFrame, df_sample: pd.DataFrame, 
                           df_dark: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        """Enhanced absorbance calculation with better error handling."""
        try:
            # Validate inputs
            if df_ref is None or df_sample is None:
                return None

            # Check wavelength consistency
            ref_wl = df_ref['Nanometers'].values
            sample_wl = df_sample['Nanometers'].values
            
            if not np.allclose(ref_wl, sample_wl, rtol=1e-3):
                st.error("Wavelength mismatch between reference and sample data.")
                return None

            # Initialize result dataframe
            result = pd.DataFrame({
                'Nanometers': ref_wl,
                'Reference_Counts': df_ref['Counts'].values,
                'Sample_Counts': df_sample['Counts'].values
            })
            
            # Handle dark correction
            dark_counts = np.zeros_like(ref_wl)
            if df_dark is not None:
                if not np.allclose(df_dark['Nanometers'].values, ref_wl, rtol=1e-3):
                    st.error("Wavelength mismatch with dark spectrum.")
                    return None
                dark_counts = df_dark['Counts'].values
                result['Dark_Counts'] = dark_counts
            
            # Apply dark correction with clipping
            ref_corrected = np.clip(result['Reference_Counts'] - dark_counts, EPSILON, None)
            sample_corrected = np.clip(result['Sample_Counts'] - dark_counts, EPSILON, None)
            
            result['Reference_Corrected'] = ref_corrected
            result['Sample_Corrected'] = sample_corrected
            
            # Calculate absorbance
            with np.errstate(divide='ignore', invalid='ignore'):
                absorbance = np.log10(ref_corrected / sample_corrected)
                absorbance = np.nan_to_num(absorbance, nan=0.0, posinf=5.0, neginf=0.0)
            
            result['Absorbance'] = absorbance
            
            return result
            
        except Exception as e:
            st.error(f"Error calculating absorbance: {str(e)}")
            return None

# --- Enhanced Plotting Functions ---
class PlotGenerator:
    """Class for generating optimized plots."""
    
    @staticmethod
    def create_base_plot(title: str, x_label: str, y_label: str, 
                        x_range: Optional[List[float]] = None, height: int = 500):
        """Create base plot configuration."""
        # Lazy import plotly (only when plotting)
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            xaxis=dict(range=x_range) if x_range else {},
            height=height,
            hovermode='x unified',
            template="plotly_white",
            margin=dict(t=60, l=60, r=20, b=60)
        )
        return fig

    @staticmethod
    def plot_counts_preview(df: pd.DataFrame, title: str, line_color: str = 'blue',
                            x_range: Optional[List[float]] = None, height: int = 250):
        """Create a simple counts vs wavelength plot for waveform previews."""
        import plotly.graph_objects as go
        fig = PlotGenerator.create_base_plot(
            title, 'Wavelength (nm)', 'Counts', x_range, height
        )
        fig.add_trace(go.Scatter(
            x=df['Nanometers'], y=df['Counts'], mode='lines',
            line=dict(color=line_color, width=2), name='Average'
        ))
        return fig

    @staticmethod
    def plot_absorbance_optimized(df: pd.DataFrame,
                                peak_info: Optional[Tuple] = None,
                                smoothed_data: Optional[np.ndarray] = None,
                                x_range: Optional[List[float]] = None):
        """Simplified absorbance plot with color gradient."""
        import plotly.graph_objects as go
        
        fig = PlotGenerator.create_base_plot(
            'Absorbance Spectrum',
            'Wavelength (nm)', 'Absorbance (AU)', x_range
        )
        
        wl = df['Nanometers'].values
        abs_data = df['Absorbance'].values
        
        # Create gradient line by splitting into segments
        segments = min(100, len(wl) - 1)
        for i in range(segments):
            idx = int(i * len(wl) / segments)
            next_idx = int((i + 1) * len(wl) / segments)
            
            # Get segment data and color
            seg_wl = wl[idx:next_idx+1]
            seg_abs = abs_data[idx:next_idx+1]
            rgb = wavelength_to_rgb(seg_wl[len(seg_wl)//2])
            
            fig.add_trace(go.Scatter(
                x=seg_wl, y=seg_abs, mode='lines',
                line=dict(color=f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})', width=3),
                showlegend=False, hovertemplate='%{x:.1f} nm: %{y:.4f} AU<extra></extra>'
            ))
        
        # Add smoothed line if available
        if smoothed_data is not None:
            fig.add_trace(go.Scatter(
                x=wl, y=smoothed_data, mode='lines', name='Smoothed',
                line=dict(color='purple', width=2, dash='dash'), visible='legendonly'
            ))
        
        # Add peak markers and labels
        if peak_info and len(peak_info[0]) > 0:
            peaks = peak_info[0]
            peak_wl = wl[peaks]
            peak_abs = abs_data[peaks]
            
            # Filter peaks by x_range if specified
            if x_range:
                mask = (peak_wl >= x_range[0]) & (peak_wl <= x_range[1])
                peak_wl = peak_wl[mask]
                peak_abs = peak_abs[mask]
            
            # Add peak markers
            fig.add_trace(go.Scatter(
                x=peak_wl, y=peak_abs,
                mode='markers', name=f'Peaks ({len(peak_wl)})',
                marker=dict(size=10, symbol='triangle-down', color='red')
            ))
            
            # Add peak labels - limit to top 10 peaks to avoid clutter
            sorted_indices = np.argsort(peak_abs)[::-1]  # Sort by absorbance descending
            for i in range(min(10, len(peak_wl))):
                idx = sorted_indices[i]
                fig.add_annotation(
                    x=peak_wl[idx],
                    y=peak_abs[idx],
                    text=f"{peak_wl[idx]:.1f} nm",
                    showarrow=True,
                    arrowhead=0,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="black",
                    ax=0,
                    ay=-40,
                    bgcolor="rgba(255, 255, 255, 0.95)",
                    bordercolor="black",
                    borderwidth=2,
                    font=dict(size=12, color="black", family="Arial")
                )
        
        return fig

# --- Enhanced File Processing ---
def process_uploaded_files(uploaded_files: List, file_type: str) -> Optional[pd.DataFrame]:
    """Process uploaded files with progress tracking."""
    if not uploaded_files:
        return None
    
    valid_dfs = []
    progress_bar = st.progress(0)
    
    for i, file in enumerate(uploaded_files):
        try:
            # Check file size
            if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                st.warning(f"File '{file.name}' is too large (>{MAX_FILE_SIZE_MB}MB)")
                continue
            
            # Read file content
            content = file.read()
            file.seek(0)  # Reset for potential re-reading
            
            # Process file
            df = read_spectral_file(content, file.name)
            if df is not None:
                valid_dfs.append(df)
            
            # Update progress
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        except Exception as e:
            st.error(f"Error processing '{file.name}': {str(e)}")
    
    progress_bar.empty()
    
    if not valid_dfs:
        st.error(f"No valid {file_type} files processed.")
        return None
    
    # Average files if multiple
    if len(valid_dfs) > 1:
        st.caption(f"Averaging {len(valid_dfs)} {file_type} files…")
        return SpectralProcessor.average_dataframes(valid_dfs, file_type)
    
    return valid_dfs[0]

# --- Waveform Preview Utilities ---
def _build_preview_dataframe(uploaded_files: List) -> Optional[pd.DataFrame]:
    """Quietly read and average multiple uploaded spectral files for preview."""
    if not uploaded_files:
        return None
    valid_dfs: List[pd.DataFrame] = []
    for file in uploaded_files:
        try:
            # Streamlit uploader provides .read(); ensure pointer reset for other uses
            content = file.read()
            file.seek(0)
            df = read_spectral_file(content, file.name)
            if df is not None:
                valid_dfs.append(df)
        except Exception:
            continue
    if not valid_dfs:
        return None
    if len(valid_dfs) == 1:
        return valid_dfs[0]
    return SpectralProcessor.average_dataframes(valid_dfs, "Preview")

def _render_waveform_preview(uploaded_files: List, label: str, color: str,
                             x_range: Optional[List[float]] = None):
    """Render averaged waveform preview for a set of uploaded files."""
    df_preview = _build_preview_dataframe(uploaded_files)
    if df_preview is None:
        return
    file_count = len(uploaded_files) if uploaded_files else 0
    title = f"{label} Waveform Preview" + (f" • Avg of {file_count} files" if file_count > 1 else "")
    fig = PlotGenerator.plot_counts_preview(df_preview, title, color, x_range, height=250)
    st.plotly_chart(fig, use_container_width=True)

# --- Enhanced Peak Detection ---
def find_absorption_peaks_enhanced(wavelengths: np.ndarray, absorbances: np.ndarray,
                                 min_height: float = 0.1, min_distance_nm: float = 15,
                                 min_prominence: float = 0.05) -> Optional[Tuple]:
    """Enhanced peak detection with better parameter validation."""
    if len(wavelengths) < 3 or len(absorbances) < 3:
        return None
    
    try:
        # Lazy import scipy (only when peak detection is actually used)
        from scipy.signal import find_peaks
        
        # Calculate distance in points
        avg_spacing = np.mean(np.diff(wavelengths))
        distance_points = max(1, int(min_distance_nm / avg_spacing))
        
        # Find peaks
        peaks, properties = find_peaks(
            absorbances,
            height=min_height if min_height > 0 else None,
            distance=distance_points,
            prominence=min_prominence if min_prominence > 0 else None
        )
        
        return peaks, properties
        
    except Exception as e:
        st.error(f"Peak detection error: {str(e)}")
        return None

# --- Utility Functions ---
def create_download_button(df: pd.DataFrame, label: str, filename: str, key: str) -> bool:
    """Create download button with better error handling."""
    try:
        buffer = io.StringIO()
        df.to_csv(buffer, index=False, float_format='%.6g')
        csv_data = buffer.getvalue().encode('utf-8')
        
        st.download_button(
            label=f"📥 {label}",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            key=key
        )
        return True
    except Exception as e:
        st.error(f"Download preparation failed: {str(e)}")
        return False

def apply_savitzky_golay_enhanced(data: np.ndarray, window_length: int, 
                                poly_order: int) -> Optional[np.ndarray]:
    """Enhanced Savitzky-Golay filter with better validation."""
    # Validation
    if window_length <= 0 or window_length % 2 == 0:
        st.warning("Window length must be positive and odd.")
        return None
    
    if poly_order < 0 or poly_order >= window_length:
        st.warning("Polynomial order must be non-negative and less than window length.")
        return None
    
    if len(data) < window_length:
        st.warning("Data length is shorter than window length.")
        return None
    
    try:
        # Lazy import scipy (only when smoothing is actually used)
        from scipy.signal import savgol_filter
        return savgol_filter(data, window_length, poly_order)
    except Exception as e:
        st.error(f"Smoothing error: {str(e)}")
        return None

# --- Main Application ---
def main():
    """Main application function with improved structure."""
    
    # Header
    st.title("Absorbance Calculator")
    st.markdown("### Spectral data processing with real-time visualization and peak analysis")
    
    # Sidebar configuration
    setup_sidebar()
    
    # Main content
    with st.container():
        # File upload section
        st.subheader("📁 Data Upload")
        ref_files, sample_files, dark_files = setup_file_upload()
        
        # Processing and results
        if ref_files and sample_files:
            process_and_display_results(ref_files, sample_files, dark_files)
        else:
            display_welcome_message()

def setup_sidebar():
    """Setup sidebar with enhanced controls."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Analysis range
        with st.expander("📊 Analysis Range", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                min_wl = st.number_input(
                    "Min λ (nm)", 
                    min_value=0.0, 
                    value=st.session_state.analysis_min_wl,
                    step=1.0
                )
            with col2:
                max_wl = st.number_input(
                    "Max λ (nm)", 
                    min_value=0.0,
                    value=st.session_state.analysis_max_wl,
                    step=1.0
                )
            
            if min_wl < max_wl:
                st.session_state.analysis_min_wl = min_wl
                st.session_state.analysis_max_wl = max_wl
                st.session_state.plot_range = [min_wl, max_wl]
            else:
                st.warning("Min wavelength must be less than max wavelength")
                st.session_state.plot_range = None
        
        # Processing options
        setup_processing_options()
        
        # Information
        setup_info_section()

def setup_processing_options():
    """Setup processing options in sidebar."""
    with st.expander("🔬 Signal Processing", expanded=False):
        st.session_state.apply_smoothing = st.checkbox(
            "Apply Smoothing", 
            value=False,
            help="Apply Savitzky-Golay smoothing to absorbance data"
        )
        
        if st.session_state.apply_smoothing:
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.sg_window = st.number_input(
                    "Window", min_value=3, value=11, step=2
                )
            with col2:
                st.session_state.sg_poly = st.number_input(
                    "Order", min_value=1, value=2, step=1
                )
    
    with st.expander("🎯 Peak Detection", expanded=True):
        st.session_state.peak_source = st.radio(
            "Peak detection on:",
            ["Raw Absorbance", "Smoothed Absorbance"],
            index=0,  # Default to Raw Absorbance
            disabled=not st.session_state.get('apply_smoothing', False)
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.peak_height = st.number_input(
                "Height", min_value=0.0, value=0.1, step=0.01, format="%.3f"
            )
        with col2:
            st.session_state.peak_distance = st.number_input(
                "Distance", min_value=0.1, value=15.0, step=0.5
            )
        with col3:
            st.session_state.peak_prominence = st.number_input(
                "Prominence", min_value=0.0, value=0.05, step=0.01, format="%.3f"
            )

def setup_info_section():
    """Setup information section in sidebar."""
    st.markdown("---")
    st.markdown("#### ℹ️ Information")
    
    with st.expander("📋 File Format"):
        st.markdown("""
        **Supported formats:**
        - Tab/space-delimited text files
        - Comma-separated values (CSV)
        
        **Required structure:**
        ```
        Nanometers    Counts
        340.0         1234.5
        341.0         1235.2
        ...
        ```
        
        **Tips:**
        - First column: wavelengths (nm)
        - Second column: intensity counts
        - Files can have headers
        - Multiple files will be averaged
        """)
    
    with st.expander("🧮 Calculations"):
        st.markdown("""
        **Absorbance Calculation:**
        ```
        A = log₁₀(I₀/I)
        ```
        Where:
        - A = Absorbance
        - I₀ = Reference intensity
        - I = Sample intensity
        
        **With dark correction:**
        ```
        A = log₁₀((I₀-Dark)/(I-Dark))
        ```
        """)

def setup_file_upload():
    """Setup file upload interface with enhanced features."""
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("##### 🔵 Reference (Blank)")
        ref_files = st.file_uploader(
            "Reference files",
            type=SUPPORTED_EXTENSIONS,
            accept_multiple_files=True,
            key="ref_uploader",
            help="Upload reference/blank spectrum files"
        )
        if ref_files:
            st.caption(f"✅ {len(ref_files)} file(s) uploaded")
            _render_waveform_preview(ref_files, "Reference", "blue", st.session_state.get('plot_range'))
    
    with cols[1]:
        st.markdown("##### 🟢 Sample")
        sample_files = st.file_uploader(
            "Sample files",
            type=SUPPORTED_EXTENSIONS,
            accept_multiple_files=True,
            key="sample_uploader",
            help="Upload sample spectrum files"
        )
        if sample_files:
            st.caption(f"✅ {len(sample_files)} file(s) uploaded")
            _render_waveform_preview(sample_files, "Sample", "green", st.session_state.get('plot_range'))
    
    with cols[2]:
        st.markdown("##### ⚫ Dark (Optional)")
        dark_files = st.file_uploader(
            "Dark files",
            type=SUPPORTED_EXTENSIONS,
            accept_multiple_files=True,
            key="dark_uploader",
            help="Upload dark spectrum files (optional)"
        )
        if dark_files:
            st.caption(f"✅ {len(dark_files)} file(s) uploaded")
            _render_waveform_preview(dark_files, "Dark", "black", st.session_state.get('plot_range'))
    
    return ref_files, sample_files, dark_files

def process_and_display_results(ref_files, sample_files, dark_files):
    """Process files and display results with enhanced organization."""
    
    # Processing
    with st.spinner("🔄 Processing spectral data..."):
        start_time = time.time()
        
        # Process files
        df_ref = process_uploaded_files(ref_files, "Reference")
        df_sample = process_uploaded_files(sample_files, "Sample") 
        df_dark = process_uploaded_files(dark_files, "Dark") if dark_files else None
        
        # Calculate absorbance
        if df_ref is not None and df_sample is not None:
            df_result = SpectralProcessor.calculate_absorbance(
                df_ref, df_sample, df_dark
            )
        else:
            df_result = None
        
        processing_time = time.time() - start_time
        st.session_state.last_calculation_time = processing_time
    
    if df_result is None:
        st.error("❌ Processing failed. Please check your files and try again.")
        return
    
    # Success message with timing (compact)
    st.caption(f"✅ Processing completed in {processing_time:.2f} s")
    
    # Apply smoothing if enabled
    smoothed_data = None
    if st.session_state.get('apply_smoothing', False):
        smoothed_data = apply_savitzky_golay_enhanced(
            df_result['Absorbance'].values,
            st.session_state.get('sg_window', 11),
            st.session_state.get('sg_poly', 2)
        )
        if smoothed_data is not None:
            df_result['Absorbance_Smoothed'] = smoothed_data
    
    # Peak detection
    peak_data = df_result['Absorbance'].values
    if (smoothed_data is not None and 
        st.session_state.get('peak_source') == "Smoothed Absorbance"):
        peak_data = smoothed_data
    
    peak_info = find_absorption_peaks_enhanced(
        df_result['Nanometers'].values,
        peak_data,
        st.session_state.get('peak_height', 0.1),
        st.session_state.get('peak_distance', 15.0),
        st.session_state.get('peak_prominence', 0.05)
    )
    
    # Display results
    display_results_tabs(df_result, smoothed_data, peak_info)

def display_results_tabs(df_result, smoothed_data, peak_info):
    """Display results in organized tabs."""
    
    # Add filename input
    st.markdown("#### 💾 Output File Name")
    col1, col2 = st.columns([3, 1])
    with col1:
        output_filename = st.text_input(
            "Enter filename (without extension)",
            value="absorbance_data",
            help="Specify the base name for downloaded files"
        )
    with col2:
        st.markdown("")  # Spacer
        st.markdown("")  # Spacer
    
    tab1, tab2, tab3 = st.tabs(["📈 Absorbance & Data", "📉 Raw Spectra", "🎯 Peak Analysis"])
    
    with tab1:
        # Display absorbance plot
        st.plotly_chart(
            PlotGenerator.plot_absorbance_optimized(
                df_result,
                peak_info,
                smoothed_data,
                st.session_state.get('plot_range')
            ),
            use_container_width=True
        )
        
        # Display data table
        display_data_table(df_result, output_filename)
    
    with tab2:
        display_raw_spectra(df_result)
    
    with tab3:
        display_peak_analysis(df_result, peak_info, output_filename)

def display_data_table(df_result, filename_base="absorbance_data"):
    """Display data table with download options."""
    st.markdown("#### 📋 Spectral Data")
    
    # Prepare display columns
    display_cols = ['Nanometers', 'Absorbance']
    if 'Absorbance_Smoothed' in df_result.columns:
        display_cols.append('Absorbance_Smoothed')
    
    # Display table
    st.dataframe(
        df_result[display_cols].round(6),
        use_container_width=True,
        height=400
    )
    
    # Download button - automatically uses filtered data when range is set
    if st.session_state.get('plot_range'):
        min_wl, max_wl = st.session_state.plot_range
        filtered_data = df_result[
            (df_result['Nanometers'] >= min_wl) & 
            (df_result['Nanometers'] <= max_wl)
        ]
        create_download_button(
            filtered_data, f"Download Data ({min_wl:.0f}-{max_wl:.0f} nm)",
            f"{filename_base}.csv", "data_download"
        )
    else:
        create_download_button(
            df_result, "Download Data", 
            f"{filename_base}.csv", "data_download"
        )

def display_raw_spectra(df_result):
    """Display raw spectral data plots."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    st.markdown("#### 📉 Raw Detector Counts")
    
    # Create subplot
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Raw Counts', 'Dark-Corrected Counts'))
    
    # Raw counts
    fig.add_trace(go.Scatter(x=df_result['Nanometers'], y=df_result['Reference_Counts'],
                           mode='lines', name='Reference', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_result['Nanometers'], y=df_result['Sample_Counts'],
                           mode='lines', name='Sample', line=dict(color='green')), row=1, col=1)
    
    if 'Dark_Counts' in df_result.columns:
        fig.add_trace(go.Scatter(x=df_result['Nanometers'], y=df_result['Dark_Counts'],
                               mode='lines', name='Dark', line=dict(color='black', dash='dot')), row=1, col=1)
    
    # Corrected counts
    fig.add_trace(go.Scatter(x=df_result['Nanometers'], y=df_result['Reference_Corrected'],
                           mode='lines', name='Reference (Corrected)', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_result['Nanometers'], y=df_result['Sample_Corrected'],
                           mode='lines', name='Sample (Corrected)', showlegend=False), row=2, col=1)
    
    fig.update_layout(height=600, template="plotly_white")
    fig.update_xaxes(title_text="Wavelength (nm)", row=2, col=1)
    fig.update_yaxes(title_text="Counts", row=1, col=1)
    fig.update_yaxes(title_text="Corrected Counts", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def display_peak_analysis(df_result, peak_info, filename_base="absorbance_data"):
    """Display peak analysis results."""
    import plotly.graph_objects as go
    st.markdown("#### 🎯 Peak Detection Results")
    
    if peak_info is None or len(peak_info[0]) == 0:
        st.info("No peaks detected with current parameters.")
        return
    
    peak_indices, properties = peak_info
    
    # Create peak summary table
    peak_data = {
        'Wavelength (nm)': df_result['Nanometers'].iloc[peak_indices].round(1),
        'Absorbance (AU)': df_result['Absorbance'].iloc[peak_indices].round(4),
        'Height (AU)': properties.get('peak_heights', [0]*len(peak_indices)),
        'Prominence (AU)': properties.get('prominences', [0]*len(peak_indices)),
        'Color Region': [get_color_name(wl) for wl in df_result['Nanometers'].iloc[peak_indices]]
    }

    peak_df = pd.DataFrame(peak_data)
    peak_df = peak_df.sort_values('Absorbance (AU)', ascending=False)
    
    # Filter by current range if set
    if st.session_state.get('plot_range'):
        min_wl, max_wl = st.session_state.plot_range
        peak_df_filtered = peak_df[
            (peak_df['Wavelength (nm)'] >= min_wl) & 
            (peak_df['Wavelength (nm)'] <= max_wl)
        ]
        st.markdown(f"**Peaks in range {min_wl:.0f}-{max_wl:.0f} nm:**")
        st.dataframe(peak_df_filtered, use_container_width=True)
           
        create_download_button(
            peak_df_filtered, "Download Peak Data",
            f"{filename_base}_peaks.csv", "peak_data"
        )
    else:
        st.dataframe(peak_df, use_container_width=True)
        create_download_button(
            peak_df, "Download Peak Data",
            f"{filename_base}_peaks.csv", "peak_data"
        )
    
    # Peak statistics
    if len(peak_df) > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Peaks", len(peak_df))
        with col2:
            st.metric("Max Absorbance", f"{peak_df['Absorbance (AU)'].max():.3f}")
        with col3:
            st.metric("Primary Peak", f"{peak_df.iloc[0]['Wavelength (nm)']:.1f} nm")
        with col4:
            st.metric("Peak Range", f"{peak_df['Wavelength (nm)'].max() - peak_df['Wavelength (nm)'].min():.1f} nm")

def display_welcome_message():
    """Display welcome message and instructions."""
    st.info("👆 Please upload Reference and Sample files to begin analysis.")
    
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown("""
        **Step 1:** Upload your spectral data files
        - **Reference/Blank**: Spectrum without sample
        - **Sample**: Spectrum with your sample
        - **Dark** (optional): Detector noise spectrum
        
        **Step 2:** Adjust settings in the sidebar
        - Set path length of your cuvette
        - Configure analysis wavelength range
        - Tune peak detection parameters
        
        **Step 3:** Analyze results
        - View absorbance spectrum with automatic peak detection
        - Export data and peak lists
        - Compare raw and processed spectra
        """)

if __name__ == "__main__":
    main() 
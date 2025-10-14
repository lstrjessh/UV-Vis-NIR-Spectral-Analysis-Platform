import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks, savgol_filter
from scipy import stats
import io
from typing import Dict, List, Tuple, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="Spectrum Viewer",
    page_icon="📊",
    layout="wide"
)

# --- Constants ---
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
SPECTRUM_TYPES = {
    'UV-Vis': {'range': (200, 800), 'unit': 'nm', 'label': 'Wavelength'},
    'IR': {'range': (400, 4000), 'unit': 'cm⁻¹', 'label': 'Wavenumber'},
    'Raman': {'range': (500, 3500), 'unit': 'cm⁻¹', 'label': 'Raman Shift'}
}

# --- Utility Functions ---
@st.cache_data
def read_spectrum_file(content: bytes, delimiter: str = None) -> pd.DataFrame:
    """Read spectrum file with automatic delimiter detection."""
    try:
        text = content.decode('utf-8')
        delimiters = [delimiter] if delimiter else [',', '\t', ' ', ';']
        
        for delim in delimiters:
            try:
                df = pd.read_csv(io.StringIO(text), delimiter=delim)
                if len(df.columns) >= 2 and len(df) > 10:
                    # Convert to numeric and clean
                    df = df.apply(pd.to_numeric, errors='coerce').dropna()
                    return df.iloc[:, :2]  # Return first two columns
            except:
                continue
        return None
    except:
        return None

def smooth_data(y: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    """Apply Savitzky-Golay smoothing."""
    if window % 2 == 0: window += 1
    if poly >= window: poly = window - 1
    return savgol_filter(y, window, poly)

def find_spectrum_peaks(x: np.ndarray, y: np.ndarray, 
                       prominence: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Find peaks in spectrum."""
    peaks, _ = find_peaks(y, prominence=prominence * (y.max() - y.min()))
    return x[peaks], y[peaks]

def calculate_stats(df: pd.DataFrame) -> Dict:
    """Calculate basic statistics."""
    x, y = df.iloc[:, 0], df.iloc[:, 1]
    return {
        'Points': len(df),
        'X Range': f"{x.min():.2f} - {x.max():.2f}",
        'Y Mean': f"{y.mean():.4f}",
        'Y Std': f"{y.std():.4f}",
        'Area': f"{np.trapz(y, x):.2f}"
    }

# --- Color Utilities for UV-Vis wavelength coloring ---
@st.cache_data
def wavelength_to_rgb(wavelength: float) -> Tuple[float, float, float]:
    """Approximate wavelength (nm) to RGB triple (0-1)."""
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

# --- Plotting Functions ---
def create_spectrum_plot(df: pd.DataFrame, name: str = "Spectrum",
                        show_smooth: bool = False, show_peaks: bool = False,
                        smooth_params: Dict = None, peak_params: Dict = None) -> go.Figure:
    """Create interactive spectrum plot."""
    x, y = df.iloc[:, 0].values, df.iloc[:, 1].values
    
    fig = go.Figure()
    
    # Main spectrum: use wavelength-colored segments if x-axis in nm (UV-Vis)
    x_label = str(df.columns[0]).lower()
    is_wavelength_nm = ('nm' in x_label) or ('wavelength' in x_label)
    if is_wavelength_nm and len(x) > 1:
        segments = min(120, max(2, len(x) - 1))
        for i in range(segments):
            idx = int(i * len(x) / segments)
            next_idx = int((i + 1) * len(x) / segments)
            seg_x = x[idx:next_idx+1]
            seg_y = y[idx:next_idx+1]
            if len(seg_x) < 2:
                continue
            r, g, b = wavelength_to_rgb(seg_x[len(seg_x)//2])
            fig.add_trace(go.Scatter(
                x=seg_x, y=seg_y, mode='lines', name=name,
                line=dict(width=3, color=f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'),
                showlegend=False,
                hovertemplate=f'{df.columns[0]}: %{{x:.1f}}<br>{df.columns[1]}: %{{y:.4f}}<extra></extra>'
            ))
    else:
        fig.add_trace(go.Scatter(
            x=x, y=y, name=name, mode='lines',
            line=dict(width=2, color=COLORS[0])
        ))
    
    # Smoothed line
    if show_smooth and smooth_params:
        y_smooth = smooth_data(y, **smooth_params)
        fig.add_trace(go.Scatter(
            x=x, y=y_smooth, name="Smoothed",
            mode='lines', line=dict(width=2, color=COLORS[1], dash='dash')
        ))
    
    # Peaks
    if show_peaks and peak_params:
        peak_x, peak_y = find_spectrum_peaks(x, y, **peak_params)
        if len(peak_x) > 0:
            fig.add_trace(go.Scatter(
                x=peak_x, y=peak_y, name=f"Peaks ({len(peak_x)})",
                mode='markers', marker=dict(size=10, color='red', symbol='triangle-down')
            ))
    
    fig.update_layout(
        xaxis_title=df.columns[0],
        yaxis_title=df.columns[1],
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_comparison_plot(data_list: List[pd.DataFrame], names: List[str]) -> go.Figure:
    """Create multi-spectrum comparison plot."""
    fig = go.Figure()
    
    for i, (df, name) in enumerate(zip(data_list, names)):
        x, y = df.iloc[:, 0], df.iloc[:, 1]
        fig.add_trace(go.Scatter(
            x=x, y=y, name=name, mode='lines',
            line=dict(width=2, color=COLORS[i % len(COLORS)])
        ))
    
    fig.update_layout(
        xaxis_title=data_list[0].columns[0],
        yaxis_title=data_list[0].columns[1],
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    
    return fig

# --- Main Application ---
def main():
    st.title("Spectral Data Visualization & Analysis")
    st.markdown("Advanced spectroscopic data visualization, preprocessing, and quality assessment platform")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Mode selection
        mode = st.radio("Mode", ["Single File", "Compare Files", "Demo"], horizontal=True)
        
        # Processing options
        st.subheader("Processing")
        show_smooth = st.checkbox("Show Smoothed")
        smooth_params = {}
        if show_smooth:
            smooth_params['window'] = st.slider("Window", 5, 51, 11, 2)
            smooth_params['poly'] = st.slider("Order", 1, 5, 3)
        
        show_peaks = st.checkbox("Show Peaks")
        peak_params = {}
        if show_peaks:
            peak_params['prominence'] = st.slider("Prominence", 0.0, 1.0, 0.1, 0.05)
    
    # Main content
    if mode == "Single File":
        handle_single_file(show_smooth, show_peaks, smooth_params, peak_params)
    elif mode == "Compare Files":
        handle_multiple_files()
    else:
        handle_demo(show_smooth, show_peaks, smooth_params, peak_params)

def handle_single_file(show_smooth, show_peaks, smooth_params, peak_params):
    """Handle single file upload and analysis."""
    file = st.file_uploader("Upload spectrum file", type=['csv', 'txt', 'dat'])
    
    if file:
        df = read_spectrum_file(file.read())
        if df is not None:
            # Create columns for layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                fig = create_spectrum_plot(
                    df, file.name, show_smooth, show_peaks, smooth_params, peak_params
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Statistics")
                stats = calculate_stats(df)
                for key, value in stats.items():
                    st.metric(key, value)
            
            # Download section
            with st.expander("Export Options"):
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download Data",
                        df.to_csv(index=False),
                        f"{file.name}_processed.csv",
                        "text/csv"
                    )
                with col2:
                    st.download_button(
                        "📥 Download Plot",
                        fig.to_html(),
                        f"{file.name}_plot.html",
                        "text/html"
                    )
        else:
            st.error("Could not read file. Please check the format.")

def handle_multiple_files():
    """Handle multiple file comparison."""
    files = st.file_uploader("Upload files to compare", type=['csv', 'txt', 'dat'], 
                            accept_multiple_files=True)
    
    if files:
        data_list = []
        names = []
        
        for file in files:
            df = read_spectrum_file(file.read())
            if df is not None:
                data_list.append(df)
                names.append(file.name)
        
        if data_list:
            fig = create_comparison_plot(data_list, names)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics comparison
            st.subheader("Comparison Statistics")
            stats_data = []
            for df, name in zip(data_list, names):
                stats = calculate_stats(df)
                stats['File'] = name
                stats_data.append(stats)
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)

def handle_demo(show_smooth, show_peaks, smooth_params, peak_params):
    """Generate and display demo spectrum."""
    st.subheader("Demo Spectrum")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        spec_type = st.selectbox("Type", list(SPECTRUM_TYPES.keys()))
    with col2:
        points = st.slider("Points", 100, 2000, 500)
    with col3:
        noise = st.slider("Noise", 0.0, 0.2, 0.05)
    
    # Generate demo data
    spec = SPECTRUM_TYPES[spec_type]
    x = np.linspace(spec['range'][0], spec['range'][1], points)
    
    # Create synthetic spectrum with multiple peaks
    y = np.zeros_like(x)
    peak_positions = np.linspace(spec['range'][0] + 100, spec['range'][1] - 100, 5)
    for pos in peak_positions:
        width = (spec['range'][1] - spec['range'][0]) / 20
        y += np.exp(-((x - pos) / width) ** 2) * np.random.uniform(0.5, 1.0)
    
    # Add granular noise
    y += np.random.normal(0, noise, len(x))
    
    # Dataframe diri 
    df = pd.DataFrame({
        f"{spec['label']} ({spec['unit']})": x,
        "Intensity": y
    })
    
    # Plot
    fig = create_spectrum_plot(
        df, f"Demo {spec_type} Spectrum", 
        show_smooth, show_peaks, smooth_params, peak_params
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 
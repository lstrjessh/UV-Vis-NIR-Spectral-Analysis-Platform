"""
Professional Calibration Modeling Application

Optimized and modular spectroscopic calibration system with advanced ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import time
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add calibration_v2 to path
import sys
sys.path.append(str(Path(__file__).parent))

# Import calibration system components
from calibration_v2.core import (
    ModelConfig, 
    CalibrationDataset,
    SpectralData
)
from calibration_v2.data import (
    StreamlitFileLoader,    
    StandardPreprocessor,
    CacheManager
)
from calibration_v2.models import ModelFactory, ModelRegistry

# Page configuration
st.set_page_config(
    page_title="Model Calibration",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}
    if 'cache_manager' not in st.session_state:
        st.session_state.cache_manager = CacheManager()
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = []

init_session_state()


class CalibrationApp:
    """Main application class for calibration modeling."""
    
    def __init__(self):
        self.loader = StreamlitFileLoader()
        self.factory = ModelFactory()
        self.registry = ModelRegistry()
        
    def run(self):
        """Run the main application."""
        self._render_header()
        self._render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📁 Data Loading", 
            "🔬 Preprocessing", 
            "🤖 Model Training",
            "📊 Results & Analysis"
        ])
        
        with tab1:
            self._render_data_loading()
        
        with tab2:
            self._render_preprocessing()
        
        with tab3:
            self._render_model_training()
        
        with tab4:
            self._render_results()
    
    def _render_header(self):
        """Render application header."""
        st.title("🔬 Calibration Modeling System")
        st.markdown("""        
        **Features:**
        - 🚀 Optimized model implementations with automatic hyperparameter tuning
        - 📈 Comprehensive metrics and visualizations
        - 🔄 Modular architecture for easy extension
        """)
    
    def _render_sidebar(self):
        """Render sidebar configuration."""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Model selection
            st.subheader("🤖 Model Selection")
            available_models = self.factory.available_models()
            
            model_groups = {
                'Linear': ['plsr', 'ridge', 'lasso'],
                'Ensemble': ['random_forest', 'xgboost', 'gradient_boosting'],
                'Neural': ['mlp', 'cnn1d'],
                'Kernel': ['svr']
            }
            
            selected_models = []
            for group, models in model_groups.items():
                st.write(f"**{group} Models**")
                for model in models:
                    model_info = next((m for m in available_models if m['name'] == model), None)
                    if model_info and model_info['available']:
                        if st.checkbox(f"{model.upper()}", value=model in ['plsr', 'random_forest'], key=f"cb_{model}"):
                            selected_models.append(model)
                    else:
                        st.checkbox(f"{model.upper()} (Not Available)", value=False, disabled=True, key=f"cb_{model}")
            
            st.session_state.selected_models = selected_models
            
            st.markdown("---")
            
            # Global settings
            st.subheader("🎯 Global Settings")
            
            cv_folds = st.slider(
                "Cross-Validation Folds",
                min_value=2,
                max_value=10,
                value=5,
                help="Number of folds for cross-validation"
            )
            
            optimization_method = st.selectbox(
                "Optimization Method",
                ["bayesian", "random_search", "grid_search"],
                index=0,
                help="Hyperparameter optimization strategy"
            )
            
            n_trials = st.slider(
                "Optimization Trials",
                min_value=10,
                max_value=100,
                value=30,
                step=10,
                help="Number of hyperparameter configurations to test"
            )
            
            early_stopping = st.checkbox(
                "Enable Early Stopping",
                value=True,
                help="Stop training when performance plateaus"
            )
            
            # Store in session state
            st.session_state.config = {
                'cv_folds': cv_folds,
                'optimization_method': optimization_method,
                'n_trials': n_trials,
                'early_stopping': early_stopping
            }
            
            st.markdown("---")
            
            # Cache management
            st.subheader("💾 Cache Management")
            cache_stats = st.session_state.cache_manager.get_stats()
            st.metric("Cache Entries", cache_stats['memory_entries'])
            st.metric("Cache Size", f"{cache_stats['memory_size_mb']:.2f} MB")
            
            if st.button("Clear Cache"):
                st.session_state.cache_manager.clear()
                st.success("Cache cleared!")
                st.rerun()
    
    def _render_data_loading(self):
        """Render data loading interface."""
        st.header("📁 Data Loading")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Upload Spectroscopic Data Files",
                type=['csv', 'txt', 'dat'],
                accept_multiple_files=True,
                help="Upload CSV files containing wavelength and absorbance data"
            )
            
            if uploaded_files:
                try:
                    with st.spinner("Loading and validating files..."):
                        dataset = self.loader.load_multiple(uploaded_files)
                        st.session_state.dataset = dataset
                    
                    # Display summary
                    st.success(f"✅ Successfully loaded {len(dataset)} spectra")
                    
                    # Dataset summary
                    summary = dataset.summary()
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Number of Spectra", summary['n_spectra'])
                    with col_b:
                        st.metric("Wavelength Points", summary['n_wavelengths'])
                    with col_c:
                        wl_range = summary['wavelength_range']
                        st.metric("Wavelength Range", f"{wl_range[0]:.0f}-{wl_range[1]:.0f} nm")
                    
                    # Concentration input
                    if not summary['has_concentrations']:
                        st.warning("⚠️ No concentrations detected in filenames. Please enter manually:")
                        
                        concentration_data = {}
                        cols = st.columns(3)
                        for i, spectrum in enumerate(dataset.spectra):
                            col = cols[i % 3]
                            with col:
                                conc = st.number_input(
                                    f"{spectrum.filename}",
                                    min_value=0.0,
                                    value=spectrum.concentration or 0.0,
                                    format="%.4f",
                                    key=f"conc_{spectrum.filename}"
                                )
                                concentration_data[spectrum.filename] = conc
                        
                        if st.button("Update Concentrations"):
                            for spectrum in dataset.spectra:
                                spectrum.concentration = concentration_data.get(spectrum.filename, 0.0)
                            st.session_state.dataset = dataset
                            st.success("Concentrations updated!")
                            st.rerun()
                    
                except Exception as e:
                    st.error(f"Error loading files: {str(e)}")
        
        with col2:
            if st.session_state.dataset:
                st.subheader("📊 Quick Visualization")
                
                # Create spectral plot
                fig = self._create_spectral_plot(st.session_state.dataset)
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_preprocessing(self):
        """Render preprocessing options."""
        st.header("🔬 Data Preprocessing")
        
        if st.session_state.dataset is None:
            st.warning("Please load data first in the Data Loading tab.")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Preprocessing Options")
            
            # Preprocessing settings
            smoothing = st.checkbox("Apply Smoothing", value=False)
            if smoothing:
                window_size = st.slider("Window Size", 3, 11, 5, step=2)
                poly_order = st.slider("Polynomial Order", 1, 5, 2)
            else:
                window_size, poly_order = 5, 2
            
            derivative = st.selectbox(
                "Derivative",
                [None, 1, 2],
                format_func=lambda x: "None" if x is None else f"{x}st derivative" if x == 1 else f"{x}nd derivative"
            )
            
            normalization = st.selectbox(
                "Normalization",
                [None, "standard", "minmax", "robust", "snv"],
                format_func=lambda x: "None" if x is None else x.upper()
            )
            
            baseline_correction = st.checkbox("Baseline Correction", value=False)
            
            # Apply preprocessing
            if st.button("Apply Preprocessing", type="primary"):
                with st.spinner("Preprocessing data..."):
                    preprocessor = StandardPreprocessor(
                        smoothing=smoothing,
                        smoothing_window=window_size if smoothing else 5,
                        smoothing_polyorder=poly_order if smoothing else 2,
                        derivative=derivative,
                        normalization=normalization,
                        baseline_correction=baseline_correction
                    )
                    
                    preprocessed_dataset = preprocessor.preprocess_dataset(st.session_state.dataset)
                    st.session_state.preprocessed_dataset = preprocessed_dataset
                    st.success("✅ Preprocessing completed!")
        
        with col2:
            st.subheader("Preview")
            
            # Show before/after comparison
            if hasattr(st.session_state, 'preprocessed_dataset'):
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Original", "Preprocessed")
                )
                
                # Original
                for spectrum in st.session_state.dataset.spectra[:5]:
                    fig.add_trace(
                        go.Scatter(
                            x=spectrum.wavelengths,
                            y=spectrum.absorbance,
                            name=spectrum.filename,
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                
                # Preprocessed
                for spectrum in st.session_state.preprocessed_dataset.spectra[:5]:
                    fig.add_trace(
                        go.Scatter(
                            x=spectrum.wavelengths,
                            y=spectrum.absorbance,
                            name=spectrum.filename,
                            showlegend=False
                        ),
                        row=1, col=2
                    )
                
                fig.update_layout(height=400)
                fig.update_xaxes(title_text="Wavelength (nm)")
                fig.update_yaxes(title_text="Absorbance", row=1, col=1)
                fig.update_yaxes(title_text="Processed", row=1, col=2)
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_model_training(self):
        """Render model training interface."""
        st.header("🤖 Model Training")
        
        # Check prerequisites
        dataset = getattr(st.session_state, 'preprocessed_dataset', st.session_state.dataset)
        
        if dataset is None:
            st.warning("Please load data first.")
            return
        
        if not st.session_state.selected_models:
            st.warning("Please select at least one model in the sidebar.")
            return
        
        # Training controls
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.info(f"**Selected Models:** {', '.join([m.upper() for m in st.session_state.selected_models])}")
        
        with col2:
            train_split = st.slider("Training Set Size", 0.5, 0.9, 0.8, 0.05)
        
        with col3:
            if st.button("🚀 Train Models", type="primary"):
                self._train_models(dataset, train_split)
        
        # Progress and results
        if st.session_state.model_results:
            st.markdown("---")
            st.subheader("Training Results")
            
            # Create results DataFrame
            results_data = []
            for name, result in st.session_state.model_results.items():
                results_data.append({
                    'Model': name.upper(),
                    'R²': f"{result.metrics.r2:.4f}",
                    'RMSE': f"{result.metrics.rmse:.4f}",
                    'MAE': f"{result.metrics.mae:.4f}",
                    'CV Score': f"{result.metrics.cv_mean:.4f} ± {result.metrics.cv_std:.4f}" if result.metrics.cv_mean else "N/A",
                    'Time (s)': f"{result.metrics.training_time:.2f}"
                })
            
            df_results = pd.DataFrame(results_data)
            st.dataframe(df_results, use_container_width=True)
            
            # Best model
            best_model = max(st.session_state.model_results.items(), 
                           key=lambda x: x[1].metrics.r2)
            st.success(f"🏆 Best Model: **{best_model[0].upper()}** (R² = {best_model[1].metrics.r2:.4f})")
    
    def _render_results(self):
        """Render results and analysis."""
        st.header("📊 Results & Analysis")
        
        if not st.session_state.model_results:
            st.warning("Please train models first.")
            return
        
        # Model comparison
        tab1, tab2, tab3, tab4 = st.tabs([
            "Model Comparison",
            "Predictions",
            "Feature Importance", 
            "Export Results"
        ])
        
        with tab1:
            self._render_model_comparison()
        
        with tab2:
            self._render_predictions()
        
        with tab3:
            self._render_feature_importance()
        
        with tab4:
            self._render_export()
    
    def _train_models(self, dataset: CalibrationDataset, train_split: float):
        """Train selected models."""
        # Prepare data
        X, y, wavelengths = dataset.to_matrix()
        
        if len(y) == 0:
            st.error("No concentration data available for training.")
            return
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_split, random_state=42
        )
        
        # Progress bar
        progress_bar = st.progress(0, text="Starting training...")
        model_results = {}
        
        # Train each model
        for i, model_name in enumerate(st.session_state.selected_models):
            progress_bar.progress(
                (i + 1) / len(st.session_state.selected_models),
                text=f"Training {model_name.upper()}..."
            )
            
            try:
                # Create model config
                config = ModelConfig(
                    name=model_name,
                    **st.session_state.config
                )
                
                # Create and train model
                model = self.factory.create(model_name, config)
                result = model.fit(X_train, y_train)
                
                # Test set evaluation
                y_pred_test = model.predict(X_test)
                test_metrics = model.calculate_metrics(y_test, y_pred_test, X_test)
                result.test_metrics = test_metrics
                
                model_results[model_name] = result
                
            except Exception as e:
                st.error(f"Error training {model_name}: {str(e)}")
        
        progress_bar.empty()
        
        # Store results
        st.session_state.model_results = model_results
        st.session_state.train_test_data = {
            'X_train': X_train,
            'X_test': X_test, 
            'y_train': y_train,
            'y_test': y_test,
            'wavelengths': wavelengths
        }
        
        st.success(f"✅ Successfully trained {len(model_results)} models!")
    
    def _render_model_comparison(self):
        """Render model comparison visualizations."""
        # Metrics comparison
        st.subheader("Performance Metrics")
        
        # Create comparison plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("R² Score", "RMSE", "MAE", "Training Time")
        )
        
        model_names = list(st.session_state.model_results.keys())
        train_r2 = [r.metrics.r2 for r in st.session_state.model_results.values()]
        test_r2 = [r.test_metrics.r2 for r in st.session_state.model_results.values()]
        rmse = [r.metrics.rmse for r in st.session_state.model_results.values()]
        mae = [r.metrics.mae for r in st.session_state.model_results.values()]
        times = [r.metrics.training_time for r in st.session_state.model_results.values()]
        
        # R² comparison
        fig.add_trace(
            go.Bar(name='Train', x=model_names, y=train_r2),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Test', x=model_names, y=test_r2),
            row=1, col=1
        )
        
        # RMSE
        fig.add_trace(
            go.Bar(x=model_names, y=rmse, showlegend=False),
            row=1, col=2
        )
        
        # MAE
        fig.add_trace(
            go.Bar(x=model_names, y=mae, showlegend=False),
            row=2, col=1
        )
        
        # Training time
        fig.add_trace(
            go.Bar(x=model_names, y=times, showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_predictions(self):
        """Render prediction plots."""
        st.subheader("Prediction Analysis")
        
        # Model selection
        model_name = st.selectbox(
            "Select Model",
            list(st.session_state.model_results.keys()),
            format_func=lambda x: x.upper()
        )
        
        result = st.session_state.model_results[model_name]
        data = st.session_state.train_test_data
        
        # Predict on test set
        model = result.model
        y_test_pred = result.model.predict(data['X_test']) if hasattr(result.model, 'predict') else data['y_test']
        
        # Create plots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Predicted vs Actual", "Residuals")
        )
        
        # Predicted vs Actual
        fig.add_trace(
            go.Scatter(
                x=data['y_test'],
                y=y_test_pred,
                mode='markers',
                name='Predictions',
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Perfect prediction line
        min_val = min(data['y_test'].min(), y_test_pred.min())
        max_val = max(data['y_test'].max(), y_test_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect',
                line=dict(dash='dash', color='red')
            ),
            row=1, col=1
        )
        
        # Residuals
        residuals = data['y_test'] - y_test_pred
        fig.add_trace(
            go.Scatter(
                x=y_test_pred,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(size=6)
            ),
            row=1, col=2
        )
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        
        fig.update_xaxes(title_text="Actual", row=1, col=1)
        fig.update_yaxes(title_text="Predicted", row=1, col=1)
        fig.update_xaxes(title_text="Predicted", row=1, col=2)
        fig.update_yaxes(title_text="Residual", row=1, col=2)
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_feature_importance(self):
        """Render feature importance analysis."""
        st.subheader("Feature Importance")
        
        # Filter models with feature importance
        models_with_importance = {
            name: result for name, result in st.session_state.model_results.items()
            if result.feature_importance
        }
        
        if not models_with_importance:
            st.info("No models with feature importance available.")
            return
        
        model_name = st.selectbox(
            "Select Model",
            list(models_with_importance.keys()),
            format_func=lambda x: x.upper()
        )
        
        result = models_with_importance[model_name]
        wavelengths = st.session_state.train_test_data['wavelengths']
        
        # Convert feature importance to wavelengths
        importance_values = []
        for i in range(len(wavelengths)):
            feature_key = f'feature_{i}'
            importance_values.append(result.feature_importance.get(feature_key, 0))
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=wavelengths,
            y=importance_values,
            mode='lines',
            name='Feature Importance',
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title=f"Feature Importance - {model_name.upper()}",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Importance",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_export(self):
        """Render export options."""
        st.subheader("Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Metrics (CSV)"):
                # Create comprehensive results DataFrame
                results_data = []
                for name, result in st.session_state.model_results.items():
                    row = {
                        'Model': name,
                        'Train_R2': result.metrics.r2,
                        'Test_R2': result.test_metrics.r2,
                        'RMSE': result.metrics.rmse,
                        'MAE': result.metrics.mae,
                        'Training_Time': result.metrics.training_time
                    }
                    if result.metrics.cv_mean:
                        row['CV_Mean'] = result.metrics.cv_mean
                        row['CV_Std'] = result.metrics.cv_std
                    results_data.append(row)
                
                df = pd.DataFrame(results_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="calibration_results.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("💾 Export Models (Pickle)"):
                import pickle
                
                # Serialize all models
                models_dict = {
                    name: result.model 
                    for name, result in st.session_state.model_results.items()
                }
                
                pickle_bytes = pickle.dumps(models_dict)
                
                st.download_button(
                    label="Download Models",
                    data=pickle_bytes,
                    file_name="calibration_models.pkl",
                    mime="application/octet-stream"
                )
    
    def _create_spectral_plot(self, dataset: CalibrationDataset) -> go.Figure:
        """Create spectral visualization."""
        fig = go.Figure()
        
        for spectrum in dataset.spectra[:10]:  # Limit to 10 for clarity
            label = f"{spectrum.filename}"
            if spectrum.concentration is not None:
                label += f" (C={spectrum.concentration:.3f})"
            
            fig.add_trace(go.Scatter(
                x=spectrum.wavelengths,
                y=spectrum.absorbance,
                mode='lines',
                name=label
            ))
        
        fig.update_layout(
            title="Spectral Data",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Absorbance",
            height=400,
            hovermode='x unified'
        )
        
        return fig


# Run application
if __name__ == "__main__":
    app = CalibrationApp()
    app.run()

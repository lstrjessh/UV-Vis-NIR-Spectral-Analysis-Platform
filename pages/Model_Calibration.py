"""
Professional Calibration Modeling Application

Optimized and modular spectroscopic calibration system with advanced ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import time
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add calibration to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="Calibration",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cached resource loaders
def get_model_factory():
    """Get ModelFactory instance."""
    from calibration.models import ModelFactory
    # Force registration of all models to ensure they're available
    from calibration.models import register_default_models
    register_default_models()
    
    # Create fresh factory
    return ModelFactory()

def get_model_registry():
    """Get ModelRegistry instance."""
    from calibration.models import ModelRegistry
    return ModelRegistry()

@st.cache_resource
def get_file_loader():
    """Get cached StreamlitFileLoader instance."""
    from calibration.data import StreamlitFileLoader
    return StreamlitFileLoader()

@st.cache_resource
def get_cache_manager():
    """Get cached CacheManager instance."""
    from calibration.data import CacheManager
    return CacheManager()

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = {}
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = []

init_session_state()


class CalibrationApp:   
    """Main application class for calibration modeling."""
    
    def __init__(self):
        # Use cached instances (lazy-loaded)
        self.loader = get_file_loader()
        self.factory = get_model_factory()
        self.registry = get_model_registry()
        
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
        st.title("Calibration")
        st.markdown("Build and optimize calibration models with automatic hyperparameter tuning.")
    
    def _render_sidebar(self):
        """Render sidebar configuration."""
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Model selection
            st.subheader("🤖 Model Selection")
            available_models = self.factory.available_models()
            
            model_groups = {
                'Linear': ['plsr', 'ridge', 'lasso', 'elastic_net'],
                'Ensemble': ['random_forest', 'xgboost'],
                'Neural': ['mlp'],
                'Kernel': ['svr']
            }
            
            selected_models = []
            for group, models in model_groups.items():
                st.write(f"**{group}**")
                for model in models:
                    model_info = next((m for m in available_models if m['name'] == model), None)
                    if model_info and model_info['available']:
                        # Add technical descriptions for each model
                        model_descriptions = {
                            'plsr': "Partial Least Squares Regression: Linear method that finds latent variables maximizing covariance between predictors and response. Excellent for multicollinearity and high-dimensional spectral data.",
                            'ridge': "Ridge Regression: L2-regularized linear regression that shrinks coefficients to prevent overfitting. Robust to multicollinearity in spectral data.",
                            'lasso': "Lasso Regression: L1-regularized linear regression that performs automatic feature selection by setting some coefficients to zero. Useful for identifying important wavelengths.",
                            'elastic_net': "Elastic Net: Combines L1 and L2 regularization, balancing feature selection (Lasso) and coefficient shrinkage (Ridge). Optimal for spectral data with correlated features.",
                            'random_forest': "Random Forest: Ensemble of decision trees using bootstrap aggregation and random feature selection. Non-parametric, handles non-linear relationships and provides feature importance. ⚠️ No scaling required (tree-based).",
                            'xgboost': "XGBoost: Extreme Gradient Boosting - Advanced gradient boosting framework with regularization, parallel processing, and early stopping. Excellent for complex non-linear patterns and high-dimensional spectral data with superior performance. ⚠️ No scaling, uses default parameters.",
                            'mlp': "Multi-Layer Perceptron: Feedforward neural network with multiple hidden layers. Captures complex non-linear patterns in spectral data through backpropagation learning.",
                            'svr': "Support Vector Regression: Kernel-based method that finds optimal hyperplane in high-dimensional feature space. Excellent for non-linear spectral relationships."
                        }
                        
                        if st.checkbox(f"{model.upper()}", value=model in ['plsr'], 
                                     help=model_descriptions.get(model, "Machine learning model for spectral calibration."), 
                                     key=f"cb_{model}"):
                            selected_models.append(model)
                    else:
                        st.checkbox(f"{model.upper()} (Not Available)", value=False, disabled=True, key=f"cb_{model}")
            
            st.session_state.selected_models = selected_models
            
            # Add warnings for models that don't use scaling
            tree_models = [model for model in selected_models if model in ['xgboost', 'random_forest']]
            if tree_models:
                model_names = ', '.join([m.upper() for m in tree_models])
                st.info(f"ℹ️ **{model_names}**: No scaling required (tree-based)")
            
            st.markdown("---")
            
            # Global settings
            st.subheader("Settings")
            
            cv_folds = st.slider("CV Folds", 2, 10, 5, 
                                help="Cross-validation folds for model evaluation. Higher values provide more robust performance estimates but increase computation time. Recommended: 5-7 for spectral data.")
            optimization_method = st.selectbox("Optimization", ["bayesian", "random_search", "grid_search"], 0,
                                             help="Hyperparameter optimization strategy. Bayesian: Uses Gaussian Process to model parameter-performance relationship. Random Search: Random sampling of parameter space. Grid Search: Exhaustive search over parameter grid.")
            n_trials = st.slider("Trials", 10, 100, 30, 10, 
                               help="Number of optimization trials for Bayesian/Random search. More trials improve parameter optimization but increase computation time. Grid search ignores this setting.")
            early_stopping = st.checkbox("Early Stopping", True, 
                                       help="Stop optimization early if no improvement is found. Reduces computation time but may miss optimal parameters in complex parameter spaces.")
            
            # Store in session state
            st.session_state.config = {
                'cv_folds': cv_folds,
                'optimization_method': optimization_method,
                'n_trials': n_trials,
                'early_stopping': early_stopping
            }
            
            st.markdown("---")
    
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
            st.subheader("Options")
            
            smoothing = st.checkbox("Smoothing", False, 
                                  help="Savitzky-Golay smoothing filter reduces noise while preserving spectral features. Uses polynomial fitting within a moving window to smooth the signal.")
            window_size = st.slider("Window", 3, 11, 5, 2, 
                                   help="Window size for Savitzky-Golay smoothing. Must be odd and greater than polynomial order. Larger windows provide more smoothing.") if smoothing else 5
            poly_order = st.slider("Poly Order", 1, 5, 2, 
                                  help="Polynomial order for Savitzky-Golay smoothing. Higher orders preserve more spectral features but may overfit noise.") if smoothing else 2
            
            derivative = st.selectbox("Derivative", [None, 1, 2], 0, 
                                    format_func=lambda x: "None" if x is None else f"{x}st" if x == 1 else f"{x}nd",
                                    help="Spectral derivatives enhance resolution and remove baseline effects. 1st derivative removes constant baseline, 2nd derivative removes linear baseline.")
            
            baseline_correction = st.checkbox("Baseline Correction", False, 
                                            help="Asymmetric Least Squares (ALS) baseline correction removes systematic baseline drift and fluorescence effects from spectral data.")
            
            if st.button("Apply", type="primary"):
                with st.spinner("Preprocessing data..."):
                    # Lazy import
                    from calibration.data import StandardPreprocessor
                    
                    preprocessor = StandardPreprocessor(
                        smoothing=smoothing,
                        smoothing_window=window_size if smoothing else 5,
                        smoothing_polyorder=poly_order if smoothing else 2,
                        derivative=derivative,
                        baseline_correction=baseline_correction
                    )
                    
                    preprocessed_dataset = preprocessor.preprocess_dataset(st.session_state.dataset)
                    st.session_state.preprocessed_dataset = preprocessed_dataset
                    st.success("✅ Preprocessing completed!")
        
        with col2:
            st.subheader("Preview")
            
            # Show before/after comparison
            if hasattr(st.session_state, 'preprocessed_dataset'):
                # Lazy import plotly
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
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
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.info(f"**Models:** {', '.join([m.upper() for m in st.session_state.selected_models])}")
            train_split = st.slider("Train Split", 0.5, 0.9, 0.6, 0.05, 
                                  help="Proportion of data used for training. Higher values use more data for training but less for validation. Recommended: 0.6-0.8 for spectral data.")
        
        with col2:
            st.info("ℹ️ **Scaling**: StandardScaler used automatically for models that require scaling")
        
        # Splitting method and train button
        col3, col4 = st.columns([2, 1])
        
        with col3:
            split_method = st.selectbox(
                "Split Method",
                ["kennard_stone", "random"],
                0,
                format_func=lambda x: "Kennard-Stone" if x == "kennard_stone" else "Random",
                help="Data splitting strategy. Kennard-Stone: Selects representative samples maximizing spectral diversity for training set. Random: Random sampling which may not capture spectral variability optimally."
            )
        
        with col4:
            if st.button("Train", type="primary"):
                self._train_models(dataset, train_split, split_method)
        
        # Progress and results
        if st.session_state.model_results:
            st.markdown("---")
            st.subheader("Training Results")
            
            # Create results DataFrame with both training and test metrics
            results_data = []
            for name, result in st.session_state.model_results.items():
                # Use TEST metrics instead of training metrics to avoid showing overfitting
                test_r2 = result.test_metrics.r2 if hasattr(result, 'test_metrics') else result.metrics.r2
                test_rmse = result.test_metrics.rmse if hasattr(result, 'test_metrics') else result.metrics.rmse
                test_mae = result.test_metrics.mae if hasattr(result, 'test_metrics') else result.metrics.mae
                
                results_data.append({
                    'Model': name.upper(),
                    'Train R²': f"{result.metrics.r2:.6f}",
                    'Test R²': f"{test_r2:.6f}",
                    'RMSE': f"{test_rmse:.6f}",
                    'MAE': f"{test_mae:.6f}",
                    'CV Score': f"{result.metrics.cv_mean:.6f} ± {result.metrics.cv_std:.6f}" if result.metrics.cv_mean else "N/A",
                    'Time (s)': f"{result.metrics.training_time:.2f}"
                })
            
            df_results = pd.DataFrame(results_data)
            st.dataframe(df_results, use_container_width=True)
            
            
            # Best model based on combined test metrics (R² > RMSE > MAE)
            best_model_item = max(
                st.session_state.model_results.items(), 
                key=lambda item: (
                    item[1].test_metrics.r2,    # 1. Maximize R²
                    -item[1].test_metrics.rmse, # 2. Minimize RMSE (by maximizing negative)
                    -item[1].test_metrics.mae   # 3. Minimize MAE (by maximizing negative)
                )
            )
            
            # Unpack the best model and its metrics
            best_name = best_model_item[0]
            best_metrics = best_model_item[1].test_metrics
            
            st.success(
                f"🏆 Best Model: **{best_name.upper()}** "
                f"(Test R²={best_metrics.r2:.6f}, RMSE={best_metrics.rmse:.6f}, MAE={best_metrics.mae:.6f})"
            )
    
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
    
    def _train_models(self, dataset, train_split: float, split_method: str = "kennard_stone"):
        """Train selected models."""
        # Lazy import
        import numpy as np
        from calibration.core import ModelConfig
        from sklearn.model_selection import train_test_split
        from utils.shared_utils import kennard_stone_split
        
        # Prepare data
        X, y, wavelengths = dataset.to_matrix()
        
        if len(y) == 0:
            st.error("No concentration data available for training.")
            return
        
        # Data validation and debugging
        n_samples, n_features = X.shape
        print(f"📊 Data Info: {n_samples} samples, {n_features} features")
        print(f"📊 Target range: {y.min():.4f} to {y.max():.4f}")
        print(f"📊 Target mean: {y.mean():.4f}, std: {y.std():.4f}")
        
        # Check for challenging dataset characteristics
        if n_samples < 50:
            print(f"⚠️ Small dataset detected ({n_samples} samples). Model performance may be limited.")
        if n_features > n_samples * 5:
            print(f"⚠️ High-dimensional data detected ({n_features} features vs {n_samples} samples). Consider dimensionality reduction.")
        if n_samples < 20:
            print("🚨 Very small dataset! Cross-validation may not be reliable. Consider collecting more data.")
            
        # Validate minimum requirements for models
        min_samples = 3  # Minimum for cross-validation
        if n_samples < min_samples:
            st.error(f"Dataset too small: {n_samples} samples. Need at least {min_samples} samples for training.")
            return
            
        # Suggest dimensionality reduction for high-dimensional data
        if n_features > n_samples * 10:
            print("💡 Consider using dimensionality reduction (PCA) for better model performance with high-dimensional data.")
        
        # Check for data issues
        if np.isnan(X).any():
            print("⚠️ Input data contains NaN values")
        if np.isnan(y).any():
            print("⚠️ Target data contains NaN values")
        if np.isinf(X).any():
            print("⚠️ Input data contains infinite values")
        if np.isinf(y).any():
            print("⚠️ Target data contains infinite values")
        
        # Split data using selected method
        if split_method == "kennard_stone":
            print("🎯 Using Kennard-Stone algorithm for optimal sample selection...")
            X_train, X_test, y_train, y_test = kennard_stone_split(
                X, y, train_size=train_split, random_state=42
            )
        else:
            print("🎲 Using random split...")
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
                result = model.fit(X_train, y_train, 
                                 scaler_type="standard")
                
                # Test set evaluation
                y_pred_test = model.predict(X_test)
                test_metrics = model.calculate_metrics(y_test, y_pred_test, X_test)
                result.test_metrics = test_metrics
                
                model_results[model_name] = result
                
            except Exception as e:
                error_msg = str(e)
                if "scipy.linalg.eigh" in error_msg:
                    st.error(f"Error training {model_name}: Dataset too small for this model. Try with more data or different model.")
                elif "Cannot use scipy.linalg.eigh for sparse A with k ≥ N" in error_msg:
                    st.error(f"Error training {model_name}: Too many components for dataset size. Try reducing components or using more data.")
                else:
                    st.error(f"Error training {model_name}: {error_msg}")
                print(f"Detailed error for {model_name}: {e}")
        
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
        # Lazy import plotly
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
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
        # Lazy import plotly
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        st.subheader("Prediction Analysis")
        
        # Model selection
        model_name = st.selectbox(
            "Select Model",
            list(st.session_state.model_results.keys()),
            format_func=lambda x: x.upper(),
            key="prediction_analysis_model_select"
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
        # Lazy import plotly
        import plotly.graph_objects as go
        
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
            format_func=lambda x: x.upper(),
            key="feature_importance_model_select"
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
    
    # UMAP visualization method removed for simplicity
    def _render_export(self):
        """Render export options."""
        st.subheader("Export Results")
        
        # Filename input
        output_filename = st.text_input(
            "Output filename (without extension)",
            value="calibration_results",
            help="Specify the base name for downloaded files"
        )
        
        st.markdown("")  # Spacer
        
        # Metrics export
        st.markdown("#### 📊 Metrics")
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
            label="📥 Download Metrics (CSV)",
            data=csv,
            file_name=f"{output_filename}.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # Individual model exports
        st.markdown("#### 🤖 Individual Models")
        
        # Create columns for individual downloads (2 per row)
        model_names = list(st.session_state.model_results.keys())
        cols_per_row = 2
        
        for i in range(0, len(model_names), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(model_names):
                    model_name = model_names[i + j]
                    result = st.session_state.model_results[model_name]
                    
                    with col:
                        self._create_model_download_button(
                            model_name, 
                            result, 
                            output_filename
                        )
        
        st.markdown("---")
        
        # Download all models
        st.markdown("#### 📦 Download All")
        if st.button("📥 Download All Models", type="primary"):
            zip_data = self._create_all_models_zip(output_filename)
            if zip_data:
                st.download_button(
                    label="💾 Save ZIP File",
                    data=zip_data,
                    file_name=f"{output_filename}_all_models.zip",
                    mime="application/zip",
                    key="download_all_zip"
                )
    
    def _create_model_download_button(self, model_name: str, result, base_filename: str):
        """Create download button for individual model."""
        import pickle
        import io
        
        try:
            # Determine model type
            is_pytorch = hasattr(result.model, 'state_dict')  # PyTorch model
            
            if is_pytorch:
                # PyTorch model - use torch.save
                import torch
                buffer = io.BytesIO()
                torch.save({
                    'model_state_dict': result.model.state_dict(),
                    'model_class': result.model.__class__.__name__,
                    'scaler_mean': result.model.scaler.mean_ if hasattr(result.model, 'scaler') else None,
                    'scaler_scale': result.model.scaler.scale_ if hasattr(result.model, 'scaler') else None,
                }, buffer)
                buffer.seek(0)
                
                st.download_button(
                    label=f"📥 {model_name.upper()} (PyTorch)",
                    data=buffer,
                    file_name=f"{base_filename}_{model_name}.pth",
                    mime="application/octet-stream",
                    key=f"download_{model_name}"
                )
            else:
                # Sklearn model - use pickle
                model_bytes = pickle.dumps(result.model)
                
                st.download_button(
                    label=f"📥 {model_name.upper()} (Pickle)",
                    data=model_bytes,
                    file_name=f"{base_filename}_{model_name}.pkl",
                    mime="application/octet-stream",
                    key=f"download_{model_name}"
                )
                
        except Exception as e:
            st.error(f"❌ {model_name.upper()}: {str(e)}")
    
    def _create_all_models_zip(self, base_filename: str) -> Optional[bytes]:
        """Create ZIP file containing all models."""
        import pickle
        import io
        import zipfile
        
        try:
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add metrics CSV
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
                csv_data = df.to_csv(index=False)
                zip_file.writestr(f"{base_filename}_metrics.csv", csv_data)
                
                # Add each model
                for model_name, result in st.session_state.model_results.items():
                    try:
                        is_pytorch = hasattr(result.model, 'state_dict')
                        
                        if is_pytorch:
                            # PyTorch model
                            import torch
                            buffer = io.BytesIO()
                            torch.save({
                                'model_state_dict': result.model.state_dict(),
                                'model_class': result.model.__class__.__name__,
                                'scaler_mean': result.model.scaler.mean_ if hasattr(result.model, 'scaler') else None,
                                'scaler_scale': result.model.scaler.scale_ if hasattr(result.model, 'scaler') else None,
                            }, buffer)
                            zip_file.writestr(
                                f"models/{model_name}.pth",
                                buffer.getvalue()
                            )
                        else:
                            # Sklearn model
                            model_bytes = pickle.dumps(result.model)
                            zip_file.writestr(
                                f"models/{model_name}.pkl",
                                model_bytes
                            )
                    except Exception as e:
                        st.warning(f"⚠️ Could not export {model_name}: {str(e)}")
                
                # Add README
                readme_content = f"""# Calibration Models Export
                
## Contents
- metrics.csv: Performance metrics for all models
- models/: Directory containing individual model files

## Models Included
{chr(10).join([f'- {name.upper()}: {type(result.model).__name__}' for name, result in st.session_state.model_results.items()])}

## Loading Models

### Sklearn Models (.pkl)
```python
import pickle
with open('model_name.pkl', 'rb') as f:
    model = pickle.load(f)
predictions = model.predict(X_test)
```

### PyTorch Models (.pth)
```python
import torch
checkpoint = torch.load('model_name.pth')
# Rebuild model architecture, then:
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                zip_file.writestr("README.md", readme_content)
            
            zip_buffer.seek(0)
            st.success(f"✅ Successfully packaged {len(st.session_state.model_results)} models!")
            return zip_buffer.getvalue()
            
        except Exception as e:
            st.error(f"❌ Error creating ZIP file: {str(e)}")
            return None
    
    def _create_spectral_plot(self, dataset):
        """Create spectral visualization."""
        # Lazy import plotly
        import plotly.graph_objects as go
        
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

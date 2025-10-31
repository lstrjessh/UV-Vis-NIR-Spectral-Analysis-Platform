# Spectral Analysis Platform

A comprehensive application for spectroscopic data analysis and machine learning model calibration. Available as both a **desktop application** (Qt/PyQt6) and a **web application** (Streamlit), this platform provides advanced tools for absorbance calculation, spectral visualization, model training, and concentration prediction.

## 🌟 Features

### Core Analysis Tools
- **Calculate Absorbance** - Process raw spectral data with dark spectrum correction, averaging, and Savitzky-Golay smoothing
- **View Spectra** - Visualize and compare multiple spectral datasets with customizable processing options and peak detection
- **Model Calibration** - Train and optimize ML models using multiple algorithms
- **Predict Concentration** - Apply trained models to new spectral data for accurate concentration predictions

### Machine Learning Models
The platform supports multiple regression algorithms:
- **Linear Models**: PLSR, Ridge Regression, Lasso, Elastic Net
- **Ensemble Models**: Random Forest, XGBoost
- **Neural Networks**: Multi-Layer Perceptron (MLP)
- **Kernel Methods**: Support Vector Regression (SVR)

### Advanced Features
- **Automated Hyperparameter Optimization** - Bayesian optimization, random search, and grid search using Optuna
- **Cross-Validation** - Built-in k-fold cross-validation for robust model evaluation
- **Interactive Visualizations** - Matplotlib (desktop) and Plotly (web) charts for data exploration
- **Model Persistence** - Save and load trained models (pickle for sklearn, PyTorch for neural networks)
- **Peak Detection** - Automated spectral peak identification with configurable parameters
- **Data Preprocessing** - Smoothing, derivatives, baseline correction, and normalization
- **Model Comparison** - Side-by-side performance metrics and visualization
- **Export Capabilities** - Export metrics, models, and analysis results to CSV/ZIP

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/lstrjessh/UV-Vis-Spectrophotometry-Platform.git
cd UV-Vis-Spectrophotometry-Platform

# Install dependencies
pip install -r requirements.txt
```

### Desktop Application (Qt)

The desktop app provides a native experience with tabbed interface:

```bash
# Run Qt desktop application
python qt_app/main_qt.py
```

**Features**:
- Modern, responsive Qt interface
- Light mode theme optimized for spectral analysis
- Real-time plot updates
- Native file dialogs
- Full-featured model training and visualization

### Web Application (Streamlit)

The web app provides browser-based access:

```bash
# Run Streamlit web application
streamlit run Main.py
```

The application will open in your default web browser at `http://localhost:8501`

**Features**:
- Multi-page navigation
- Interactive Plotly visualizations
- Upload interface for spectral files
- Comprehensive model training dashboard

## 🚀 Quick Start

### Desktop App Workflow

1. **Calculate Absorbance**
   - Select reference, sample, and optional dark spectrum files
   - Configure smoothing and peak detection parameters
   - View absorbance spectrum with detected peaks
   - Export results to CSV

2. **View Spectra**
   - Load multiple spectral files
   - Apply normalization and smoothing
   - Detect and visualize peaks
   - Compare spectra side-by-side

3. **Model Calibration**
   - Load calibration dataset (CSV files with concentration data)
   - Configure preprocessing (smoothing, derivatives, baseline correction)
   - Select models to train (PLSR, Random Forest, XGBoost, etc.)
   - Configure training parameters (CV folds, optimization method, train/test split)
   - View training results with metrics table
   - Analyze models using comparison plots, prediction plots, and feature importance
   - Export models and metrics

4. **Predict Concentration**
   - Load trained model
   - Upload new spectral data
   - Get concentration predictions with confidence intervals

### Web App Workflow

Similar workflow through browser-based interface with tabbed navigation.

## 📊 Sample Data

The platform includes sample spectral data files in `sample_data/` with concentration standards ranging from 0.1 to 1.0 units. These files demonstrate the expected data format and can be used for testing:

```
sample_data/
├── 0.1_a.csv, 0.1_b.csv
├── 0.2_a.csv, 0.2_b.csv
├── ...
└── 1.0_a.csv, 1.0_b.csv
```

**Expected CSV Format**:
- Wavelength column: "Nanometers" or similar (detected automatically)
- Absorbance/Intensity column: "Counts" or similar (detected automatically)
- Optional: Concentration can be extracted from filename or provided manually

## 🔧 Building Executable (Windows)

Package the desktop application into a standalone executable:

```bash
# Install PyInstaller
pip install pyinstaller

# Build executable using the provided spec file
pyinstaller SpectralAnalysis.spec
```

The executable will be created in `dist/SpectralAnalysis/SpectralAnalysis.exe`

**Note**: Build artifacts (`dist/`, `build/`) are excluded from Git. Do not commit executable files.

## 📁 Project Structure

```
thesis_webapp/
├── Main.py                      # Streamlit web application entry point
├── qt_app/                      # Desktop Qt application
│   ├── main_qt.py              # Qt application entry point
│   ├── spectral_processing.py  # Core spectral processing functions
│   └── views/                  # Qt view components
│       ├── home.py             # Home/dashboard view
│       ├── absorbance_view.py  # Absorbance calculation interface
│       ├── viewer_view.py      # Spectrum viewer interface
│       ├── calibration_view.py # Model calibration interface
│       └── predict_view.py     # Concentration prediction interface
├── pages/                       # Streamlit page modules
│   ├── Calculate_Absorbance.py
│   ├── View_Spectra.py
│   ├── Model_Calibration.py
│   ├── Predict_Concentration.py
│   └── Provide_Feedback.py
├── calibration/                 # Core calibration framework
│   ├── core/                   # Base classes and interfaces
│   │   ├── base_model.py       # Abstract model base classes
│   │   ├── data_structures.py # Data containers (SpectralData, CalibrationDataset)
│   │   ├── exceptions.py       # Custom exceptions
│   │   └── interfaces.py       # Abstract interfaces
│   ├── data/                   # Data handling modules
│   │   ├── loader.py           # CSV data loading (Qt & Streamlit compatible)
│   │   └── preprocessor.py     # Data preprocessing (smoothing, derivatives, etc.)
│   ├── models/                 # Machine learning implementations
│   │   ├── linear_models.py    # PLSR, Ridge, Lasso, Elastic Net
│   │   ├── ensemble_models.py  # Random Forest, XGBoost
│   │   ├── neural_models.py    # PyTorch MLP
│   │   ├── svm_models.py       # Support Vector Regression
│   │   ├── pipeline_models.py  # Model pipelines with preprocessing
│   │   └── registry.py         # Model registration system
│   └── utils/                  # Calibration utilities
│       ├── export.py           # Model export tools
│       ├── metrics.py          # Evaluation metrics
│       └── optimization.py     # Optimization helpers
├── utils/                       # Shared utilities
│   ├── shared_utils.py        # Common functions (Kennard-Stone split, etc.)
│   └── optimization_helpers.py
├── sample_data/                 # Sample spectral data files
├── SpectralAnalysis.spec        # PyInstaller configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 📋 Requirements

### Core Dependencies
- **Python**: 3.8 or higher
- **Streamlit**: ≥1.28.0 (web app)
- **PyQt6**: ≥6.0.0 (desktop app)
- **Pandas**: ≥2.0.0
- **NumPy**: ≥1.24.0
- **SciPy**: ≥1.10.0
- **Scikit-learn**: ≥1.3.0
- **Matplotlib**: ≥3.7.0 (desktop plots)
- **Plotly**: ≥5.15.0 (web visualizations)

### Machine Learning
- **XGBoost**: ≥1.7.0
- **PyTorch**: ≥2.0.0 (for neural network models)
- **Optuna**: ≥3.0.0 (hyperparameter optimization)

### Additional Tools
- **Joblib**: ≥1.3.0 (model serialization)
- **PyWavelets**: ≥1.4.0 (wavelet transforms)

See `requirements.txt` for complete list with version constraints.

## 🔬 Model Training Workflow

### 1. Data Preparation
- Load CSV files containing spectral data with wavelength and absorbance columns
- Ensure concentration values are either:
  - Embedded in filenames (e.g., `0.5_sample.csv`)
  - Or manually entered through the interface

### 2. Preprocessing Configuration
- **Smoothing**: Savitzky-Golay filter with configurable window size and polynomial order
- **Derivatives**: Optional 1st or 2nd derivative for baseline removal
- **Baseline Correction**: Asymmetric Least Squares (ALS) baseline correction

### 3. Model Selection & Training
- Select models from linear, ensemble, neural, and kernel-based options
- Configure training parameters:
  - Train/test split ratio (Kennard-Stone or random)
  - Cross-validation folds
  - Hyperparameter optimization method (Bayesian, random search, grid search)
  - Number of optimization trials

### 4. Results Analysis
- **Training Metrics Tab**: View R², RMSE, MAE, and training time for all models
- **Model Comparison Tab**: Visualize metrics across models with bar charts
- **Predictions Tab**: Predicted vs actual scatter plots and residual analysis
- **Feature Importance Tab**: Wavelength importance plots (for applicable models)
- **Export Tab**: Download individual models or all models as ZIP

### 5. Model Export
- Export trained models as `.pkl` (sklearn) or `.pth` (PyTorch)
- Export metrics and hyperparameters as CSV/JSON
- Package all models with documentation in ZIP format

## 🎯 Use Cases

- **Research Laboratories**: Calibrate models for concentration prediction from spectroscopic measurements
- **Quality Control**: Build predictive models for material analysis
- **Educational**: Teach spectroscopic analysis and machine learning concepts
- **Process Development**: Optimize calibration models for industrial applications

## 🐛 Troubleshooting

### Desktop App Issues

**Import Errors**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

**Plot Not Rendering**: Ensure Matplotlib backend is properly configured (QtAgg should be used automatically)

**Model Training Fails**: Check that:
- Dataset has sufficient samples (minimum 3 for cross-validation)
- Concentration data is present in filenames or entered manually
- Wavelength grids match across all spectra

### Web App Issues

**Port Already in Use**: Change the port:
```bash
streamlit run Main.py --server.port 8502
```

**Large File Upload**: Increase file size limit in Streamlit config

### Build Issues

**PyInstaller Errors**: 
- Ensure all dependencies are installed
- Check `SpectralAnalysis.spec` for correct paths
- Large builds may take several minutes

## 🤝 Contributing

This project is developed for research and educational purposes. Contributions are welcome!

## 📄 License

This project is developed for research and educational purposes as part of thesis work in spectroscopic analysis.

## 📧 Contact

For questions or feedback, please open an issue on the GitHub repository.

---

**Note**: Build artifacts (`dist/`, `build/`, `*.exe`, `*.dll`) are excluded from the repository. Executables should be distributed via GitHub Releases rather than committing them to the repository.

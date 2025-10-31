# Spectral Analysis Platform

A comprehensive web-based application for spectroscopic data analysis and machine learning model calibration. Developed as a complimentary tool for camera-based single-beam transmission grating spectrophotometry, this platform provides advanced tools for absorbance calculation, model training, and concentration prediction.

## Features

### 🔬 Core Analysis Tools
- **Calculate Absorbance** - Process raw spectral data with dark spectrum correction, averaging, and advanced smoothing algorithms
- **Model Calibration** - Build and optimize ML models with multiple algorithms including:
  - Partial Least Squares Regression (PLSR)
  - Support Vector Regression (SVR)
  - Random Forest
  - XGBoost
  - Neural Networks (PyTorch-based)
- **Predict Concentration** - Apply trained models to predict analyte concentrations with batch processing support
- **Provide Feedback** - Submit feedback and suggestions to improve the platform

### 🚀 Advanced Features
- **Automated Hyperparameter Optimization** - Uses Optuna for intelligent parameter tuning
- **Cross-Validation** - Built-in k-fold cross-validation for robust model evaluation
- **Interactive Visualizations** - Plotly-based charts for data exploration and model analysis
- **Model Persistence** - Save and load trained models with joblib serialization
- **Peak Detection** - Automated spectral peak identification and analysis
- **Data Preprocessing** - Comprehensive data cleaning and normalization tools

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd thesis_webapp

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run as web application
streamlit run Main.py
```

The application will open in your default web browser at `http://localhost:8501`

## Requirements

### Core Dependencies
- Python 3.8+
- Streamlit 1.28+
- Pandas 2.0+
- NumPy 1.24+
- SciPy 1.10+
- Scikit-learn 1.3+
- Plotly 5.15+

### Machine Learning
- XGBoost 1.7+
- PyTorch 2.0+ (for neural networks)
- Optuna 3.0+ (for hyperparameter optimization)

### Additional Tools
- Joblib 1.3+ (for model serialization)

See `requirements.txt` for complete list of dependencies.

## Sample Data

The platform includes sample spectral data files (`sample_data/`) with concentration standards ranging from 0.1 to 1.0 units. These files demonstrate the expected data format and can be used for testing and calibration purposes.

## Project Structure

```
thesis_webapp/
├── Main.py                    # Main Streamlit application entry point
├── pages/                     # Streamlit page modules
│   ├── Calculate_Absorbance.py    # Absorbance calculation interface
│   ├── Model_Calibration.py       # Model training and optimization
│   ├── Predict_Concentration.py   # Concentration prediction
│   └── Provide_Feedback.py        # User feedback system
├── calibration/               # Core calibration framework
│   ├── core/                  # Base classes and interfaces
│   │   ├── base_model.py      # Abstract model base classes
│   │   ├── data_structures.py # Data containers and types
│   │   ├── exceptions.py      # Custom exceptions
│   │   └── interfaces.py      # Abstract interfaces
│   ├── data/                  # Data handling modules
│   │   ├── cache_manager.py   # Caching system
│   │   ├── loader.py          # Data loading utilities
│   │   └── preprocessor.py    # Data preprocessing
│   ├── models/                # Machine learning implementations
│   │   ├── ensemble_models.py # Random Forest, XGBoost
│   │   ├── linear_models.py   # PLSR implementations
│   │   ├── neural_models.py   # PyTorch neural networks
│   │   ├── pipeline_models.py # Model pipelines
│   │   ├── registry.py        # Model registration system
│   │   └── svm_models.py      # Support Vector models
│   └── utils/                 # Calibration utilities
│       ├── export.py          # Model export tools
│       ├── metrics.py         # Evaluation metrics
│       └── optimization.py    # Optimization helpers
├── utils/                     # Shared utilities
│   ├── optimization_helpers.py # Optimization utilities
│   └── shared_utils.py        # Common functions
├── sample_data/               # Sample spectral data files
└── requirements.txt           # Python dependencies
```

## Getting Started

1. **Calculate Absorbance**: Upload your spectral data files and process them with dark spectrum correction and smoothing
2. **Model Calibration**: Use the sample data or your own datasets to train predictive models
3. **Predict Concentration**: Apply trained models to new spectral data for concentration prediction
4. **Provide Feedback**: Share your experience and suggestions for platform improvement

## License

This project is developed for research and educational purposes as part of thesis work in spectroscopic analysis.


# Spectral Analysis Platform

A comprehensive application for analyzing spectroscopic data and building machine learning models for concentration prediction. Developed to complement a portable CMOS camera-based spectrophotometer for heavy metal detection, built as part of an undergraduate thesis project.

Available as both a **desktop application** (Qt) and **web application** (Streamlit) to suit different workflows and deployment scenarios.

## Features

### Desktop Application (Qt)

![Capture View](images/capture_view.png)
*Real-time spectrum acquisition with camera-based spectrometer*

![Calibration View](images/calibration_view.png)
*Machine learning model training and evaluation interface*

The desktop application provides:

- **Live Spectrum Capture** - Real-time data acquisition from camera-based spectrometers with adjustable exposure, gain, and smoothing controls
- **Wavelength Calibration** - Polynomial calibration using known spectral lines for accurate wavelength mapping
- **Absorbance Calculation** - Process raw intensity data with dark spectrum correction and reference normalization
- **Spectrum Viewer** - Visualize and compare multiple spectra with interactive plots
- **Model Training** - Build and evaluate machine learning models with automatic hyperparameter optimization
- **Concentration Prediction** - Apply trained models to new spectral measurements

### Web Application (Streamlit)

The web interface offers a simplified workflow for spectral analysis:

- **Calculate Absorbance** - Batch process spectral files with preprocessing options
- **View Spectra** - Upload and overlay multiple spectrum files with peak detection
- **Model Calibration** - Train predictive models using various ML algorithms
- **Predict Concentration** - Load trained models and predict from new data

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Desktop Application

```bash
python qt_app/main_qt.py
```

### Web Application

```bash
streamlit run Main.py
```

Access the web interface at `http://localhost:8501`

## Machine Learning Models

The platform supports multiple regression algorithms:

- **Partial Least Squares Regression (PLSR)** - Standard method for spectroscopic data
- **Linear Models** - Ridge, Lasso, and Elastic Net regression
- **Tree-Based Models** - Random Forest and XGBoost with ensemble learning
- **Neural Networks** - PyTorch-based models with customizable architecture
- **Support Vector Regression** - Kernel-based regression with RBF and linear kernels

All models include:
- Automatic hyperparameter optimization using Bayesian search
- Cross-validation for robust performance evaluation
- Preprocessing pipelines (smoothing, derivatives, baseline correction)
- Model export and persistence for deployment

## Data Format

CSV files should contain wavelength and intensity/absorbance columns. The platform automatically detects column headers like "Nanometers", "Wavelength", "Counts", "Absorbance", etc.

For calibration, filenames can include concentration values (e.g., `0.5_a.csv` for 0.5 concentration units) or concentrations can be entered manually.

Example data is provided in `sample_data/` directory.

## Building Executable (Windows)

Create a standalone executable using PyInstaller:

```bash
pip install pyinstaller
pyinstaller SpectralAnalysis.spec
```

Output will be in `dist/SpectralAnalysis/`

## Project Structure

```
spectral_analysis_platform/
├── Main.py                  # Streamlit web app entry point
├── qt_app/                  # Desktop Qt application
│   ├── main_qt.py          # Qt app entry point
│   ├── views/              # UI view components
│   ├── camera_thread.py    # Camera interface thread
│   └── spec_functions.py   # Spectroscopy utilities
├── pages/                   # Streamlit pages
├── calibration/             # Core ML framework
│   ├── core/               # Base classes and interfaces
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # ML model implementations
│   └── utils/              # Metrics, export, optimization
├── sample_data/             # Example spectral files
└── requirements.txt         # Python dependencies
```

## Troubleshooting

**Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`

**Port conflict**: Run Streamlit on a different port: `streamlit run Main.py --server.port 8502`

**Camera not detected**: Check camera permissions and ensure OpenCV can access the device

**Model training errors**: Verify you have at least 3 samples for cross-validation and that all spectra use consistent wavelength ranges

## License

Developed for research and educational purposes as part of undergraduate thesis work.
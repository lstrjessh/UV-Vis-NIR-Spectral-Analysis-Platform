# Spectral Analysis Platform

A cross-platform application for spectroscopic data analysis and machine learning model calibration. Primariy developed as a complimentary tool for my proprietary camera-based single-beam transmission grating spectrophotometer.

## Features

- **Calculate Absorbance** - Process raw spectral data with dark spectrum correction and smoothing
- **Model Calibration** - Build ML models (PLSR, SVR, Random Forest, XGBoost, Neural Networks) with automated hyperparameter optimization
- **Predict Concentration** - Apply trained models to predict analyte concentrations

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
streamlit run main.py
```

The application will open in your default web browser at `http://localhost:8501`

## Requirements

- Python 3.8+
- Streamlit 1.28+
- NumPy, Pandas, SciPy
- Scikit-learn
- PyTorch (optional, for neural networks)
- XGBoost

See `requirements.txt` for complete list of dependencies.

## Project Structure

```
thesis_webapp/
├── main.py                 # Main application entry point   # Build script for executables
├── pages/                  # Streamlit pages
│   ├── Calculate_Absorbance.py
│   ├── Model_Calibration.py
│   ├── Predict_Concentration.py
│   └── Provide_Feedback.py
├── calibration/            # Calibration module
│   ├── core/              # Core classes and interfaces
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # ML model implementations
│   └── utils/             # Utilities
└── requirements.txt        # Python dependencies
```

## License

This project is for research and educational purposes.


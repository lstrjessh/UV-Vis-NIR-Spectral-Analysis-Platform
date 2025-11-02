# Spectral Analysis Platform

A user-friendly application for analyzing spectroscopic data and building machine learning models for concentration prediction. This platform was developed to complement a portable CMOS camera-based spectrophotometer for heavy metal detection, built as part of my undergraduate thesis project.

Available as both a **desktop app** (Qt) and **web app** (Streamlit), so you can choose what works best for your workflow.

## 🌟 What It Does

- **Calculate Absorbance** - Process raw spectra with dark spectrum correction and smoothing
- **Visualize Spectra** - View and compare multiple spectra with interactive plots
- **Train ML Models** - Build predictive models using various algorithms (PLSR, Random Forest, XGBoost, Neural Networks, and more)
- **Predict Concentrations** - Use trained models to predict concentrations from new spectral data

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Run the Desktop App

```bash
python qt_app/main_qt.py
```

### Run the Web App

```bash
streamlit run Main.py
```

Then open `http://localhost:8501` in your browser.

## 📊 Typical Workflow

1. **Load your data** - Upload CSV files with wavelength and absorbance/intensity columns
2. **Preprocess** - Apply smoothing, derivatives, or baseline correction as needed
3. **Train models** - Select algorithms, configure hyperparameters, and let the platform optimize them
4. **Analyze results** - Compare models, visualize predictions, and check feature importance
5. **Predict** - Load a trained model and predict concentrations for new samples

## 📦 Sample Data

The `sample_data/` folder contains example files showing the expected format. Files are named with concentration values (e.g., `0.1_a.csv` for 0.1 concentration units).

CSV files should have:
- Wavelength column (automatically detected as "Nanometers" or similar)
- Absorbance/Intensity column (automatically detected as "Counts" or similar)

## 🔧 Build Executable (Windows)

```bash
pip install pyinstaller
pyinstaller SpectralAnalysis.spec
```

The executable will be in `dist/SpectralAnalysis/SpectralAnalysis.exe`

## 📋 Key Features

- **Multiple ML Models**: PLSR, Ridge, Lasso, Random Forest, XGBoost, Neural Networks (PyTorch), SVR
- **Auto Optimization**: Bayesian optimization finds the best hyperparameters automatically
- **Visualizations**: Interactive plots for comparing models and analyzing predictions
- **Export Everything**: Save models, metrics, and results in various formats

## 📁 Project Structure

```
thesis_webapp/
├── Main.py              # Streamlit web app entry point
├── qt_app/              # Desktop Qt application
│   ├── main_qt.py      # Qt entry point
│   └── views/          # UI components
├── pages/               # Streamlit pages
├── calibration/         # Core ML framework
│   ├── core/           # Base classes
│   ├── data/           # Data loading & preprocessing
│   ├── models/         # ML model implementations
│   └── utils/          # Metrics, export, optimization
├── sample_data/         # Example spectral files
└── requirements.txt     # Dependencies
```

## 🐛 Common Issues

**Import errors?** Make sure all dependencies are installed: `pip install -r requirements.txt`

**Port already in use?** Run Streamlit on a different port: `streamlit run Main.py --server.port 8502`

**Model training fails?** Check that:
- You have at least 3 samples (for cross-validation)
- Concentration values are in filenames or entered manually
- All spectra use the same wavelength grid

## 📄 License

This project was developed for research and educational purposes as part of undergraduate thesis work.

---

**Note**: Build artifacts are excluded from the repository. Executables should be distributed via GitHub Releases.
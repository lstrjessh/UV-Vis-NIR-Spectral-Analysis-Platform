from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QHBoxLayout, 
    QCheckBox, QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem, 
    QComboBox, QGroupBox, QGridLayout, QScrollArea, QTabWidget, QProgressDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPalette
from calibration.data.loader import CSVDataLoader
from calibration.core.data_structures import CalibrationDataset
from calibration.models import ModelFactory, register_default_models
from calibration.core import ModelConfig
from utils.shared_utils import kennard_stone_split
import numpy as np
import pandas as pd
from calibration.data.preprocessor import StandardPreprocessor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class CalibrationView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(100, 20, 100, 20)
        layout.setSpacing(18)

        # Title
        title = QLabel("Model Calibration")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        subtitle = QLabel("Train machine learning models for spectral concentration prediction")
        subtitle.setStyleSheet("color: #444444; font-size: 14px; margin-bottom: 8px;")
        layout.addWidget(subtitle)

        # 1. Data Loader
        loader_group = QGroupBox("Step 1: Load Calibration Dataset")
        loader_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("📂 Select Dataset Files...")
        self.load_btn.setMinimumHeight(44)
        self.load_btn.setToolTip(
            "Load spectral calibration dataset files.\n\n"
            "📋 Supported: CSV, TXT, DAT files\n"
            "📊 Required: wavelength and absorbance columns\n"
            "🏷️ Concentration: Extracted from filename or entered manually\n"
            "💡 Example filename: 0.5_replicate1.csv (0.5 = concentration)"
        )
        self.load_btn.clicked.connect(self._load_files)
        loader_layout.addWidget(self.load_btn)
        
        self.load_status = QLabel("No dataset loaded")
        self.load_status.setStyleSheet("color: #555555; font-size: 13px;")
        loader_layout.addWidget(self.load_status)
        loader_layout.addStretch()
        
        loader_group.setLayout(loader_layout)
        layout.addWidget(loader_group)

        # 2. Preprocessing
        prep_group = QGroupBox("Step 2: Preprocessing Configuration")
        prep_layout = QGridLayout()
        prep_layout.setSpacing(12)
        
        # Smoothing
        self.cb_smooth = QCheckBox("Apply Smoothing")
        self.cb_smooth.setToolTip(
            "Savitzky-Golay smoothing reduces noise while preserving spectral features.\n\n"
            "📊 Uses polynomial fitting within a moving window\n"
            "🔬 Helps improve signal-to-noise ratio\n"
            "⚠️ Can remove important features if over-applied"
        )
        self.cb_smooth.stateChanged.connect(self._render_previews)
        prep_layout.addWidget(self.cb_smooth, 0, 0, 1, 2)
        
        window_label = QLabel("Window Size:")
        window_label.setToolTip(
            "Must be odd and greater than polynomial order.\n\n"
            "⬆️ Larger windows = more smoothing\n"
            "💡 Recommended: 11"
        )
        prep_layout.addWidget(window_label, 1, 0)
        self.spin_window = QSpinBox()
        self.spin_window.setRange(3, 51)
        self.spin_window.setSingleStep(2)
        self.spin_window.setValue(11)
        self.spin_window.setToolTip("Window size for Savitzky-Golay smoothing (must be odd)")
        self.spin_window.valueChanged.connect(self._render_previews)
        prep_layout.addWidget(self.spin_window, 1, 1)
        
        poly_label = QLabel("Polynomial Order:")
        poly_label.setToolTip(
            "Higher orders preserve more spectral features.\n\n"
            "⚠️ Must be less than window size\n"
            "💡 Recommended: 2-3"
        )
        prep_layout.addWidget(poly_label, 2, 0)
        self.spin_poly = QSpinBox()
        self.spin_poly.setRange(1, 9)
        self.spin_poly.setValue(2)
        self.spin_poly.setToolTip("Polynomial order for smoothing (typically 2-3)")
        self.spin_poly.valueChanged.connect(self._render_previews)
        prep_layout.addWidget(self.spin_poly, 2, 1)
        
        deriv_label = QLabel("Derivative:")
        deriv_label.setToolTip(
            "Spectral derivatives enhance resolution and remove baseline effects.\n\n"
            "📊 1st derivative: Removes constant baseline\n"
            "📉 2nd derivative: Removes linear baseline\n"
            "⚠️ Amplifies noise - use with smoothing"
        )
        prep_layout.addWidget(deriv_label, 3, 0)
        self.combo_deriv = QComboBox()
        self.combo_deriv.addItems(["None", "1st Derivative", "2nd Derivative"])
        self.combo_deriv.setToolTip("Apply spectral derivative transformation")
        self.combo_deriv.currentTextChanged.connect(self._render_previews)
        prep_layout.addWidget(self.combo_deriv, 3, 1)
        
        self.cb_baseline = QCheckBox("Baseline Correction")
        self.cb_baseline.setToolTip(
            "Asymmetric Least Squares (ALS) baseline correction.\n\n"
            "📊 Removes systematic baseline drift\n"
            "🔬 Corrects fluorescence effects\n"
            "💡 Useful for noisy or drifting baselines"
        )
        self.cb_baseline.stateChanged.connect(self._render_previews)
        prep_layout.addWidget(self.cb_baseline, 4, 0, 1, 2)
        
        prep_layout.setColumnStretch(1, 1)
        prep_group.setLayout(prep_layout)
        layout.addWidget(prep_group)

        # 3. Preview
        preview_group = QGroupBox("Step 3: Spectral Preview")
        preview_layout = QHBoxLayout()
        preview_layout.setSpacing(16)
        
        self.fig_orig = FigureCanvas(Figure(figsize=(5, 3), facecolor='none'))
        self.fig_orig.setMinimumHeight(250)
        self.fig_prep = FigureCanvas(Figure(figsize=(5, 3), facecolor='none'))
        self.fig_prep.setMinimumHeight(250)
        
        preview_layout.addWidget(self.fig_orig)
        preview_layout.addWidget(self.fig_prep)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # 4. Model Selection
        model_group = QGroupBox("Step 4: Select Models to Train")
        model_grid = QGridLayout()
        model_grid.setSpacing(8)
        
        # Linear Models
        linear_label = QLabel("Linear Models:")
        linear_label.setStyleSheet("font-weight: 600; font-size: 14px;")
        linear_label.setToolTip("Linear regression models with regularization")
        model_grid.addWidget(linear_label, 0, 0, 1, 4)
        
        self.cb_plsr = QCheckBox("PLSR")
        self.cb_plsr.setChecked(True)
        self.cb_plsr.setToolTip(
            "Partial Least Squares Regression\n\n"
            "📊 Finds latent variables maximizing covariance\n"
            "🔬 Excellent for multicollinearity\n"
            "⭐ Best for high-dimensional spectral data\n"
            "✅ Handles correlated features well"
        )
        model_grid.addWidget(self.cb_plsr, 1, 0)
        
        self.cb_ridge = QCheckBox("Ridge Regression")
        self.cb_ridge.setToolTip(
            "Ridge Regression (L2 Regularization)\n\n"
            "📊 Shrinks coefficients to prevent overfitting\n"
            "🔬 Robust to multicollinearity\n"
            "⚖️ Balances bias-variance tradeoff\n"
            "💡 Good for spectral data with noise"
        )
        model_grid.addWidget(self.cb_ridge, 1, 1)
        
        self.cb_lasso = QCheckBox("Lasso")
        self.cb_lasso.setToolTip(
            "Lasso Regression (L1 Regularization)\n\n"
            "📊 Performs automatic feature selection\n"
            "🎯 Sets some coefficients to zero\n"
            "🔍 Identifies important wavelengths\n"
            "💡 Use when features are sparse"
        )
        model_grid.addWidget(self.cb_lasso, 1, 2)
        
        self.cb_elastic = QCheckBox("ElasticNet")
        self.cb_elastic.setToolTip(
            "Elastic Net (L1 + L2 Regularization)\n\n"
            "📊 Combines Lasso and Ridge benefits\n"
            "🎯 Feature selection + shrinkage\n"
            "⭐ Optimal for correlated features\n"
            "💡 Best of both regularization worlds"
        )
        model_grid.addWidget(self.cb_elastic, 1, 3)
        
        # Ensemble Models
        ensemble_label = QLabel("Ensemble Models (Tree-Based):")
        ensemble_label.setStyleSheet("font-weight: 600; font-size: 14px; margin-top: 8px;")
        ensemble_label.setToolTip("Tree-based ensemble methods (no scaling needed)")
        model_grid.addWidget(ensemble_label, 2, 0, 1, 4)
        
        self.cb_rf = QCheckBox("Random Forest")
        self.cb_rf.setToolTip(
            "Random Forest Regression\n\n"
            "🌲 Ensemble of decision trees\n"
            "📊 Non-parametric, handles non-linearity\n"
            "🎯 Provides feature importance\n"
            "⚠️ No scaling required (tree-based)\n"
            "💡 Robust to outliers and noise"
        )
        model_grid.addWidget(self.cb_rf, 3, 0)
        
        self.cb_xgb = QCheckBox("XGBoost")
        self.cb_xgb.setToolTip(
            "Extreme Gradient Boosting\n\n"
            "🚀 Advanced gradient boosting framework\n"
            "⚡ Fast with parallel processing\n"
            "🎯 Built-in regularization\n"
            "⚠️ No scaling, uses default parameters\n"
            "⭐ Often best for complex patterns\n"
            "💡 Excellent for high-dimensional data"
        )
        model_grid.addWidget(self.cb_xgb, 3, 1)
        
        # Neural & Kernel
        other_label = QLabel("Neural & Kernel Models:")
        other_label.setStyleSheet("font-weight: 600; font-size: 14px; margin-top: 8px;")
        other_label.setToolTip("Advanced non-linear models")
        model_grid.addWidget(other_label, 4, 0, 1, 4)
        
        self.cb_mlp = QCheckBox("MLP (Neural Network)")
        self.cb_mlp.setToolTip(
            "Multi-Layer Perceptron Neural Network\n\n"
            "🧠 Feedforward neural network\n"
            "📊 Captures complex non-linear patterns\n"
            "🔄 Learns through backpropagation\n"
            "⚠️ Requires more data than linear models\n"
            "💡 Powerful but can overfit"
        )
        model_grid.addWidget(self.cb_mlp, 5, 0)
        
        self.cb_svr = QCheckBox("SVR (Support Vector)")
        self.cb_svr.setToolTip(
            "Support Vector Regression\n\n"
            "🎯 Kernel-based non-linear regression\n"
            "📊 Optimal hyperplane in feature space\n"
            "🔬 Excellent for non-linear relationships\n"
            "⚠️ Can be slow with large datasets\n"
            "💡 Works well with RBF kernel"
        )
        model_grid.addWidget(self.cb_svr, 5, 1)
        
        model_grid.setColumnStretch(4, 1)
        model_group.setLayout(model_grid)
        layout.addWidget(model_group)

        # 5. Training Configuration
        train_group = QGroupBox("Step 5: Training Parameters")
        train_layout = QGridLayout()
        train_layout.setSpacing(12)
        
        split_label = QLabel("Split Method:")
        split_label.setToolTip(
            "Data splitting strategy for train/test sets.\n\n"
            "🎯 Kennard-Stone: Maximizes spectral diversity\n"
            "🎲 Random: Simple random sampling\n"
            "💡 Recommended: Kennard-Stone for spectral data"
        )
        train_layout.addWidget(split_label, 0, 0)
        self.combo_split = QComboBox()
        self.combo_split.addItems(["Kennard-Stone", "Random"])
        self.combo_split.setToolTip(
            "Kennard-Stone selects representative samples for better model generalization"
        )
        train_layout.addWidget(self.combo_split, 0, 1)
        
        ratio_label = QLabel("Train Split Ratio:")
        ratio_label.setToolTip(
            "Proportion of data for training vs validation.\n\n"
            "⬆️ Higher = more training data\n"
            "⬇️ Lower = more validation data\n"
            "💡 Recommended: 0.6-0.8"
        )
        train_layout.addWidget(ratio_label, 0, 2)
        self.split_train = QDoubleSpinBox()
        self.split_train.setRange(0.5, 0.95)
        self.split_train.setSingleStep(0.05)
        self.split_train.setValue(0.7)
        self.split_train.setToolTip("Fraction of data used for training (0.7 = 70% train, 30% test)")
        train_layout.addWidget(self.split_train, 0, 3)
        
        cv_label = QLabel("CV Folds:")
        cv_label.setToolTip(
            "Cross-validation folds for model evaluation.\n\n"
            "📊 Higher = more robust estimates\n"
            "⚠️ Higher = longer computation time\n"
            "💡 Recommended: 5-7 for spectral data"
        )
        train_layout.addWidget(cv_label, 1, 0)
        self.spin_cv = QSpinBox()
        self.spin_cv.setRange(2, 10)
        self.spin_cv.setValue(5)
        self.spin_cv.setToolTip("Number of folds for k-fold cross-validation")
        train_layout.addWidget(self.spin_cv, 1, 1)
        
        opt_label = QLabel("Optimization:")
        opt_label.setToolTip(
            "Hyperparameter optimization strategy.\n\n"
            "🧠 Bayesian: Smart search using Gaussian Process\n"
            "🎲 Random: Random sampling of parameters\n"
            "🔍 Grid: Exhaustive search (slowest)\n"
            "💡 Recommended: Bayesian for best results"
        )
        train_layout.addWidget(opt_label, 1, 2)
        self.combo_opt = QComboBox()
        self.combo_opt.addItems(["bayesian", "random_search", "grid_search"])
        self.combo_opt.setToolTip("Bayesian optimization is most efficient for finding optimal parameters")
        train_layout.addWidget(self.combo_opt, 1, 3)
        
        trials_label = QLabel("Optimization Trials:")
        trials_label.setToolTip(
            "Number of optimization iterations.\n\n"
            "⬆️ More trials = better parameters\n"
            "⬆️ More trials = longer computation\n"
            "🔍 Grid search ignores this setting\n"
            "💡 Recommended: 30-50"
        )
        train_layout.addWidget(trials_label, 2, 0)
        self.spin_trials = QSpinBox()
        self.spin_trials.setRange(10, 200)
        self.spin_trials.setSingleStep(10)
        self.spin_trials.setValue(30)
        self.spin_trials.setToolTip("Number of hyperparameter trials (Bayesian/Random search only)")
        train_layout.addWidget(self.spin_trials, 2, 1)
        
        self.cb_earlystopping = QCheckBox("Enable Early Stopping")
        self.cb_earlystopping.setChecked(True)
        self.cb_earlystopping.setToolTip(
            "Stop optimization early if no improvement.\n\n"
            "⚡ Reduces computation time\n"
            "⚠️ May miss optimal parameters\n"
            "💡 Generally safe to enable"
        )
        train_layout.addWidget(self.cb_earlystopping, 2, 2, 1, 2)
        
        train_layout.setColumnStretch(4, 1)
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)

        # Train button and status
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
        self.train_btn = QPushButton("🚀 Train Selected Models")
        self.train_btn.setMinimumHeight(44)
        self.train_btn.setEnabled(False)
        self.train_btn.setToolTip(
            "Train all selected models with automatic hyperparameter optimization.\n\n"
            "⚙️ Preprocessing applied first\n"
            "🎯 Bayesian optimization for best parameters\n"
            "📊 Cross-validation for robust evaluation\n"
            "⏱️ May take several minutes depending on models and data size"
        )
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                font-size: 15px;
                font-weight: 600;
            }
            QPushButton:hover:enabled {
                background-color: #45a049;
            }
            QPushButton:pressed:enabled {
                background-color: #3d8b40;
            }
        """)
        self.train_btn.clicked.connect(self._train_models)
        button_layout.addWidget(self.train_btn)
        
        layout.addLayout(button_layout)

        self.status = QLabel("Load a dataset to begin")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet("""
            padding: 12px;
            border-radius: 6px;
            background: palette(midlight);
            font-size: 13px;
        """)
        layout.addWidget(self.status)

        # 6. Results
        results_group = QGroupBox("Training Results")
        results_layout = QVBoxLayout()
        
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "Model", "Train R²", "Test R²", "RMSE", "MAE", "Time (s)"
        ])
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(200)
        # (moved into Training Metrics tab below)
        
        self.export_btn = QPushButton("💾 Export Metrics to CSV...")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_metrics)
        # (moved into Training Metrics tab below)
        
        # Analysis tabs (Model Comparison, Predictions, Feature Importance, Export)
        # Placed inside results group to avoid layout squashing
        self.tabs = QTabWidget()
        # Training Metrics tab
        self.tab_metrics = QWidget()
        metrics_layout = QVBoxLayout(self.tab_metrics)
        metrics_layout.addWidget(self.table)
        metrics_layout.addWidget(self.export_btn)
        self.tabs.addTab(self.tab_metrics, "Training Metrics")

        # Model Comparison tab
        self.tab_compare = QWidget()
        compare_layout = QVBoxLayout(self.tab_compare)
        self.fig_compare = FigureCanvas(Figure(figsize=(8, 8), facecolor='none'))
        self.fig_compare.setMinimumHeight(600)
        compare_layout.addWidget(self.fig_compare)
        self.tabs.addTab(self.tab_compare, "Model Comparison")

        # Predictions tab
        self.tab_predict = QWidget()
        predict_layout = QVBoxLayout(self.tab_predict)
        # selector
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Model:"))
        self.combo_predict_model = QComboBox()
        selector_row.addWidget(self.combo_predict_model)
        selector_row.addStretch(1)
        predict_layout.addLayout(selector_row)
        # plots
        self.fig_pred_vs_actual = FigureCanvas(Figure(figsize=(5.5, 5.5), facecolor='none'))
        self.fig_pred_vs_actual.setMinimumHeight(420)
        self.fig_residuals = FigureCanvas(Figure(figsize=(5.5, 5.5), facecolor='none'))
        self.fig_residuals.setMinimumHeight(420)
        plots_row = QHBoxLayout()
        plots_row.addWidget(self.fig_pred_vs_actual)
        plots_row.addWidget(self.fig_residuals)
        predict_layout.addLayout(plots_row)
        self.tabs.addTab(self.tab_predict, "Predictions")

        # Feature Importance tab
        self.tab_importance = QWidget()
        importance_layout = QVBoxLayout(self.tab_importance)
        sel_imp_row = QHBoxLayout()
        sel_imp_row.addWidget(QLabel("Model:"))
        self.combo_importance_model = QComboBox()
        sel_imp_row.addWidget(self.combo_importance_model)
        sel_imp_row.addStretch(1)
        importance_layout.addLayout(sel_imp_row)
        self.fig_importance = FigureCanvas(Figure(figsize=(10, 5), facecolor='none'))
        self.fig_importance.setMinimumHeight(380)
        importance_layout.addWidget(self.fig_importance)
        self.tabs.addTab(self.tab_importance, "Feature Importance")

        # Export tab (models only)
        self.tab_export = QWidget()
        export_layout = QVBoxLayout(self.tab_export)
        selector_row_exp = QHBoxLayout()
        selector_row_exp.addWidget(QLabel("Model:"))
        self.combo_export_model = QComboBox()
        selector_row_exp.addWidget(self.combo_export_model)
        selector_row_exp.addStretch(1)
        export_layout.addLayout(selector_row_exp)
        self.btn_export_selected = QPushButton("📥 Export Selected Model")
        self.btn_export_selected.clicked.connect(self._export_selected_model)
        self.btn_export_all = QPushButton("📦 Export All Models (ZIP)")
        self.btn_export_all.clicked.connect(self._export_all_zip)
        export_layout.addWidget(self.btn_export_selected)
        export_layout.addWidget(self.btn_export_all)
        export_layout.addStretch(1)
        self.tabs.addTab(self.tab_export, "Export")

        # Best model label (under tabs)
        self.best_label = QLabel("")
        self.best_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.best_label.setStyleSheet("""
            padding: 10px; border-radius: 6px; background: #e3f2fd; font-size: 13px;
        """)

        # Add tabs and best label inside results group
        results_layout.addWidget(self.tabs)
        results_layout.addWidget(self.best_label)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        layout.addStretch()
        scroll.setWidget(content)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

        self._dataset = None
        self._train_results = None
        self._model_results = {}
        self._train_test_data = None
        self._initialize_previews()

    def _initialize_previews(self):
        for fig_canvas in [self.fig_orig, self.fig_prep]:
            # Theme: force light
            bg_color = '#ffffff'
            text_color = '#000000'
            
            fig_canvas.figure.clear()
            fig_canvas.figure.patch.set_facecolor('none')
            ax = fig_canvas.figure.add_subplot(111)
            ax.set_facecolor(bg_color)
            ax.text(0.5, 0.5, "No data loaded", ha='center', va='center', 
                   fontsize=12, color=text_color, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(text_color)
            ax.spines['left'].set_color(text_color)
            fig_canvas.figure.tight_layout()
            fig_canvas.draw_idle()

    def _load_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Dataset Files", "", 
            "Data Files (*.csv *.txt *.dat);;All Files (*)"
        )
        if not files:
            return
        
        self.status.setText("⏳ Loading dataset...")
        self.status.setStyleSheet("background: #fff3e0; color: #ef6c00; padding: 12px; border-radius: 6px;")
        
        loader = CSVDataLoader(cache_enabled=False)
        spectra = []
        errors = 0
        for p in files:
            try:
                spectra.append(loader.load_file(p))
            except Exception:
                errors += 1
        
        if not spectra:
            self.status.setText("❌ Failed to load any files")
            self.status.setStyleSheet("background: #ffebee; color: #c62828; padding: 12px; border-radius: 6px;")
            return
        
        self._dataset = CalibrationDataset(spectra=spectra, name="Loaded Dataset")
        summary = self._dataset.summary()
        
        msg = (
            f"✅ Loaded {len(self._dataset)} spectra • "
            f"{int(summary['n_wavelengths'])} wavelengths • "
            f"Range: {summary['wavelength_range'][0]:.0f}-{summary['wavelength_range'][1]:.0f} nm"
        )
        if errors:
            msg += f" • {errors} failed"
        
        self.load_status.setText(f"{len(self._dataset)} spectra loaded")
        self.load_status.setStyleSheet("color: palette(text); font-size: 13px; font-weight: 600;")
        self.status.setText(msg)
        self.status.setStyleSheet("background: #e8f5e9; color: #2e7d32; padding: 12px; border-radius: 6px;")
        self.train_btn.setEnabled(True)
        self._render_previews()

    def _get_preprocessor(self):
        return StandardPreprocessor(
            smoothing=self.cb_smooth.isChecked(),
            smoothing_window=int(self.spin_window.value()),
            smoothing_polyorder=int(self.spin_poly.value()),
            derivative={
                "None": None, 
                "1st Derivative": 1, 
                "2nd Derivative": 2
            }[self.combo_deriv.currentText()],
            baseline_correction=self.cb_baseline.isChecked(),
        )

    def _render_previews(self):
        if not self._dataset or not self._dataset.spectra:
            return
        
        # Theme: force light
        bg_color = '#ffffff'
        text_color = '#000000'
        
        # Original spectra
        self.fig_orig.figure.clear()
        self.fig_orig.figure.patch.set_facecolor('none')
        ax1 = self.fig_orig.figure.add_subplot(111)
        ax1.set_facecolor(bg_color)
        
        # Color palette for different samples
        colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0', 
                 '#00BCD4', '#FFEB3B', '#795548', '#E91E63', '#607D8B']
        
        n = min(10, len(self._dataset.spectra))
        for idx, s in enumerate(self._dataset.spectra[:n]):
            color = colors[idx % len(colors)]
            ax1.plot(s.wavelengths, s.absorbance, color=color, alpha=0.7, linewidth=1.5)
        
        ax1.set_title(f'Raw Spectra (showing {n} of {len(self._dataset)})', 
                     fontsize=12, fontweight='bold', color=text_color, pad=10)
        ax1.set_xlabel('Wavelength (nm)', fontsize=10, color=text_color)
        ax1.set_ylabel('Absorbance', fontsize=10, color=text_color)
        ax1.tick_params(colors=text_color, labelsize=9)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_color(text_color)
        ax1.spines['left'].set_color(text_color)
        ax1.grid(True, alpha=0.2, linestyle='--')
        self.fig_orig.figure.tight_layout()
        self.fig_orig.draw_idle()
        
        # Preprocessed spectra
        prep = self._get_preprocessor()
        prepped = prep.preprocess_dataset(self._dataset)
        
        self.fig_prep.figure.clear()
        self.fig_prep.figure.patch.set_facecolor('none')
        ax2 = self.fig_prep.figure.add_subplot(111)
        ax2.set_facecolor(bg_color)
        
        # Use same color palette for preprocessed spectra to match originals
        for idx, s in enumerate(prepped.spectra[:n]):
            color = colors[idx % len(colors)]
            ax2.plot(s.wavelengths, s.absorbance, color=color, alpha=0.7, linewidth=1.5)
        
        ax2.set_title(f'Preprocessed Spectra (showing {n})', 
                     fontsize=12, fontweight='bold', color=text_color, pad=10)
        ax2.set_xlabel('Wavelength (nm)', fontsize=10, color=text_color)
        ax2.set_ylabel('Processed Value', fontsize=10, color=text_color)
        ax2.tick_params(colors=text_color, labelsize=9)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_color(text_color)
        ax2.spines['left'].set_color(text_color)
        ax2.grid(True, alpha=0.2, linestyle='--')
        self.fig_prep.figure.tight_layout()
        self.fig_prep.draw_idle()

    def _selected_models(self):
        sel = []
        if self.cb_plsr.isChecked(): sel.append('plsr')
        if self.cb_ridge.isChecked(): sel.append('ridge')
        if self.cb_lasso.isChecked(): sel.append('lasso')
        if self.cb_elastic.isChecked(): sel.append('elastic_net')
        if self.cb_rf.isChecked(): sel.append('random_forest')
        if self.cb_xgb.isChecked(): sel.append('xgboost')
        if self.cb_svr.isChecked(): sel.append('svr')
        if self.cb_mlp.isChecked(): sel.append('mlp')
        return sel

    def _train_models(self) -> None:
        if not self._dataset:
            return
        
        models = self._selected_models()
        if not models:
            self.status.setText("⚠️ Please select at least one model to train")
            self.status.setStyleSheet("background: #fff3e0; color: #ef6c00; padding: 12px; border-radius: 6px;")
            return
        
        self.status.setText(f"⏳ Training {len(models)} model(s)...")
        self.status.setStyleSheet("background: #e3f2fd; color: #1565c0; padding: 12px; border-radius: 6px;")
        self.train_btn.setEnabled(False)
        
        # Preprocess
        preproc = self._get_preprocessor()
        dataset = preproc.preprocess_dataset(self._dataset)
        X, y, wavelengths = dataset.to_matrix()
        
        if len(y) == 0:
            self.status.setText("❌ No concentration data available in dataset")
            self.status.setStyleSheet("background: #ffebee; color: #c62828; padding: 12px; border-radius: 6px;")
            self.train_btn.setEnabled(True)
            return
        
        # Split data
        if self.combo_split.currentText().lower().startswith("kennard"):
            X_train, X_test, y_train, y_test = kennard_stone_split(
                X, y, train_size=float(self.split_train.value()), random_state=42
            )
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=float(self.split_train.value()), random_state=42
            )
        
        # Train models
        register_default_models()
        factory = ModelFactory()
        results = []
        self.table.setRowCount(0)

        # Progress dialog
        total_models = len(models)
        progress = QProgressDialog("Training models...", "Cancel", 0, total_models, self)
        progress.setWindowTitle("Training")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        
        for idx, name in enumerate(models):
            self.status.setText(f"⏳ Training {name.upper()} ({idx+1}/{len(models)})...")
            progress.setLabelText(f"Training {name.upper()} ({idx+1}/{total_models})")
            progress.setValue(idx)
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
            if progress.wasCanceled():
                self.status.setText("❌ Training cancelled")
                self.status.setStyleSheet("background: #ffebee; color: #c62828; padding: 12px; border-radius: 6px;")
                break
            try:
                config = ModelConfig(
                    name=name,
                    cv_folds=int(self.spin_cv.value()),
                    optimization_method=self.combo_opt.currentText(),
                    n_trials=int(self.spin_trials.value()),
                    early_stopping=self.cb_earlystopping.isChecked(),
                )
                model = factory.create(name, config)
                res = model.fit(X_train, y_train, scaler_type="standard")
                y_pred = model.predict(X_test)
                test_metrics = model.calculate_metrics(y_test, y_pred, X_test)
                res.test_metrics = test_metrics
                results.append((name, res))
                
                # Add to table
                r = self.table.rowCount()
                self.table.insertRow(r)
                self.table.setItem(r, 0, QTableWidgetItem(name.upper()))
                self.table.setItem(r, 1, QTableWidgetItem(f"{res.metrics.r2:.10f}"))
                self.table.setItem(r, 2, QTableWidgetItem(f"{test_metrics.r2:.10f}"))
                self.table.setItem(r, 3, QTableWidgetItem(f"{test_metrics.rmse:.10f}"))
                self.table.setItem(r, 4, QTableWidgetItem(f"{test_metrics.mae:.10f}"))
                self.table.setItem(r, 5, QTableWidgetItem(f"{res.metrics.training_time:.2f}"))
                
                # Color-code R² values
                r2_item = self.table.item(r, 2)
                if test_metrics.r2 >= 0.9:
                    r2_item.setForeground(Qt.GlobalColor.darkGreen)
                elif test_metrics.r2 >= 0.7:
                    r2_item.setForeground(Qt.GlobalColor.darkYellow)
                else:
                    r2_item.setForeground(Qt.GlobalColor.red)
                    
            except Exception as e:
                r = self.table.rowCount()
                self.table.insertRow(r)
                self.table.setItem(r, 0, QTableWidgetItem(name.upper()))
                error_item = QTableWidgetItem(f"ERROR: {str(e)[:30]}")
                error_item.setForeground(Qt.GlobalColor.red)
                self.table.setItem(r, 1, error_item)
        
        self.table.resizeColumnsToContents()
        self._train_results = results
        # Also store as dict and train/test data for analysis
        self._model_results = {name: res for name, res in results}
        self._train_test_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'wavelengths': wavelengths,
        }

        # Populate selectors
        self.combo_predict_model.clear()
        self.combo_predict_model.addItems(list(self._model_results.keys()))
        self.combo_predict_model.currentTextChanged.connect(self._render_predictions_tab)

        self.combo_importance_model.clear()
        # Only models that expose feature_importance
        importance_models = [n for n, r in self._model_results.items() if getattr(r, 'feature_importance', None)]
        self.combo_importance_model.addItems(importance_models)
        self.combo_importance_model.currentTextChanged.connect(self._render_importance_tab)

        self.combo_export_model.clear()
        self.combo_export_model.addItems(list(self._model_results.keys()))

        # Render tabs
        self._render_comparison_tab()
        self._render_predictions_tab()
        self._render_importance_tab()

        # Best model selection
        try:
            best_name, best_res = max(
                self._model_results.items(),
                key=lambda kv: (
                    getattr(kv[1], 'test_metrics').r2,
                    -getattr(kv[1], 'test_metrics').rmse,
                    -getattr(kv[1], 'test_metrics').mae,
                )
            )
            self.best_label.setText(
                f"🏆 Best Model: {best_name.upper()} • R²={best_res.test_metrics.r2:.3f} • "
                f"RMSE={best_res.test_metrics.rmse:.3f} • MAE={best_res.test_metrics.mae:.3f}"
            )
        except Exception:
            self.best_label.setText("")
        
        progress.setValue(total_models)
        progress.close()
        self.status.setText(f"✅ Training complete • {len(results)} model(s) trained successfully")
        self.status.setStyleSheet("background: #e8f5e9; color: #2e7d32; padding: 12px; border-radius: 6px;")
        self.train_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

    def _export_metrics(self) -> None:
        if not self._train_results:
            return
        
        rows = []
        for name, res in self._train_results:
            tm = res.test_metrics
            row = {
                'Model': name,
                'Train_R2': res.metrics.r2,
                'Test_R2': tm.r2,
                'RMSE': tm.rmse,
                'MAE': tm.mae,
                'Training_Time_s': res.metrics.training_time,
            }
            if res.metrics.cv_mean:
                row['CV_Mean_R2'] = res.metrics.cv_mean
                row['CV_Std_R2'] = res.metrics.cv_std
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Metrics", "calibration_results.csv", 
            "CSV Files (*.csv)"
        )
        if not path:
            return
        
        try:
            df.to_csv(path, index=False)
            self.status.setText(f"💾 Exported to: {path}")
            self.status.setStyleSheet("background: #e3f2fd; color: #1565c0; padding: 12px; border-radius: 6px;")
        except Exception as e:
            self.status.setText(f"❌ Export failed: {str(e)}")
            self.status.setStyleSheet("background: #ffebee; color: #c62828; padding: 12px; border-radius: 6px;")

    # ---------- Analysis Tab Renderers ----------
    def _apply_axes_theme(self, ax, title: str, xlabel: str = None, ylabel: str = None):
        bg_color = '#ffffff'
        text_color = '#000000'
        ax.set_facecolor(bg_color)
        ax.set_title(title, fontsize=12, fontweight='bold', color=text_color, pad=10)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=10, color=text_color)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=10, color=text_color)
        ax.tick_params(colors=text_color, labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(text_color)
        ax.spines['left'].set_color(text_color)
        ax.grid(True, alpha=0.2, linestyle='--')

    def _render_comparison_tab(self):
        if not self._model_results:
            self.fig_compare.figure.clear()
            self.fig_compare.draw_idle()
            return
        import numpy as np
        fig = self.fig_compare.figure
        fig.clear()
        axs = fig.subplots(2, 2)
        names = list(self._model_results.keys())
        train_r2 = [res.metrics.r2 for res in self._model_results.values()]
        test_r2 = [res.test_metrics.r2 for res in self._model_results.values()]
        rmse = [res.metrics.rmse for res in self._model_results.values()]
        mae = [res.metrics.mae for res in self._model_results.values()]
        times = [res.metrics.training_time for res in self._model_results.values()]
        x = np.arange(len(names))
        # R2
        ax = axs[0,0]
        width = 0.35
        ax.bar(x - width/2, train_r2, width, label='Train')
        ax.bar(x + width/2, test_r2, width, label='Test')
        ax.set_xticks(x, [n.upper() for n in names], rotation=45)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('right')
        ax.legend()
        self._apply_axes_theme(ax, 'R² Score')
        # RMSE
        ax = axs[0,1]
        ax.bar(x, rmse)
        ax.set_xticks(x, [n.upper() for n in names], rotation=45)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('right')
        self._apply_axes_theme(ax, 'RMSE')
        # MAE
        ax = axs[1,0]
        ax.bar(x, mae)
        ax.set_xticks(x, [n.upper() for n in names], rotation=45)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('right')
        self._apply_axes_theme(ax, 'MAE')
        # Time
        ax = axs[1,1]
        ax.bar(x, times)
        ax.set_xticks(x, [n.upper() for n in names], rotation=45)
        for lbl in ax.get_xticklabels():
            lbl.set_ha('right')
        self._apply_axes_theme(ax, 'Training Time (s)')
        # Increase vertical spacing and bottom margin so bottom-row tick labels have room
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.45, bottom=0.22)
        self.fig_compare.draw_idle()

    def _render_predictions_tab(self):
        if not self._model_results or not self._train_test_data:
            self.fig_pred_vs_actual.figure.clear()
            self.fig_residuals.figure.clear()
            self.fig_pred_vs_actual.draw_idle()
            self.fig_residuals.draw_idle()
            return
        name = self.combo_predict_model.currentText()
        if not name:
            return
        res = self._model_results.get(name)
        if not res:
            return
        X_test = self._train_test_data['X_test']
        y_test = self._train_test_data['y_test']
        try:
            y_pred = res.model.predict(X_test)
        except Exception:
            y_pred = y_test
        # Pred vs Actual
        fig1 = self.fig_pred_vs_actual.figure
        fig1.clear()
        ax1 = fig1.add_subplot(111)
        ax1.scatter(y_test, y_pred, s=20)
        min_val = float(min(y_test.min(), y_pred.min()))
        max_val = float(max(y_test.max(), y_pred.max()))
        ax1.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red')
        self._apply_axes_theme(ax1, 'Predicted vs Actual', 'Actual', 'Predicted')
        fig1.tight_layout()
        self.fig_pred_vs_actual.draw_idle()
        # Residuals
        fig2 = self.fig_residuals.figure
        fig2.clear()
        ax2 = fig2.add_subplot(111)
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, s=18)
        ax2.axhline(0.0, linestyle='--', color='gray')
        self._apply_axes_theme(ax2, 'Residuals', 'Predicted', 'Residual')
        fig2.tight_layout()
        self.fig_residuals.draw_idle()

    def _render_importance_tab(self):
        if not self._model_results or not self._train_test_data:
            self.fig_importance.figure.clear()
            self.fig_importance.draw_idle()
            return
        name = self.combo_importance_model.currentText()
        if not name:
            self.fig_importance.figure.clear()
            self.fig_importance.draw_idle()
            return
        res = self._model_results.get(name)
        if not res or not getattr(res, 'feature_importance', None):
            self.fig_importance.figure.clear()
            self.fig_importance.draw_idle()
            return
        wavelengths = self._train_test_data['wavelengths']
        importance_values = []
        for i in range(len(wavelengths)):
            key = f'feature_{i}'
            importance_values.append(res.feature_importance.get(key, 0))
        fig = self.fig_importance.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(wavelengths, importance_values, linewidth=1.2)
        self._apply_axes_theme(ax, f'Feature Importance - {name.upper()}', 'Wavelength (nm)', 'Importance')
        fig.tight_layout()
        self.fig_importance.draw_idle()

    # ---------- Export helpers ----------
    def _export_selected_model(self) -> None:
        if not self._model_results:
            return
        name = self.combo_export_model.currentText()
        if not name:
            return
        res = self._model_results.get(name)
        if not res:
            return
        # Choose file path
        default_name = f"{name}.pkl" if not hasattr(res.model, 'state_dict') else f"{name}.pth"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Model", default_name, "Model Files (*.pkl *.pth)"
        )
        if not path:
            return
        try:
            if hasattr(res.model, 'state_dict'):
                import torch
                torch.save({
                    'model_state_dict': res.model.state_dict(),
                    'model_class': res.model.__class__.__name__,
                    'scaler_mean': getattr(getattr(res.model, 'scaler', None), 'mean_', None),
                    'scaler_scale': getattr(getattr(res.model, 'scaler', None), 'scale_', None),
                }, path)
            else:
                import pickle
                with open(path, 'wb') as f:
                    f.write(pickle.dumps(res.model))
            self.status.setText(f"💾 Model exported to: {path}")
            self.status.setStyleSheet("background: #e3f2fd; color: #1565c0; padding: 12px; border-radius: 6px;")
        except Exception as e:
            self.status.setText(f"❌ Model export failed: {e}")
            self.status.setStyleSheet("background: #ffebee; color: #c62828; padding: 12px; border-radius: 6px;")

    def _export_all_zip(self) -> None:
        if not self._model_results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save All Models", "calibration_models.zip", "ZIP Files (*.zip)"
        )
        if not path:
            return
        import io, zipfile, pickle
        try:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
                # metrics.csv
                import csv
                metrics_io = io.StringIO()
                writer = csv.writer(metrics_io)
                writer.writerow(["Model","Train_R2","Test_R2","RMSE","MAE","Training_Time"])
                for name, res in self._model_results.items():
                    writer.writerow([
                        name,
                        f"{res.metrics.r2:.6f}",
                        f"{res.test_metrics.r2:.6f}" if hasattr(res, 'test_metrics') else "",
                        f"{res.metrics.rmse:.6f}",
                        f"{res.metrics.mae:.6f}",
                        f"{res.metrics.training_time:.2f}",
                    ])
                z.writestr("metrics.csv", metrics_io.getvalue())
                # models
                for name, res in self._model_results.items():
                    try:
                        if hasattr(res.model, 'state_dict'):
                            import torch
                            model_buf = io.BytesIO()
                            torch.save({
                                'model_state_dict': res.model.state_dict(),
                                'model_class': res.model.__class__.__name__,
                                'scaler_mean': getattr(getattr(res.model, 'scaler', None), 'mean_', None),
                                'scaler_scale': getattr(getattr(res.model, 'scaler', None), 'scale_', None),
                            }, model_buf)
                            z.writestr(f"models/{name}.pth", model_buf.getvalue())
                        else:
                            model_bytes = pickle.dumps(res.model)
                            z.writestr(f"models/{name}.pkl", model_bytes)
                    except Exception as e:
                        z.writestr(f"models/{name}.ERROR.txt", str(e))
            with open(path, 'wb') as f:
                f.write(buf.getvalue())
            self.status.setText(f"💾 All models exported to: {path}")
            self.status.setStyleSheet("background: #e3f2fd; color: #1565c0; padding: 12px; border-radius: 6px;")
        except Exception as e:
            self.status.setText(f"❌ ZIP export failed: {e}")
            self.status.setStyleSheet("background: #ffebee; color: #c62828; padding: 12px; border-radius: 6px;")
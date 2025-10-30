from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QHBoxLayout, QCheckBox, QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem, QComboBox, QGroupBox, QFrame
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
        layout = QVBoxLayout()
        layout.addWidget(QLabel("<b>🔬 Model Calibration</b>", parent=self))

        # Section 1: Data Loader
        loader_box = QGroupBox("1. Load Spectral Dataset")
        loader_layout = QHBoxLayout()
        self.load_btn = QPushButton("Select Dataset Files…")
        self.load_btn.clicked.connect(self._load_files)
        loader_layout.addWidget(self.load_btn)
        loader_box.setLayout(loader_layout)
        layout.addWidget(loader_box)

        # Section 2: Preprocessing
        prep_box = QGroupBox("2. Preprocessing Options")
        prep_layout = QHBoxLayout()
        self.cb_smooth = QCheckBox("Smoothing")
        self.spin_window = QSpinBox(); self.spin_window.setRange(3, 51); self.spin_window.setSingleStep(2); self.spin_window.setValue(11)
        self.spin_poly = QSpinBox(); self.spin_poly.setRange(1, 9); self.spin_poly.setValue(2)
        self.combo_deriv = QComboBox(); self.combo_deriv.addItems(["None", "1st Deriv", "2nd Deriv"])
        self.cb_baseline = QCheckBox("Baseline Corr.")
        prep_layout.addWidget(self.cb_smooth); prep_layout.addWidget(QLabel("Win")); prep_layout.addWidget(self.spin_window)
        prep_layout.addWidget(QLabel("Poly")); prep_layout.addWidget(self.spin_poly)
        prep_layout.addWidget(QLabel("Deriv.")); prep_layout.addWidget(self.combo_deriv)
        prep_layout.addWidget(self.cb_baseline)
        prep_box.setLayout(prep_layout)
        layout.addWidget(prep_box)

        # Section 3: Quick Preview
        prev_box = QGroupBox("3. Quick Spectral Preview (before / after preprocessing)")
        prev_layout = QHBoxLayout()
        self.fig_orig = FigureCanvas(Figure(figsize=(3, 2)))
        self.fig_prep = FigureCanvas(Figure(figsize=(3, 2)))
        prev_layout.addWidget(self.fig_orig)
        prev_layout.addWidget(self.fig_prep)
        prev_box.setLayout(prev_layout)
        layout.addWidget(prev_box)

        # Section 4: Model selection (compact grid by family)
        from PyQt6.QtWidgets import QGridLayout
        model_box = QGroupBox()
        model_box.setTitle("4. Choose Models to Train")
        model_box.setStyleSheet("QGroupBox {font-weight:bold; font-size:15px; border: 1px solid #ccc; margin-top: 8px; background: #f8f8fa; padding:7px 7px 5px 7px;}")
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(4)
        row = 0
        # Linear
        grid.addWidget(QLabel("<b>Linear</b>"), row, 0)
        self.cb_plsr = QCheckBox("PLSR"); self.cb_plsr.setChecked(True); grid.addWidget(self.cb_plsr, row, 1)
        self.cb_ridge = QCheckBox("Ridge"); grid.addWidget(self.cb_ridge, row, 2)
        self.cb_lasso = QCheckBox("Lasso"); grid.addWidget(self.cb_lasso, row, 3)
        self.cb_elastic = QCheckBox("ElasticNet"); grid.addWidget(self.cb_elastic, row, 4)
        row += 1
        # Ensemble
        grid.addWidget(QLabel("<b>Ensemble</b>"), row, 0)
        self.cb_rf = QCheckBox("RandomForest"); grid.addWidget(self.cb_rf, row, 1)
        self.cb_xgb = QCheckBox("XGBoost"); grid.addWidget(self.cb_xgb, row, 2)
        row += 1
        # Neural
        grid.addWidget(QLabel("<b>Neural</b>"), row, 0)
        self.cb_mlp = QCheckBox("MLP"); grid.addWidget(self.cb_mlp, row, 1)
        row += 1
        # Kernel
        grid.addWidget(QLabel("<b>Kernel</b>"), row, 0)
        self.cb_svr = QCheckBox("SVR"); grid.addWidget(self.cb_svr, row, 1)
        grid.setColumnStretch(5, 1)
        model_box.setLayout(grid)
        layout.addWidget(model_box)

        # Section 5: Training controls
        train_config_box = QGroupBox("5. Training Configuration")
        train_layout = QHBoxLayout()
        # Split method
        train_layout.addWidget(QLabel("Split Method:"))
        self.combo_split = QComboBox(); self.combo_split.addItems(["Kennard-Stone", "Random"])
        train_layout.addWidget(self.combo_split)
        # CV
        train_layout.addWidget(QLabel("CV folds:")); self.spin_cv = QSpinBox(); self.spin_cv.setRange(2, 10); self.spin_cv.setValue(5)
        train_layout.addWidget(self.spin_cv)
        # Optimization method
        train_layout.addWidget(QLabel("Optimization:"))
        self.combo_opt = QComboBox(); self.combo_opt.addItems(["bayesian", "random_search", "grid_search"])
        train_layout.addWidget(self.combo_opt)
        # n_trials
        train_layout.addWidget(QLabel("Trials:")); self.spin_trials = QSpinBox(); self.spin_trials.setRange(10, 200); self.spin_trials.setSingleStep(10); self.spin_trials.setValue(30)
        # Early stopping
        train_layout.addWidget(self.spin_trials)
        self.cb_earlystopping = QCheckBox("Early Stopping"); self.cb_earlystopping.setChecked(True)
        train_layout.addWidget(self.cb_earlystopping)
        # Train split ratio
        train_layout.addWidget(QLabel("Train split:"))
        self.split_train = QDoubleSpinBox(); self.split_train.setRange(0.5, 0.95); self.split_train.setSingleStep(0.05); self.split_train.setValue(0.6)
        train_layout.addWidget(self.split_train)
        train_config_box.setLayout(train_layout)
        layout.addWidget(train_config_box)

        # Button and status
        self.train_btn = QPushButton("Train Selected Models")
        self.train_btn.clicked.connect(self._train_models)
        layout.addWidget(self.train_btn)

        self.status = QLabel("<i>No dataset loaded</i>")
        layout.addWidget(self.status)

        # Section 6: Results table
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Model", "Train R2", "Test R2", "RMSE", "MAE", "Time (s)"])
        layout.addWidget(self.table)
        self.export_btn = QPushButton("Export Metrics CSV…")
        self.export_btn.clicked.connect(self._export_metrics)
        layout.addWidget(self.export_btn)

        self.setLayout(layout)
        self._dataset = None
        self._train_results = None

    def _load_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Select Dataset Files", "", "Data Files (*.csv *.txt *.dat)")
        if not files:
            return
        loader = CSVDataLoader(cache_enabled=False)
        spectra = []
        errors = 0
        for p in files:
            try:
                spectra.append(loader.load_file(p))
            except Exception:
                errors += 1
        if not spectra:
            self.status.setText("Failed to load any files.")
            return
        self._dataset = CalibrationDataset(spectra=spectra, name="Loaded Dataset")
        summary = self._dataset.summary()
        msg = (
            f"Loaded {len(self._dataset)} spectra"
            f" • Wavelengths: {int(summary['n_wavelengths'])}"
            f" • Range: {summary['wavelength_range'][0]:.0f}-{summary['wavelength_range'][1]:.0f} nm"
        )
        if errors:
            msg += f" • {errors} file(s) failed"
        self.status.setText(msg)
        self._render_previews()

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

    def _get_preprocessor(self):
        return StandardPreprocessor(
            smoothing=self.cb_smooth.isChecked(),
            smoothing_window=int(self.spin_window.value()),
            smoothing_polyorder=int(self.spin_poly.value()),
            derivative={"None": None, "1st Deriv": 1, "2nd Deriv": 2}[self.combo_deriv.currentText()],
            baseline_correction=self.cb_baseline.isChecked(),
        )

    def _render_previews(self):
        # orig
        ax = self.fig_orig.figure.subplots(); ax.clear()
        prep = self._get_preprocessor()
        if not self._dataset or not self._dataset.spectra:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            self.fig_orig.draw_idle()
            self.fig_prep.figure.clear(); self.fig_prep.draw_idle(); return
        n = min(10, len(self._dataset.spectra))
        for s in self._dataset.spectra[:n]:
            ax.plot(s.wavelengths, s.absorbance, color='C0', alpha=0.7)
        ax.set_title("Raw Spectra (up to 10)", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        self.fig_orig.draw_idle()
        # preprocessed
        ax2 = self.fig_prep.figure.subplots(); ax2.clear()
        prepped = prep.preprocess_dataset(self._dataset)
        for s in prepped.spectra[:n]:
            ax2.plot(s.wavelengths, s.absorbance, color='C3', alpha=0.7)
        ax2.set_title("Preprocessed", fontsize=11)
        ax2.set_xticks([]); ax2.set_yticks([])
        self.fig_prep.draw_idle()

    def _train_models(self) -> None:
        if not self._dataset:
            self.status.setText("Load a dataset first.")
            return
        models = self._selected_models()
        if not models:
            self.status.setText("Select at least one model.")
            return
        # Preprocess data
        preproc = self._get_preprocessor()
        dataset = preproc.preprocess_dataset(self._dataset)
        X, y, wavelengths = dataset.to_matrix()
        if len(y) == 0:
            self.status.setText("No concentration data available.")
            return
        # Split
        if self.combo_split.currentText().lower().startswith("kennard"):
            X_train, X_test, y_train, y_test = kennard_stone_split(
                X, y, train_size=float(self.split_train.value()), random_state=42
            )
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=float(self.split_train.value()), random_state=42
            )
        register_default_models()
        factory = ModelFactory()
        results = []
        self.table.setRowCount(0)
        for name in models:
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
                r = self.table.rowCount(); self.table.insertRow(r)
                self.table.setItem(r, 0, QTableWidgetItem(name.upper()))
                self.table.setItem(r, 1, QTableWidgetItem(f"{res.metrics.r2:.6f}"))
                self.table.setItem(r, 2, QTableWidgetItem(f"{test_metrics.r2:.6f}"))
                self.table.setItem(r, 3, QTableWidgetItem(f"{test_metrics.rmse:.6f}"))
                self.table.setItem(r, 4, QTableWidgetItem(f"{test_metrics.mae:.6f}"))
                self.table.setItem(r, 5, QTableWidgetItem(f"{res.metrics.training_time:.2f}"))
            except Exception as e:
                r = self.table.rowCount(); self.table.insertRow(r)
                self.table.setItem(r, 0, QTableWidgetItem(name.upper()))
                self.table.setItem(r, 1, QTableWidgetItem("ERR"))
                self.table.setItem(r, 2, QTableWidgetItem(str(e)))
        self._train_results = results
        self.status.setText(f"Trained {len(results)} model(s)")

    def _export_metrics(self) -> None:
        if not self._train_results:
            self.status.setText("No results to export.")
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
                'Time_s': res.metrics.training_time,
            }
            if res.metrics.cv_mean:
                row['CV_Mean'] = res.metrics.cv_mean
                row['CV_Std'] = res.metrics.cv_std
            rows.append(row)
        df = pd.DataFrame(rows)
        path, _ = QFileDialog.getSaveFileName(self, "Save Metrics CSV", "calibration_results.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            df.to_csv(path, index=False)
            self.status.setText(f"Saved: {path}")
        except Exception as e:
            self.status.setText(f"Export failed: {e}")



from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QTableWidget, 
    QTableWidgetItem, QGroupBox, QHBoxLayout, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import pickle
import numpy as np
import pandas as pd
from calibration.data.loader import CSVDataLoader
from calibration.core.data_structures import CalibrationDataset


class PredictView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        
        # Main scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(100, 20, 100, 20)
        layout.setSpacing(18)
        
        # Title
        title = QLabel("Predict Concentration")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        subtitle = QLabel("Apply trained models to new spectral data for concentration predictions")
        subtitle.setStyleSheet("color: #444444; font-size: 14px; margin-bottom: 8px;")
        layout.addWidget(subtitle)

        # Step 1: Load Model
        model_group = QGroupBox("Step 1: Load Trained Model")
        model_layout = QHBoxLayout()
        
        self.model_btn = QPushButton("📂 Load Model File...")
        self.model_btn.setMinimumHeight(44)
        self.model_btn.setToolTip(
            "Load a previously trained calibration model.\n\n"
            "📋 Supported: .pkl (scikit-learn models)\n"
            "⚠️ .pth (PyTorch) not yet fully supported\n"
            "💡 Use models from Calibration tab"
        )
        self.model_btn.clicked.connect(self._pick_model)
        model_layout.addWidget(self.model_btn)
        
        self.model_status = QLabel("No model loaded")
        self.model_status.setStyleSheet("color: #555555; font-size: 13px;")
        model_layout.addWidget(self.model_status)
        model_layout.addStretch()
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Step 2: Load Data
        data_group = QGroupBox("Step 2: Select Input Spectra")
        data_layout = QHBoxLayout()
        
        self.data_btn = QPushButton("📊 Select Spectral Files...")
        self.data_btn.setMinimumHeight(44)
        self.data_btn.setToolTip(
            "Select spectral files to predict concentrations.\n\n"
            "📋 Supported: CSV, TXT, DAT files\n"
            "📊 Must match model's wavelength range\n"
            "🔢 Batch processing supported\n"
            "💡 Apply same preprocessing as training"
        )
        self.data_btn.clicked.connect(self._pick_data)
        data_layout.addWidget(self.data_btn)
        
        self.data_status = QLabel("No data loaded")
        self.data_status.setStyleSheet("color: #555555; font-size: 13px;")
        data_layout.addWidget(self.data_status)
        data_layout.addStretch()
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        # Status
        self.status = QLabel("Load model and data to begin predictions")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet("""
            padding: 12px;
            border-radius: 6px;
            background: palette(midlight);
            font-size: 13px;
        """)
        layout.addWidget(self.status)

        # Results
        results_group = QGroupBox("Step 3: Prediction Results")
        results_layout = QVBoxLayout()
        
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["File", "Predicted Concentration"])
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(300)
        results_layout.addWidget(self.table)

        self.export_btn = QPushButton("💾 Export Predictions to CSV...")
        self.export_btn.setMinimumHeight(40)
        self.export_btn.setToolTip(
            "Export prediction results to CSV file.\n\n"
            "📋 Format: filename, predicted concentration\n"
            "💾 Can be opened in Excel or other tools\n"
            "📊 Includes all batch predictions"
        )
        self.export_btn.clicked.connect(self._export_predictions)
        self.export_btn.setEnabled(False)
        results_layout.addWidget(self.export_btn)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        layout.addStretch(1)
        
        scroll.setWidget(content)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

        self.setLayout(main_layout)

    def _pick_model(self) -> None:
        file, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Models (*.pkl *.pth)")
        if file:
            self._model_path = file
            import os
            model_name = os.path.basename(file)
            self.model_status.setText(f"✅ Loaded: {model_name}")
            self.model_status.setStyleSheet("color: palette(text); font-size: 13px; font-weight: 600;")
            self.status.setText(f"📦 Model loaded: {model_name}")
            self.status.setStyleSheet("background: #e8f5e9; color: #2e7d32; padding: 12px; border-radius: 6px;")

    def _pick_data(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Select Spectra Files", "", "Data Files (*.csv *.txt *.dat)")
        if files:
            self._data_paths = files
            self.data_status.setText(f"✅ {len(files)} file(s) loaded")
            self.data_status.setStyleSheet("color: palette(text); font-size: 13px; font-weight: 600;")
            self.status.setText(f"📊 Processing {len(files)} spectra...")
            self.status.setStyleSheet("background: #e3f2fd; color: #1565c0; padding: 12px; border-radius: 6px;")
            self._run_predictions()

    def _run_predictions(self) -> None:
        if not hasattr(self, '_model_path') or not hasattr(self, '_data_paths'):
            return
        if self._model_path.endswith('.pth'):
            self.status.setText("⚠️ PyTorch .pth loading not yet supported in PyQt UI.")
            self.status.setStyleSheet("background: #fff3e0; color: #ef6c00; padding: 12px; border-radius: 6px;")
            return
        try:
            with open(self._model_path, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            self.status.setText(f"❌ Failed to load model: {e}")
            self.status.setStyleSheet("background: #ffebee; color: #c62828; padding: 12px; border-radius: 6px;")
            return
        # Load data
        loader = CSVDataLoader(cache_enabled=False)
        spectra = []
        for p in self._data_paths:
            try:
                spectra.append(loader.load_file(p))
            except Exception:
                continue
        if not spectra:
            self.status.setText("❌ Failed to load spectra files")
            self.status.setStyleSheet("background: #ffebee; color: #c62828; padding: 12px; border-radius: 6px;")
            return
        dataset = CalibrationDataset(spectra=spectra, name="Prediction Dataset")
        X, _, _ = dataset.to_matrix()
        try:
            y_pred = model.predict(X)
        except Exception as e:
            self.status.setText(f"❌ Prediction failed: {e}")
            self.status.setStyleSheet("background: #ffebee; color: #c62828; padding: 12px; border-radius: 6px;")
            return
        self.table.setRowCount(0)
        for s, y in zip(spectra, y_pred):
            r = self.table.rowCount(); self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(s.filename))
            self.table.setItem(r, 1, QTableWidgetItem(f"{float(y):.6f}"))
        self.table.resizeColumnsToContents()
        self._pred_rows = [(s.filename, float(y)) for s, y in zip(spectra, y_pred)]
        self.status.setText(f"✅ Successfully predicted {len(self._pred_rows)} sample(s)")
        self.status.setStyleSheet("background: #e8f5e9; color: #2e7d32; padding: 12px; border-radius: 6px;")
        self.export_btn.setEnabled(True)

    def _export_predictions(self) -> None:
        if not hasattr(self, '_pred_rows') or not self._pred_rows:
            self.status.setText("⚠️ No predictions to export")
            self.status.setStyleSheet("background: #fff3e0; color: #ef6c00; padding: 12px; border-radius: 6px;")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Predictions CSV", "predictions.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            df = pd.DataFrame(self._pred_rows, columns=["File", "Predicted_Concentration"])
            df.to_csv(path, index=False)
            self.status.setText(f"💾 Saved to: {path}")
            self.status.setStyleSheet("background: #e3f2fd; color: #1565c0; padding: 12px; border-radius: 6px;")
        except Exception as e:
            self.status.setText(f"❌ Export failed: {e}")
            self.status.setStyleSheet("background: #ffebee; color: #c62828; padding: 12px; border-radius: 6px;")



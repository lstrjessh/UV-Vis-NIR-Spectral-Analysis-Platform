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
        layout.setContentsMargins(60, 20, 60, 20)
        layout.setSpacing(20)
        
        # Title
        title = QLabel("🎯 Predict Concentration")
        title.setProperty("class", "title")
        layout.addWidget(title)

        subtitle = QLabel("Apply trained models to new spectral data for concentration predictions")
        subtitle.setProperty("class", "subtitle")
        layout.addWidget(subtitle)

        # Two-column layout
        main_content = QHBoxLayout()
        main_content.setSpacing(20)
        
        # LEFT COLUMN - Model & Data Loading
        left_column = QVBoxLayout()
        left_column.setSpacing(15)

        # Load Model
        model_group = QGroupBox("📦 Load Model")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(10)
        
        self.model_btn = QPushButton("📂 Select Model...")
        self.model_btn.setMinimumHeight(40)
        self.model_btn.setToolTip("Load trained calibration model (.pkl)")
        self.model_btn.clicked.connect(self._pick_model)
        model_layout.addWidget(self.model_btn)
        
        self.model_status = QLabel("No model loaded")
        self.model_status.setStyleSheet("color: #888; font-size: 12px;")
        model_layout.addWidget(self.model_status)
        
        model_group.setLayout(model_layout)
        left_column.addWidget(model_group)

        # Load Data
        data_group = QGroupBox("📊 Load Spectra")
        data_layout = QVBoxLayout()
        data_layout.setSpacing(10)
        
        self.data_btn = QPushButton("� Select Files...")
        self.data_btn.setMinimumHeight(40)
        self.data_btn.setToolTip("Select spectral files to predict")
        self.data_btn.clicked.connect(self._pick_data)
        data_layout.addWidget(self.data_btn)
        
        self.data_status = QLabel("No data loaded")
        self.data_status.setStyleSheet("color: #888; font-size: 12px;")
        data_layout.addWidget(self.data_status)
        
        data_group.setLayout(data_layout)
        left_column.addWidget(data_group)

        # Status
        self.status = QLabel("Load model and data to begin")
        self.status.setProperty("class", "info-box")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setWordWrap(True)
        left_column.addWidget(self.status)

        left_column.addStretch()
        
        # RIGHT COLUMN - Results
        right_column = QVBoxLayout()
        right_column.setSpacing(15)

        # Results table
        results_group = QGroupBox("📋 Prediction Results")
        results_layout = QVBoxLayout()
        
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["File", "Predicted Concentration"])
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(400)
        results_layout.addWidget(self.table)

        self.export_btn = QPushButton("💾 Export to CSV")
        self.export_btn.setMinimumHeight(40)
        self.export_btn.setToolTip("Save predictions to CSV file")
        self.export_btn.clicked.connect(self._export_predictions)
        self.export_btn.setEnabled(False)
        results_layout.addWidget(self.export_btn)
        
        results_group.setLayout(results_layout)
        right_column.addWidget(results_group)
        
        # Add columns to main content
        main_content.addLayout(left_column, stretch=1)
        main_content.addLayout(right_column, stretch=2)
        
        layout.addLayout(main_content)
        layout.addStretch()
        
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



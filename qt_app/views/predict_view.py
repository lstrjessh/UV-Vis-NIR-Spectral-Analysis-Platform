from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem
import pickle
import numpy as np
import pandas as pd
from calibration.data.loader import CSVDataLoader
from calibration.core.data_structures import CalibrationDataset


class PredictView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Predict Concentration"))

        self.model_btn = QPushButton("Load Trained Model…")
        self.model_btn.clicked.connect(self._pick_model)
        layout.addWidget(self.model_btn)

        self.data_btn = QPushButton("Select Input Spectra…")
        self.data_btn.clicked.connect(self._pick_data)
        layout.addWidget(self.data_btn)

        self.status = QLabel("No model or data selected")
        layout.addWidget(self.status)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["File", "Predicted Concentration"])
        layout.addWidget(self.table)

        self.export_btn = QPushButton("Export Predictions CSV…")
        self.export_btn.clicked.connect(self._export_predictions)
        layout.addWidget(self.export_btn)

        layout.addStretch(1)

        self.setLayout(layout)

    def _pick_model(self) -> None:
        file, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Models (*.pkl *.pth)")
        if file:
            self._model_path = file
            self.status.setText(f"Model: {file}")

    def _pick_data(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Select Spectra Files", "", "Data Files (*.csv *.txt *.dat)")
        if files:
            self._data_paths = files
            self.status.setText(f"Data files: {len(files)} selected")
            self._run_predictions()

    def _run_predictions(self) -> None:
        if not hasattr(self, '_model_path') or not hasattr(self, '_data_paths'):
            return
        if self._model_path.endswith('.pth'):
            self.status.setText("PyTorch .pth loading not yet supported in PyQt UI.")
            return
        try:
            with open(self._model_path, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            self.status.setText(f"Failed to load model: {e}")
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
            self.status.setText("Failed to load spectra.")
            return
        dataset = CalibrationDataset(spectra=spectra, name="Prediction Dataset")
        X, _, _ = dataset.to_matrix()
        try:
            y_pred = model.predict(X)
        except Exception as e:
            self.status.setText(f"Prediction failed: {e}")
            return
        self.table.setRowCount(0)
        for s, y in zip(spectra, y_pred):
            r = self.table.rowCount(); self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(s.filename))
            self.table.setItem(r, 1, QTableWidgetItem(f"{float(y):.6f}"))
        self._pred_rows = [(s.filename, float(y)) for s, y in zip(spectra, y_pred)]
        self.status.setText(f"Predicted {len(self._pred_rows)} sample(s)")

    def _export_predictions(self) -> None:
        if not hasattr(self, '_pred_rows') or not self._pred_rows:
            self.status.setText("No predictions to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Predictions CSV", "predictions.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            df = pd.DataFrame(self._pred_rows, columns=["File", "Predicted"])
            df.to_csv(path, index=False)
            self.status.setText(f"Saved: {path}")
        except Exception as e:
            self.status.setText(f"Export failed: {e}")



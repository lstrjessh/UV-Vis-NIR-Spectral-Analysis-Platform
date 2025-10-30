from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QHBoxLayout, QCheckBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from qt_app.spectral_processing import (
    read_spectral_file_from_path,
    find_peaks_absorbance,
)


class ViewerView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.paths = []

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Spectrum Viewer"))

        top = QHBoxLayout()
        btn = QPushButton("Select Spectrum Files…")
        btn.clicked.connect(self._pick_files)
        top.addWidget(btn)
        self.cb_normalize = QCheckBox("Normalize")
        self.cb_smooth = QCheckBox("Smooth")
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.1, 10.0)
        self.spin_sigma.setSingleStep(0.1)
        self.spin_sigma.setValue(1.0)
        self.cb_peaks = QCheckBox("Show peaks")
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(0.0, 10.0)
        self.spin_height.setSingleStep(0.001)
        self.spin_height.setValue(0.01)
        self.spin_distance = QDoubleSpinBox()
        self.spin_distance.setRange(1.0, 100.0)
        self.spin_distance.setSingleStep(0.5)
        self.spin_distance.setValue(5.0)
        self.spin_prom = QDoubleSpinBox()
        self.spin_prom.setRange(0.0, 10.0)
        self.spin_prom.setSingleStep(0.001)
        self.spin_prom.setValue(0.01)
        top.addWidget(self.cb_normalize)
        top.addWidget(self.cb_smooth)
        top.addWidget(QLabel("σ"))
        top.addWidget(self.spin_sigma)
        top.addWidget(self.cb_peaks)
        top.addWidget(QLabel("H"))
        top.addWidget(self.spin_height)
        top.addWidget(QLabel("D (nm)"))
        top.addWidget(self.spin_distance)
        top.addWidget(QLabel("Prom"))
        top.addWidget(self.spin_prom)
        layout.addLayout(top)

        self.status = QLabel("No files selected")
        layout.addWidget(self.status)

        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Spectrum", "Wavelength (nm)", "Value"])
        layout.addWidget(self.table)

        self.setLayout(layout)

    def _pick_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Select Spectrum Files", "", "Data Files (*.csv *.txt *.dat)")
        if files:
            self.paths = files
            self.status.setText(f"Loaded {len(files)} file(s)")
            self._render_plot()

    def _render_plot(self) -> None:
        if not self.paths:
            return
        # Load all spectra
        spectra = []
        for p in self.paths:
            df = read_spectral_file_from_path(p)
            if df is None:
                continue
            wl = df['Nanometers'].values
            y = df['Counts'].values
            spectra.append((p, wl, y))
        if not spectra:
            self.status.setText("Failed to parse files")
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.table.setRowCount(0)
        for idx, (name, wl, y) in enumerate(spectra):
            yy = y.copy()
            if self.cb_smooth.isChecked():
                try:
                    from scipy.ndimage import gaussian_filter1d
                    yy = gaussian_filter1d(yy, sigma=float(self.spin_sigma.value()))
                except Exception:
                    pass
            if self.cb_normalize.isChecked():
                ymin = float(np.min(yy))
                ymax = float(np.max(yy))
                if ymax - ymin > 0:
                    yy = (yy - ymin) / (ymax - ymin)
            ax.plot(wl, yy, label=name.split('/')[-1])
            if self.cb_peaks.isChecked():
                res = find_peaks_absorbance(
                    wl, yy,
                    height=float(self.spin_height.value()),
                    distance_nm=float(self.spin_distance.value()),
                    prominence=float(self.spin_prom.value()),
                )
                if res and len(res[0]) > 0:
                    pk_idx = res[0]
                    ax.scatter(wl[pk_idx], yy[pk_idx], marker='D', s=30)
                    # Append to table
                    for i in pk_idx:
                        r = self.table.rowCount()
                        self.table.insertRow(r)
                        self.table.setItem(r, 0, QTableWidgetItem(name.split('/')[-1]))
                        self.table.setItem(r, 1, QTableWidgetItem(f"{wl[i]:.1f}"))
                        self.table.setItem(r, 2, QTableWidgetItem(f"{yy[i]:.4f}"))
        ax.set_title('Spectrum Overlay Viewer')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Intensity')
        ax.legend(loc='upper right', fontsize='small')
        self.figure.tight_layout()
        self.canvas.draw_idle()



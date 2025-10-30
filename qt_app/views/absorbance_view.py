from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QHBoxLayout, QCheckBox, QSpinBox, QDoubleSpinBox, QGroupBox, QFrame, QToolTip
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from qt_app.spectral_processing import (
    read_spectral_file_from_path,
    average_counts_dataframes,
    calculate_absorbance,
    savgol_smooth,
    find_peaks_absorbance,
)


class AbsorbanceView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.ref_paths = []
        self.sample_paths = []
        self.dark_paths = []

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 20, 24, 16)
        layout.setSpacing(14)

        # Main Title (center, enlarge, color)
        title = QLabel("<b>🧪 Calculate Absorbance</b>")
        font = QFont(); font.setPointSize(20); font.setBold(True)
        title.setFont(font)
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        title.setStyleSheet("padding:12px 0 8px 0; color:#175793;")
        layout.addWidget(title)

        # 1. File picker
        picker_group = QGroupBox('1. Select Files for Each Spectrum Type')
        picker_group.setStyleSheet('QGroupBox{font-weight:bold; font-size:15px; border:1px solid #bbb; margin-top:8px; background:#f8fafc;}')
        pickers = QHBoxLayout()
        btn_ref = QPushButton('Reference Files…'); btn_ref.setMinimumWidth(170); btn_ref.setToolTip('Upload blank or baseline spectral files. These will be averaged.'); btn_ref.setStyleSheet('font-size:13px; font-weight:bold;')
        btn_ref.clicked.connect(self._select_ref)
        pickers.addWidget(btn_ref)
        btn_sample = QPushButton('Sample Files…'); btn_sample.setMinimumWidth(170); btn_sample.setToolTip('Upload sample spectral files. These will be averaged per group.'); btn_sample.setStyleSheet('font-size:13px; font-weight:bold;')
        btn_sample.clicked.connect(self._select_sample)
        pickers.addWidget(btn_sample)
        btn_dark = QPushButton('Dark Files (optional)…'); btn_dark.setMinimumWidth(170); btn_dark.setToolTip('Upload dark/noise spectra (optional).'); btn_dark.setStyleSheet('font-size:13px; font-weight:bold;')
        btn_dark.clicked.connect(self._select_dark)
        pickers.addWidget(btn_dark)
        picker_group.setLayout(pickers)
        layout.addWidget(picker_group)

        # 2. Preview: light bg, clear border, equal spacing
        prev_group = QGroupBox('2. Quick Spectrum Preview')
        prev_group.setStyleSheet('QGroupBox{font-weight:bold; font-size:15px; border:1px solid #ccd; margin-top:10px; background:#f6faf8;}')
        prev_layout = QHBoxLayout(); prev_layout.setSpacing(8)
        self.prev_ref = FigureCanvas(Figure(figsize=(2.9, 2.2)))
        self.prev_sample = FigureCanvas(Figure(figsize=(2.9, 2.2)))
        self.prev_dark = FigureCanvas(Figure(figsize=(2.9, 2.2)))
        for f in [self.prev_ref, self.prev_sample, self.prev_dark]:
            f.setStyleSheet('background:white; border:1px solid #eef; border-radius:7px;')
        prev_layout.addWidget(self.prev_ref)
        vline = QFrame()
        vline.setFrameShape(QFrame.Shape.VLine)
        vline.setFrameShadow(QFrame.Shadow.Sunken)
        vline.setStyleSheet('color: #ddeeff; border: 1px dashed #aaccff; min-width:8px; max-width:12px;')
        prev_layout.addWidget(vline)
        prev_layout.addWidget(self.prev_sample)
        vline2 = QFrame()
        vline2.setFrameShape(QFrame.Shape.VLine)
        vline2.setFrameShadow(QFrame.Shadow.Sunken)
        vline2.setStyleSheet('color: #ddeeff; border: 1px dashed #aaccff; min-width:8px; max-width:12px;')
        prev_layout.addWidget(vline2)
        prev_layout.addWidget(self.prev_dark)
        prev_group.setLayout(prev_layout)
        layout.addWidget(prev_group)

        # 3. Status message: colored, bordered banner
        self.status = QLabel('<i>No files selected</i>')
        self.status.setWordWrap(True)
        self.status.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.status.setStyleSheet('QLabel { border:1px solid #bcd; background:#eef9fd; color:#245181; border-radius:7px; padding:7px 8px; font-size:14px; margin:6px 4px;}')
        layout.addWidget(self.status)

        # 4. Analysis Controls section
        controls_group = QGroupBox('3. Analysis Controls')
        controls_group.setStyleSheet('QGroupBox{font-weight:bold; font-size:15px; border:1px solid #ccd; margin-top:8px; background:#f7f8fb;}')
        controls = QHBoxLayout(); controls.setSpacing(10)
        self.cb_smooth = QCheckBox('Smoothing'); self.cb_smooth.setToolTip('Apply Savitzky-Golay smoothing filter for noise reduction.')
        self.spin_window = QSpinBox(); self.spin_window.setRange(3, 101); self.spin_window.setSingleStep(2); self.spin_window.setValue(11); self.spin_window.setToolTip('Window size for smoothing, odd integers.')
        self.spin_poly = QSpinBox(); self.spin_poly.setRange(1, 9); self.spin_poly.setValue(2); self.spin_poly.setToolTip('Polynomial order for smoothing.')
        self.spin_peak_height = QDoubleSpinBox(); self.spin_peak_height.setRange(0.0, 10.0); self.spin_peak_height.setSingleStep(0.01); self.spin_peak_height.setValue(0.1); self.spin_peak_height.setToolTip('Minimum peak height in absorbance units for detection.')
        self.spin_peak_distance = QDoubleSpinBox(); self.spin_peak_distance.setRange(0.1, 1000.0); self.spin_peak_distance.setSingleStep(0.5); self.spin_peak_distance.setValue(15.0); self.spin_peak_distance.setToolTip('Minimum peak separation, nm.')
        self.spin_peak_prom = QDoubleSpinBox(); self.spin_peak_prom.setRange(0.0, 10.0); self.spin_peak_prom.setSingleStep(0.01); self.spin_peak_prom.setValue(0.05); self.spin_peak_prom.setToolTip('Minimum prominence (vertical drop) required to recognize a peak.')
        controls.addWidget(self.cb_smooth)
        controls.addWidget(QLabel('Win')); controls.addWidget(self.spin_window)
        controls.addWidget(QLabel('Poly')); controls.addWidget(self.spin_poly)
        controls.addWidget(QLabel('Height')); controls.addWidget(self.spin_peak_height)
        controls.addWidget(QLabel('Dist (nm)')); controls.addWidget(self.spin_peak_distance)
        controls.addWidget(QLabel('Prom')); controls.addWidget(self.spin_peak_prom)
        for i in range(controls.count()):
            w = controls.itemAt(i).widget(); w.setStyleSheet('font-size:13px;') if hasattr(w, 'setStyleSheet') else None
        controls_group.setLayout(controls)
        layout.addWidget(controls_group)

        # 5. Output (bigger, styled action buttons, white border plot)
        outputs_group = QGroupBox('4. Absorbance Analysis Output')
        outputs_group.setStyleSheet('QGroupBox{font-weight:bold; font-size:15px; border:1px solid #b7d5fa; margin-top:8px; background:#fdfcfa;}')
        outputs_layout = QVBoxLayout(); outputs_layout.setSpacing(8)
        self.figure = Figure(figsize=(7, 4)); self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet('background:#fff; border-radius:12px; border:1.7px solid #eaf0ff;')
        outputs_layout.addWidget(self.canvas)
        self.run_btn = QPushButton('Compute Absorbance'); self.run_btn.setMinimumWidth(210)
        self.run_btn.setToolTip('Process selected spectra and display absorbance curve, peaks.')
        self.run_btn.setStyleSheet('font-size:16px; background:#1278f2; color:white; font-weight:bold; border-radius:10px; padding:8px 12px;')
        self.run_btn.clicked.connect(self._compute)
        outputs_layout.addWidget(self.run_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.export_btn = QPushButton('Export Data as CSV…'); self.export_btn.setMinimumWidth(210)
        self.export_btn.setToolTip('Export last computed absorbance data as CSV for further analysis.')
        self.export_btn.setStyleSheet('font-size:14px; background:#f5f9fd; color:#246; border-radius:7px; padding:7px 8px; margin-top:6px; border:1px solid #d1dbe8;')
        self.export_btn.clicked.connect(self._export_csv)
        outputs_layout.addWidget(self.export_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        outputs_group.setLayout(outputs_layout)
        layout.addWidget(outputs_group)

        self.setLayout(layout)
        self._update_status()
        self._update_previews()

    def _select_ref(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Select Reference Files", "", "Data Files (*.csv *.txt *.dat)")
        if files:
            self.ref_paths = files
        self._update_status()
        self._update_previews()

    def _select_sample(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Select Sample Files", "", "Data Files (*.csv *.txt *.dat)")
        if files:
            self.sample_paths = files
        self._update_status()
        self._update_previews()

    def _select_dark(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Select Dark Files", "", "Data Files (*.csv *.txt *.dat)")
        if files:
            self.dark_paths = files
        self._update_status()
        self._update_previews()

    def _update_status(self) -> None:
        self.status.setText(
            f"Reference: {len(self.ref_paths)} • Sample: {len(self.sample_paths)} • Dark: {len(self.dark_paths)}"
        )

    def _update_previews(self) -> None:
        # For each group, show averaged preview spectrum if files exist
        self._draw_preview(self.prev_ref, self.ref_paths, "Reference")
        self._draw_preview(self.prev_sample, self.sample_paths, "Sample")
        self._draw_preview(self.prev_dark, self.dark_paths, "Dark")

    def _draw_preview(self, canvas, files, label):
        fig = canvas.figure
        fig.clear()
        ax = fig.subplots()
        # Remove axis ticks and labels for clean look
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if not files:
            ax.text(0.5, 0.5, f"No {label}\nfiles", fontsize=10, ha='center', va='center')
            ax.set_title(f"{label} Preview", fontsize=12, fontweight='bold', pad=6)
            fig.subplots_adjust(left=0, right=1, top=0.8, bottom=0.15)
            canvas.draw_idle()
            return
        dfs = [read_spectral_file_from_path(p) for p in files]
        dfs = [d for d in dfs if d is not None]
        import numpy as np
        if not dfs:
            ax.text(0.5, 0.5, f"No {label}\nfiles", fontsize=10, ha='center', va='center')
            ax.set_title(f"{label} Preview", fontsize=12, fontweight='bold', pad=6)
            fig.subplots_adjust(left=0, right=1, top=0.8, bottom=0.15)
            canvas.draw_idle()
            return
        ref = dfs[0]['Nanometers'].values
        valid = all(np.allclose(ref, d['Nanometers'].values, rtol=1e-3) for d in dfs)
        if not valid:
            ax.text(0.5, 0.5, "Wavelength\nmismatch!", fontsize=10, ha='center', va='center', color='red')
            ax.set_title(f"{label} Preview", fontsize=12, fontweight='bold', pad=6)
            fig.subplots_adjust(left=0, right=1, top=0.8, bottom=0.15)
            canvas.draw_idle()
            return
        if len(dfs) == 1:
            df = dfs[0]
            ax.plot(df['Nanometers'], df['Counts'], color='C0', linewidth=1.4)
        else:
            avg = np.mean(np.column_stack([d['Counts'].values for d in dfs]), axis=1)
            ax.plot(ref, avg, color='C0', linewidth=1.4)
        ax.set_title(f"{label} Preview", fontsize=12, fontweight='bold', pad=6)
        fig.subplots_adjust(left=0, right=1, top=0.8, bottom=0.15)
        canvas.draw_idle()

    def _compute(self) -> None:
        if not self.ref_paths or not self.sample_paths:
            self.status.setText("Please select both Reference and Sample files")
            return
        ref_dfs = [read_spectral_file_from_path(p) for p in self.ref_paths]
        ref_dfs = [d for d in ref_dfs if d is not None]
        sample_dfs = [read_spectral_file_from_path(p) for p in self.sample_paths]
        sample_dfs = [d for d in sample_dfs if d is not None]
        dark_df = None
        if self.dark_paths:
            dark_dfs = [read_spectral_file_from_path(p) for p in self.dark_paths]
            dark_dfs = [d for d in dark_dfs if d is not None]
            dark_df = average_counts_dataframes(dark_dfs) if dark_dfs else None
        ref_df = average_counts_dataframes(ref_dfs)
        sample_df = average_counts_dataframes(sample_dfs)
        if ref_df is None or sample_df is None:
            self.status.setText("Failed to parse or align files. Ensure identical wavelength grids.")
            return
        result = calculate_absorbance(ref_df, sample_df, dark_df)
        if result is None:
            self.status.setText("Absorbance calculation failed (wavelength mismatch or invalid data)")
            return
        smoothed = None
        if self.cb_smooth.isChecked():
            smoothed = savgol_smooth(
                result['Absorbance'].values,
                window=int(self.spin_window.value()),
                poly=int(self.spin_poly.value()),
            )
        peaks = find_peaks_absorbance(
            result['Nanometers'].values,
            smoothed if smoothed is not None else result['Absorbance'].values,
            height=float(self.spin_peak_height.value()),
            distance_nm=float(self.spin_peak_distance.value()),
            prominence=float(self.spin_peak_prom.value()),
        )
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(result['Nanometers'], result['Absorbance'], label='Absorbance', color='C0', linewidth=2)
        if smoothed is not None:
            ax.plot(result['Nanometers'], smoothed, label='Smoothed', color='C3', linestyle='--', alpha=0.95)
        if peaks and len(peaks[0]) > 0:
            idx = peaks[0]
            ax.scatter(result['Nanometers'].iloc[idx], (smoothed if smoothed is not None else result['Absorbance'].values)[idx],
                       color='red', marker='v', label='Peaks', zorder=10)
        ax.set_title('Absorbance Spectrum', fontsize=14, pad=12)
        ax.set_xlabel('Wavelength (nm)', fontsize=12, labelpad=6)
        ax.set_ylabel('Absorbance (AU)', fontsize=12, labelpad=6)
        ax.legend(fontsize=10, frameon=False)
        ax.tick_params(axis='x', labelrotation=15, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        self.figure.subplots_adjust(left=0.13, right=0.98, bottom=0.16, top=0.88)
        self.canvas.draw_idle()
        self.status.setText("Computed absorbance successfully")
        self._last_result = result

    def _export_csv(self) -> None:
        if not hasattr(self, '_last_result') or self._last_result is None:
            self.status.setText("Nothing to export. Compute first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "absorbance_data.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            self._last_result[['Nanometers', 'Absorbance']].to_csv(path, index=False)
            self.status.setText(f"Saved: {path}")
        except Exception as e:
            self.status.setText(f"Export failed: {e}")



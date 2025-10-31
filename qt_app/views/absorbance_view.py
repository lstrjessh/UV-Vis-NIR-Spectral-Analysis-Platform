from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QHBoxLayout, 
    QCheckBox, QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout, QScrollArea
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPalette

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

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
        
        # Main scroll area for better responsiveness
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(20)

        # Title
        title = QLabel("Calculate Absorbance")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(title)

        subtitle = QLabel("Upload reference, sample, and optional dark spectra to calculate absorbance")
        subtitle.setStyleSheet("color: palette(mid); font-size: 14px; margin-bottom: 8px;")
        layout.addWidget(subtitle)

        # 1. File Selection
        file_group = QGroupBox("Step 1: Select Spectral Files")
        file_layout = QGridLayout()
        file_layout.setSpacing(12)
        
        # Reference
        ref_label = QLabel("Reference (Blank):")
        ref_label.setStyleSheet("font-weight: 600;")
        self.btn_ref = QPushButton("Choose Files...")
        self.btn_ref.setToolTip("Select reference/blank spectral files")
        self.btn_ref.clicked.connect(self._select_ref)
        self.ref_count = QLabel("0 files")
        self.ref_count.setStyleSheet("color: palette(mid);")
        
        file_layout.addWidget(ref_label, 0, 0)
        file_layout.addWidget(self.btn_ref, 0, 1)
        file_layout.addWidget(self.ref_count, 0, 2)
        
        # Sample
        sample_label = QLabel("Sample:")
        sample_label.setStyleSheet("font-weight: 600;")
        self.btn_sample = QPushButton("Choose Files...")
        self.btn_sample.setToolTip("Select sample spectral files")
        self.btn_sample.clicked.connect(self._select_sample)
        self.sample_count = QLabel("0 files")
        self.sample_count.setStyleSheet("color: palette(mid);")
        
        file_layout.addWidget(sample_label, 1, 0)
        file_layout.addWidget(self.btn_sample, 1, 1)
        file_layout.addWidget(self.sample_count, 1, 2)
        
        # Dark (optional)
        dark_label = QLabel("Dark (Optional):")
        dark_label.setStyleSheet("font-weight: 600;")
        self.btn_dark = QPushButton("Choose Files...")
        self.btn_dark.setToolTip("Select dark/noise spectral files (optional)")
        self.btn_dark.clicked.connect(self._select_dark)
        self.dark_count = QLabel("0 files")
        self.dark_count.setStyleSheet("color: palette(mid);")
        
        file_layout.addWidget(dark_label, 2, 0)
        file_layout.addWidget(self.btn_dark, 2, 1)
        file_layout.addWidget(self.dark_count, 2, 2)
        
        file_layout.setColumnStretch(1, 1)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # 2. Preview
        preview_group = QGroupBox("Step 2: Spectrum Preview")
        preview_layout = QHBoxLayout()
        preview_layout.setSpacing(16)
        
        self.prev_ref = FigureCanvas(Figure(figsize=(4, 3), facecolor='none'))
        self.prev_sample = FigureCanvas(Figure(figsize=(4, 3), facecolor='none'))
        self.prev_dark = FigureCanvas(Figure(figsize=(4, 3), facecolor='none'))
        
        for canvas in [self.prev_ref, self.prev_sample, self.prev_dark]:
            canvas.setMinimumHeight(200)
            preview_layout.addWidget(canvas)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # 3. Analysis Controls
        controls_group = QGroupBox("Step 3: Analysis Parameters")
        controls_layout = QGridLayout()
        controls_layout.setSpacing(12)
        
        # Smoothing
        self.cb_smooth = QCheckBox("Apply Smoothing")
        self.cb_smooth.setToolTip("Apply Savitzky-Golay smoothing filter")
        controls_layout.addWidget(self.cb_smooth, 0, 0, 1, 2)
        
        controls_layout.addWidget(QLabel("Window Size:"), 1, 0)
        self.spin_window = QSpinBox()
        self.spin_window.setRange(3, 101)
        self.spin_window.setSingleStep(2)
        self.spin_window.setValue(11)
        controls_layout.addWidget(self.spin_window, 1, 1)
        
        controls_layout.addWidget(QLabel("Polynomial Order:"), 2, 0)
        self.spin_poly = QSpinBox()
        self.spin_poly.setRange(1, 9)
        self.spin_poly.setValue(2)
        controls_layout.addWidget(self.spin_poly, 2, 1)
        
        # Peak detection
        peak_label = QLabel("Peak Detection:")
        peak_label.setStyleSheet("font-weight: 600; margin-top: 8px;")
        controls_layout.addWidget(peak_label, 3, 0, 1, 2)
        
        controls_layout.addWidget(QLabel("Min Height:"), 4, 0)
        self.spin_peak_height = QDoubleSpinBox()
        self.spin_peak_height.setRange(0.0, 10.0)
        self.spin_peak_height.setSingleStep(0.01)
        self.spin_peak_height.setValue(0.1)
        controls_layout.addWidget(self.spin_peak_height, 4, 1)
        
        controls_layout.addWidget(QLabel("Min Distance (nm):"), 5, 0)
        self.spin_peak_distance = QDoubleSpinBox()
        self.spin_peak_distance.setRange(0.1, 1000.0)
        self.spin_peak_distance.setSingleStep(0.5)
        self.spin_peak_distance.setValue(15.0)
        controls_layout.addWidget(self.spin_peak_distance, 5, 1)
        
        controls_layout.addWidget(QLabel("Min Prominence:"), 6, 0)
        self.spin_peak_prom = QDoubleSpinBox()
        self.spin_peak_prom.setRange(0.0, 10.0)
        self.spin_peak_prom.setSingleStep(0.01)
        self.spin_peak_prom.setValue(0.05)
        controls_layout.addWidget(self.spin_peak_prom, 6, 1)
        
        controls_layout.setColumnStretch(1, 1)
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # 4. Results
        results_group = QGroupBox("Step 4: Absorbance Results")
        results_layout = QVBoxLayout()
        results_layout.setSpacing(12)
        
        # Main plot with proper sizing
        self.figure = Figure(figsize=(10, 5), facecolor='none')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(400)
        results_layout.addWidget(self.canvas)
        
        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
        self.run_btn = QPushButton("🔬 Compute Absorbance")
        self.run_btn.setToolTip("Calculate absorbance spectrum")
        self.run_btn.setMinimumHeight(44)
        self.run_btn.clicked.connect(self._compute)
        button_layout.addWidget(self.run_btn)
        
        self.export_btn = QPushButton("💾 Export CSV")
        self.export_btn.setToolTip("Export absorbance data")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_csv)
        button_layout.addWidget(self.export_btn)
        
        results_layout.addLayout(button_layout)
        
        # Status
        self.status = QLabel("Ready to process spectra")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet("""
            padding: 12px;
            border-radius: 6px;
            background: palette(midlight);
            font-size: 13px;
        """)
        results_layout.addWidget(self.status)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        layout.addStretch()
        scroll.setWidget(content)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        
        self._update_previews()

    def _select_ref(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Reference Files", "", 
            "Data Files (*.csv *.txt *.dat);;All Files (*)"
        )
        if files:
            self.ref_paths = files
            self.ref_count.setText(f"{len(files)} file(s)")
            self._update_previews()

    def _select_sample(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Sample Files", "", 
            "Data Files (*.csv *.txt *.dat);;All Files (*)"
        )
        if files:
            self.sample_paths = files
            self.sample_count.setText(f"{len(files)} file(s)")
            self._update_previews()

    def _select_dark(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Dark Files", "", 
            "Data Files (*.csv *.txt *.dat);;All Files (*)"
        )
        if files:
            self.dark_paths = files
            self.dark_count.setText(f"{len(files)} file(s)")
            self._update_previews()

    def _update_previews(self) -> None:
        self._draw_preview(self.prev_ref, self.ref_paths, "Reference")
        self._draw_preview(self.prev_sample, self.sample_paths, "Sample")
        self._draw_preview(self.prev_dark, self.dark_paths, "Dark")

    def _draw_preview(self, canvas, files, label):
        fig = canvas.figure
        fig.clear()
        
        # Theme: force light
        bg_color = '#ffffff'
        text_color = '#000000'
        
        fig.patch.set_facecolor('none')
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg_color)
        
        if not files:
            ax.text(0.5, 0.5, f"No {label.lower()} files selected", 
                   ha='center', va='center', fontsize=11, color=text_color, alpha=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(text_color)
            ax.spines['left'].set_color(text_color)
            ax.set_title(f"{label} Preview", fontsize=12, fontweight='bold', 
                        color=text_color, pad=10)
            fig.tight_layout()
            canvas.draw_idle()
            return
        
        dfs = [read_spectral_file_from_path(p) for p in files]
        dfs = [d for d in dfs if d is not None]
        
        if not dfs:
            ax.text(0.5, 0.5, f"Failed to load {label.lower()} files", 
                   ha='center', va='center', fontsize=11, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
            canvas.draw_idle()
            return
        
        import numpy as np
        ref = dfs[0]['Nanometers'].values
        valid = all(np.allclose(ref, d['Nanometers'].values, rtol=1e-3) for d in dfs)
        
        if not valid:
            ax.text(0.5, 0.5, "Wavelength mismatch!", 
                   ha='center', va='center', fontsize=11, color='red')
            canvas.draw_idle()
            return
        
        if len(dfs) == 1:
            ax.plot(dfs[0]['Nanometers'], dfs[0]['Counts'], 
                   color='#2196F3', linewidth=2, alpha=0.9)
        else:
            avg = np.mean(np.column_stack([d['Counts'].values for d in dfs]), axis=1)
            ax.plot(ref, avg, color='#2196F3', linewidth=2, alpha=0.9)
            ax.fill_between(ref, avg * 0.95, avg * 1.05, alpha=0.2, color='#2196F3')
        
        ax.set_title(f"{label} Preview ({len(files)} file{'s' if len(files) > 1 else ''})", 
                    fontsize=12, fontweight='bold', color=text_color, pad=10)
        ax.set_xlabel('Wavelength (nm)', fontsize=10, color=text_color)
        ax.set_ylabel('Counts', fontsize=10, color=text_color)
        ax.tick_params(colors=text_color, labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(text_color)
        ax.spines['left'].set_color(text_color)
        ax.grid(True, alpha=0.2, linestyle='--')
        
        fig.tight_layout()
        canvas.draw_idle()

    def _compute(self) -> None:
        if not self.ref_paths or not self.sample_paths:
            self.status.setText("⚠️ Please select both reference and sample files")
            self.status.setStyleSheet("background: #ffebee; color: #c62828; padding: 12px; border-radius: 6px;")
            return
        
        self.status.setText("⏳ Processing spectra...")
        self.status.setStyleSheet("background: #fff3e0; color: #ef6c00; padding: 12px; border-radius: 6px;")
        
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
            self.status.setText("❌ Failed to parse files - ensure wavelength grids match")
            self.status.setStyleSheet("background: #ffebee; color: #c62828; padding: 12px; border-radius: 6px;")
            return
        
        result = calculate_absorbance(ref_df, sample_df, dark_df)
        if result is None:
            self.status.setText("❌ Absorbance calculation failed")
            self.status.setStyleSheet("background: #ffebee; color: #c62828; padding: 12px; border-radius: 6px;")
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
        
        # Plot results - force light theme
        bg_color = '#ffffff'
        text_color = '#000000'
        
        self.figure.clear()
        self.figure.patch.set_facecolor('none')
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(bg_color)
        
        ax.plot(result['Nanometers'], result['Absorbance'], 
               label='Absorbance', color='#2196F3', linewidth=2.5, alpha=0.9)
        
        if smoothed is not None:
            ax.plot(result['Nanometers'], smoothed, 
                   label='Smoothed', color='#FF9800', linestyle='--', linewidth=2, alpha=0.85)
        
        if peaks and len(peaks[0]) > 0:
            idx = peaks[0]
            peak_values = (smoothed if smoothed is not None else result['Absorbance'].values)[idx]
            ax.scatter(result['Nanometers'].iloc[idx], peak_values,
                      color='#F44336', marker='v', s=100, label='Peaks', zorder=10, 
                      edgecolors='white', linewidths=1.5)
            
            # Annotate peaks
            for i, (wl, val) in enumerate(zip(result['Nanometers'].iloc[idx], peak_values)):
                if i < 5:  # Limit annotations
                    ax.annotate(f'{wl:.1f}nm', xy=(wl, val), xytext=(0, 10),
                              textcoords='offset points', ha='center', fontsize=9,
                              color=text_color, bbox=dict(boxstyle='round,pad=0.3', 
                              facecolor=bg_color, edgecolor='#F44336', alpha=0.8))
        
        ax.set_title('Absorbance Spectrum Analysis', fontsize=16, fontweight='bold', 
                    color=text_color, pad=15)
        ax.set_xlabel('Wavelength (nm)', fontsize=13, labelpad=10, color=text_color)
        ax.set_ylabel('Absorbance (AU)', fontsize=13, labelpad=10, color=text_color)
        ax.tick_params(colors=text_color, labelsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(text_color)
        ax.spines['left'].set_color(text_color)
        ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, 
                 facecolor=bg_color, edgecolor=text_color)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
        
        self.figure.tight_layout()
        self.canvas.draw_idle()
        
        peak_count = len(peaks[0]) if peaks and len(peaks[0]) > 0 else 0
        self.status.setText(f"✅ Analysis complete • {peak_count} peak(s) detected")
        self.status.setStyleSheet("background: #e8f5e9; color: #2e7d32; padding: 12px; border-radius: 6px;")
        self.export_btn.setEnabled(True)
        self._last_result = result

    def _export_csv(self) -> None:
        if not hasattr(self, '_last_result') or self._last_result is None:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Absorbance Data", "absorbance_data.csv", 
            "CSV Files (*.csv)"
        )
        if not path:
            return
        
        try:
            self._last_result[['Nanometers', 'Absorbance']].to_csv(path, index=False)
            self.status.setText(f"💾 Exported to: {path}")
            self.status.setStyleSheet("background: #e3f2fd; color: #1565c0; padding: 12px; border-radius: 6px;")
        except Exception as e:
            self.status.setText(f"❌ Export failed: {e}")
            self.status.setStyleSheet("background: #ffebee; color: #c62828; padding: 12px; border-radius: 6px;")
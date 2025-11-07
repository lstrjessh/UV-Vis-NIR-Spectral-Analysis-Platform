from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QHBoxLayout, 
    QCheckBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem, QGroupBox,
    QGridLayout, QScrollArea, QHeaderView, QMessageBox, QSlider
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPalette
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import csv
from datetime import datetime

from qt_app.spectral_processing import (
    read_spectral_file_from_path,
    find_peaks_absorbance,
)
from qt_app.spec_functions import savitzky_golay, peakIndexes


class ViewerView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.paths = []

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(60, 20, 60, 20)
        layout.setSpacing(20)

        # Title
        title = QLabel("👁️ Spectrum Viewer")
        title.setProperty("class", "title")
        layout.addWidget(title)

        subtitle = QLabel("Load and visualize multiple spectra with customizable processing options")
        subtitle.setProperty("class", "subtitle")
        layout.addWidget(subtitle)
        
        # Controls section at top (full width, horizontal layout)
        controls_container = QHBoxLayout()
        controls_container.setSpacing(15)
        
        # File selection (compact)
        file_group = QGroupBox("📁 Files")
        file_layout = QVBoxLayout()
        file_layout.setSpacing(5)
        
        btn = QPushButton("📂 Select Files...")
        btn.setMinimumHeight(35)
        btn.setToolTip("Load multiple spectral files for comparison")
        btn.clicked.connect(self._pick_files)
        file_layout.addWidget(btn)
        
        self.file_count = QLabel("No files loaded")
        self.file_count.setStyleSheet("color: #888888; font-size: 9pt;")
        file_layout.addWidget(self.file_count)
        
        file_group.setLayout(file_layout)
        controls_container.addWidget(file_group)

        # Processing options (horizontal)
        options_group = QGroupBox("⚙️ Processing")
        options_layout = QHBoxLayout()
        options_layout.setSpacing(15)
        
        # Column 1: Basic processing
        col1 = QVBoxLayout()
        col1.setSpacing(5)
        self.cb_normalize = QCheckBox("Normalize")
        self.cb_normalize.setToolTip("Scale each spectrum to 0-1 range")
        self.cb_normalize.stateChanged.connect(self._render_plot)
        col1.addWidget(self.cb_normalize)
        
        self.cb_smooth = QCheckBox("Smooth")
        self.cb_smooth.setToolTip("Apply Gaussian smoothing")
        self.cb_smooth.stateChanged.connect(self._render_plot)
        col1.addWidget(self.cb_smooth)
        
        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(QLabel("σ:"))
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.1, 10.0)
        self.spin_sigma.setSingleStep(0.1)
        self.spin_sigma.setValue(1.0)
        self.spin_sigma.setMaximumWidth(70)
        self.spin_sigma.setToolTip("Smoothing strength")
        self.spin_sigma.valueChanged.connect(self._render_plot)
        smooth_layout.addWidget(self.spin_sigma)
        smooth_layout.addStretch()
        col1.addLayout(smooth_layout)
        options_layout.addLayout(col1)
        
        # Separator
        sep1 = QLabel("│")
        sep1.setStyleSheet("color: #cccccc; font-size: 14pt;")
        options_layout.addWidget(sep1)
        
        # Column 2: Peak detection
        col2 = QVBoxLayout()
        col2.setSpacing(5)
        self.cb_peaks = QCheckBox("Detect Peaks")
        self.cb_peaks.setToolTip("Identify spectral maxima")
        self.cb_peaks.stateChanged.connect(self._render_plot)
        col2.addWidget(self.cb_peaks)
        
        self.cb_label_peaks = QCheckBox("Label Peaks")
        self.cb_label_peaks.setToolTip("Show wavelength labels")
        self.cb_label_peaks.stateChanged.connect(self._render_plot)
        col2.addWidget(self.cb_label_peaks)
        
        col2.addWidget(QLabel(" "))  # Spacer
        options_layout.addLayout(col2)
        
        # Separator
        sep2 = QLabel("│")
        sep2.setStyleSheet("color: #cccccc; font-size: 14pt;")
        options_layout.addWidget(sep2)
        
        # Column 3: Peak parameters (with sliders)
        col3 = QVBoxLayout()
        col3.setSpacing(8)
        
        # Height slider
        height_row = QHBoxLayout()
        height_row.addWidget(QLabel("Height:"))
        self.slider_height = QSlider(Qt.Orientation.Horizontal)
        self.slider_height.setRange(0, 1000)  # 0.00 to 10.00 with 0.01 step
        self.slider_height.setValue(1)  # 0.01
        self.slider_height.setMinimumWidth(120)
        self.slider_height.setToolTip("Min peak intensity (0.00 - 10.00)")
        self.slider_height.valueChanged.connect(self._on_height_changed)
        height_row.addWidget(self.slider_height)
        self.label_height = QLabel("0.01")
        self.label_height.setMinimumWidth(40)
        self.label_height.setStyleSheet("color: #0066cc; font-weight: bold;")
        height_row.addWidget(self.label_height)
        col3.addLayout(height_row)
        
        # Distance slider
        distance_row = QHBoxLayout()
        distance_row.addWidget(QLabel("Distance:"))
        self.slider_distance = QSlider(Qt.Orientation.Horizontal)
        self.slider_distance.setRange(10, 1000)  # 1.0 to 100.0 with 0.1 step
        self.slider_distance.setValue(50)  # 5.0 nm
        self.slider_distance.setMinimumWidth(120)
        self.slider_distance.setToolTip("Min wavelength separation (1.0 - 100.0 nm)")
        self.slider_distance.valueChanged.connect(self._on_distance_changed)
        distance_row.addWidget(self.slider_distance)
        self.label_distance = QLabel("5.0 nm")
        self.label_distance.setMinimumWidth(50)
        self.label_distance.setStyleSheet("color: #0066cc; font-weight: bold;")
        distance_row.addWidget(self.label_distance)
        col3.addLayout(distance_row)
        
        # Prominence slider
        prom_row = QHBoxLayout()
        prom_row.addWidget(QLabel("Prominence:"))
        self.slider_prom = QSlider(Qt.Orientation.Horizontal)
        self.slider_prom.setRange(0, 1000)  # 0.00 to 10.00 with 0.01 step
        self.slider_prom.setValue(1)  # 0.01
        self.slider_prom.setMinimumWidth(120)
        self.slider_prom.setToolTip("Peak prominence (0.00 - 10.00)")
        self.slider_prom.valueChanged.connect(self._on_prom_changed)
        prom_row.addWidget(self.slider_prom)
        self.label_prom = QLabel("0.01")
        self.label_prom.setMinimumWidth(40)
        self.label_prom.setStyleSheet("color: #0066cc; font-weight: bold;")
        prom_row.addWidget(self.label_prom)
        col3.addLayout(prom_row)
        options_layout.addLayout(col3)
        
        # Separator
        sep3 = QLabel("│")
        sep3.setStyleSheet("color: #cccccc; font-size: 14pt;")
        options_layout.addWidget(sep3)
        
        # Column 4: View mode
        col4 = QVBoxLayout()
        col4.setSpacing(5)
        view_label = QLabel("View Mode:")
        view_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        col4.addWidget(view_label)
        
        self.cb_side_by_side = QCheckBox("Side-by-Side")
        self.cb_side_by_side.setToolTip("Separate subplots instead of overlay")
        self.cb_side_by_side.stateChanged.connect(self._render_plot)
        col4.addWidget(self.cb_side_by_side)
        
        col4.addWidget(QLabel(" "))  # Spacer
        options_layout.addLayout(col4)
        
        options_layout.addStretch()
        options_group.setLayout(options_layout)
        controls_container.addWidget(options_group, stretch=1)
        
        layout.addLayout(controls_container)
        
        # Spectrum plot (full width, large)
        plot_group = QGroupBox("📈 Spectrum Visualization")
        plot_layout = QVBoxLayout()
        
        self.figure = Figure(figsize=(14, 8), facecolor='none')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(500)
        plot_layout.addWidget(self.canvas)
        
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)

        # Peak table (full width, below plot)
        table_group = QGroupBox("🎯 Detected Peaks")
        table_layout = QVBoxLayout()
        
        # Export button row
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        self.export_peaks_btn = QPushButton("💾 Export Peaks to CSV")
        self.export_peaks_btn.setMaximumWidth(200)
        self.export_peaks_btn.setToolTip("Export detected peaks to CSV file")
        self.export_peaks_btn.clicked.connect(self._export_peaks_csv)
        export_layout.addWidget(self.export_peaks_btn)
        table_layout.addLayout(export_layout)
        
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Spectrum", "Wavelength (nm)", "Intensity"])
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(300)  # Increased from 200
        self.table.setMaximumHeight(500)  # Increased from 200
        # Set equal column widths
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table_layout.addWidget(self.table)
        
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)

        layout.addStretch()
        scroll.setWidget(content)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

        self._initialize_plot()

    def _initialize_plot(self):
        # Theme: force light
        bg_color = '#ffffff'
        text_color = '#000000'
        
        self.figure.clear()
        self.figure.patch.set_facecolor('none')
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(bg_color)
        ax.text(0.5, 0.5, "No spectra loaded\n\nClick 'Select Spectrum Files' to begin", 
               ha='center', va='center', fontsize=14, color=text_color, alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(text_color)
        ax.spines['left'].set_color(text_color)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _on_height_changed(self, value):
        """Update height label and rerender plot"""
        self.label_height.setText(f"{value / 100.0:.2f}")
        self._render_plot()
    
    def _on_distance_changed(self, value):
        """Update distance label and rerender plot"""
        self.label_distance.setText(f"{value / 10.0:.1f} nm")
        self._render_plot()
    
    def _on_prom_changed(self, value):
        """Update prominence label and rerender plot"""
        self.label_prom.setText(f"{value / 100.0:.2f}")
        self._render_plot()

    def _pick_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Spectrum Files", "", 
            "Data Files (*.csv *.txt *.dat);;All Files (*)"
        )
        if files:
            self.paths = files
            self.file_count.setText(f"{len(files)} file(s) loaded")
            self._render_plot()

    def _render_plot(self) -> None:
        if not self.paths:
            return
        
        # Load spectra
        spectra = []
        for p in self.paths:
            df = read_spectral_file_from_path(p)
            if df is None:
                continue
            wl = df['Nanometers'].values
            y = df['Counts'].values
            name = p.split('/')[-1].split('\\')[-1]  # Cross-platform filename
            spectra.append((name, wl, y))
        
        if not spectra:
            return
        
        # Theme: force light
        bg_color = '#ffffff'
        text_color = '#000000'
        
        # Clear plot
        self.figure.clear()
        self.figure.patch.set_facecolor('none')
        
        # Clear peak table
        self.table.setRowCount(0)
        
        # Color palette
        colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0', 
                 '#00BCD4', '#FFEB3B', '#795548', '#E91E63', '#607D8B']
        
        # Determine layout: side-by-side or overlay
        side_by_side = self.cb_side_by_side.isChecked()
        
        if side_by_side and len(spectra) > 1:
            # Side-by-side comparison mode
            n_spectra = len(spectra)
            # Calculate grid dimensions (prefer columns)
            if n_spectra <= 2:
                n_rows, n_cols = 1, n_spectra
            elif n_spectra <= 4:
                n_rows, n_cols = 2, 2
            elif n_spectra <= 6:
                n_rows, n_cols = 2, 3
            elif n_spectra <= 9:
                n_rows, n_cols = 3, 3
            else:
                n_rows = int(np.ceil(np.sqrt(n_spectra)))
                n_cols = int(np.ceil(n_spectra / n_rows))
            
            for idx, (name, wl, y) in enumerate(spectra):
                yy = y.copy()
                
                # Apply smoothing
                if self.cb_smooth.isChecked():
                    try:
                        from scipy.ndimage import gaussian_filter1d
                        yy = gaussian_filter1d(yy, sigma=float(self.spin_sigma.value()))
                    except Exception:
                        pass
                
                # Apply normalization
                if self.cb_normalize.isChecked():
                    ymin = float(np.min(yy))
                    ymax = float(np.max(yy))
                    if ymax - ymin > 0:
                        yy = (yy - ymin) / (ymax - ymin)
                
                color = colors[idx % len(colors)]
                
                # Create subplot
                ax = self.figure.add_subplot(n_rows, n_cols, idx + 1)
                ax.set_facecolor(bg_color)
                ax.plot(wl, yy, color=color, linewidth=2, alpha=0.85)
                
                # Peak detection and labeling
                if self.cb_peaks.isChecked():
                    res = find_peaks_absorbance(
                        wl, yy,
                        height=float(self.slider_height.value() / 100.0),
                        distance_nm=float(self.slider_distance.value() / 10.0),
                        prominence=float(self.slider_prom.value() / 100.0),
                    )
                    if res and len(res[0]) > 0:
                        pk_idx = res[0]
                        ax.scatter(wl[pk_idx], yy[pk_idx], marker='D', s=50, 
                                 color=color, edgecolors='white', linewidths=1.5, 
                                 zorder=10)
                        
                        # Label peaks if enabled
                        if self.cb_label_peaks.isChecked():
                            for i in pk_idx:
                                wavelength = round(wl[i], 1)
                                height = yy[i]
                                ax.annotate(
                                    f'{wavelength}nm',
                                    xy=(wl[i], height),
                                    xytext=(0, 8),
                                    textcoords='offset points',
                                    fontsize=8,
                                    ha='center',
                                    bbox=dict(boxstyle='round,pad=0.3', 
                                            facecolor='yellow', alpha=0.7, 
                                            edgecolor=color, linewidth=1)
                                )
                        
                        # Add to table
                        for i in pk_idx:
                            r = self.table.rowCount()
                            self.table.insertRow(r)
                            self.table.setItem(r, 0, QTableWidgetItem(name))
                            self.table.setItem(r, 1, QTableWidgetItem(f"{wl[i]:.2f}"))
                            self.table.setItem(r, 2, QTableWidgetItem(f"{yy[i]:.6f}"))
                
                # Subplot formatting
                ax.set_title(name, fontsize=10, fontweight='bold', color=text_color, pad=5)
                ax.set_xlabel('Wavelength (nm)', fontsize=9, color=text_color)
                ylabel = 'Norm.' if self.cb_normalize.isChecked() else 'Intensity'
                ax.set_ylabel(ylabel, fontsize=9, color=text_color)
                ax.tick_params(colors=text_color, labelsize=8)
                ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color(text_color)
                ax.spines['left'].set_color(text_color)
            
        else:
            # Overlay mode (original behavior)
            ax = self.figure.add_subplot(111)
            ax.set_facecolor(bg_color)
            
            for idx, (name, wl, y) in enumerate(spectra):
                yy = y.copy()
                
                # Apply smoothing
                if self.cb_smooth.isChecked():
                    try:
                        from scipy.ndimage import gaussian_filter1d
                        yy = gaussian_filter1d(yy, sigma=float(self.spin_sigma.value()))
                    except Exception:
                        pass
                
                # Apply normalization
                if self.cb_normalize.isChecked():
                    ymin = float(np.min(yy))
                    ymax = float(np.max(yy))
                    if ymax - ymin > 0:
                        yy = (yy - ymin) / (ymax - ymin)
                
                color = colors[idx % len(colors)]
                ax.plot(wl, yy, label=name, color=color, linewidth=2, alpha=0.85)
                
                # Peak detection and labeling
                if self.cb_peaks.isChecked():
                    res = find_peaks_absorbance(
                        wl, yy,
                        height=float(self.slider_height.value() / 100.0),
                        distance_nm=float(self.slider_distance.value() / 10.0),
                        prominence=float(self.slider_prom.value() / 100.0),
                    )
                    if res and len(res[0]) > 0:
                        pk_idx = res[0]
                        ax.scatter(wl[pk_idx], yy[pk_idx], marker='D', s=60, 
                                 color=color, edgecolors='white', linewidths=1.5, 
                                 zorder=10)
                        
                        # Label peaks if enabled (limit to top 10 per spectrum for readability)
                        if self.cb_label_peaks.isChecked():
                            # Sort by intensity and take top peaks
                            peak_heights = yy[pk_idx]
                            if len(pk_idx) > 10:
                                top_indices = np.argsort(peak_heights)[-10:]
                                labeled_peaks = pk_idx[top_indices]
                            else:
                                labeled_peaks = pk_idx
                            
                            for i in labeled_peaks:
                                wavelength = round(wl[i], 1)
                                height = yy[i]
                                ax.annotate(
                                    f'{wavelength}nm',
                                    xy=(wl[i], height),
                                    xytext=(0, 8),
                                    textcoords='offset points',
                                    fontsize=7,
                                    ha='center',
                                    bbox=dict(boxstyle='round,pad=0.2', 
                                            facecolor='yellow', alpha=0.6, 
                                            edgecolor=color, linewidth=0.5)
                                )
                        
                        # Add to table
                        for i in pk_idx:
                            r = self.table.rowCount()
                            self.table.insertRow(r)
                            self.table.setItem(r, 0, QTableWidgetItem(name))
                            self.table.setItem(r, 1, QTableWidgetItem(f"{wl[i]:.2f}"))
                            self.table.setItem(r, 2, QTableWidgetItem(f"{yy[i]:.6f}"))
            
            ax.set_title('Multi-Spectrum Overlay Viewer', fontsize=16, fontweight='bold', 
                        color=text_color, pad=15)
            ax.set_xlabel('Wavelength (nm)', fontsize=13, labelpad=10, color=text_color)
            ylabel = 'Normalized Intensity' if self.cb_normalize.isChecked() else 'Intensity (Counts)'
            ax.set_ylabel(ylabel, fontsize=13, labelpad=10, color=text_color)
            ax.tick_params(colors=text_color, labelsize=11)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(text_color)
            ax.spines['left'].set_color(text_color)
            
            if len(spectra) <= 10:
                ax.legend(loc='best', fontsize=10, frameon=True, fancybox=True, 
                         shadow=True, facecolor=bg_color, edgecolor=text_color)
            else:
                ax.text(0.98, 0.98, f'{len(spectra)} spectra', 
                       transform=ax.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor=bg_color, 
                                edgecolor=text_color, alpha=0.8),
                       color=text_color, fontsize=11)
            
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
        
        self.figure.tight_layout()
        self.canvas.draw_idle()
    
    def _export_peaks_csv(self) -> None:
        """Export detected peaks to CSV file."""
        if self.table.rowCount() == 0:
            QMessageBox.warning(self, "No Peaks", "No peaks detected to export.\n\nEnable 'Detect Peaks' and ensure peaks are found.")
            return
        
        # Generate default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"detected_peaks_{timestamp}.csv"
        
        # Open save dialog
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Peaks to CSV",
            default_filename,
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                headers = []
                for col in range(self.table.columnCount()):
                    headers.append(self.table.horizontalHeaderItem(col).text())
                writer.writerow(headers)
                
                # Write data rows
                for row in range(self.table.rowCount()):
                    row_data = []
                    for col in range(self.table.columnCount()):
                        item = self.table.item(row, col)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)
            
            QMessageBox.information(
                self,
                "Export Successful",
                f"Peaks exported successfully!\n\n{self.table.rowCount()} peaks saved to:\n{filename}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export peaks:\n{str(e)}"
            )
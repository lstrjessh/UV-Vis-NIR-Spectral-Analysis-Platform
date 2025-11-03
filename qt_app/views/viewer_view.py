from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QHBoxLayout, 
    QCheckBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem, QGroupBox,
    QGridLayout, QScrollArea
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPalette
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

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(100, 20, 100, 20)
        layout.setSpacing(18)

        # Title
        title = QLabel("Spectrum Viewer")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        subtitle = QLabel("Load and visualize multiple spectra with customizable processing options")
        subtitle.setStyleSheet("color: #444444; font-size: 14px; margin-bottom: 8px;")
        layout.addWidget(subtitle)

        # File selection
        file_group = QGroupBox("Load Spectral Files")
        file_layout = QHBoxLayout()
        
        btn = QPushButton("📂 Select Spectrum Files...")
        btn.setMinimumHeight(44)
        btn.setToolTip(
            "Load multiple spectral files for comparison.\n\n"
            "📋 Supported: CSV, TXT, DAT files\n"
            "📊 Overlay and compare spectra\n"
            "🎨 Automatic color coding\n"
            "💡 Great for replicate comparison"
        )
        btn.clicked.connect(self._pick_files)
        file_layout.addWidget(btn)
        
        self.file_count = QLabel("No files loaded")
        self.file_count.setStyleSheet("color: #555555; font-size: 13px;")
        file_layout.addWidget(self.file_count)
        file_layout.addStretch()
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Processing options
        options_group = QGroupBox("Processing Options")
        options_layout = QGridLayout()
        options_layout.setSpacing(12)
        
        # Normalization
        self.cb_normalize = QCheckBox("Normalize Intensity")
        self.cb_normalize.setToolTip(
            "Scale each spectrum to 0-1 range.\n\n"
            "📊 Enables intensity comparison\n"
            "📉 Removes absolute scale differences\n"
            "💡 Useful for comparing spectra with different magnitudes"
        )
        self.cb_normalize.stateChanged.connect(self._render_plot)
        options_layout.addWidget(self.cb_normalize, 0, 0)
        
        # Smoothing
        self.cb_smooth = QCheckBox("Apply Smoothing")
        self.cb_smooth.setToolTip(
            "Apply Gaussian smoothing filter.\n\n"
            "📊 Reduces high-frequency noise\n"
            "🔬 Preserves general spectral shape\n"
            "💡 Adjust sigma for smoothing strength"
        )
        self.cb_smooth.stateChanged.connect(self._render_plot)
        options_layout.addWidget(self.cb_smooth, 0, 1)
        
        sigma_label = QLabel("Smoothing σ:")
        sigma_label.setToolTip(
            "Gaussian filter standard deviation.\n\n"
            "⬆️ Higher = more smoothing\n"
            "⬇️ Lower = less smoothing\n"
            "💡 Typical: 1.0-2.0 for spectral data"
        )
        options_layout.addWidget(sigma_label, 1, 0)
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.1, 10.0)
        self.spin_sigma.setSingleStep(0.1)
        self.spin_sigma.setValue(1.0)
        self.spin_sigma.setToolTip("Standard deviation for Gaussian smoothing (sigma)")
        self.spin_sigma.valueChanged.connect(self._render_plot)
        options_layout.addWidget(self.spin_sigma, 1, 1)
        
        # Peak detection
        self.cb_peaks = QCheckBox("Show Peaks")
        self.cb_peaks.setToolTip(
            "Detect and mark spectral peaks.\n\n"
            "🎯 Identifies maxima in spectra\n"
            "💎 Marks peaks with diamond markers\n"
            "📋 Lists peaks in table below"
        )
        self.cb_peaks.stateChanged.connect(self._render_plot)
        options_layout.addWidget(self.cb_peaks, 2, 0, 1, 2)
        
        height_label = QLabel("Min Height:")
        height_label.setToolTip(
            "Minimum peak height threshold.\n\n"
            "📏 Lower for normalized spectra\n"
            "⬆️ Higher = fewer, stronger peaks\n"
            "💡 Adjust based on data scale"
        )
        options_layout.addWidget(height_label, 3, 0)
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(0.0, 10.0)
        self.spin_height.setSingleStep(0.01)
        self.spin_height.setValue(0.01)
        self.spin_height.setToolTip("Minimum intensity for peak detection")
        self.spin_height.valueChanged.connect(self._render_plot)
        options_layout.addWidget(self.spin_height, 3, 1)
        
        distance_label = QLabel("Min Distance (nm):")
        distance_label.setToolTip(
            "Minimum separation between peaks.\n\n"
            "📏 Prevents detecting nearby peaks\n"
            "⬇️ Lower = detects close peaks\n"
            "💡 Typical: 5-20 nm"
        )
        options_layout.addWidget(distance_label, 4, 0)
        self.spin_distance = QDoubleSpinBox()
        self.spin_distance.setRange(1.0, 100.0)
        self.spin_distance.setSingleStep(0.5)
        self.spin_distance.setValue(5.0)
        self.spin_distance.setToolTip("Minimum distance between peaks in nanometers")
        self.spin_distance.valueChanged.connect(self._render_plot)
        options_layout.addWidget(self.spin_distance, 4, 1)
        
        prom_label = QLabel("Min Prominence:")
        prom_label.setToolTip(
            "Peak prominence threshold.\n\n"
            "📏 How much peak stands out\n"
            "⬆️ Higher = only prominent peaks\n"
            "💡 Typical: 0.01-0.1"
        )
        options_layout.addWidget(prom_label, 5, 0)
        self.spin_prom = QDoubleSpinBox()
        self.spin_prom.setRange(0.0, 10.0)
        self.spin_prom.setSingleStep(0.01)
        self.spin_prom.setValue(0.01)
        self.spin_prom.setToolTip("Minimum peak prominence")
        self.spin_prom.valueChanged.connect(self._render_plot)
        options_layout.addWidget(self.spin_prom, 5, 1)
        
        options_layout.setColumnStretch(1, 1)
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Plot
        plot_group = QGroupBox("Spectrum Visualization")
        plot_layout = QVBoxLayout()
        
        self.figure = Figure(figsize=(10, 6), facecolor='none')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(450)
        plot_layout.addWidget(self.canvas)
        
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)

        # Peak table
        table_group = QGroupBox("Detected Peaks")
        table_layout = QVBoxLayout()
        
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Spectrum", "Wavelength (nm)", "Intensity"])
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(200)
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
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(bg_color)
        
        # Clear peak table
        self.table.setRowCount(0)
        
        # Color palette
        colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0', 
                 '#00BCD4', '#FFEB3B', '#795548', '#E91E63', '#607D8B']
        
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
            
            # Peak detection
            if self.cb_peaks.isChecked():
                res = find_peaks_absorbance(
                    wl, yy,
                    height=float(self.spin_height.value()),
                    distance_nm=float(self.spin_distance.value()),
                    prominence=float(self.spin_prom.value()),
                )
                if res and len(res[0]) > 0:
                    pk_idx = res[0]
                    ax.scatter(wl[pk_idx], yy[pk_idx], marker='D', s=60, 
                             color=color, edgecolors='white', linewidths=1.5, 
                             zorder=10)
                    
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
        
        # Resize table columns
        self.table.resizeColumnsToContents()
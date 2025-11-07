"""
Spectrum Capture View for Qt Application
Real-time spectral data acquisition from camera-based spectrometer
Based on PySpectrometer2 by Les Wright
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QSlider, QCheckBox, QSpinBox, QGridLayout, QScrollArea, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import cv2
import time
from pathlib import Path

from qt_app.camera_thread import CameraThread
from qt_app.spec_functions import (
    savitzky_golay, peakIndexes,
    generateGraticule
)


class CaptureView(QWidget):
    """Main view for live spectrum capture and calibration."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # State variables
        self.current_frame = None
        self.intensity = np.zeros(800)
        self.wavelengthData = []
        self.hold_peaks = False
        self.calibration_points = []  # [(pixel, wavelength), ...]
        self.click_positions = []  # Pixel positions from plot clicks
        
        # Polynomial calibration system (improved over Theremino's 2-point linear)
        self.calib_peaks = []  # List of calibration peaks: [(pixel_pos, known_wavelength), ...]
        self.calib_polynomial_degree = 2  # Default to quadratic fit
        self.calib_polynomial_coeffs = None  # Stores polynomial coefficients after fit
        
        # Theremino-style trim/calibration points (kept for backward compatibility)
        self.trim_point1 = 436.0  # First calibration wavelength (nm)
        self.trim_point2 = 546.0  # Second calibration wavelength (nm)
        self.trim_mode = False    # When true, can drag spectrum to align
        self.drag_start_x = None
        self.initial_nm_min = None
        self.initial_nm_max = None
        
        # Camera settings
        self.frame_width = 1920  # Default to 1080p
        self.frame_height = 1080
        self.camera_exposure = -1  # Auto exposure
        self.camera_fps = 30
        
        # ROI settings for spectral line extraction (absolute position on frame)
        self.roi_x_start = 0    # X position of left edge of ROI
        self.roi_x_end = 1120   # X position of right edge of ROI
        self.roi_y_start = 544  # Y position of top of ROI (default: center)
        self.roi_height = 30    # Height of ROI
        
        # Processing settings
        self.savpoly = 7
        self.mindist = 50
        self.thresh = 20
        
        # Theremino-style processing parameters
        self.filter_strength = 30  # 0-100, noise filtering
        self.rising_speed = 50     # 0-100, how fast to track increases
        self.falling_speed = 10    # 0-100, how fast to track decreases
        self.flip_horizontal = True  # Flip camera image horizontally by default
        
        # Temporal filtering state
        self.spec_array = None  # Smoothed intensity with rising/falling speed
        self.spec_filtered = None  # Further filtered with noise filter
        
        # Irradiance correction (Theremino-style spectral response calibration)
        self.irradiance_correction_enabled = True  # Always enabled
        # Theremino uses intensity scale 0-800 with noise=2.2
        # We use scale 0-300, so scale the noise: 2.2 * (300/800) = 0.825
        self.average_bg_noise = 0.825  # Scaled from Theremino's 2.2 for our 0-300 range
        self.uv380_coeff = 2.568  # UV region correction coefficient
        self.ir780_coeff = 1.41   # IR region correction coefficient
        self.init_irradiance_coefficients()  # Initialize correction lookup table
        
        # Performance optimization
        self.frame_skip = 0  # Skip frames for performance
        self.frame_counter = 0
        self.update_interval = 100  # ms between graph updates (was 50)
        
        # Initialize calibration to 300-850 nm linear scale
        self.nm_min = 300.0
        self.nm_max = 850.0
        self.wavelengthData = np.linspace(self.nm_min, self.nm_max, self.frame_width).tolist()
        
        # Generate graticule
        self.graticule_data = generateGraticule(self.wavelengthData)
        self.tens = self.graticule_data[0]
        self.fifties = self.graticule_data[1]
        
        # Initialize UI
        self.init_ui()
        
        # Try to load saved calibration
        self.load_calibration_from_file()
        
        # Initialize camera thread
        self.camera_thread = None
    
    def init_irradiance_coefficients(self):
        """
        Initialize irradiance correction coefficients from Theremino Spectrometer.
        These correct for spectral sensitivity variations of the camera sensor.
        Covers 380-780 nm range with 401 data points.
        """
        # Theremino's standard wavelength irradiance correction coefficients (380-780 nm, 1nm steps)
        irradiance_coeffs = [
            2.568, 2.568, 2.568, 2.568, 2.568, 2.568, 2.562, 2.532, 2.504, 2.364, 2.282, 2.258, 2.19, 2.072, 1.94, 1.836, 1.753, 1.642, 1.555, 1.49,
            1.423, 1.37, 1.3, 1.243, 1.214, 1.18, 1.155, 1.155, 1.16, 1.165, 1.175, 1.188, 1.204, 1.221, 1.25, 1.263, 1.258, 1.277, 1.291, 1.32,
            1.348, 1.376, 1.381, 1.376, 1.359, 1.344, 1.326, 1.313, 1.292, 1.273, 1.254, 1.24, 1.218, 1.204, 1.186, 1.173, 1.147, 1.117, 1.089, 1.064,
            1.049, 1.04, 1.038, 1.036, 1.036, 1.036, 1.036, 1.038, 1.041, 1.046, 1.049, 1.051, 1.052, 1.055, 1.057, 1.057, 1.056, 1.054, 1.053, 1.049,
            1.043, 1.035, 1.028, 1.02, 1.01, 0.996, 0.979, 0.963, 0.945, 0.927, 0.908, 0.893, 0.88, 0.868, 0.857, 0.846, 0.838, 0.828, 0.82, 0.813,
            0.81, 0.808, 0.808, 0.809, 0.812, 0.814, 0.816, 0.819, 0.822, 0.827, 0.832, 0.84, 0.85, 0.859, 0.868, 0.874, 0.88, 0.886, 0.891, 0.896,
            0.9, 0.904, 0.908, 0.91, 0.909, 0.908, 0.906, 0.903, 0.899, 0.895, 0.891, 0.887, 0.883, 0.878, 0.873, 0.867, 0.86, 0.852, 0.845, 0.839,
            0.833, 0.827, 0.822, 0.817, 0.813, 0.808, 0.805, 0.803, 0.802, 0.8, 0.798, 0.796, 0.795, 0.794, 0.794, 0.794, 0.793, 0.793, 0.794, 0.789,
            0.786, 0.784, 0.788, 0.793, 0.797, 0.8, 0.804, 0.808, 0.812, 0.815, 0.818, 0.821, 0.824, 0.825, 0.827, 0.827, 0.827, 0.826, 0.824, 0.823,
            0.822, 0.822, 0.821, 0.82, 0.818, 0.817, 0.815, 0.813, 0.809, 0.806, 0.803, 0.802, 0.802, 0.801, 0.8, 0.798, 0.796, 0.794, 0.792, 0.791,
            0.791, 0.792, 0.795, 0.798, 0.802, 0.806, 0.81, 0.814, 0.82, 0.825, 0.832, 0.839, 0.845, 0.852, 0.858, 0.864, 0.87, 0.877, 0.885, 0.895,
            0.905, 0.915, 0.924, 0.935, 0.943, 0.952, 0.958, 0.967, 0.975, 0.984, 0.993, 1.001, 1.012, 1.02, 1.028, 1.031, 1.036, 1.039, 1.042, 1.046,
            1.05, 1.055, 1.061, 1.065, 1.069, 1.071, 1.071, 1.069, 1.066, 1.064, 1.062, 1.062, 1.063, 1.064, 1.064, 1.062, 1.058, 1.053, 1.048, 1.044,
            1.043, 1.044, 1.045, 1.044, 1.041, 1.036, 1.029, 1.023, 1.018, 1.015, 1.01, 1.007, 1.005, 1.003, 1, 0.995, 0.992, 0.991, 0.99, 0.987,
            0.985, 0.984, 0.983, 0.98, 0.978, 0.979, 0.982, 0.985, 0.991, 0.996, 1.002, 1.003, 1.003, 1.001, 1.003, 1.005, 1.008, 1.011, 1.016, 1.018,
            1.017, 1.016, 1.018, 1.026, 1.032, 1.039, 1.045, 1.053, 1.057, 1.057, 1.06, 1.064, 1.07, 1.073, 1.076, 1.081, 1.086, 1.087, 1.089, 1.091,
            1.093, 1.095, 1.098, 1.102, 1.104, 1.108, 1.111, 1.115, 1.117, 1.118, 1.122, 1.126, 1.128, 1.131, 1.132, 1.134, 1.138, 1.14, 1.142, 1.144,
            1.145, 1.146, 1.148, 1.148, 1.145, 1.147, 1.141, 1.138, 1.14, 1.14, 1.14, 1.143, 1.145, 1.149, 1.15, 1.152, 1.156, 1.156, 1.157, 1.161,
            1.165, 1.17, 1.172, 1.174, 1.176, 1.179, 1.183, 1.186, 1.189, 1.192, 1.204, 1.2, 1.205, 1.21, 1.215, 1.219, 1.225, 1.23, 1.244, 1.241,
            1.246, 1.252, 1.257, 1.264, 1.271, 1.278, 1.283, 1.299, 1.299, 1.309, 1.314, 1.322, 1.332, 1.34, 1.347, 1.367, 1.368, 1.376, 1.387, 1.398, 1.41
        ]
        
        # Create dictionary mapping wavelength (380-780 nm) to correction coefficient
        self.irradiance_dict = {}
        for i, coeff in enumerate(irradiance_coeffs):
            wavelength = 380 + i  # Start at 380 nm, increment by 1 nm
            self.irradiance_dict[wavelength] = coeff
        
        print(f"Irradiance correction initialized: {len(self.irradiance_dict)} wavelength points (380-780 nm)")
    
    def apply_irradiance_correction(self, intensity_data):
        """
        Apply Theremino-style irradiance correction to intensity data.
        Corrects for spectral sensitivity variations of the camera sensor.
        
        Args:
            intensity_data: numpy array of intensity values for each pixel
            
        Returns:
            numpy array of corrected intensity values
        """
        if not self.irradiance_correction_enabled:
            return intensity_data
        
        corrected = np.zeros_like(intensity_data)
        max_value = 300.0  # Maximum intensity value for clamping
        
        for i in range(len(intensity_data)):
            # Get wavelength for this pixel position
            if i < len(self.wavelengthData):
                wavelength = self.wavelengthData[i]
            else:
                # Fallback: no correction
                corrected[i] = intensity_data[i]
                continue
            
            # Round wavelength to nearest integer for lookup
            wl_int = int(round(wavelength))
            
            # Apply correction based on wavelength region
            if wl_int < 380:
                # UV region: use fixed coefficient
                corrected_value = (intensity_data[i] - self.average_bg_noise) * self.uv380_coeff
                corrected_value = max(0, corrected_value)  # No negative values
                corrected[i] = min(corrected_value, max_value)
                
            elif wl_int <= 780:
                # Visible region: use lookup table
                if wl_int in self.irradiance_dict:
                    coeff = self.irradiance_dict[wl_int]
                    corrected_value = (intensity_data[i] - self.average_bg_noise) * coeff
                    corrected_value = max(0, corrected_value)
                    corrected[i] = min(corrected_value, max_value)
                else:
                    # Wavelength not in table, use nearest or no correction
                    corrected[i] = intensity_data[i]
                    
            else:
                # IR region: use fixed coefficient
                corrected_value = intensity_data[i] * self.ir780_coeff
                corrected[i] = min(corrected_value, max_value)
        
        return corrected
        
    def init_ui(self):
        """Initialize the user interface."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(60, 20, 60, 20)
        layout.setSpacing(20)
        
        # Title Section
        title = QLabel("📷 Live Spectrum Capture")
        title.setProperty("class", "title")
        layout.addWidget(title)
        
        subtitle = QLabel("Real-time spectral analysis from camera-based spectrometer with precise wavelength calibration and built-in irradiance and dark current correction")
        subtitle.setProperty("class", "subtitle")
        layout.addWidget(subtitle)
        
        # Status bar (moved to top for better visibility)
        self.status_label = QLabel(f"✓ Ready | Default calibration: 300-1100 nm")
        self.status_label.setProperty("class", "status")
        layout.addWidget(self.status_label)
        
        # Main Content Grid (Camera + Spectrum side by side on wider screens)
        main_content = QHBoxLayout()
        main_content.setSpacing(20)
        
        # Left Column - Camera Controls & Preview
        left_column = QVBoxLayout()
        left_column.setSpacing(15)
        
        # Camera control
        camera_group = self.create_camera_controls()
        left_column.addWidget(camera_group)
        
        # Camera preview
        preview_group = QGroupBox("📹 Camera Feed")
        preview_layout = QVBoxLayout()
        
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 240)
        self.camera_label.setMaximumHeight(400)
        self.camera_label.setStyleSheet("""
            background-color: #1a1a1a; 
            border: 2px solid #667eea;
            border-radius: 8px;
            color: #ffffff;
            font-size: 13pt;
            font-weight: 500;
        """)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setText("📷 Camera not started\n\nClick 'Start Camera' to begin")
        preview_layout.addWidget(self.camera_label)
        
        preview_group.setLayout(preview_layout)
        left_column.addWidget(preview_group)
        
        # Camera settings (collapsed)
        settings_group = self.create_camera_settings()
        left_column.addWidget(settings_group)
        
        left_column.addStretch()
        
        # Right Column - Spectrum Display
        right_column = QVBoxLayout()
        right_column.setSpacing(15)
        
        # Spectrum graph
        spectrum_group = QGroupBox("📈 Live Spectrum Analysis")
        spectrum_layout = QVBoxLayout()
        
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(500)
        self.ax = self.figure.add_subplot(111)
        self.spec_ax = self.ax  # Alias for trim mode compatibility
        self.spec_canvas = self.canvas  # Alias for trim mode compatibility
        
        # Connect click event for calibration
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        
        spectrum_layout.addWidget(self.canvas)
        spectrum_group.setLayout(spectrum_layout)
        right_column.addWidget(spectrum_group)
        
        # Processing controls (compact)
        controls_group = self.create_processing_controls()
        right_column.addWidget(controls_group)
        
        # Add columns to main content
        main_content.addLayout(left_column, stretch=1)
        main_content.addLayout(right_column, stretch=2)
        
        layout.addLayout(main_content)
        
        # Bottom section - Calibration (full width)
        cal_group = self.create_calibration_section()
        layout.addWidget(cal_group)
        
        layout.addStretch()
        
        scroll.setWidget(content)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        
        # Update timer for spectrum rendering
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_spectrum)
        
    def create_camera_controls(self):
        """Create camera control section."""
        group = QGroupBox("Camera Controls")
        layout = QGridLayout()
        layout.setSpacing(10)
        
        # Start/Stop button
        self.start_btn = QPushButton("▶ Start Camera")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setToolTip("Start live camera capture")
        self.start_btn.clicked.connect(self.toggle_camera)
        layout.addWidget(self.start_btn, 0, 0, 1, 2)
        
        # Camera device
        layout.addWidget(QLabel("Camera Device:"), 1, 0)
        self.device_spin = QSpinBox()
        self.device_spin.setRange(0, 10)
        self.device_spin.setValue(1)
        self.device_spin.setToolTip("Camera device ID (1 for default)")
        layout.addWidget(self.device_spin, 1, 1)
        
        group.setLayout(layout)
        return group
    
    def create_camera_settings(self):
        """Create camera settings section."""
        group = QGroupBox("Camera Settings")
        layout = QGridLayout()
        layout.setSpacing(10)
        
        # Resolution dropdown
        layout.addWidget(QLabel("Resolution:"), 0, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "640 × 480",
            "1280 × 720",
            "1920 × 1080"
        ])
        self.resolution_combo.setCurrentIndex(2)  # Default to 1920x1080
        self.resolution_combo.setToolTip("Camera resolution - lower is faster")
        layout.addWidget(self.resolution_combo, 0, 1)
        
        # FPS
        layout.addWidget(QLabel("Frame Rate:"), 1, 0)
        fps_layout = QHBoxLayout()
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(5, 60)
        self.fps_spin.setValue(30)
        self.fps_spin.setToolTip("Frames per second (lower = less CPU)")
        self.fps_label_val = QLabel("30 fps")
        self.fps_spin.valueChanged.connect(lambda v: self.fps_label_val.setText(f"{v} fps"))
        fps_layout.addWidget(self.fps_spin)
        fps_layout.addWidget(self.fps_label_val)
        layout.addLayout(fps_layout, 1, 1)
        
        # Exposure
        layout.addWidget(QLabel("Exposure:"), 2, 0)
        exposure_layout = QHBoxLayout()
        self.exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.exposure_slider.setRange(-13, 0)  # OpenCV exposure values
        self.exposure_slider.setValue(-1)
        self.exposure_slider.setToolTip("Camera exposure (-13 to 0, -1 = auto)")
        self.exposure_slider.valueChanged.connect(self.on_exposure_changed)
        self.exposure_label = QLabel("Auto")
        exposure_layout.addWidget(self.exposure_slider)
        exposure_layout.addWidget(self.exposure_label)
        layout.addLayout(exposure_layout, 2, 1)
        
        # --- ROI Settings ---
        roi_separator = QLabel("═══ Region of Interest (ROI) ═══")
        roi_separator.setStyleSheet("color: #0066cc; font-weight: bold; margin-top: 10px;")
        roi_separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(roi_separator, 3, 0, 1, 2)
        
        # ROI X Start Position
        layout.addWidget(QLabel("ROI X Start:"), 4, 0)
        roi_x_start_layout = QHBoxLayout()
        self.roi_x_start_slider = QSlider(Qt.Orientation.Horizontal)
        self.roi_x_start_slider.setRange(0, 1920)
        self.roi_x_start_slider.setValue(0)
        self.roi_x_start_slider.setToolTip("X coordinate of left edge of ROI")
        self.roi_x_start_slider.valueChanged.connect(self.on_roi_x_start_changed)
        self.roi_x_start_label = QLabel("0 px")
        roi_x_start_layout.addWidget(self.roi_x_start_slider)
        roi_x_start_layout.addWidget(self.roi_x_start_label)
        layout.addLayout(roi_x_start_layout, 4, 1)
        
        # ROI X End Position
        layout.addWidget(QLabel("ROI X End:"), 5, 0)
        roi_x_end_layout = QHBoxLayout()
        self.roi_x_end_slider = QSlider(Qt.Orientation.Horizontal)
        self.roi_x_end_slider.setRange(0, 1920)
        self.roi_x_end_slider.setValue(1100)  # Default to 1100px
        self.roi_x_end_slider.setToolTip("X coordinate of right edge of ROI")
        self.roi_x_end_slider.valueChanged.connect(self.on_roi_x_end_changed)
        self.roi_x_end_label = QLabel("1100 px")
        roi_x_end_layout.addWidget(self.roi_x_end_slider)
        roi_x_end_layout.addWidget(self.roi_x_end_label)
        layout.addLayout(roi_x_end_layout, 5, 1)
        
        # ROI Y Position (absolute)
        layout.addWidget(QLabel("ROI Y Start:"), 6, 0)
        roi_y_layout = QHBoxLayout()
        self.roi_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.roi_y_slider.setRange(0, 1080)  # Max common resolution height
        self.roi_y_slider.setValue(544)  # Default to 544px
        self.roi_y_slider.setToolTip("Y coordinate of top of ROI (pixels from top of frame)")
        self.roi_y_slider.valueChanged.connect(self.on_roi_y_changed)
        self.roi_y_label = QLabel("544 px")
        roi_y_layout.addWidget(self.roi_y_slider)
        roi_y_layout.addWidget(self.roi_y_label)
        layout.addLayout(roi_y_layout, 6, 1)
        
        # ROI Height
        layout.addWidget(QLabel("ROI Height:"), 7, 0)
        roi_h_layout = QHBoxLayout()
        self.roi_h_slider = QSlider(Qt.Orientation.Horizontal)
        self.roi_h_slider.setRange(10, 300)
        self.roi_h_slider.setValue(30)  # Default to 30px
        self.roi_h_slider.setToolTip("Height of spectral line region (pixels)")
        self.roi_h_slider.valueChanged.connect(self.on_roi_h_changed)
        self.roi_h_label = QLabel("30 px")
        roi_h_layout.addWidget(self.roi_h_slider)
        roi_h_layout.addWidget(self.roi_h_label)
        layout.addLayout(roi_h_layout, 7, 1)
        
        # Update interval (performance)
        layout.addWidget(QLabel("Update Interval:"), 8, 0)
        interval_layout = QHBoxLayout()
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(33, 500)
        self.interval_spin.setValue(100)
        self.interval_spin.setSingleStep(10)
        self.interval_spin.setSuffix(" ms")
        self.interval_spin.setToolTip("Graph update interval (higher = less CPU, lower = smoother)")
        self.interval_spin.valueChanged.connect(self.on_interval_changed)
        interval_layout.addWidget(self.interval_spin)
        interval_layout.addWidget(QLabel("(↑ = less lag)"))
        layout.addLayout(interval_layout, 8, 1)
        
        # Apply button
        self.apply_settings_btn = QPushButton("🔄 Apply Resolution/FPS")
        self.apply_settings_btn.setMinimumHeight(35)
        self.apply_settings_btn.setToolTip("Restart camera with new resolution/FPS settings")
        self.apply_settings_btn.clicked.connect(self.apply_camera_settings)
        layout.addWidget(self.apply_settings_btn, 9, 0, 1, 2)
        
        # Reset camera settings button
        self.reset_camera_btn = QPushButton("🔄 Reset Camera Settings to Defaults")
        self.reset_camera_btn.setMinimumHeight(30)
        self.reset_camera_btn.setStyleSheet("background-color: #ff9800; color: white;")
        self.reset_camera_btn.setToolTip("Reset all camera settings to default values")
        self.reset_camera_btn.clicked.connect(self.reset_camera_settings)
        layout.addWidget(self.reset_camera_btn, 10, 0, 1, 2)
        
        group.setLayout(layout)
        return group
    
    def create_processing_controls(self):
        """Create spectrum processing controls - compact version."""
        from PyQt6.QtWidgets import QTabWidget
        
        group = QGroupBox("⚙️ Spectrum Processing")
        main_layout = QVBoxLayout()
        
        # Tabs for organization
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        
        # Tab 1: Temporal Filtering (Theremino)
        filter_tab = QWidget()
        filter_layout = QGridLayout(filter_tab)
        filter_layout.setSpacing(10)
        
        filter_layout.addWidget(QLabel("Filter Strength:"), 0, 0)
        filter_h = QHBoxLayout()
        self.filter_slider = QSlider(Qt.Orientation.Horizontal)
        self.filter_slider.setRange(0, 100)
        self.filter_slider.setValue(30)
        self.filter_slider.setToolTip("Noise filtering (0=none, 100=max)")
        self.filter_slider.valueChanged.connect(self.on_filter_changed)
        self.filter_label = QLabel("30")
        self.filter_label.setMinimumWidth(35)
        filter_h.addWidget(self.filter_slider)
        filter_h.addWidget(self.filter_label)
        filter_layout.addLayout(filter_h, 0, 1)
        
        filter_layout.addWidget(QLabel("Rising Speed:"), 1, 0)
        rising_h = QHBoxLayout()
        self.rising_slider = QSlider(Qt.Orientation.Horizontal)
        self.rising_slider.setRange(0, 100)
        self.rising_slider.setValue(50)
        self.rising_slider.setToolTip("Response for intensity increases")
        self.rising_slider.valueChanged.connect(self.on_rising_changed)
        self.rising_label = QLabel("50")
        self.rising_label.setMinimumWidth(35)
        rising_h.addWidget(self.rising_slider)
        rising_h.addWidget(self.rising_label)
        filter_layout.addLayout(rising_h, 1, 1)
        
        filter_layout.addWidget(QLabel("Falling Speed:"), 2, 0)
        falling_h = QHBoxLayout()
        self.falling_slider = QSlider(Qt.Orientation.Horizontal)
        self.falling_slider.setRange(0, 100)
        self.falling_slider.setValue(10)
        self.falling_slider.setToolTip("Response for intensity decreases")
        self.falling_slider.valueChanged.connect(self.on_falling_changed)
        self.falling_label = QLabel("10")
        self.falling_label.setMinimumWidth(35)
        falling_h.addWidget(self.falling_slider)
        falling_h.addWidget(self.falling_label)
        filter_layout.addLayout(falling_h, 2, 1)
        
        tabs.addTab(filter_tab, "🔧 Filtering")
        
        # Tab 2: Peak Detection
        peak_tab = QWidget()
        peak_layout = QGridLayout(peak_tab)
        peak_layout.setSpacing(10)
        
        peak_layout.addWidget(QLabel("Savgol Order:"), 0, 0)
        savpoly_h = QHBoxLayout()
        self.savpoly_slider = QSlider(Qt.Orientation.Horizontal)
        self.savpoly_slider.setRange(0, 15)
        self.savpoly_slider.setValue(7)
        self.savpoly_slider.setToolTip("Savitzky-Golay filter order")
        self.savpoly_slider.valueChanged.connect(self.on_savpoly_changed)
        self.savpoly_label = QLabel("7")
        self.savpoly_label.setMinimumWidth(35)
        savpoly_h.addWidget(self.savpoly_slider)
        savpoly_h.addWidget(self.savpoly_label)
        peak_layout.addLayout(savpoly_h, 0, 1)
        
        peak_layout.addWidget(QLabel("Peak Distance:"), 1, 0)
        mindist_h = QHBoxLayout()
        self.mindist_slider = QSlider(Qt.Orientation.Horizontal)
        self.mindist_slider.setRange(1, 100)
        self.mindist_slider.setValue(50)
        self.mindist_slider.setToolTip("Min distance between peaks")
        self.mindist_slider.valueChanged.connect(self.on_mindist_changed)
        self.mindist_label = QLabel("50")
        self.mindist_label.setMinimumWidth(35)
        mindist_h.addWidget(self.mindist_slider)
        mindist_h.addWidget(self.mindist_label)
        peak_layout.addLayout(mindist_h, 1, 1)
        
        peak_layout.addWidget(QLabel("Label Threshold:"), 2, 0)
        thresh_h = QHBoxLayout()
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(20)
        self.thresh_slider.setToolTip("Peak labeling threshold")
        self.thresh_slider.valueChanged.connect(self.on_thresh_changed)
        self.thresh_label = QLabel("20")
        self.thresh_label.setMinimumWidth(35)
        thresh_h.addWidget(self.thresh_slider)
        thresh_h.addWidget(self.thresh_label)
        peak_layout.addLayout(thresh_h, 2, 1)
        
        tabs.addTab(peak_tab, "🎯 Peaks")
        
        # Tab 3: Display Options
        display_tab = QWidget()
        display_layout = QVBoxLayout(display_tab)
        display_layout.setSpacing(10)
        
        # Wavelength range settings
        range_group = QGroupBox("Wavelength Display Range")
        range_layout = QGridLayout()
        range_layout.setSpacing(10)
        
        range_layout.addWidget(QLabel("Min Wavelength:"), 0, 0)
        nm_min_layout = QHBoxLayout()
        self.nm_min_spin = QSpinBox()
        self.nm_min_spin.setRange(200, 1000)
        self.nm_min_spin.setValue(300)
        self.nm_min_spin.setSuffix(" nm")
        self.nm_min_spin.setToolTip("Minimum wavelength to display")
        self.nm_min_spin.valueChanged.connect(self.on_wavelength_range_changed)
        nm_min_layout.addWidget(self.nm_min_spin)
        range_layout.addLayout(nm_min_layout, 0, 1)
        
        range_layout.addWidget(QLabel("Max Wavelength:"), 1, 0)
        nm_max_layout = QHBoxLayout()
        self.nm_max_spin = QSpinBox()
        self.nm_max_spin.setRange(300, 1200)
        self.nm_max_spin.setValue(850)
        self.nm_max_spin.setSuffix(" nm")
        self.nm_max_spin.setToolTip("Maximum wavelength to display")
        self.nm_max_spin.valueChanged.connect(self.on_wavelength_range_changed)
        nm_max_layout.addWidget(self.nm_max_spin)
        range_layout.addLayout(nm_max_layout, 1, 1)
        
        self.apply_range_btn = QPushButton("✓ Apply Range")
        self.apply_range_btn.setToolTip("Apply new wavelength range to spectrum")
        self.apply_range_btn.clicked.connect(self.apply_wavelength_range)
        range_layout.addWidget(self.apply_range_btn, 2, 0, 1, 2)
        
        range_group.setLayout(range_layout)
        display_layout.addWidget(range_group)
        
        # Display options
        self.hold_peaks_cb = QCheckBox("Hold Peak Maxima")
        self.hold_peaks_cb.setToolTip("Hold maximum intensity values")
        self.hold_peaks_cb.stateChanged.connect(self.on_hold_peaks_changed)
        display_layout.addWidget(self.hold_peaks_cb)
        
        self.flip_horizontal_cb = QCheckBox("Flip Camera Horizontally")
        self.flip_horizontal_cb.setChecked(True)  # Default to flipped
        self.flip_horizontal_cb.setToolTip("Mirror camera image")
        self.flip_horizontal_cb.stateChanged.connect(self.on_flip_changed)
        display_layout.addWidget(self.flip_horizontal_cb)
        
        display_layout.addStretch()
        
        tabs.addTab(display_tab, "📊 Display")
        
        main_layout.addWidget(tabs)
        
        # Save button at bottom
        button_layout = QHBoxLayout()
        self.save_spectrum_btn = QPushButton("💾 Save Spectrum")
        self.save_spectrum_btn.setToolTip("Save current spectrum to CSV")
        self.save_spectrum_btn.clicked.connect(self.save_spectrum)
        button_layout.addWidget(self.save_spectrum_btn)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)
        
        group.setLayout(main_layout)
        return group
    
    def create_calibration_section(self):
        """Create calibration controls with polynomial fitting support."""
        group = QGroupBox("Wavelength Calibration - Polynomial Fit (Drag Peaks to Known Lines)")
        layout = QVBoxLayout()
        
        info = QLabel(
            "📌 Advanced Polynomial Calibration:\n"
            "1. Point spectrometer at known emission source (e.g., fluorescent lamp, LED)\n"
            "2. Add calibration points by clicking 'Add Calibration Point' and entering known wavelengths\n"
            "3. Click 'Apply Calibration' to compute the polynomial fit\n"
            "4. Polynomial fit automatically adjusts the entire wavelength scale\n"
            "5. Use 2+ points for linear, 3+ for quadratic, 4+ for cubic calibration"
        )
        info.setProperty("class", "info-box")
        layout.addWidget(info)
        
        # Calibration mode controls
        calib_controls = QHBoxLayout()
        
        # Polynomial degree selector
        calib_controls.addWidget(QLabel("Polynomial Fit Degree:"))
        self.poly_degree_spin = QSpinBox()
        self.poly_degree_spin.setRange(1, 5)
        self.poly_degree_spin.setValue(2)
        self.poly_degree_spin.setToolTip("Polynomial degree: 1=linear, 2=quadratic, 3=cubic, etc.")
        self.poly_degree_spin.valueChanged.connect(self.on_poly_degree_changed)
        calib_controls.addWidget(self.poly_degree_spin)
        
        calib_controls.addStretch()
        layout.addLayout(calib_controls)
        
        # Calibration points management
        points_layout = QVBoxLayout()
        points_header = QHBoxLayout()
        
        points_label = QLabel("Calibration Points:")
        points_label.setStyleSheet("font-weight: bold; font-size: 10pt;")
        points_header.addWidget(points_label)
        
        self.add_point_btn = QPushButton("➕ Add Point")
        self.add_point_btn.setToolTip("Add a new calibration point")
        self.add_point_btn.clicked.connect(self.add_calibration_point)
        points_header.addWidget(self.add_point_btn)
        
        self.clear_points_btn = QPushButton("🗑️ Clear All")
        self.clear_points_btn.setToolTip("Clear all calibration points and reset to default linear scale")
        self.clear_points_btn.clicked.connect(self.clear_calibration_points)
        points_header.addWidget(self.clear_points_btn)
        
        self.load_calib_btn = QPushButton("📁 Load Calibration")
        self.load_calib_btn.setToolTip("Load saved calibration from file")
        self.load_calib_btn.clicked.connect(self.load_calibration_button_clicked)
        points_header.addWidget(self.load_calib_btn)
        
        self.apply_calib_btn = QPushButton("✓ Apply Calibration")
        self.apply_calib_btn.setToolTip("Compute and apply polynomial fit to wavelength scale")
        self.apply_calib_btn.clicked.connect(self.apply_polynomial_calibration)
        self.apply_calib_btn.setProperty("class", "primary")
        points_header.addWidget(self.apply_calib_btn)
        
        self.save_calib_btn = QPushButton("💾 Save Calibration")
        self.save_calib_btn.setToolTip("Save current calibration to file and cache")
        self.save_calib_btn.clicked.connect(self.save_calibration_to_file)
        self.save_calib_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        points_header.addWidget(self.save_calib_btn)
        
        points_header.addStretch()
        points_layout.addLayout(points_header)
        
        # Table to display calibration points
        self.calib_table = QTableWidget()
        self.calib_table.setColumnCount(4)
        self.calib_table.setHorizontalHeaderLabels(["Pixel Position", "Known Wavelength (nm)", "Current λ (nm)", "Actions"])
        self.calib_table.horizontalHeader().setStretchLastSection(False)
        self.calib_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.calib_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.calib_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.calib_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.calib_table.setColumnWidth(3, 100)  # Increased to 100px for Edit + Delete buttons
        self.calib_table.verticalHeader().setDefaultSectionSize(45)  # Set row height to 45px (increased from 35px)
        self.calib_table.setMinimumHeight(300)  # Increased from 200 to 300
        self.calib_table.setMaximumHeight(800)  # Increased from 600 to 800
        self.calib_table.setToolTip("List of calibration points for polynomial fitting")
        points_layout.addWidget(self.calib_table)
        
        layout.addLayout(points_layout)
        
        # Calibration status
        self.calib_status_label = QLabel("No calibration applied - using default linear scale")
        self.calib_status_label.setProperty("class", "info-box")
        self.calib_status_label.setStyleSheet("font-size: 9pt; padding: 8px;")
        layout.addWidget(self.calib_status_label)
        
        group.setLayout(layout)
        return group
    
    def toggle_camera(self):
        """Start or stop the camera."""
        if self.camera_thread is None or not self.camera_thread.isRunning():
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera capture."""
        device_id = self.device_spin.value()
        
        # Get resolution from dropdown
        resolution_map = {
            0: (640, 480),
            1: (1280, 720),
            2: (1920, 1080)
        }
        self.frame_width, self.frame_height = resolution_map[self.resolution_combo.currentIndex()]
        fps = self.fps_spin.value()
        
        self.camera_thread = CameraThread(
            device_id=device_id,
            width=self.frame_width,
            height=self.frame_height,
            fps=fps
        )
        
        self.camera_thread.frame_ready.connect(self.on_frame_ready)
        self.camera_thread.error_occurred.connect(self.on_camera_error)
        
        self.camera_thread.start()
        self.update_timer.start(self.update_interval)
        
        self.start_btn.setText("⏸ Stop Camera")
        self.camera_label.setText("Camera starting...")
        self.device_spin.setEnabled(False)
        
        # Disable resolution/FPS changes while running
        self.resolution_combo.setEnabled(False)
        self.fps_spin.setEnabled(False)
        self.apply_settings_btn.setEnabled(False)
        
        # Apply camera settings after a short delay
        QTimer.singleShot(500, self.apply_current_camera_settings)
    
    def stop_camera(self):
        """Stop camera capture."""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        
        self.update_timer.stop()
        self.start_btn.setText("▶ Start Camera")
        self.camera_label.setText("Camera stopped")
        self.device_spin.setEnabled(True)
        
        # Re-enable settings
        self.resolution_combo.setEnabled(True)
        self.fps_spin.setEnabled(True)
        self.apply_settings_btn.setEnabled(True)
    
    def on_frame_ready(self, frame):
        """Handle new frame from camera."""
        self.current_frame = frame
        
        # Flip horizontal if enabled
        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)
        
        # Frame skip optimization - only process every Nth frame for display
        self.frame_counter += 1
        if self.frame_counter % 2 != 0:  # Skip every other frame for preview
            return
        
        # Display full frame with ROI overlay (throttled)
        if self.frame_counter % 4 == 0:  # Update preview less frequently
            self.display_camera_frame(frame.copy())
        
        # Extract spectral line using absolute ROI position
        try:
            x = max(0, self.roi_x_start)
            x_end = min(frame.shape[1], self.roi_x_end)
            w = x_end - x
            
            y = self.roi_y_start
            h = self.roi_height
            
            # Ensure we don't exceed frame bounds
            if y < 0:
                y = 0
            if y + h > frame.shape[0]:
                h = frame.shape[0] - y
            if h <= 0 or w <= 0:
                return
            
            cropped = frame[y:y+h, x:x_end]
            
            # Process intensity data - Theremino style with grayscale (monochrome camera)
            # Convert to grayscale and sum across ROI height
            bwimage = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY).astype(np.float32)
            
            # Average intensity across height (simpler for monochrome)
            intensity_raw = np.mean(bwimage, axis=0)
            
            # Initialize arrays if needed
            if self.spec_array is None or len(self.spec_array) != len(intensity_raw):
                self.spec_array = intensity_raw.copy()
                self.spec_filtered = intensity_raw.copy()
                self.intensity = intensity_raw.copy()
            
            # Apply Theremino-style Rising/Falling speed filtering
            k_speed_up = self.rising_speed / 100.0
            k_speed_down = self.falling_speed / 100.0
            
            # For each pixel, apply different speed based on if intensity is rising or falling
            mask_rising = intensity_raw > self.spec_array
            self.spec_array[mask_rising] += (intensity_raw[mask_rising] - self.spec_array[mask_rising]) * k_speed_up
            self.spec_array[~mask_rising] += (intensity_raw[~mask_rising] - self.spec_array[~mask_rising]) * k_speed_down
            
            # Apply noise filter (bidirectional smoothing like Theremino)
            k_filter = (100 - self.filter_strength) / 100.0 + 0.1
            
            # Forward pass
            v = self.spec_filtered[0]
            for i in range(len(self.spec_array)):
                vnew = self.spec_array[i]
                v += (vnew - v) * k_filter
                self.spec_filtered[i] = v
            
            # Backward pass  
            v = self.spec_filtered[-1]
            for i in range(len(self.spec_array) - 1, -1, -1):
                vnew = self.spec_array[i]
                v += (vnew - v) * k_filter
                self.spec_filtered[i] = v
            
            # Apply irradiance correction (compensate for camera spectral sensitivity)
            self.spec_filtered = self.apply_irradiance_correction(self.spec_filtered)
            
            # Store final filtered result
            if self.hold_peaks:
                self.intensity = np.maximum(self.intensity, self.spec_filtered)
            else:
                self.intensity = self.spec_filtered.copy()
                
        except Exception as e:
            print(f"Frame processing error: {e}")
    
    def display_camera_frame(self, frame):
        """Display full camera frame with ROI bounding box overlay."""
        h, w = frame.shape[:2]
        
        # Calculate ROI bounds
        roi_x = max(0, min(self.roi_x_start, w))
        roi_x_end = max(roi_x + 1, min(self.roi_x_end, w))
        roi_y = max(0, min(self.roi_y_start, h))
        roi_h = self.roi_height
        roi_y_end = min(roi_y + roi_h, h)
        
        # Draw ROI bounding box (green) - entire box is used for data extraction
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x_end, roi_y_end), (0, 255, 0), 2)
        
        # Add text overlay with ROI info
        cv2.putText(frame, f"ROI: X={roi_x}-{roi_x_end}, Y={roi_y}-{roi_y_end}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {w}x{h}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Convert to QImage
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.camera_label.setPixmap(scaled_pixmap)
    
    def update_spectrum(self):
        """Update spectrum plot - optimized version."""
        # Clear and setup plot
        self.ax.clear()
        
        # Use wavelength scale on x-axis
        if len(self.wavelengthData) > 0:
            self.ax.set_xlim(self.wavelengthData[0], self.wavelengthData[-1])
        else:
            self.ax.set_xlim(0, len(self.intensity))
            
        self.ax.set_ylim(0, 300)
        self.ax.set_xlabel('Wavelength (nm)', fontsize=9)
        self.ax.set_ylabel('Intensity', fontsize=9)
        self.ax.set_title('Real-time Spectrum', fontsize=11, fontweight='bold')
        self.ax.grid(True, alpha=0.2, linewidth=0.5)
        
        # Draw graticule - simplified (wavelength-based)
        if len(self.wavelengthData) > 0:
            for pos in self.tens[::2]:  # Draw every other line for performance
                if 0 <= pos < len(self.wavelengthData):
                    wl = self.wavelengthData[pos]
                    self.ax.axvline(x=wl, color='lightgray', linewidth=0.5, alpha=0.3)
            
            for pos_data in self.fifties:
                pos = pos_data[0]
                label = pos_data[1]
                if 0 <= pos < len(self.wavelengthData):
                    wl = self.wavelengthData[pos]
                    self.ax.axvline(x=wl, color='gray', linewidth=0.8, alpha=0.5)
                    self.ax.text(wl, 290, f'{label}nm', fontsize=7, ha='center')
        
        # Process intensity data
        intensity_to_plot = self.intensity.copy()
        
        if not self.hold_peaks and len(intensity_to_plot) > 17:
            try:
                intensity_to_plot = savitzky_golay(intensity_to_plot, 17, self.savpoly)
            except:
                pass
        
        # Plot using wavelength on x-axis
        if len(self.wavelengthData) > 0:
            # Extract wavelength range corresponding to ROI
            roi_wavelengths = self.wavelengthData[self.roi_x_start:self.roi_x_end]
            if len(roi_wavelengths) == len(intensity_to_plot):
                x_data = roi_wavelengths
            else:
                x_data = np.arange(len(intensity_to_plot))
        else:
            x_data = np.arange(len(intensity_to_plot))
        
        # Plot as filled area for better performance
        self.ax.fill_between(x_data, 0, intensity_to_plot, alpha=0.7, 
                            color='steelblue', edgecolor='darkblue', linewidth=0.5)
        
        # Find and label peaks - limit number for performance
        if self.thresh > 0 and len(intensity_to_plot) > 0:
            try:
                peaks = peakIndexes(intensity_to_plot, thres=self.thresh/max(intensity_to_plot), min_dist=self.mindist)
                # Limit to top 10 peaks for performance
                if len(peaks) > 10:
                    peak_heights = [intensity_to_plot[p] for p in peaks]
                    top_indices = np.argsort(peak_heights)[-10:]
                    peaks = peaks[top_indices]
                
                for peak in peaks:
                    if peak < len(x_data):
                        wavelength = round(x_data[peak], 1)
                        height = intensity_to_plot[peak]
                        self.ax.plot(x_data[peak], height, 'ro', markersize=3)
                        self.ax.annotate(
                            f'{wavelength}nm',
                            xy=(x_data[peak], height),
                            xytext=(0, 8),
                            textcoords='offset points',
                            fontsize=7,
                            ha='center',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6, linewidth=0.5)
                        )
            except:
                pass
        
        # Draw without animation for performance
        self.canvas.draw_idle()
    
    def on_plot_click(self, event):
        """Handle click on spectrum plot."""
        # Click handling removed - no longer needed
        pass
    
    def on_exposure_changed(self, value):
        """Handle exposure slider change."""
        self.camera_exposure = value
        if value == -1:
            self.exposure_label.setText("Auto")
        else:
            self.exposure_label.setText(str(value))
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.set_camera_exposure(value)
    
    def on_roi_x_start_changed(self, value):
        """Handle ROI X start position slider change."""
        self.roi_x_start = value
        self.roi_x_start_label.setText(f"{value} px")
        # Ensure X end is after X start
        if self.roi_x_end <= self.roi_x_start:
            self.roi_x_end = self.roi_x_start + 10
            self.roi_x_end_slider.setValue(self.roi_x_end)
    
    def on_roi_x_end_changed(self, value):
        """Handle ROI X end position slider change."""
        self.roi_x_end = value
        self.roi_x_end_label.setText(f"{value} px")
        # Ensure X end is after X start
        if self.roi_x_end <= self.roi_x_start:
            self.roi_x_start = max(0, self.roi_x_end - 10)
            self.roi_x_start_slider.setValue(self.roi_x_start)
    
    def on_roi_y_changed(self, value):
        """Handle ROI Y position slider change."""
        self.roi_y_start = value
        self.roi_y_label.setText(f"{value} px")
    
    def on_roi_h_changed(self, value):
        """Handle ROI height slider change."""
        self.roi_height = value
        self.roi_h_label.setText(f"{value} px")
    
    def on_filter_changed(self, value):
        """Handle filter strength slider change."""
        self.filter_strength = value
        self.filter_label.setText(str(value))
    
    def on_rising_changed(self, value):
        """Handle rising speed slider change."""
        self.rising_speed = value
        self.rising_label.setText(str(value))
    
    def on_falling_changed(self, value):
        """Handle falling speed slider change."""
        self.falling_speed = value
        self.falling_label.setText(str(value))
    
    def on_flip_changed(self, state):
        """Handle flip horizontal checkbox."""
        self.flip_horizontal = (state == Qt.CheckState.Checked.value)
    
    def on_interval_changed(self, value):
        """Handle update interval change."""
        self.update_interval = value
        if self.update_timer.isActive():
            self.update_timer.setInterval(value)
    
    def apply_camera_settings(self):
        """Apply resolution/FPS settings (requires camera restart)."""
        was_running = self.camera_thread and self.camera_thread.isRunning()
        if was_running:
            self.stop_camera()
            QTimer.singleShot(500, self.start_camera)
        else:
            # Get resolution from dropdown
            resolution_map = {
                0: (640, 480),
                1: (1280, 720),
                2: (1920, 1080)
            }
            self.frame_width, self.frame_height = resolution_map[self.resolution_combo.currentIndex()]
            self.intensity = np.zeros(self.frame_width)
            # Regenerate calibration for new width
            self.nm_min = 300.0
            self.nm_max = 1100.0
            self.wavelengthData = np.linspace(self.nm_min, self.nm_max, self.frame_width).tolist()
            self.graticule_data = generateGraticule(self.wavelengthData)
            self.tens = self.graticule_data[0]
            self.fifties = self.graticule_data[1]
    
    def reset_camera_settings(self):
        """Reset all camera settings to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset Camera Settings?",
            "This will reset:\n"
            "• Resolution to 1920×1080\n"
            "• Frame rate to 30 fps\n"
            "• Exposure to Auto\n"
            "• Brightness to 0\n"
            "• Contrast to 1.0\n"
            "• ROI to full frame (1920×1080)\n"
            "• ROI Y Start to 544px\n"
            "• ROI Height to 30px\n\n"
            "Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Reset UI controls
            self.resolution_combo.setCurrentIndex(2)  # 1920×1080
            self.fps_spin.setValue(30)
            self.exposure_slider.setValue(-1)
            self.brightness_slider.setValue(0)
            self.contrast_slider.setValue(100)
            
    def reset_camera_settings(self):
        """Reset all camera settings to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset Camera Settings?",
            "This will reset:\n"
            "• Resolution to 1920×1080\n"
            "• Frame rate to 30 fps\n"
            "• Exposure to Auto\n"
            "• ROI to full frame (1920×1080)\n"
            "• ROI Y Start to 544px\n"
            "• ROI Height to 30px\n\n"
            "Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Reset UI controls
            self.resolution_combo.setCurrentIndex(2)  # 1920×1080
            self.fps_spin.setValue(30)
            self.exposure_slider.setValue(-1)
            
            # Reset ROI controls
            self.roi_x_start_slider.setValue(0)
            self.roi_x_end_slider.setValue(1100)
            self.roi_y_slider.setValue(544)
            self.roi_h_slider.setValue(30)
            
            # Reset internal values
            self.frame_width = 1920
            self.frame_height = 1080
            self.camera_exposure = -1
            self.camera_fps = 30
            self.roi_x_start = 0
            self.roi_x_end = 1100
            self.roi_y_start = 544
            self.roi_height = 30
            
            self.status_label.setText("✓ Camera settings reset to defaults")
            QMessageBox.information(self, "Settings Reset", "Camera settings have been reset to defaults.\n\nClick 'Apply Settings' to restart the camera with these settings.")
    
    def apply_current_camera_settings(self):
        """Apply current camera settings to running camera."""
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.set_camera_exposure(self.camera_exposure)
    
    def on_hold_peaks_changed(self, state):
        """Handle hold peaks checkbox."""
        self.hold_peaks = (state == Qt.CheckState.Checked.value)
        if not self.hold_peaks:
            self.intensity = np.zeros(self.frame_width)
    
    def on_wavelength_range_changed(self):
        """Handle wavelength range spin box changes."""
        # Ensure min is less than max
        if self.nm_min_spin.value() >= self.nm_max_spin.value():
            self.nm_max_spin.setValue(self.nm_min_spin.value() + 50)
    
    def apply_wavelength_range(self):
        """Apply new wavelength range to spectrum display."""
        new_min = float(self.nm_min_spin.value())
        new_max = float(self.nm_max_spin.value())
        
        if new_min >= new_max:
            QMessageBox.warning(self, "Invalid Range", 
                              f"Minimum wavelength ({new_min} nm) must be less than maximum ({new_max} nm).")
            return
        
        # Update wavelength range
        self.nm_min = new_min
        self.nm_max = new_max
        
        # Regenerate wavelength data with new range
        self.wavelengthData = np.linspace(self.nm_min, self.nm_max, self.frame_width).tolist()
        
        # Regenerate graticule
        self.graticule_data = generateGraticule(self.wavelengthData)
        self.tens = self.graticule_data[0]
        self.fifties = self.graticule_data[1]
        
        self.status_label.setText(f"✓ Wavelength range updated: {self.nm_min:.1f} - {self.nm_max:.1f} nm")
        
        QMessageBox.information(self, "Range Updated", 
                              f"Wavelength display range updated to:\n{self.nm_min:.1f} - {self.nm_max:.1f} nm\n\n"
                              f"Note: This only changes the display scale.\n"
                              f"For accurate wavelength calibration, use the calibration section below.")
    
    def on_poly_degree_changed(self, value):
        """Handle polynomial degree change."""
        self.calib_polynomial_degree = value
        if len(self.calib_peaks) > 0:
            self.status_label.setText(f"Polynomial degree set to {value}. Click 'Apply Calibration' to recalculate.")
    
    def add_calibration_point(self):
        """Add a new calibration point by entering current and known wavelengths."""
        from PyQt6.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Calibration Point")
        dialog.setMinimumWidth(400)
        layout = QFormLayout()
        
        # Instructions
        instruction_label = QLabel(
            "<b>⚠️ Important:</b> Use wavelengths from the <b>CURRENT</b> spectrum display!<br>"
            "<i>If your spectrum looks wrong, click 'Reset Calibration' first.</i>"
        )
        instruction_label.setWordWrap(True)
        instruction_label.setStyleSheet("background: #fff3cd; padding: 8px; border-radius: 4px; color: #856404; border: 1px solid #ffc107;")
        layout.addRow(instruction_label)
        
        # Current wavelength (what the spectrum shows now)
        layout.addRow(QLabel("<b>Step 1: Current Wavelength</b>"))
        current_wl_spin = QDoubleSpinBox()
        current_wl_spin.setRange(200.0, 1200.0)
        current_wl_spin.setValue(400.0)
        current_wl_spin.setDecimals(2)
        current_wl_spin.setSuffix(" nm")
        current_wl_spin.setToolTip("Enter the wavelength value shown on a peak in your spectrum RIGHT NOW")
        layout.addRow("Current Peak Wavelength:", current_wl_spin)
        
        # Known wavelength (what it should actually be)
        layout.addRow(QLabel("<b>Step 2: Known Wavelength</b>"))
        known_wl_spin = QDoubleSpinBox()
        known_wl_spin.setRange(200.0, 1200.0)
        known_wl_spin.setValue(435.8)
        known_wl_spin.setDecimals(2)
        known_wl_spin.setSuffix(" nm")
        known_wl_spin.setToolTip("Enter the actual known wavelength of this emission line")
        layout.addRow("Known Peak Wavelength:", known_wl_spin)
        
        # Quick presets for known wavelengths
        layout.addRow(QLabel("<i>Quick presets for known wavelength:</i>"))
        preset_layout = QHBoxLayout()
        
        presets = [
            ("Hg 404.7", 404.7),
            ("Hg 435.8", 435.8),
            ("Hg 546.1", 546.1),
            ("Hg 577.0", 577.0),
            ("He 587.6", 587.6),
            ("Ne 585.2", 585.2),
        ]
        
        for label, wl in presets:
            btn = QPushButton(label)
            btn.setToolTip(f"Set known wavelength to {wl} nm")
            btn.clicked.connect(lambda checked, w=wl: known_wl_spin.setValue(w))
            preset_layout.addWidget(btn)
        
        layout.addRow(preset_layout)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            current_wl = current_wl_spin.value()
            known_wl = known_wl_spin.value()
            
            # Find the pixel position that corresponds to the current wavelength
            if len(self.wavelengthData) > 0:
                wavelength_array = np.array(self.wavelengthData)
                pixel_distances = np.abs(wavelength_array - current_wl)
                pixel = int(np.argmin(pixel_distances))
                
                # Add the calibration point
                self.calib_peaks.append((pixel, known_wl))
                print(f"DEBUG: Added calibration point: pixel {pixel} (current: {current_wl:.2f} nm) → known: {known_wl:.2f} nm")
                print(f"DEBUG: Total calibration points: {len(self.calib_peaks)}")
                
                # Update the table
                self.update_calibration_table()
                
                self.status_label.setText(f"✓ Added: {current_wl:.2f} nm → {known_wl:.2f} nm (Total: {len(self.calib_peaks)} points)")
            else:
                QMessageBox.warning(self, "Error", "No wavelength data available. Start the camera first.")
    
    def clear_calibration_points(self):
        """Clear all calibration points and reset to default linear scale."""
        reply = QMessageBox.question(
            self, 
            "Clear Calibration?",
            "This will:\n"
            "• Clear all calibration points\n"
            "• Reset to default linear wavelength scale (300-850 nm)\n"
            "• Remove polynomial calibration\n\n"
            "Are you sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Clear calibration data
            self.calib_peaks = []
            self.calib_polynomial_coeffs = None
            
            # Reset to default linear calibration (300-850 nm)
            self.nm_min = 300.0
            self.nm_max = 850.0
            self.wavelengthData = np.linspace(self.nm_min, self.nm_max, self.frame_width).tolist()
            
            # Regenerate graticule
            self.graticule_data = generateGraticule(self.wavelengthData)
            self.tens = self.graticule_data[0]
            self.fifties = self.graticule_data[1]
            
            # Update UI
            self.update_calibration_table()
            self.calib_status_label.setText(f"✓ Cleared - Reset to default linear scale: {self.nm_min:.1f} - {self.nm_max:.1f} nm")
            self.calib_status_label.setStyleSheet("background: #e3f2fd; color: #1565c0; font-size: 9pt; padding: 8px; border: 2px solid #2196f3; border-radius: 6px;")
            self.status_label.setText("Calibration cleared and reset to default")
            
            # Update range spinboxes
            self.nm_min_spin.setValue(int(self.nm_min))
            self.nm_max_spin.setValue(int(self.nm_max))
    
    def load_calibration_button_clicked(self):
        """Load calibration from file when button is clicked."""
        cal_file = Path(__file__).parent.parent / 'caldata_polynomial.json'
        
        if not cal_file.exists():
            QMessageBox.warning(
                self,
                "No Saved Calibration",
                f"No saved calibration file found.\n\n"
                f"Expected file: {cal_file.name}\n\n"
                f"Please create a calibration first by:\n"
                f"1. Adding calibration points\n"
                f"2. Applying the calibration\n"
                f"3. Saving with 'Save Calibration' button"
            )
            return
        
        success = self.load_calibration_from_file()
        
        if success:
            QMessageBox.information(
                self,
                "Calibration Loaded",
                f"Calibration loaded successfully!\n\n"
                f"File: {cal_file.name}\n"
                f"Points: {len(self.calib_peaks)}\n"
                f"Polynomial Degree: {self.calib_polynomial_degree}\n"
                f"Range: {self.nm_min:.1f} - {self.nm_max:.1f} nm"
            )
            self.status_label.setText(f"✓ Calibration loaded from {cal_file.name}")
        else:
            QMessageBox.critical(
                self,
                "Load Failed",
                f"Failed to load calibration from {cal_file.name}\n\n"
                f"The file may be corrupted or incompatible.\n"
                f"Check the console for error details."
            )
    
    def update_calibration_table(self):
        """Update the calibration points table."""
        self.calib_table.setRowCount(len(self.calib_peaks))
        
        for i, (pixel, known_wl) in enumerate(self.calib_peaks):
            # Pixel position
            self.calib_table.setItem(i, 0, QTableWidgetItem(str(pixel)))
            
            # Known wavelength
            self.calib_table.setItem(i, 1, QTableWidgetItem(f"{known_wl:.2f}"))
            
            # Current wavelength (from current calibration)
            if pixel < len(self.wavelengthData):
                current_wl = self.wavelengthData[pixel]
                self.calib_table.setItem(i, 2, QTableWidgetItem(f"{current_wl:.2f}"))
            else:
                self.calib_table.setItem(i, 2, QTableWidgetItem("N/A"))
            
            # Actions: Edit and Delete buttons
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            actions_layout.setSpacing(2)
            
            # Edit button
            edit_btn = QPushButton("✏️")
            edit_btn.setToolTip("Edit this calibration point")
            edit_btn.clicked.connect(lambda checked, idx=i: self.edit_calibration_point(idx))
            edit_btn.setMaximumWidth(40)
            edit_btn.setMaximumHeight(32)
            edit_btn.setStyleSheet("padding: 2px; font-size: 12px;")
            actions_layout.addWidget(edit_btn)
            
            # Delete button
            delete_btn = QPushButton("🗑️")
            delete_btn.setToolTip("Remove this calibration point")
            delete_btn.clicked.connect(lambda checked, idx=i: self.remove_calibration_point(idx))
            delete_btn.setMaximumWidth(40)
            delete_btn.setMaximumHeight(32)
            delete_btn.setStyleSheet("padding: 2px; font-size: 12px;")
            actions_layout.addWidget(delete_btn)
            
            self.calib_table.setCellWidget(i, 3, actions_widget)
    
    def remove_calibration_point(self, index):
        """Remove a calibration point by index."""
        if 0 <= index < len(self.calib_peaks):
            removed = self.calib_peaks.pop(index)
            self.update_calibration_table()
            self.status_label.setText(f"Removed calibration point: {removed[0]} px → {removed[1]} nm")
    
    def edit_calibration_point(self, index):
        """Edit an existing calibration point."""
        if not (0 <= index < len(self.calib_peaks)):
            return
        
        from PyQt6.QtWidgets import QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox
        
        # Get current values
        current_pixel, current_known_wl = self.calib_peaks[index]
        current_display_wl = self.wavelengthData[current_pixel] if current_pixel < len(self.wavelengthData) else 0.0
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Calibration Point")
        dialog.setMinimumWidth(400)
        layout = QFormLayout()
        
        # Instructions
        instruction_label = QLabel(
            "<b>Edit Calibration Point</b><br>"
            "<i>Modify the current or known wavelength values for this calibration point.</i>"
        )
        instruction_label.setWordWrap(True)
        instruction_label.setStyleSheet("background: #e3f2fd; padding: 8px; border-radius: 4px; color: #1565c0; border: 1px solid #2196f3;")
        layout.addRow(instruction_label)
        
        # Current wavelength (what the spectrum shows now)
        layout.addRow(QLabel("<b>Current Peak Wavelength:</b>"))
        current_wl_spin = QDoubleSpinBox()
        current_wl_spin.setRange(200.0, 1200.0)
        current_wl_spin.setValue(current_display_wl)
        current_wl_spin.setDecimals(2)
        current_wl_spin.setSuffix(" nm")
        current_wl_spin.setToolTip("Enter the wavelength value shown on the peak in your current spectrum")
        layout.addRow("Current Wavelength:", current_wl_spin)
        
        # Known wavelength (what it should actually be)
        layout.addRow(QLabel("<b>Known Peak Wavelength:</b>"))
        known_wl_spin = QDoubleSpinBox()
        known_wl_spin.setRange(200.0, 1200.0)
        known_wl_spin.setValue(current_known_wl)
        known_wl_spin.setDecimals(2)
        known_wl_spin.setSuffix(" nm")
        known_wl_spin.setToolTip("Enter the actual known wavelength of this emission line")
        layout.addRow("Known Wavelength:", known_wl_spin)
        
        # Quick presets for known wavelengths
        layout.addRow(QLabel("<i>Quick presets for known wavelength:</i>"))
        preset_layout = QHBoxLayout()
        
        presets = [
            ("Hg 404.7", 404.7),
            ("Hg 435.8", 435.8),
            ("Hg 546.1", 546.1),
            ("Hg 577.0", 577.0),
            ("He 587.6", 587.6),
            ("Ne 585.2", 585.2),
        ]
        
        for label, wl in presets:
            btn = QPushButton(label)
            btn.setToolTip(f"Set known wavelength to {wl} nm")
            btn.clicked.connect(lambda checked, w=wl: known_wl_spin.setValue(w))
            preset_layout.addWidget(btn)
        
        layout.addRow(preset_layout)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_current_wl = current_wl_spin.value()
            new_known_wl = known_wl_spin.value()
            
            # Find the pixel position that corresponds to the new current wavelength
            if len(self.wavelengthData) > 0:
                wavelength_array = np.array(self.wavelengthData)
                pixel_distances = np.abs(wavelength_array - new_current_wl)
                new_pixel = int(np.argmin(pixel_distances))
                
                # Update the calibration point
                self.calib_peaks[index] = (new_pixel, new_known_wl)
                
                print(f"DEBUG: Edited calibration point {index}: pixel {new_pixel} (current: {new_current_wl:.2f} nm) → known: {new_known_wl:.2f} nm")
                
                # Update the table
                self.update_calibration_table()
                
                self.status_label.setText(f"✓ Edited point {index+1}: {new_current_wl:.2f} nm → {new_known_wl:.2f} nm")
            else:
                QMessageBox.warning(self, "Error", "No wavelength data available. Start the camera first.")
    
    def apply_polynomial_calibration(self):
        """Apply polynomial fit to wavelength calibration."""
        if len(self.calib_peaks) < 2:
            QMessageBox.warning(self, "Insufficient Points", 
                              "Need at least 2 calibration points for calibration.\n"
                              "Add more points using the 'Add Point' button.")
            return
        
        # Check if we have enough points for the polynomial degree
        if len(self.calib_peaks) < self.calib_polynomial_degree + 1:
            QMessageBox.warning(self, "Insufficient Points", 
                              f"Need at least {self.calib_polynomial_degree + 1} points for degree {self.calib_polynomial_degree} polynomial.\n"
                              f"Either add more points or reduce polynomial degree.")
            return
        
        try:
            # Extract pixel positions and known wavelengths
            pixels = np.array([p[0] for p in self.calib_peaks])
            wavelengths = np.array([p[1] for p in self.calib_peaks])
            
            # Sort by pixel position for better polynomial fitting
            sort_indices = np.argsort(pixels)
            pixels = pixels[sort_indices]
            wavelengths = wavelengths[sort_indices]
            
            # Debug info
            print(f"\n=== Polynomial Calibration Debug ===")
            print(f"Frame width: {self.frame_width}")
            print(f"Polynomial degree: {self.calib_polynomial_degree}")
            print(f"Calibration points (sorted by pixel):")
            for px, wl in zip(pixels, wavelengths):
                print(f"  Pixel {px:4d} → {wl:.2f} nm")
            
            # Validate that pixels are within frame bounds
            if np.any(pixels < 0) or np.any(pixels >= self.frame_width):
                QMessageBox.warning(self, "Invalid Pixels",
                                  f"Some pixel positions are out of range (0-{self.frame_width-1}).\n"
                                  f"Pixels: {pixels}\n"
                                  f"Please check your calibration points.")
                return
            
            # Check for monotonicity (wavelengths should generally increase or decrease with pixel)
            wavelength_diffs = np.diff(wavelengths)
            if not (np.all(wavelength_diffs > 0) or np.all(wavelength_diffs < 0)):
                reply = QMessageBox.question(self, "Non-monotonic Wavelengths",
                                            "Warning: Your wavelength calibration points are not monotonic.\n"
                                            "This may indicate incorrect pixel positions or wavelengths.\n\n"
                                            "Do you want to continue anyway?",
                                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                            QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    return
            
            # Fit polynomial: wavelength = f(pixel)
            self.calib_polynomial_coeffs = np.polyfit(pixels, wavelengths, self.calib_polynomial_degree)
            
            print(f"\nPolynomial coefficients (high to low degree):")
            for i, coeff in enumerate(self.calib_polynomial_coeffs):
                print(f"  x^{self.calib_polynomial_degree - i}: {coeff:.6e}")
            
            # Apply polynomial to all pixel positions
            pixel_array = np.arange(self.frame_width)
            wavelengthData_full = np.polyval(self.calib_polynomial_coeffs, pixel_array)
            
            # Clamp to reasonable bounds to avoid excessive extrapolation on display
            # Use display range settings as bounds
            wavelengthData_clamped = np.clip(wavelengthData_full, self.nm_min, self.nm_max)
            self.wavelengthData = wavelengthData_clamped.tolist()
            
            # Get actual wavelength range from the data
            calib_wl_min = float(np.min(wavelengths))
            calib_wl_max = float(np.max(wavelengths))
            actual_wl_min = float(np.min(wavelengthData_clamped))
            actual_wl_max = float(np.max(wavelengthData_clamped))
            
            print(f"\nCalibration wavelength info:")
            print(f"  Raw polynomial range: {wavelengthData_full[0]:.2f} - {wavelengthData_full[-1]:.2f} nm")
            print(f"  Clamped to display range: {actual_wl_min:.2f} - {actual_wl_max:.2f} nm")
            print(f"  Calibration point range: {calib_wl_min:.2f} - {calib_wl_max:.2f} nm")
            print(f"  Display range settings: {self.nm_min:.2f} - {self.nm_max:.2f} nm")
            print(f"  Direction: {'increasing' if wavelengthData_full[-1] > wavelengthData_full[0] else 'decreasing'}")
            
            # Regenerate graticule
            self.graticule_data = generateGraticule(self.wavelengthData)
            self.tens = self.graticule_data[0]
            self.fifties = self.graticule_data[1]
            
            # Calculate residuals (error at each calibration point)
            predicted = np.polyval(self.calib_polynomial_coeffs, pixels)
            residuals = wavelengths - predicted
            rms_error = np.sqrt(np.mean(residuals**2))
            
            print(f"\nCalibration quality:")
            print(f"  RMS error: {rms_error:.3f} nm")
            print(f"  Max error: {np.max(np.abs(residuals)):.3f} nm")
            print("="*40 + "\n")
            
            # Update status
            degree_name = ["constant", "linear", "quadratic", "cubic", "quartic", "quintic"]
            degree_str = degree_name[self.calib_polynomial_degree] if self.calib_polynomial_degree < len(degree_name) else f"degree-{self.calib_polynomial_degree}"
            
            self.calib_status_label.setText(
                f"✓ {degree_str.capitalize()} calibration applied with {len(self.calib_peaks)} points\n"
                f"Calibrated range: {calib_wl_min:.2f} - {calib_wl_max:.2f} nm | RMS Error: {rms_error:.3f} nm\n"
                f"Display range: {self.nm_min:.2f} - {self.nm_max:.2f} nm"
            )
            self.calib_status_label.setStyleSheet("background: #e8f5e9; color: #2e7d32; font-size: 9pt; padding: 8px; border: 2px solid #4caf50; border-radius: 6px;")
            
            self.status_label.setText(f"✓ Calibration applied: {degree_str}, RMS error {rms_error:.3f} nm")
            
            # Update table to show new current wavelengths
            self.update_calibration_table()
            
            # Save calibration to file
            self.save_calibration()
            
            QMessageBox.information(self, "Calibration Applied", 
                                  f"{degree_str.capitalize()} polynomial calibration applied successfully!\n\n"
                                  f"Calibration Points: {len(self.calib_peaks)}\n"
                                  f"Calibrated Range: {calib_wl_min:.2f} - {calib_wl_max:.2f} nm\n"
                                  f"Display Range: {self.nm_min:.2f} - {self.nm_max:.2f} nm\n"
                                  f"RMS Error: {rms_error:.3f} nm\n\n"
                                  f"Your display range settings remain unchanged.\n"
                                  f"Check the terminal/console for detailed calibration info.")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"\n=== Calibration Error ===")
            print(error_details)
            print("="*40 + "\n")
            QMessageBox.critical(self, "Calibration Error", 
                               f"Failed to apply calibration:\n{str(e)}\n\n"
                               f"Check the terminal/console for detailed error information.")
    
    def load_preset(self, element):
        """Load preset emission lines for common elements."""
        presets = {
            'mercury': [
                (404.7, "Hg violet"),
                (435.8, "Hg blue"),
                (546.1, "Hg green"),
                (577.0, "Hg yellow 1"),
                (579.1, "Hg yellow 2"),
            ],
            'helium': [
                (388.9, "He violet"),
                (447.1, "He blue"),
                (471.3, "He blue-green"),
                (492.2, "He cyan"),
                (501.6, "He green"),
                (587.6, "He yellow"),
                (667.8, "He red"),
                (706.5, "He deep red"),
            ],
            'neon': [
                (540.1, "Ne green"),
                (585.2, "Ne yellow"),
                (614.3, "Ne orange"),
                (638.3, "Ne red-orange"),
                (650.7, "Ne red"),
                (703.2, "Ne deep red"),
                (724.5, "Ne infrared edge"),
            ],
            'argon': [
                (696.5, "Ar red 1"),
                (706.7, "Ar red 2"),
                (727.3, "Ar red 3"),
                (738.4, "Ar red 4"),
                (763.5, "Ar infrared 1"),
                (811.5, "Ar infrared 2"),
                (826.5, "Ar infrared 3"),
            ]
        }
        
        if element not in presets:
            return
        
        lines = presets[element]
        
        # Ask user if they want to clear existing points
        if len(self.calib_peaks) > 0:
            reply = QMessageBox.question(self, "Clear Existing Points?",
                                        "Do you want to clear existing calibration points before loading preset?",
                                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                        QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.calib_peaks = []
        
        # Add preset lines at evenly spaced pixel positions
        num_lines = len(lines)
        pixel_spacing = self.frame_width // (num_lines + 1)
        
        for i, (wavelength, name) in enumerate(lines):
            pixel = pixel_spacing * (i + 1)
            self.calib_peaks.append((pixel, wavelength))
        
        self.update_calibration_table()
        self.status_label.setText(f"✓ Loaded {element.capitalize()} preset: {num_lines} emission lines")
        
        QMessageBox.information(self, "Preset Loaded",
                              f"Loaded {num_lines} {element.capitalize()} emission lines.\n\n"
                              "After adding these preset lines, click 'Apply Calibration' to compute the polynomial fit.")
    
    def save_calibration(self):
        """Save current calibration to file (called automatically after applying calibration)."""
        if self.calib_polynomial_coeffs is None:
            return
        
        try:
            cal_file = Path(__file__).parent.parent / 'caldata_polynomial.json'
            import json
            
            calibration_data = {
                'frame_width': self.frame_width,
                'polynomial_degree': self.calib_polynomial_degree,
                'polynomial_coeffs': self.calib_polynomial_coeffs.tolist(),
                'calibration_points': self.calib_peaks,
                'nm_min': self.nm_min,
                'nm_max': self.nm_max
            }
            
            with open(cal_file, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            print(f"Calibration auto-saved to {cal_file}")
        except Exception as e:
            print(f"Failed to auto-save calibration: {e}")
    
    def save_calibration_to_file(self):
        """Save current calibration to file (triggered by Save button)."""
        if self.calib_polynomial_coeffs is None and len(self.calib_peaks) == 0:
            QMessageBox.warning(self, "No Calibration",
                              "No calibration to save. Add calibration points and apply calibration first.")
            return
        
        try:
            cal_file = Path(__file__).parent.parent / 'caldata_polynomial.json'
            import json
            
            calibration_data = {
                'frame_width': self.frame_width,
                'polynomial_degree': self.calib_polynomial_degree,
                'polynomial_coeffs': self.calib_polynomial_coeffs.tolist() if self.calib_polynomial_coeffs is not None else None,
                'calibration_points': self.calib_peaks,
                'nm_min': self.nm_min,
                'nm_max': self.nm_max,
                'wavelength_data': self.wavelengthData[:100]  # Save first 100 points as sample
            }
            
            with open(cal_file, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            QMessageBox.information(self, "Calibration Saved",
                                  f"Calibration saved successfully!\n\n"
                                  f"File: {cal_file.name}\n"
                                  f"Points: {len(self.calib_peaks)}\n"
                                  f"Polynomial Degree: {self.calib_polynomial_degree}\n"
                                  f"Range: {self.nm_min:.1f} - {self.nm_max:.1f} nm")
            
            self.status_label.setText(f"✓ Calibration saved to {cal_file.name}")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Failed to save calibration:\n{str(e)}")
    
    def load_calibration_from_file(self):
        """Load calibration from file."""
        try:
            cal_file = Path(__file__).parent.parent / 'caldata_polynomial.json'
            if not cal_file.exists():
                return False
            
            import json
            with open(cal_file, 'r') as f:
                calibration_data = json.load(f)
            
            # Check if frame width matches
            if calibration_data.get('frame_width') != self.frame_width:
                print(f"Warning: Saved calibration is for width {calibration_data.get('frame_width')}, current is {self.frame_width}")
                return False
            
            # Restore calibration
            self.calib_polynomial_degree = calibration_data.get('polynomial_degree', 2)
            if calibration_data.get('polynomial_coeffs') is not None:
                self.calib_polynomial_coeffs = np.array(calibration_data['polynomial_coeffs'])
            else:
                self.calib_polynomial_coeffs = None
                
            self.calib_peaks = calibration_data.get('calibration_points', [])
            
            # Recalculate wavelength data using the polynomial
            if self.calib_polynomial_coeffs is not None:
                pixel_array = np.arange(self.frame_width)
                wavelengthData_full = np.polyval(self.calib_polynomial_coeffs, pixel_array)
                # Clamp to display range to avoid excessive extrapolation
                wavelengthData_clamped = np.clip(wavelengthData_full, self.nm_min, self.nm_max)
                self.wavelengthData = wavelengthData_clamped.tolist()
            else:
                # No polynomial coeffs, use linear scale with current display range
                self.wavelengthData = np.linspace(self.nm_min, self.nm_max, self.frame_width).tolist()
            
            # DON'T update nm_min/nm_max spinboxes - keep user's display range settings
            
            # Regenerate graticule
            self.graticule_data = generateGraticule(self.wavelengthData)
            self.tens = self.graticule_data[0]
            self.fifties = self.graticule_data[1]
            
            # Update UI
            self.poly_degree_spin.setValue(self.calib_polynomial_degree)
            self.update_calibration_table()
            
            # DON'T update range spinboxes - keep user's display range settings
            
            self.calib_status_label.setText(
                f"✓ Loaded calibration with {len(self.calib_peaks)} points\n"
                f"Display range: {self.nm_min:.1f} - {self.nm_max:.1f} nm"
            )
            self.calib_status_label.setStyleSheet("background: #e8f5e9; color: #2e7d32; font-size: 9pt; padding: 8px; border: 2px solid #4caf50; border-radius: 6px;")
            
            print(f"Calibration loaded from {cal_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"Failed to load calibration: Invalid JSON format - {e}")
            print(f"The calibration file may be corrupted. You can delete it manually at:")
            print(f"  {Path(__file__).parent.parent / 'caldata_polynomial.json'}")
            return False
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            return False
    
    def set_trim_preset(self, point1, point2):
        """Set trim point preset values (deprecated - use polynomial calibration instead)."""
        self.trim_point1 = float(point1)
        self.trim_point2 = float(point2)
        self.status_label.setText(f"Trim points set to {point1}nm and {point2}nm (legacy mode)")
    
    def on_trim1_changed(self, value):
        """Handle trim point 1 change (deprecated)."""
        self.trim_point1 = float(value)
    
    def on_trim2_changed(self, value):
        """Handle trim point 2 change (deprecated)."""
        self.trim_point2 = float(value)
    
    def on_trim_mode_changed(self, state):
        """Handle trim mode checkbox (deprecated - use calibration mode instead)."""
        pass
    
    def on_spectrum_mouse_press(self, event):
        """Handle mouse press on spectrum plot (deprecated trim mode)."""
        pass
    
    def on_spectrum_mouse_move(self, event):
        """Handle mouse move on spectrum plot (deprecated trim mode)."""
        pass
    
    def on_spectrum_mouse_release(self, event):
        """Handle mouse release on spectrum plot (deprecated trim mode)."""
        pass
    
    def update_spectrum_plot(self):
        """Force spectrum plot update during drag operations (deprecated)."""
        self.update_spectrum()
    
    def on_savpoly_changed(self, value):
        """Handle savgol polynomial slider."""
        self.savpoly = value
        self.savpoly_label.setText(str(value))
    
    def on_mindist_changed(self, value):
        """Handle peak distance slider."""
        self.mindist = value
        self.mindist_label.setText(str(value))
    
    def on_thresh_changed(self, value):
        """Handle threshold slider."""
        self.thresh = value
        self.thresh_label.setText(str(value))
    
    def save_spectrum(self):
        """Save current spectrum to CSV file."""
        if len(self.intensity) == 0:
            QMessageBox.warning(self, "No Data", "No spectrum data to save")
            return
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"spectrum-{timestamp}.csv"
        
        try:
            with open(filename, 'w') as f:
                f.write('Wavelength,Intensity\n')
                for i, intensity_val in enumerate(self.intensity):
                    if i < len(self.wavelengthData):
                        wavelength = self.wavelengthData[i]
                        f.write(f'{wavelength:.2f},{intensity_val:.2f}\n')
            
            self.status_label.setText(f"✓ Spectrum saved: {filename}")
            QMessageBox.information(self, "Success", f"Spectrum saved to:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save spectrum:\n{str(e)}")
    
    def on_camera_error(self, error_msg):
        """Handle camera errors."""
        QMessageBox.critical(self, "Camera Error", error_msg)
        self.stop_camera()
    
    def closeEvent(self, event):
        """Clean up when widget is closed."""
        self.stop_camera()
        event.accept()

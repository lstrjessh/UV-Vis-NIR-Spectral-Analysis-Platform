"""
Spectrum Capture View for Qt Application
Real-time spectral data acquisition from camera-based spectrometer
Based on PySpectrometer2 by Les Wright
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QSlider, QCheckBox, QSpinBox, QGridLayout, QScrollArea, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy, QComboBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import cv2
import time
from pathlib import Path

from qt_app.camera_thread import CameraThread
from qt_app.spec_functions import (
    wavelength_to_rgb, savitzky_golay, peakIndexes,
    readcal, writecal, generateGraticule
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
        self.measure_mode = False
        self.calibration_mode = False
        self.calibration_points = []  # [(pixel, wavelength), ...]
        self.click_positions = []  # Pixel positions from plot clicks
        
        # Theremino-style trim/calibration points
        self.trim_point1 = 436.0  # First calibration wavelength (nm)
        self.trim_point2 = 546.0  # Second calibration wavelength (nm)
        self.trim_mode = False    # When true, can drag spectrum to align
        self.drag_start_x = None
        self.initial_nm_min = None
        self.initial_nm_max = None
        
        # Camera settings
        self.frame_width = 1280  # Default to 720p
        self.frame_height = 720
        self.camera_exposure = -1  # Auto exposure
        self.camera_brightness = 0
        self.camera_contrast = 1.0
        self.camera_fps = 30
        
        # ROI settings for spectral line extraction (absolute position on frame)
        self.roi_x_start = 0    # X position of left edge of ROI
        self.roi_x_end = 1280   # X position of right edge of ROI
        self.roi_y_start = 320  # Y position of top of ROI (default: center)
        self.roi_height = 80    # Height of ROI
        
        # Processing settings
        self.savpoly = 7
        self.mindist = 50
        self.thresh = 20
        
        # Theremino-style processing parameters
        self.filter_strength = 30  # 0-100, noise filtering
        self.rising_speed = 50     # 0-100, how fast to track increases
        self.falling_speed = 10    # 0-100, how fast to track decreases
        self.flip_horizontal = False  # Flip camera image horizontally
        
        # Temporal filtering state
        self.spec_array = None  # Smoothed intensity with rising/falling speed
        self.spec_filtered = None  # Further filtered with noise filter
        
        # Performance optimization
        self.frame_skip = 0  # Skip frames for performance
        self.frame_counter = 0
        self.update_interval = 100  # ms between graph updates (was 50)
        
        # Initialize calibration
        cal_file = Path(__file__).parent.parent / 'caldata.txt'
        caldata = readcal(self.frame_width, str(cal_file))
        self.wavelengthData = caldata[0]
        self.cal_msg1 = caldata[1]
        self.cal_msg2 = caldata[2]
        self.cal_msg3 = caldata[3]
        
        # Initialize wavelength range from calibration data
        if len(self.wavelengthData) > 0:
            self.nm_min = float(self.wavelengthData[0])
            self.nm_max = float(self.wavelengthData[-1])
        else:
            # Default range if no calibration
            self.nm_min = 380.0
            self.nm_max = 780.0
        
        # Generate graticule
        self.graticule_data = generateGraticule(self.wavelengthData)
        self.tens = self.graticule_data[0]
        self.fifties = self.graticule_data[1]
        
        # Initialize UI
        self.init_ui()
        
        # Initialize camera thread
        self.camera_thread = None
        
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
        
        subtitle = QLabel("Real-time spectral analysis from camera-based spectrometer with Theremino-style calibration")
        subtitle.setProperty("class", "subtitle")
        layout.addWidget(subtitle)
        
        # Status bar (moved to top for better visibility)
        self.status_label = QLabel(f"✓ Ready | {self.cal_msg1}")
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
        
        self.figure = Figure(figsize=(10, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(350)
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
        self.device_spin.setValue(0)
        self.device_spin.setToolTip("Camera device ID (0 for default)")
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
        self.resolution_combo.setCurrentIndex(0)  # Default to 640x480
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
        
        # Brightness
        layout.addWidget(QLabel("Brightness:"), 3, 0)
        brightness_layout = QHBoxLayout()
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setToolTip("Camera brightness adjustment")
        self.brightness_slider.valueChanged.connect(self.on_brightness_changed)
        self.brightness_label = QLabel("0")
        brightness_layout.addWidget(self.brightness_slider)
        brightness_layout.addWidget(self.brightness_label)
        layout.addLayout(brightness_layout, 3, 1)
        
        # Contrast
        layout.addWidget(QLabel("Contrast:"), 4, 0)
        contrast_layout = QHBoxLayout()
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.setToolTip("Camera contrast adjustment (100 = default)")
        self.contrast_slider.valueChanged.connect(self.on_contrast_changed)
        self.contrast_label = QLabel("1.0")
        contrast_layout.addWidget(self.contrast_slider)
        contrast_layout.addWidget(self.contrast_label)
        layout.addLayout(contrast_layout, 4, 1)
        
        # --- ROI Settings ---
        roi_separator = QLabel("═══ Region of Interest (ROI) ═══")
        roi_separator.setStyleSheet("color: #0066cc; font-weight: bold; margin-top: 10px;")
        roi_separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(roi_separator, 5, 0, 1, 2)
        
        # ROI X Start Position
        layout.addWidget(QLabel("ROI X Start:"), 6, 0)
        roi_x_start_layout = QHBoxLayout()
        self.roi_x_start_slider = QSlider(Qt.Orientation.Horizontal)
        self.roi_x_start_slider.setRange(0, 1920)
        self.roi_x_start_slider.setValue(0)
        self.roi_x_start_slider.setToolTip("X coordinate of left edge of ROI")
        self.roi_x_start_slider.valueChanged.connect(self.on_roi_x_start_changed)
        self.roi_x_start_label = QLabel("0 px")
        roi_x_start_layout.addWidget(self.roi_x_start_slider)
        roi_x_start_layout.addWidget(self.roi_x_start_label)
        layout.addLayout(roi_x_start_layout, 6, 1)
        
        # ROI X End Position
        layout.addWidget(QLabel("ROI X End:"), 7, 0)
        roi_x_end_layout = QHBoxLayout()
        self.roi_x_end_slider = QSlider(Qt.Orientation.Horizontal)
        self.roi_x_end_slider.setRange(0, 1920)
        self.roi_x_end_slider.setValue(1280)
        self.roi_x_end_slider.setToolTip("X coordinate of right edge of ROI")
        self.roi_x_end_slider.valueChanged.connect(self.on_roi_x_end_changed)
        self.roi_x_end_label = QLabel("1280 px")
        roi_x_end_layout.addWidget(self.roi_x_end_slider)
        roi_x_end_layout.addWidget(self.roi_x_end_label)
        layout.addLayout(roi_x_end_layout, 7, 1)
        
        # ROI Y Position (absolute)
        layout.addWidget(QLabel("ROI Y Start:"), 8, 0)
        roi_y_layout = QHBoxLayout()
        self.roi_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.roi_y_slider.setRange(0, 1080)  # Max common resolution height
        self.roi_y_slider.setValue(320)
        self.roi_y_slider.setToolTip("Y coordinate of top of ROI (pixels from top of frame)")
        self.roi_y_slider.valueChanged.connect(self.on_roi_y_changed)
        self.roi_y_label = QLabel("320 px")
        roi_y_layout.addWidget(self.roi_y_slider)
        roi_y_layout.addWidget(self.roi_y_label)
        layout.addLayout(roi_y_layout, 8, 1)
        
        # ROI Height
        layout.addWidget(QLabel("ROI Height:"), 9, 0)
        roi_h_layout = QHBoxLayout()
        self.roi_h_slider = QSlider(Qt.Orientation.Horizontal)
        self.roi_h_slider.setRange(10, 300)
        self.roi_h_slider.setValue(80)
        self.roi_h_slider.setToolTip("Height of spectral line region (pixels)")
        self.roi_h_slider.valueChanged.connect(self.on_roi_h_changed)
        self.roi_h_label = QLabel("80 px")
        roi_h_layout.addWidget(self.roi_h_slider)
        roi_h_layout.addWidget(self.roi_h_label)
        layout.addLayout(roi_h_layout, 9, 1)
        
        # Update interval (performance)
        layout.addWidget(QLabel("Update Interval:"), 10, 0)
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
        layout.addLayout(interval_layout, 10, 1)
        
        # Apply button
        self.apply_settings_btn = QPushButton("🔄 Apply Resolution/FPS")
        self.apply_settings_btn.setMinimumHeight(35)
        self.apply_settings_btn.setToolTip("Restart camera with new resolution/FPS settings")
        self.apply_settings_btn.clicked.connect(self.apply_camera_settings)
        layout.addWidget(self.apply_settings_btn, 11, 0, 1, 2)
        
        group.setLayout(layout)
        return group
    
    def create_processing_controls(self):
        """Create spectrum processing controls - compact version."""
        from PyQt6.QtWidgets import QTabWidget
        
        group = QGroupBox("⚙️ Processing & Display")
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
        
        self.hold_peaks_cb = QCheckBox("Hold Peak Maxima")
        self.hold_peaks_cb.setToolTip("Hold maximum intensity values")
        self.hold_peaks_cb.stateChanged.connect(self.on_hold_peaks_changed)
        display_layout.addWidget(self.hold_peaks_cb)
        
        self.measure_cb = QCheckBox("Measure Mode (Click to Read)")
        self.measure_cb.setToolTip("Click spectrum to measure wavelength")
        self.measure_cb.stateChanged.connect(self.on_measure_changed)
        display_layout.addWidget(self.measure_cb)
        
        self.flip_horizontal_cb = QCheckBox("Flip Camera Horizontally")
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
        """Create calibration controls."""
        group = QGroupBox("Wavelength Calibration (Theremino Style)")
        layout = QVBoxLayout()
        
        info = QLabel(
            "📌 Theremino-Style Calibration:\n"
            "1. Point spectrometer at known source (e.g., fluorescent lamp with Hg peaks)\n"
            "2. Set two known wavelengths (Trim Points) below\n"
            "3. Enable 'Trim/Align Mode' and drag spectrum left/right to align peaks\n"
            "4. Spectrum will automatically adjust wavelength scale as you drag"
        )
        info.setProperty("class", "info-box")
        layout.addWidget(info)
        
        # Trim point inputs
        trim_layout = QGridLayout()
        
        # Preset buttons
        preset_label = QLabel("Presets:")
        trim_layout.addWidget(preset_label, 0, 0)
        
        preset_buttons = QHBoxLayout()
        self.preset_436_546_btn = QPushButton("436-546 nm")
        self.preset_436_546_btn.setToolTip("Mercury: 436nm (blue) and 546nm (green)")
        self.preset_436_546_btn.clicked.connect(lambda: self.set_trim_preset(436, 546))
        preset_buttons.addWidget(self.preset_436_546_btn)
        
        self.preset_436_692_btn = QPushButton("436-692 nm")
        self.preset_436_692_btn.setToolTip("Mercury 436nm and Ruby 692nm")
        self.preset_436_692_btn.clicked.connect(lambda: self.set_trim_preset(436, 692))
        preset_buttons.addWidget(self.preset_436_692_btn)
        
        self.preset_405_546_btn = QPushButton("405-546 nm")
        self.preset_405_546_btn.setToolTip("Mercury 405nm (violet) and 546nm (green)")
        self.preset_405_546_btn.clicked.connect(lambda: self.set_trim_preset(405, 546))
        preset_buttons.addWidget(self.preset_405_546_btn)
        
        trim_layout.addLayout(preset_buttons, 0, 1, 1, 2)
        
        # Trim Point 1
        trim_layout.addWidget(QLabel("Trim Point 1:"), 1, 0)
        self.trim1_spin = QSpinBox()
        self.trim1_spin.setRange(200, 1000)
        self.trim1_spin.setValue(436)
        self.trim1_spin.setSuffix(" nm")
        self.trim1_spin.setToolTip("First calibration wavelength (e.g., Hg 436nm)")
        self.trim1_spin.valueChanged.connect(self.on_trim1_changed)
        trim_layout.addWidget(self.trim1_spin, 1, 1)
        
        # Trim Point 2
        trim_layout.addWidget(QLabel("Trim Point 2:"), 1, 2)
        self.trim2_spin = QSpinBox()
        self.trim2_spin.setRange(200, 1000)
        self.trim2_spin.setValue(546)
        self.trim2_spin.setSuffix(" nm")
        self.trim2_spin.setToolTip("Second calibration wavelength (e.g., Hg 546nm)")
        self.trim2_spin.valueChanged.connect(self.on_trim2_changed)
        trim_layout.addWidget(self.trim2_spin, 1, 3)
        
        layout.addLayout(trim_layout)
        
        # Trim mode controls
        trim_controls = QHBoxLayout()
        
        self.trim_mode_cb = QCheckBox("Trim/Align Mode")
        self.trim_mode_cb.setToolTip("Enable to drag spectrum and align with trim points")
        self.trim_mode_cb.stateChanged.connect(self.on_trim_mode_changed)
        self.trim_mode_cb.setStyleSheet("font-weight: bold; color: #0066cc;")
        trim_controls.addWidget(self.trim_mode_cb)
        
        self.show_trim_markers_cb = QCheckBox("Show Trim Markers")
        self.show_trim_markers_cb.setChecked(True)
        self.show_trim_markers_cb.setToolTip("Show vertical lines at trim point wavelengths")
        trim_controls.addWidget(self.show_trim_markers_cb)
        
        trim_controls.addStretch()
        layout.addLayout(trim_controls)
        
        # Instructions when trim mode is active
        self.trim_instructions = QLabel(
            "🖱️ Trim Mode Active: Click and drag spectrum LEFT or RIGHT to align peaks with trim markers"
        )
        self.trim_instructions.setProperty("class", "warning-box")
        self.trim_instructions.setVisible(False)
        layout.addWidget(self.trim_instructions)
        
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
            
        self.ax.set_ylim(0, 270)
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
                    self.ax.text(wl, 260, f'{label}nm', fontsize=7, ha='center')
        
        # Process intensity data
        intensity_to_plot = self.intensity.copy()
        
        if not self.hold_peaks and len(intensity_to_plot) > 17:
            try:
                intensity_to_plot = savitzky_golay(intensity_to_plot, 17, self.savpoly)
            except:
                pass
        
        # Plot using wavelength on x-axis
        if len(self.wavelengthData) > 0 and len(self.wavelengthData) == len(intensity_to_plot):
            x_data = self.wavelengthData
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
        
        # Show calibration points (old method - removed)
        
        # Show trim point markers (Theremino-style calibration)
        if self.show_trim_markers_cb.isChecked():
            # Draw vertical lines at trim points
            ylim = self.ax.get_ylim()
            
            # Trim point 1
            self.ax.axvline(x=self.trim_point1, color='red', linewidth=2, linestyle='--', alpha=0.8, label=f'Trim 1: {self.trim_point1}nm')
            self.ax.text(self.trim_point1, ylim[1] * 0.95, f'{self.trim_point1}nm', 
                        fontsize=9, ha='center', va='top', color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', linewidth=2))
            
            # Trim point 2
            self.ax.axvline(x=self.trim_point2, color='blue', linewidth=2, linestyle='--', alpha=0.8, label=f'Trim 2: {self.trim_point2}nm')
            self.ax.text(self.trim_point2, ylim[1] * 0.95, f'{self.trim_point2}nm', 
                        fontsize=9, ha='center', va='top', color='blue', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='blue', linewidth=2))
        
        # Draw without animation for performance
        self.canvas.draw_idle()
    
    def on_plot_click(self, event):
        """Handle click on spectrum plot."""
        if event.inaxes != self.ax:
            return
        
        if self.calibration_mode and event.xdata is not None:
            pixel = int(round(event.xdata))
            if 0 <= pixel < self.frame_width:
                self.click_positions.append(pixel)
                self.status_label.setText(f"Calibration point added: {pixel} px")
    
    def on_exposure_changed(self, value):
        """Handle exposure slider change."""
        self.camera_exposure = value
        if value == -1:
            self.exposure_label.setText("Auto")
        else:
            self.exposure_label.setText(str(value))
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.set_camera_exposure(value)
    
    def on_brightness_changed(self, value):
        """Handle brightness slider change."""
        self.camera_brightness = value
        self.brightness_label.setText(str(value))
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.set_camera_brightness(value / 100.0)
    
    def on_contrast_changed(self, value):
        """Handle contrast slider change."""
        self.camera_contrast = value / 100.0
        self.contrast_label.setText(f"{self.camera_contrast:.1f}")
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.set_camera_contrast(self.camera_contrast)
    
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
            cal_file = Path(__file__).parent.parent / 'caldata.txt'
            caldata = readcal(self.frame_width, str(cal_file))
            self.wavelengthData = caldata[0]
            self.graticule_data = generateGraticule(self.wavelengthData)
            self.tens = self.graticule_data[0]
            self.fifties = self.graticule_data[1]
    
    def apply_current_camera_settings(self):
        """Apply current camera settings to running camera."""
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.set_camera_exposure(self.camera_exposure)
            self.camera_thread.set_camera_brightness(self.camera_brightness / 100.0)
            self.camera_thread.set_camera_contrast(self.camera_contrast)
    
    def on_hold_peaks_changed(self, state):
        """Handle hold peaks checkbox."""
        self.hold_peaks = (state == Qt.CheckState.Checked.value)
        if not self.hold_peaks:
            self.intensity = np.zeros(self.frame_width)
    
    def on_measure_changed(self, state):
        """Handle measure mode checkbox."""
        self.measure_mode = (state == Qt.CheckState.Checked.value)
        # Measure mode is independent of trim mode
    
    def set_trim_preset(self, point1, point2):
        """Set trim point preset values."""
        self.trim1_spin.setValue(point1)
        self.trim2_spin.setValue(point2)
        self.status_label.setText(f"Trim points set to {point1}nm and {point2}nm")
    
    def on_trim1_changed(self, value):
        """Handle trim point 1 change."""
        self.trim_point1 = float(value)
    
    def on_trim2_changed(self, value):
        """Handle trim point 2 change."""
        self.trim_point2 = float(value)
    
    def on_trim_mode_changed(self, state):
        """Handle trim mode checkbox."""
        self.trim_mode = (state == Qt.CheckState.Checked.value)
        self.trim_instructions.setVisible(self.trim_mode)
        
        if self.trim_mode:
            self.status_label.setText("🖱️ Trim Mode: Click and drag spectrum to align peaks")
            # Connect mouse events to spectrum plot
            self.spec_canvas.mpl_connect('button_press_event', self.on_spectrum_mouse_press)
            self.spec_canvas.mpl_connect('motion_notify_event', self.on_spectrum_mouse_move)
            self.spec_canvas.mpl_connect('button_release_event', self.on_spectrum_mouse_release)
        else:
            self.status_label.setText("Trim Mode disabled")
            # Disconnect mouse events
            self.spec_canvas.mpl_disconnect('button_press_event')
            self.spec_canvas.mpl_disconnect('motion_notify_event')
            self.spec_canvas.mpl_disconnect('button_release_event')
    
    def on_spectrum_mouse_press(self, event):
        """Handle mouse press on spectrum plot."""
        if not self.trim_mode or event.inaxes != self.spec_ax:
            return
        
        # Record drag start position
        self.drag_start_x = event.xdata
        self.initial_nm_min = self.nm_min
        self.initial_nm_max = self.nm_max
    
    def on_spectrum_mouse_move(self, event):
        """Handle mouse move on spectrum plot - adjust wavelength scale."""
        if not self.trim_mode or self.drag_start_x is None or event.inaxes != self.spec_ax:
            return
        
        # Calculate drag delta in wavelength units
        drag_delta = event.xdata - self.drag_start_x
        
        # Adjust wavelength range by dragging
        # Moving right = shift spectrum left = increase wavelength values
        self.nm_min = self.initial_nm_min + drag_delta
        self.nm_max = self.initial_nm_max + drag_delta
        
        # Update wavelength calibration
        self.wavelengthData = np.linspace(self.nm_min, self.nm_max, self.frame_width)
        
        # Update status
        self.status_label.setText(f"Wavelength range: {self.nm_min:.1f} - {self.nm_max:.1f} nm (drag: {drag_delta:+.1f})")
        
        # Redraw spectrum with new wavelength scale
        self.update_spectrum_plot()
    
    def on_spectrum_mouse_release(self, event):
        """Handle mouse release on spectrum plot."""
        if not self.trim_mode:
            return
        
        # Reset drag tracking
        self.drag_start_x = None
        self.initial_nm_min = None
        self.initial_nm_max = None
        
        self.status_label.setText(f"✓ Calibration adjusted: {self.nm_min:.1f} - {self.nm_max:.1f} nm")
    
    def update_spectrum_plot(self):
        """Force spectrum plot update during drag operations."""
        # Just call the regular update_spectrum which will redraw with new wavelength scale
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

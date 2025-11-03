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
        
        # Camera settings
        self.frame_width = 1280  # Default to 720p
        self.frame_height = 720
        self.camera_exposure = -1  # Auto exposure
        self.camera_brightness = 0
        self.camera_contrast = 1.0
        self.camera_fps = 30
        
        # ROI settings for spectral line extraction (absolute position on frame)
        self.roi_y_start = 320  # Y position of top of ROI (default: center)
        self.roi_height = 80    # Height of ROI
        
        # Processing settings
        self.savpoly = 7
        self.mindist = 50
        self.thresh = 20
        
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
        layout.setContentsMargins(100, 20, 100, 20)
        layout.setSpacing(18)
        
        # Title
        title = QLabel("Live Spectrum Capture")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        subtitle = QLabel("Real-time spectral analysis from camera-based spectrometer")
        subtitle.setStyleSheet("color: #444444; font-size: 14px; margin-bottom: 8px;")
        layout.addWidget(subtitle)
        
        # Camera control
        camera_group = self.create_camera_controls()
        layout.addWidget(camera_group)
        
        # Camera settings
        settings_group = self.create_camera_settings()
        layout.addWidget(settings_group)
        
        # Camera preview
        preview_group = QGroupBox("Camera Preview")
        preview_layout = QVBoxLayout()
        
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(800, 100)
        self.camera_label.setMaximumHeight(150)
        self.camera_label.setStyleSheet("background-color: #000000; border: 1px solid #cccccc;")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setText("Camera not started")
        preview_layout.addWidget(self.camera_label)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Spectrum graph
        spectrum_group = QGroupBox("Spectrum Analysis")
        spectrum_layout = QVBoxLayout()
        
        self.figure = Figure(figsize=(12, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(350)
        self.ax = self.figure.add_subplot(111)
        
        # Connect click event for calibration
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        
        spectrum_layout.addWidget(self.canvas)
        spectrum_group.setLayout(spectrum_layout)
        layout.addWidget(spectrum_group)
        
        # Processing controls
        controls_group = self.create_processing_controls()
        layout.addWidget(controls_group)
        
        # Calibration section
        cal_group = self.create_calibration_section()
        layout.addWidget(cal_group)
        
        # Status bar
        self.status_label = QLabel(f"Status: {self.cal_msg1} | {self.cal_msg3}")
        self.status_label.setStyleSheet("color: #0066cc; font-weight: bold; padding: 5px;")
        layout.addWidget(self.status_label)
        
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
        
        # ROI Y Position (absolute)
        layout.addWidget(QLabel("ROI Y Position:"), 6, 0)
        roi_y_layout = QHBoxLayout()
        self.roi_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.roi_y_slider.setRange(0, 1080)  # Max common resolution height
        self.roi_y_slider.setValue(320)
        self.roi_y_slider.setToolTip("Y coordinate of top of ROI (pixels from top of frame)")
        self.roi_y_slider.valueChanged.connect(self.on_roi_y_changed)
        self.roi_y_label = QLabel("320 px")
        roi_y_layout.addWidget(self.roi_y_slider)
        roi_y_layout.addWidget(self.roi_y_label)
        layout.addLayout(roi_y_layout, 6, 1)
        
        # ROI Height
        layout.addWidget(QLabel("ROI Height:"), 7, 0)
        roi_h_layout = QHBoxLayout()
        self.roi_h_slider = QSlider(Qt.Orientation.Horizontal)
        self.roi_h_slider.setRange(10, 300)
        self.roi_h_slider.setValue(80)
        self.roi_h_slider.setToolTip("Height of spectral line region (pixels)")
        self.roi_h_slider.valueChanged.connect(self.on_roi_h_changed)
        self.roi_h_label = QLabel("80 px")
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
        
        group.setLayout(layout)
        return group
    
    def create_processing_controls(self):
        """Create spectrum processing controls."""
        group = QGroupBox("Processing Controls")
        layout = QGridLayout()
        layout.setSpacing(10)
        
        # Mode toggles
        self.hold_peaks_cb = QCheckBox("Hold Peaks")
        self.hold_peaks_cb.setToolTip("Hold maximum intensity values")
        self.hold_peaks_cb.stateChanged.connect(self.on_hold_peaks_changed)
        layout.addWidget(self.hold_peaks_cb, 0, 0)
        
        self.measure_cb = QCheckBox("Measure Mode")
        self.measure_cb.setToolTip("Click on spectrum to measure wavelength")
        self.measure_cb.stateChanged.connect(self.on_measure_changed)
        layout.addWidget(self.measure_cb, 0, 1)
        
        # Savitzky-Golay filter
        layout.addWidget(QLabel("Savgol Polynomial:"), 1, 0)
        savpoly_layout = QHBoxLayout()
        self.savpoly_slider = QSlider(Qt.Orientation.Horizontal)
        self.savpoly_slider.setRange(0, 15)
        self.savpoly_slider.setValue(7)
        self.savpoly_slider.setToolTip("Savitzky-Golay filter polynomial order")
        self.savpoly_slider.valueChanged.connect(self.on_savpoly_changed)
        self.savpoly_label = QLabel("7")
        savpoly_layout.addWidget(self.savpoly_slider)
        savpoly_layout.addWidget(self.savpoly_label)
        layout.addLayout(savpoly_layout, 1, 1)
        
        # Peak width
        layout.addWidget(QLabel("Peak Distance:"), 2, 0)
        mindist_layout = QHBoxLayout()
        self.mindist_slider = QSlider(Qt.Orientation.Horizontal)
        self.mindist_slider.setRange(1, 100)
        self.mindist_slider.setValue(50)
        self.mindist_slider.setToolTip("Minimum distance between peaks")
        self.mindist_slider.valueChanged.connect(self.on_mindist_changed)
        self.mindist_label = QLabel("50")
        mindist_layout.addWidget(self.mindist_slider)
        mindist_layout.addWidget(self.mindist_label)
        layout.addLayout(mindist_layout, 2, 1)
        
        # Threshold
        layout.addWidget(QLabel("Label Threshold:"), 3, 0)
        thresh_layout = QHBoxLayout()
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(20)
        self.thresh_slider.setToolTip("Threshold for peak labeling")
        self.thresh_slider.valueChanged.connect(self.on_thresh_changed)
        self.thresh_label = QLabel("20")
        thresh_layout.addWidget(self.thresh_slider)
        thresh_layout.addWidget(self.thresh_label)
        layout.addLayout(thresh_layout, 3, 1)
        
        # Save button
        self.save_btn = QPushButton("💾 Save Spectrum")
        self.save_btn.setMinimumHeight(35)
        self.save_btn.setToolTip("Save spectrum as CSV file")
        self.save_btn.clicked.connect(self.save_spectrum)
        layout.addWidget(self.save_btn, 4, 0, 1, 2)
        
        group.setLayout(layout)
        return group
    
    def create_calibration_section(self):
        """Create calibration controls."""
        group = QGroupBox("Wavelength Calibration")
        layout = QVBoxLayout()
        
        info = QLabel(
            "📌 Calibration Steps:\n"
            "1. Point spectrometer at known light source (e.g., fluorescent lamp)\n"
            "2. Enable 'Calibration Mode' and click on spectrum peaks\n"
            "3. Enter known wavelengths for each clicked position\n"
            "4. Click 'Apply Calibration' to compute wavelength mapping"
        )
        info.setStyleSheet("color: #555; background: #f0f0f0; padding: 10px; border-radius: 5px;")
        layout.addWidget(info)
        
        # Calibration controls
        cal_controls = QHBoxLayout()
        
        self.cal_mode_cb = QCheckBox("Calibration Mode")
        self.cal_mode_cb.setToolTip("Click on spectrum to record calibration points")
        self.cal_mode_cb.stateChanged.connect(self.on_calibration_mode_changed)
        cal_controls.addWidget(self.cal_mode_cb)
        
        self.clear_cal_btn = QPushButton("Clear Points")
        self.clear_cal_btn.setToolTip("Clear all calibration points")
        self.clear_cal_btn.clicked.connect(self.clear_calibration_points)
        cal_controls.addWidget(self.clear_cal_btn)
        
        self.apply_cal_btn = QPushButton("Apply Calibration")
        self.apply_cal_btn.setToolTip("Apply wavelength calibration")
        self.apply_cal_btn.clicked.connect(self.apply_calibration)
        cal_controls.addWidget(self.apply_cal_btn)
        
        self.finalize_cal_btn = QPushButton("✓ Finalize Calibration")
        self.finalize_cal_btn.setToolTip("Compute and save wavelength mapping")
        self.finalize_cal_btn.clicked.connect(self.finalize_calibration)
        self.finalize_cal_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        cal_controls.addWidget(self.finalize_cal_btn)
        
        cal_controls.addStretch()
        layout.addLayout(cal_controls)
        
        # Calibration points table
        self.cal_table = QTableWidget()
        self.cal_table.setColumnCount(2)
        self.cal_table.setHorizontalHeaderLabels(["Pixel", "Wavelength (nm)"])
        self.cal_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.cal_table.setMaximumHeight(150)
        layout.addWidget(self.cal_table)
        
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
        
        # Frame skip optimization - only process every Nth frame for display
        self.frame_counter += 1
        if self.frame_counter % 2 != 0:  # Skip every other frame for preview
            return
        
        # Extract spectral line using absolute ROI position
        try:
            y = self.roi_y_start
            h = self.roi_height
            
            # Ensure we don't exceed frame bounds
            if y < 0:
                y = 0
            if y + h > frame.shape[0]:
                h = frame.shape[0] - y
            if h <= 0:
                return
            
            cropped = frame[y:y+h, 0:min(self.frame_width, frame.shape[1])]
            
            # Display cropped region (throttled)
            if self.frame_counter % 4 == 0:  # Update preview less frequently
                self.display_camera_frame(cropped)
            
            # Process intensity data - vectorized for performance
            bwimage = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            rows, cols = bwimage.shape
            halfway = int(rows / 2)
            
            # Vectorized 3-row averaging (much faster than loop)
            if halfway > 0 and halfway < rows - 1:
                intensity_new = (
                    bwimage[halfway - 1, :cols].astype(np.float32) +
                    bwimage[halfway, :cols].astype(np.float32) +
                    bwimage[halfway + 1, :cols].astype(np.float32)
                ) / 3
                
                # Resize if needed to match expected width
                if cols != len(self.intensity):
                    self.intensity = np.zeros(cols)
                
                if self.hold_peaks:
                    self.intensity = np.maximum(self.intensity[:cols], intensity_new)
                else:
                    self.intensity[:cols] = intensity_new
        except Exception as e:
            print(f"Frame processing error: {e}")
    
    def display_camera_frame(self, frame):
        """Display camera frame in preview label with ROI markers."""
        # Draw spectral line marker
        h, w = frame.shape[:2]
        halfway = int(h / 2)
        
        # Draw center line (where data is extracted)
        cv2.line(frame, (0, halfway - 1), (w, halfway - 1), (0, 255, 0), 2)
        cv2.line(frame, (0, halfway + 1), (w, halfway + 1), (0, 255, 0), 2)
        
        # Draw ROI boundaries (top and bottom of cropped region)
        cv2.line(frame, (0, 0), (w, 0), (255, 255, 0), 2)  # Top
        cv2.line(frame, (0, h - 1), (w, h - 1), (255, 255, 0), 2)  # Bottom
        
        # Add text overlay with ROI info
        roi_end = self.roi_y_start + self.roi_height
        cv2.putText(frame, f"ROI: Y={self.roi_y_start}-{roi_end} (H={self.roi_height}px)", 
                   (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
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
        self.ax.set_xlim(0, len(self.intensity))
        self.ax.set_ylim(0, 270)
        self.ax.set_xlabel('Pixel Position', fontsize=9)
        self.ax.set_ylabel('Intensity', fontsize=9)
        self.ax.set_title('Real-time Spectrum', fontsize=11, fontweight='bold')
        self.ax.grid(True, alpha=0.2, linewidth=0.5)
        
        # Draw graticule - simplified
        for pos in self.tens[::2]:  # Draw every other line for performance
            if 0 <= pos < len(self.intensity):
                self.ax.axvline(x=pos, color='lightgray', linewidth=0.5, alpha=0.3)
        
        for pos_data in self.fifties:
            pos = pos_data[0]
            label = pos_data[1]
            if 0 <= pos < len(self.intensity):
                self.ax.axvline(x=pos, color='gray', linewidth=0.8, alpha=0.5)
                self.ax.text(pos, 260, f'{label}nm', fontsize=7, ha='center')
        
        # Process intensity data
        intensity_to_plot = self.intensity.copy()
        
        if not self.hold_peaks and len(intensity_to_plot) > 17:
            try:
                intensity_to_plot = savitzky_golay(intensity_to_plot, 17, self.savpoly)
            except:
                pass
        
        # Optimized plotting - plot as single line instead of individual segments
        x_data = np.arange(len(intensity_to_plot))
        
        # Create color array based on wavelengths
        colors = []
        step = max(1, len(intensity_to_plot) // 400)  # Sample colors, don't compute all
        for i in range(0, len(intensity_to_plot), step):
            if i < len(self.wavelengthData):
                wavelength = round(self.wavelengthData[i])
                rgb = wavelength_to_rgb(wavelength)
                colors.append((rgb[0]/255, rgb[1]/255, rgb[2]/255))
        
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
                    if peak < len(self.wavelengthData):
                        wavelength = round(self.wavelengthData[peak], 1)
                        height = intensity_to_plot[peak]
                        self.ax.plot(peak, height, 'ro', markersize=3)
                        self.ax.annotate(
                            f'{wavelength}nm',
                            xy=(peak, height),
                            xytext=(0, 8),
                            textcoords='offset points',
                            fontsize=7,
                            ha='center',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6, linewidth=0.5)
                        )
            except:
                pass
        
        # Show calibration points
        if self.calibration_mode:
            for pixel in self.click_positions:
                if 0 <= pixel < len(self.intensity):
                    self.ax.plot(pixel, 250, 'r*', markersize=12)
        
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
    
    def on_roi_y_changed(self, value):
        """Handle ROI Y position slider change."""
        self.roi_y_start = value
        self.roi_y_label.setText(f"{value} px")
    
    def on_roi_h_changed(self, value):
        """Handle ROI height slider change."""
        self.roi_height = value
        self.roi_h_label.setText(f"{value} px")
    
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
        if self.measure_mode:
            self.calibration_mode = False
            self.cal_mode_cb.setChecked(False)
    
    def on_calibration_mode_changed(self, state):
        """Handle calibration mode checkbox."""
        self.calibration_mode = (state == Qt.CheckState.Checked.value)
        if self.calibration_mode:
            self.measure_mode = False
            self.measure_cb.setChecked(False)
    
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
    
    def clear_calibration_points(self):
        """Clear all calibration points."""
        self.click_positions.clear()
        self.calibration_points.clear()
        self.cal_table.setRowCount(0)
        self.status_label.setText("Calibration points cleared")
    
    def apply_calibration(self):
        """Apply wavelength calibration."""
        if len(self.click_positions) < 3:
            QMessageBox.warning(
                self,
                "Insufficient Data",
                "Please select at least 3 calibration points on the spectrum.\n\n"
                "Enable 'Calibration Mode' and click on known spectral peaks."
            )
            return
        
        # Update table with clicked positions
        self.cal_table.setRowCount(len(self.click_positions))
        for i, pixel in enumerate(self.click_positions):
            pixel_item = QTableWidgetItem(str(pixel))
            pixel_item.setFlags(pixel_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.cal_table.setItem(i, 0, pixel_item)
            self.cal_table.setItem(i, 1, QTableWidgetItem(""))
        
        QMessageBox.information(
            self,
            "Enter Wavelengths",
            f"Please enter known wavelengths (in nm) for the {len(self.click_positions)} selected peaks.\n\n"
            "Common fluorescent lamp peaks:\n"
            "• 405.4 nm (Mercury)\n"
            "• 436.6 nm (Mercury)\n"
            "• 546.5 nm (Mercury)\n"
            "• 611.6 nm (Europium)"
        )
        
        # Wait for user to fill in wavelengths
        QTimer.singleShot(100, self.check_calibration_table)
    
    def check_calibration_table(self):
        """Check if calibration table is filled and apply."""
        # This is a simplified version - in practice, you'd want a dialog or button to confirm
        pass
    
    def finalize_calibration(self):
        """Finalize calibration after wavelengths are entered."""
        self.calibration_points.clear()
        
        for i in range(self.cal_table.rowCount()):
            pixel_item = self.cal_table.item(i, 0)
            wavelength_item = self.cal_table.item(i, 1)
            
            if pixel_item and wavelength_item:
                try:
                    pixel = int(pixel_item.text())
                    wavelength = float(wavelength_item.text())
                    self.calibration_points.append((pixel, wavelength))
                except ValueError:
                    continue
        
        if len(self.calibration_points) < 3:
            QMessageBox.warning(self, "Error", "Please enter valid wavelengths for all points")
            return
        
        # Write calibration file
        cal_file = Path(__file__).parent.parent / 'caldata.txt'
        if writecal(self.calibration_points, str(cal_file)):
            # Reload calibration
            caldata = readcal(self.frame_width, str(cal_file))
            self.wavelengthData = caldata[0]
            self.cal_msg1 = caldata[1]
            self.cal_msg2 = caldata[2]
            self.cal_msg3 = caldata[3]
            
            # Regenerate graticule
            self.graticule_data = generateGraticule(self.wavelengthData)
            self.tens = self.graticule_data[0]
            self.fifties = self.graticule_data[1]
            
            self.status_label.setText(f"✓ {self.cal_msg1} | {self.cal_msg3}")
            
            QMessageBox.information(
                self,
                "Calibration Success",
                f"Wavelength calibration applied!\n\n{self.cal_msg1}\n{self.cal_msg2}\n{self.cal_msg3}"
            )
            
            # Clear calibration mode
            self.calibration_mode = False
            self.cal_mode_cb.setChecked(False)
            self.click_positions.clear()
        else:
            QMessageBox.critical(self, "Error", "Failed to save calibration data")
    
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

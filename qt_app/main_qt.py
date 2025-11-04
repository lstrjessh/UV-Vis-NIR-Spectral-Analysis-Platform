from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor, QIcon
from pathlib import Path
import sys

from qt_app.views.home import HomeView
from qt_app.views.capture_view import CaptureView
from qt_app.views.absorbance_view import AbsorbanceView
from qt_app.views.viewer_view import ViewerView
from qt_app.views.calibration_view import CalibrationView
from qt_app.views.predict_view import PredictView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("🔬 Spectral Analysis Platform - Advanced ML Calibration Tool")
        self.resize(1650, 1000)
        
        # Set window icon if available
        icon_path = Path(__file__).parent.parent / 'assets' / 'app.ico'
        if not icon_path.exists():
            # Try alternative paths
            icon_path = Path(__file__).parent.parent / 'app.ico'
        if not icon_path.exists():
            # Try PNG as fallback
            icon_path = Path(__file__).parent.parent / 'assets' / 'app.png'
        if not icon_path.exists():
            icon_path = Path(__file__).parent.parent / 'app.png'
        
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        
        # Apply modern styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: palette(window);
            }
            QTabWidget::pane {
                border: 1px solid palette(mid);
                border-radius: 6px;
                background-color: palette(base);
                margin-top: -1px;
            }
            QTabBar::tab {
                background-color: palette(button);
                color: palette(text);
                border: 1px solid palette(mid);
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 10px 20px;
                margin-right: 2px;
                font-weight: 500;
                font-size: 13px;
            }
            QTabBar::tab:selected {
                background-color: palette(base);
                border-bottom: 2px solid palette(highlight);
            }
            QTabBar::tab:hover:!selected {
                background-color: palette(midlight);
            }
            QTabBar::tab:!selected {
                margin-top: 3px;
            }
            /* Make checkboxes more visible application-wide */
            QCheckBox {
                font-size: 14px;
                font-weight: 600;
                padding: 2px 0px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 3px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #2196F3;
                background: #ffffff;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #1565c0;
                background: #2196F3;
            }
            QCheckBox::indicator:unchecked:hover {
                border-color: #1565c0;
            }
            QCheckBox::indicator:disabled {
                border: 2px solid palette(mid);
                background: palette(midlight);
            }
        """)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.setToolTip(
            "Navigate between different analysis tools.\n\n"
            "💡 Hover over any control to see detailed help\n"
            "📚 Start with Home for an overview"
        )

        # Add tabs with tooltips
        home_widget = HomeView(self)
        home_widget.setToolTip("Overview of available features and quick start guide")
        self.tabs.addTab(home_widget, "🏠 Home")
        
        capture_widget = CaptureView(self)
        capture_widget.setToolTip("Real-time spectral data capture from camera-based spectrometer")
        self.tabs.addTab(capture_widget, "📷 Capture")
        
        abs_widget = AbsorbanceView(self)
        abs_widget.setToolTip("Calculate absorbance from reference and sample spectra with peak detection")
        self.tabs.addTab(abs_widget, "📊 Absorbance")
        
        viewer_widget = ViewerView(self)
        viewer_widget.setToolTip("View and compare multiple spectra with processing options")
        self.tabs.addTab(viewer_widget, "👁 Viewer")
        
        calib_widget = CalibrationView(self)
        calib_widget.setToolTip("Train machine learning models for concentration prediction")
        self.tabs.addTab(calib_widget, "🔬 Calibration")
        
        predict_widget = PredictView(self)
        predict_widget.setToolTip("Apply trained models to predict concentrations in new spectra")
        self.tabs.addTab(predict_widget, "🎯 Predict")

        self.setCentralWidget(self.tabs)


def setup_theme(app: QApplication) -> None:
    """Setup light-only theme"""
    app.setStyle("Fusion")
    light_palette = QPalette()
    light_palette.setColor(QPalette.ColorRole.Window, QColor(248, 249, 250))
    light_palette.setColor(QPalette.ColorRole.WindowText, QColor(33, 37, 41))
    light_palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    light_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(248, 249, 250))
    light_palette.setColor(QPalette.ColorRole.Text, QColor(33, 37, 41))
    light_palette.setColor(QPalette.ColorRole.Button, QColor(233, 236, 239))
    light_palette.setColor(QPalette.ColorRole.ButtonText, QColor(33, 37, 41))
    light_palette.setColor(QPalette.ColorRole.Link, QColor(0, 123, 255))
    light_palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 123, 255))
    light_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    light_palette.setColor(QPalette.ColorRole.Mid, QColor(206, 212, 218))
    light_palette.setColor(QPalette.ColorRole.Midlight, QColor(222, 226, 230))
    app.setPalette(light_palette)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Spectral Analysis Platform")
    
    # Setup adaptive theme
    setup_theme(app)
    
    # Load modern stylesheet
    style_path = Path(__file__).parent / 'resources' / 'modern_style.qss'
    if style_path.exists():
        with open(style_path, 'r') as f:
            app.setStyleSheet(f.read())
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
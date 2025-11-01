from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor, QIcon
from pathlib import Path
import sys

from qt_app.views.home import HomeView
from qt_app.views.absorbance_view import AbsorbanceView
from qt_app.views.viewer_view import ViewerView
from qt_app.views.calibration_view import CalibrationView
from qt_app.views.predict_view import PredictView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Spectral Analysis Platform")
        self.resize(1400, 900)
        
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

        # Add tabs
        self.tabs.addTab(HomeView(self), "🏠 Home")
        self.tabs.addTab(AbsorbanceView(self), "📊 Absorbance")
        self.tabs.addTab(ViewerView(self), "👁 Viewer")
        self.tabs.addTab(CalibrationView(self), "🔬 Calibration")
        self.tabs.addTab(PredictView(self), "🎯 Predict")

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
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
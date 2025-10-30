from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt6.QtCore import Qt
import sys

from qt_app.views.home import HomeView
from qt_app.views.absorbance_view import AbsorbanceView
from qt_app.views.viewer_view import ViewerView
from qt_app.views.calibration_view import CalibrationView
from qt_app.views.predict_view import PredictView
from qt_app.views.feedback_view import FeedbackView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Spectral Analysis Platform (PyQt)")
        self.resize(1200, 800)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setMovable(True)

        # Tabs mirroring the Streamlit pages
        self.tabs.addTab(HomeView(self), "Home")
        self.tabs.addTab(AbsorbanceView(self), "Calculate Absorbance")
        self.tabs.addTab(ViewerView(self), "View Spectra")
        self.tabs.addTab(CalibrationView(self), "Model Calibration")
        self.tabs.addTab(PredictView(self), "Predict Concentration")
        self.tabs.addTab(FeedbackView(self), "Provide Feedback")

        self.setCentralWidget(self.tabs)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Spectral Analysis Platform (PyQt)")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()



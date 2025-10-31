from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout, QFrame
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt


class HomeView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.setContentsMargins(48, 48, 48, 48)
        layout.setSpacing(32)

        # Title
        title = QLabel("Spectral Analysis Platform")
        title_font = QFont()
        title_font.setPointSize(32)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel(
            "Advanced tools for spectroscopic data analysis and machine learning"
        )
        subtitle_font = QFont()
        subtitle_font.setPointSize(14)
        subtitle.setFont(subtitle_font)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: palette(mid); margin-bottom: 20px;")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # Feature cards
        cards = QGridLayout()
        cards.setSpacing(20)
        
        features = [
            ("🧪", "Calculate Absorbance", 
             "Process reference, sample, and dark spectra to calculate absorbance with peak detection"),
            ("📊", "View Spectra", 
             "Visualize and compare multiple spectral datasets with customizable processing"),
            ("🔬", "Model Calibration", 
             "Train machine learning models for concentration prediction using various algorithms"),
            ("🎯", "Predict Concentration", 
             "Apply trained models to new spectra for accurate concentration predictions"),
        ]
        
        for idx, (icon, title_text, desc) in enumerate(features):
            row = idx // 2
            col = idx % 2
            
            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                    background: palette(base);
                    border: 1px solid palette(mid);
                    border-radius: 12px;
                    padding: 20px;
                }
                QFrame:hover {
                    border: 2px solid #2196F3;
                }
            """)
            
            card_layout = QVBoxLayout(card)
            card_layout.setSpacing(8)
            
            # Icon
            icon_label = QLabel(icon)
            icon_font = QFont()
            icon_font.setPointSize(36)
            icon_label.setFont(icon_font)
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(icon_label)
            
            # Title
            title_label = QLabel(title_text)
            title_font = QFont()
            title_font.setPointSize(16)
            title_font.setBold(True)
            title_label.setFont(title_font)
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(title_label)
            
            # Description
            desc_label = QLabel(desc)
            desc_label.setWordWrap(True)
            desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            desc_label.setStyleSheet("color: palette(mid); font-size: 13px;")
            card_layout.addWidget(desc_label)
            
            cards.addWidget(card, row, col)
        
        layout.addLayout(cards)
        
        # Instructions
        instructions = QLabel(
            "Use the tabs above to navigate between features. "
            "Start by loading your spectral data in the 'Calculate Absorbance' or 'View Spectra' tab."
        )
        instructions.setWordWrap(True)
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setStyleSheet("""
            background: palette(midlight);
            padding: 16px;
            border-radius: 8px;
            font-size: 13px;
            margin-top: 20px;
        """)
        layout.addWidget(instructions)
        
        layout.addStretch()
        self.setLayout(layout)
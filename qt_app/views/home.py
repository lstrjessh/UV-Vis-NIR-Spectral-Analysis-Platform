from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout, QFrame, QScrollArea
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt


class HomeView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        
        # Main scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(80, 30, 80, 30)
        layout.setSpacing(30)

        # Title
        title = QLabel("🔬 Spectral Analysis Platform")
        title.setProperty("class", "title")
        # Set font for emoji rendering
        title_font = QFont()
        title_font.setPointSize(28)
        title_font.setBold(True)
        title_font.setFamily("Segoe UI, Segoe UI Emoji")
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel(
            "Advanced tools for spectroscopic data analysis and machine learning\n"
            "Build calibration models, analyze spectra, and predict concentrations"
        )
        subtitle.setProperty("class", "subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # Feature cards
        cards = QGridLayout()
        cards.setSpacing(25)
        cards.setContentsMargins(0, 20, 0, 20)
        
        features = [
            ("📷", "Live Capture", 
             "Real-time spectral data capture from camera-based spectrometer. "
             "Perform wavelength calibration, peak detection, and save spectra for analysis.",
             "• Real-time camera preview\n• Wavelength calibration\n• Peak detection & labeling\n• Export to CSV",
             "#667eea"),
            ("📊", "Calculate Absorbance", 
             "Process reference, sample, and dark spectra to calculate absorbance with automatic peak detection. "
             "Apply smoothing filters and customize peak detection parameters.",
             "• Reference/blank spectrum\n• Sample spectrum\n• Optional dark correction\n• Peak detection & analysis",
             "#f093fb"),
            ("👁️", "View Spectra", 
             "Visualize and compare multiple spectral datasets with customizable processing options. "
             "Apply normalization, smoothing, and peak detection to multiple spectra simultaneously.",
             "• Multi-spectrum overlay\n• Normalization options\n• Gaussian smoothing\n• Interactive peak detection",
             "#4facfe"),
            ("🤖", "Model Calibration", 
             "Train machine learning models for concentration prediction using various algorithms including "
             "PLSR, Ridge, Lasso, Random Forest, XGBoost, Neural Networks, and SVR. Features automatic "
             "hyperparameter optimization and cross-validation.",
             "• Multiple ML algorithms\n• Bayesian optimization\n• K-fold cross-validation\n• Preprocessing options",
             "#a8e063"),
            ("🎯", "Predict Concentration", 
             "Apply trained calibration models to new spectral data for accurate concentration predictions. "
             "Load saved models and batch process multiple samples.",
             "• Load trained models\n• Batch predictions\n• Export results to CSV\n• Quick analysis",
             "#ffa502"),
        ]
        
        for idx, (icon, title_text, desc, features_text, color) in enumerate(features):
            row = idx // 3
            col = idx % 3
            
            card = QFrame()
            card.setProperty("class", "card-elevated")
            card.setStyleSheet(f"""
                QFrame {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                stop:0 #ffffff, stop:1 #fafbff);
                    border: none;
                    border-radius: 16px;
                    padding: 20px;
                }}
                QFrame:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                                stop:0 #ffffff, stop:1 #f0f4ff);
                    border: 2px solid {color};
                }}
            """)
            card.setToolTip(features_text)
            
            card_layout = QVBoxLayout(card)
            card_layout.setSpacing(12)
            card_layout.setContentsMargins(15, 15, 15, 15)
            
            # Icon - display emoji directly without background circle
            icon_label = QLabel(icon)
            # Set font for emoji rendering
            icon_font = QFont()
            icon_font.setPointSize(80)
            icon_font.setFamily("Segoe UI Emoji")
            icon_label.setFont(icon_font)
            icon_label.setStyleSheet("background: transparent; font-size: 80pt;")
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_label.setMinimumHeight(100)
            
            card_layout.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignCenter)
            
            # Title
            title_label = QLabel(title_text)
            title_label.setStyleSheet(f"""
                font-size: 16pt; 
                font-weight: 700; 
                color: {color};
                background: transparent;
            """)
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            card_layout.addWidget(title_label)
            
            # Description
            desc_label = QLabel(desc)
            desc_label.setWordWrap(True)
            desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            desc_label.setStyleSheet("""
                color: #5a5a5a; 
                font-size: 11pt; 
                line-height: 1.5;
                background: transparent;
            """)
            card_layout.addWidget(desc_label)
            
            card_layout.addStretch()
            
            cards.addWidget(card, row, col)
        
        layout.addLayout(cards)
        
        # Instructions
        instructions = QLabel(
            "🚀 <b>Quick Start:</b> Use the tabs above to navigate between features<br>"
            "💡 <b>Tip:</b> Hover over controls to see detailed tooltips explaining each parameter<br>"
            "🔧 <b>Hardware:</b> Use '<b>Capture</b>' tab for real-time spectrometer data acquisition<br>"
            "📂 <b>Analysis:</b> Load spectral data in the '<b>Absorbance</b>' or '<b>Viewer</b>' tab to begin"
        )
        # Set font for emoji rendering
        inst_font = QFont()
        inst_font.setPointSize(10)
        inst_font.setFamily("Segoe UI, Segoe UI Emoji")
        instructions.setFont(inst_font)
        instructions.setWordWrap(True)
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setTextFormat(Qt.TextFormat.RichText)
        instructions.setProperty("class", "info-box")
        layout.addWidget(instructions)
        
        # Add quick info box
        info_frame = QFrame()
        info_frame.setProperty("class", "card")
        info_layout = QVBoxLayout(info_frame)
        info_layout.setSpacing(15)
        
        info_title = QLabel("📋 Supported File Formats")
        info_title.setProperty("class", "section-header")
        # Set font for emoji rendering
        info_title_font = QFont()
        info_title_font.setPointSize(14)
        info_title_font.setFamily("Segoe UI, Segoe UI Emoji")
        info_title.setFont(info_title_font)
        info_layout.addWidget(info_title)
        
        format_text = QLabel(
            "• <b>CSV files (.csv)</b> - Comma-separated values<br>"
            "• <b>Text files (.txt)</b> - Tab or space delimited<br>"
            "• <b>Data files (.dat)</b> - Standard spectroscopy format<br><br>"
            "<b>Required structure:</b> Two columns with wavelength (nm) and intensity/counts"
        )
        format_text.setTextFormat(Qt.TextFormat.RichText)
        format_text.setWordWrap(True)
        format_text.setStyleSheet("font-size: 11pt; line-height: 1.6; background: transparent;")
        info_layout.addWidget(format_text)
        
        layout.addWidget(info_frame)
        
        layout.addStretch()
        
        scroll.setWidget(content)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
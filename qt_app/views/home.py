from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout, QFrame, QTextBrowser, QScrollArea
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
        layout.setContentsMargins(150, 30, 150, 30)
        layout.setSpacing(25)

        # Title
        title = QLabel("🔬 Spectral Analysis Platform")
        title_font = QFont()
        title_font.setPointSize(32)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel(
            "Advanced tools for spectroscopic data analysis and machine learning\n"
            "Build calibration models, analyze spectra, and predict concentrations"
        )
        subtitle_font = QFont()
        subtitle_font.setPointSize(14)
        subtitle.setFont(subtitle_font)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #444444; margin-bottom: 20px;")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # Feature cards
        cards = QGridLayout()
        cards.setSpacing(20)
        cards.setContentsMargins(0, 10, 0, 10)
        
        features = [
            ("📊", "Calculate Absorbance", 
             "Process reference, sample, and dark spectra to calculate absorbance with automatic peak detection. "
             "Apply smoothing filters and customize peak detection parameters.",
             "• Reference/blank spectrum\n• Sample spectrum\n• Optional dark correction\n• Peak detection & analysis"),
            ("👁", "View Spectra", 
             "Visualize and compare multiple spectral datasets with customizable processing options. "
             "Apply normalization, smoothing, and peak detection to multiple spectra simultaneously.",
             "• Multi-spectrum overlay\n• Normalization options\n• Gaussian smoothing\n• Interactive peak detection"),
            ("🤖", "Model Calibration", 
             "Train machine learning models for concentration prediction using various algorithms including "
             "PLSR, Ridge, Lasso, Random Forest, XGBoost, Neural Networks, and SVR. Features automatic "
             "hyperparameter optimization and cross-validation.",
             "• Multiple ML algorithms\n• Bayesian optimization\n• K-fold cross-validation\n• Preprocessing options"),
            ("🎯", "Predict Concentration", 
             "Apply trained calibration models to new spectral data for accurate concentration predictions. "
             "Load saved models and batch process multiple samples.",
             "• Load trained models\n• Batch predictions\n• Export results to CSV\n• Quick analysis"),
        ]
        
        for idx, (icon, title_text, desc, features_text) in enumerate(features):
            row = idx // 2
            col = idx % 2
            
            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                    background: #ffffff;
                    border: 2px solid #d0d0d0;
                    border-radius: 8px;
                    padding: 18px;
                    max-height: 220px;
                }
                QFrame:hover {
                    border: 2px solid #2196F3;
                    background: #f5f9ff;
                }
            """)
            # Add tooltip with features
            card.setToolTip(features_text)
            
            card_layout = QVBoxLayout(card)
            card_layout.setSpacing(6)
            card_layout.setContentsMargins(8, 8, 8, 8)
            
            # Icon
            icon_label = QLabel(icon)
            icon_font = QFont()
            icon_font.setPointSize(28)
            icon_label.setFont(icon_font)
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_label.setMaximumHeight(35)
            card_layout.addWidget(icon_label)
            
            # Title
            title_label = QLabel(title_text)
            title_font = QFont()
            title_font.setPointSize(14)
            title_font.setBold(True)
            title_label.setFont(title_font)
            title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title_label.setStyleSheet("color: #1a1a1a;")
            title_label.setMaximumHeight(25)
            card_layout.addWidget(title_label)
            
            # Description
            desc_label = QLabel(desc)
            desc_label.setWordWrap(True)
            desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            desc_label.setStyleSheet("color: #3a3a3a; font-size: 11px; padding: 2px;")
            desc_label.setMaximumHeight(120)
            card_layout.addWidget(desc_label)
            
            cards.addWidget(card, row, col)
        
        layout.addLayout(cards)
        
        # Instructions
        instructions = QLabel(
            "🚀 <b>Quick Start:</b> Use the tabs above to navigate between features.<br>"
            "💡 <b>Tip:</b> Hover over controls to see detailed tooltips explaining each parameter.<br>"
            "📂 <b>Getting Started:</b> Load your spectral data in the '<b>Absorbance</b>' tab to begin analysis."
        )
        instructions.setWordWrap(True)
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setTextFormat(Qt.TextFormat.RichText)
        instructions.setStyleSheet("""
            background: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
            font-size: 14px;
            margin-top: 20px;
            border: 1px solid #2196F3;
            color: #1a1a1a;
        """)
        layout.addWidget(instructions)
        
        # Add quick info box
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin-top: 12px;
                border: 1px solid #dee2e6;
            }
        """)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setSpacing(10)
        
        info_title = QLabel("📋 Supported File Formats")
        info_title_font = QFont()
        info_title_font.setBold(True)
        info_title_font.setPointSize(12)
        info_title.setFont(info_title_font)
        info_title.setStyleSheet("color: #1a1a1a;")
        info_layout.addWidget(info_title)
        
        format_text = QLabel(
            "• CSV files (.csv) - Comma-separated values\n"
            "• Text files (.txt) - Tab or space delimited\n"
            "• Data files (.dat) - Standard spectroscopy format\n\n"
            "<b>Required structure:</b> Two columns with wavelength (nm) and intensity/counts"
        )
        format_text.setTextFormat(Qt.TextFormat.RichText)
        format_text.setWordWrap(True)
        format_text.setStyleSheet("color: #2a2a2a; font-size: 12px;")
        info_layout.addWidget(format_text)
        
        layout.addWidget(info_frame)
        
        layout.addStretch()
        
        scroll.setWidget(content)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
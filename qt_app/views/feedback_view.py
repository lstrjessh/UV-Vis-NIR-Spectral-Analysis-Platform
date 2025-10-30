from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton


class FeedbackView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("User Feedback & Support"))
        self.text = QTextEdit()
        self.text.setPlaceholderText("Share your feedback, issues, or feature requests…")
        layout.addWidget(self.text)
        submit = QPushButton("Submit Feedback")
        submit.clicked.connect(self._submit)
        layout.addWidget(submit)
        self.status = QLabel()
        layout.addWidget(self.status)
        layout.addStretch(1)
        self.setLayout(layout)

    def _submit(self) -> None:
        content = self.text.toPlainText().strip()
        if content:
            self.status.setText("Thank you for your feedback!")
            self.text.clear()
        else:
            self.status.setText("Please enter some feedback before submitting.")



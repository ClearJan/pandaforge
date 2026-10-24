"""Visual theme: a restrained modern engineering dark theme (QSS)."""

DARK_QSS = """
QMainWindow, QDialog { background-color: #1e2227; }
QWidget { color: #d6dbe1; font-size: 12px; }
QDockWidget { titlebar-close-icon: none; }
QDockWidget::title {
    background: #262b31; padding: 5px 8px; border-bottom: 1px solid #333a42;
    font-weight: 600;
}
QGroupBox {
    border: 1px solid #333a42; border-radius: 4px; margin-top: 10px;
    padding: 8px 6px 6px 6px; font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; color: #7fb4e6; }
QPushButton {
    background-color: #2d3540; border: 1px solid #3d4754; border-radius: 3px;
    padding: 5px 12px; min-height: 18px;
}
QPushButton:hover { background-color: #39424f; }
QPushButton:pressed { background-color: #262d37; }
QPushButton:disabled { color: #6a7480; background-color: #262b31; }
QPushButton#dangerButton { border-color: #8c3b3b; background-color: #4a2727; }
QPushButton#dangerButton:hover { background-color: #5c3030; }
QPushButton#primaryButton { border-color: #2d5f8f; background-color: #24507a; }
QPushButton#primaryButton:hover { background-color: #2b5e91; }
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
    background-color: #14171b; border: 1px solid #3d4754; border-radius: 3px;
    padding: 2px 4px; selection-background-color: #2d5f8f;
}
QSlider::groove:horizontal { height: 4px; background: #3d4754; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #7fb4e6; width: 12px; margin: -5px 0; border-radius: 6px;
}
QSlider::sub-page:horizontal { background: #2d5f8f; border-radius: 2px; }
QTableWidget { background-color: #14171b; gridline-color: #2c333b; }
QHeaderView::section {
    background-color: #262b31; border: none; border-bottom: 1px solid #333a42;
    padding: 3px;
}
QTextEdit, QPlainTextEdit { background-color: #14171b; border: 1px solid #3d4754; }
QStatusBar { background: #262b31; }
QLabel[state="ok"] { color: #7ed491; }
QLabel[state="err"] { color: #ef7b7b; }
QLabel[state="warn"] { color: #e6c07b; }
QListWidget { background-color: #14171b; border: 1px solid #3d4754; }
QTabWidget::pane { border: 1px solid #333a42; }
QTabBar::tab {
    background: #262b31; padding: 4px 12px; border: 1px solid #333a42;
    border-bottom: none;
}
QTabBar::tab:selected { background: #2d3540; color: #ffffff; }
"""

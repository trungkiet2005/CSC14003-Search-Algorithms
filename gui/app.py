# File 1: app.py - PyQt6 version
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QTabWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import sys

from .comparison.comparison_tab_new import ComparisonTab
from .visualization.visualization_tab import VisualizationTab


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Swarm Intelligence Studio")
        self.setGeometry(100, 100, 1400, 800)
        
        # Set dark theme stylesheet
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QTabWidget::pane {
                border: 1px solid #3a3a3a;
                border-radius: 8px;
                background-color: #1a1a1a;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                color: #ffffff;
                padding: 8px 16px;
                margin: 1px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-size: 12px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #00A86B;
            }
            QTabBar::tab:hover {
                background-color: #3a3a3a;
            }
        """)
        
        # Main container
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Tab widget
        self.tabview = QTabWidget()
        self.tabview.setDocumentMode(True)
        layout.addWidget(self.tabview)
        
        # Create tabs
        self.visualization_widget = QWidget()
        self.comparison_widget = QWidget()
        
        self.tabview.addTab(self.visualization_widget, "🔬 Algorithm Visualization")
        self.tabview.addTab(self.comparison_widget, "⚖️ Performance Comparison")
        
        # Initialize tab content
        self.visualization_tab = VisualizationTab(self.visualization_widget)
        self.comparison_tab = ComparisonTab(self.comparison_widget)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    # Set application-wide font
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    window = App()
    window.show()
    sys.exit(app.exec())
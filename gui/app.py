"""gui/app.py - Main application window with tab-based interface"""

import customtkinter
import tkinter
from .comparison_tab import ComparisonTab
from .visualization_tab import VisualizationTab

customtkinter.set_appearance_mode("white")
customtkinter.set_default_color_theme("blue")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Search Algorithms Visualization & Analysis")
        self.geometry("1600x850")
        
        # Configure grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create tabview
        self.tabview = customtkinter.CTkTabview(self, width=1550)
        self.tabview.grid(row=0, column=0, padx=20, pady=(20, 20), sticky="nsew")
        
        # Add tabs
        self.tabview.add("🔍 Swarm Visualization")
        self.tabview.add("⚖️ Algorithm Comparison")
        
        # Create tab content
        self.visualization_tab = VisualizationTab(
            self.tabview.tab("🔍 Swarm Visualization")
        )
        
        self.comparison_tab = ComparisonTab(
            self.tabview.tab("⚖️ Algorithm Comparison")
        )
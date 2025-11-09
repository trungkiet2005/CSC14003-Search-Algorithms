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
        self.grid_rowconfigure(1, weight=1)

        # Create header
        self.header_frame = customtkinter.CTkFrame(self, height=80, corner_radius=0)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        self.header_frame.grid_columnconfigure(0, weight=1)
        
        # Logo/Title
        self.logo_label = customtkinter.CTkLabel(
            self.header_frame, 
            text="🔬 Algorithm Lab", 
            font=customtkinter.CTkFont(size=28, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=30, pady=(15, 5))
        
        # Subtitle
        self.subtitle_label = customtkinter.CTkLabel(
            self.header_frame, 
            text="Comprehensive Analysis & Visualization of Optimization Algorithms", 
            font=customtkinter.CTkFont(size=13),
            text_color=("gray60", "gray40")
        )
        self.subtitle_label.grid(row=1, column=0, padx=30, pady=(0, 15))

        # Create tabview
        self.tabview = customtkinter.CTkTabview(self, width=1550)
        self.tabview.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        
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
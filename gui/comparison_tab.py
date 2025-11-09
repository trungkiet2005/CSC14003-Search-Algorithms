"""gui/comparison_tab.py - Algorithm comparison tab"""

import customtkinter
import tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os

from gui.experiment_runner import ComparisonRunner


class ComparisonTab:
    def __init__(self, parent):
        self.parent = parent
        self.is_running = False
        self.current_view = "convergence"
        self.metric_data = {}
        
        # Configure grid
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        
        self._create_sidebar()
        self._create_results_area()
        self._init_placeholder_text()
        
    def _create_sidebar(self):
        """Create sidebar with controls"""
        self.sidebar_frame = customtkinter.CTkFrame(self.parent, width=300, corner_radius=10)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        self.sidebar_frame.grid_rowconfigure(10, weight=1)
        
        # Title
        title = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Algorithm Comparison",
            font=customtkinter.CTkFont(size=16, weight="bold")
        )
        title.grid(row=0, column=0, padx=20, pady=(20, 5), sticky="w")
        
        subtitle = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Compare algorithm pairs on benchmarks",
            font=customtkinter.CTkFont(size=11),
            text_color=("gray60", "gray40")
        )
        subtitle.grid(row=1, column=0, padx=20, pady=(0, 15), sticky="w")
        
        # Separator
        sep1 = customtkinter.CTkFrame(self.sidebar_frame, height=2, fg_color=("gray70", "gray30"))
        sep1.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        # Comparison selection
        comp_label = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Select Comparison",
            font=customtkinter.CTkFont(size=13, weight="bold")
        )
        comp_label.grid(row=3, column=0, padx=20, pady=(10, 5), sticky="w")
        
        self.comparison_menu = customtkinter.CTkOptionMenu(
            self.sidebar_frame,
            values=[
                "ACO vs SA (TSP)",
                "PSO vs HC (Rastrigin)",
                "ABC vs GA (Rastrigin)",
                "FA vs SA (Ackley)",
                "CS vs SA (Ackley)"
            ],
            command=self.change_experiment,
            width=260,
            height=35,
            font=customtkinter.CTkFont(size=12)
        )
        self.comparison_menu.grid(row=4, column=0, padx=20, pady=(0, 15))
        
        # Parameters
        params_label = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Parameters",
            font=customtkinter.CTkFont(size=13, weight="bold")
        )
        params_label.grid(row=5, column=0, padx=20, pady=(10, 10), sticky="w")
        
        self.params_frame = customtkinter.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.params_frame.grid(row=6, column=0, padx=20, pady=(0, 15), sticky="ew")
        
        # Separator
        sep2 = customtkinter.CTkFrame(self.sidebar_frame, height=2, fg_color=("gray70", "gray30"))
        sep2.grid(row=7, column=0, padx=20, pady=15, sticky="ew")
        
        # Run button
        self.run_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="▶ Run Comparison",
            command=self.run_experiment,
            width=260,
            height=40,
            font=customtkinter.CTkFont(size=14, weight="bold"),
            fg_color=("#2CC985", "#2FA572"),
            hover_color=("#28B574", "#298F64")
        )
        self.run_button.grid(row=8, column=0, padx=20, pady=10)
        
        # Save button
        self.save_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="💾 Save Figure",
            command=self.save_figure,
            width=260,
            height=35,
            font=customtkinter.CTkFont(size=12),
            state="disabled"
        )
        self.save_button.grid(row=9, column=0, padx=20, pady=(0, 10))
        
        # Status
        self.status_label = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Ready",
            font=customtkinter.CTkFont(size=11),
            text_color=("gray50", "gray50")
        )
        self.status_label.grid(row=11, column=0, padx=20, pady=(10, 20), sticky="s")
        
        # Set default
        self.change_experiment(self.comparison_menu.get())
        
    def _create_results_area(self):
        """Create results display area"""
        self.results_frame = customtkinter.CTkFrame(self.parent, corner_radius=10)
        self.results_frame.grid(row=0, column=1, padx=(10, 0), pady=0, sticky="nsew")
        self.results_frame.grid_rowconfigure(2, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title = customtkinter.CTkLabel(
            self.results_frame,
            text="📈 Comparison Results",
            font=customtkinter.CTkFont(size=16, weight="bold")
        )
        title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        # Metric buttons
        self.metrics_button_frame = customtkinter.CTkFrame(self.results_frame, fg_color="transparent")
        self.metrics_button_frame.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.metrics_button_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        self.convergence_btn = customtkinter.CTkButton(
            self.metrics_button_frame, text="Convergence",
            command=lambda: self.show_metric_view("convergence")
        )
        self.convergence_btn.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        
        self.complexity_btn = customtkinter.CTkButton(
            self.metrics_button_frame, text="Complexity",
            command=lambda: self.show_metric_view("complexity")
        )
        self.complexity_btn.grid(row=0, column=1, padx=5, sticky="ew")
        
        self.robustness_btn = customtkinter.CTkButton(
            self.metrics_button_frame, text="Robustness",
            command=lambda: self.show_metric_view("robustness")
        )
        self.robustness_btn.grid(row=0, column=2, padx=5, sticky="ew")
        
        self.scalability_btn = customtkinter.CTkButton(
            self.metrics_button_frame, text="Scalability",
            command=lambda: self.show_metric_view("scalability")
        )
        self.scalability_btn.grid(row=0, column=3, padx=(5, 0), sticky="ew")
        
        # Content frame
        self.metric_content_frame = customtkinter.CTkFrame(self.results_frame)
        self.metric_content_frame.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.metric_content_frame.grid_rowconfigure(0, weight=1)
        self.metric_content_frame.grid_columnconfigure(0, weight=1)
        
        # Display frame
        self.metric_display = customtkinter.CTkFrame(self.metric_content_frame, fg_color="transparent")
        self.metric_display.grid(row=0, column=0, sticky="nsew")
        self.metric_display.grid_rowconfigure(0, weight=1)
        self.metric_display.grid_columnconfigure(0, weight=1)
        
    def _init_placeholder_text(self):
        """Initialize placeholder text"""
        placeholder = "Run a comparison to see\ndetailed performance metrics."
        self.metric_data = {
            'convergence': placeholder,
            'complexity': placeholder,
            'robustness': placeholder,
            'scalability': placeholder
        }
        self.show_metric_view(self.current_view)
        
    def show_metric_view(self, view_name: str):
        """Display the selected metric view"""
        self.current_view = view_name
        
        # Clear previous content
        for widget in self.metric_display.winfo_children():
            widget.destroy()
        
        # Highlight button
        buttons = {
            "convergence": self.convergence_btn,
            "complexity": self.complexity_btn,
            "robustness": self.robustness_btn,
            "scalability": self.scalability_btn
        }
        for name, btn in buttons.items():
            if name == view_name:
                btn.configure(fg_color=customtkinter.ThemeManager.theme["CTkButton"]["hover_color"])
            else:
                btn.configure(fg_color=customtkinter.ThemeManager.theme["CTkButton"]["fg_color"])
        
        # Get data
        data = self.metric_data.get(view_name)
        
        if isinstance(data, plt.Figure):
            canvas = FigureCanvasTkAgg(data, master=self.metric_display)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        else:
            textbox = customtkinter.CTkTextbox(
                self.metric_display,
                font=customtkinter.CTkFont(size=11, family="Courier"),
                wrap="word"
            )
            textbox.pack(fill="both", expand=True, padx=5, pady=5)
            textbox.insert("0.0", str(data))
            textbox.configure(state="disabled")
            
    def change_experiment(self, experiment: str):
        """Update parameter inputs based on selected experiment"""
        # Clear old widgets
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        
        if "TSP" in experiment:
            # Number of cities
            cities_label = customtkinter.CTkLabel(self.params_frame, text="Number of Cities:", font=customtkinter.CTkFont(size=11))
            cities_label.grid(row=0, column=0, padx=0, pady=(5, 2), sticky="w")
            self.n_cities_entry = customtkinter.CTkEntry(self.params_frame, placeholder_text="20", width=260, height=30)
            self.n_cities_entry.grid(row=1, column=0, padx=0, pady=(0, 8), sticky="ew")
            
            # Max iterations
            iter_label = customtkinter.CTkLabel(self.params_frame, text="Max Iterations:", font=customtkinter.CTkFont(size=11))
            iter_label.grid(row=2, column=0, padx=0, pady=(5, 2), sticky="w")
            self.max_iter_entry = customtkinter.CTkEntry(self.params_frame, placeholder_text="100", width=260, height=30)
            self.max_iter_entry.grid(row=3, column=0, padx=0, pady=(0, 8), sticky="ew")

            # Number of runs
            runs_label = customtkinter.CTkLabel(self.params_frame, text="Number of Runs:", font=customtkinter.CTkFont(size=11))
            runs_label.grid(row=4, column=0, padx=0, pady=(5, 2), sticky="w")
            self.n_runs_entry = customtkinter.CTkEntry(self.params_frame, placeholder_text="5", width=260, height=30)
            self.n_runs_entry.grid(row=5, column=0, padx=0, pady=(0, 0), sticky="ew")
        else:
            # Dimensions
            dim_label = customtkinter.CTkLabel(self.params_frame, text="Dimensions:", font=customtkinter.CTkFont(size=11))
            dim_label.grid(row=0, column=0, padx=0, pady=(5, 2), sticky="w")
            self.dim_entry = customtkinter.CTkEntry(self.params_frame, placeholder_text="10", width=260, height=30)
            self.dim_entry.grid(row=1, column=0, padx=0, pady=(0, 8), sticky="ew")
            
            # Max iterations
            iter_label = customtkinter.CTkLabel(self.params_frame, text="Max Iterations:", font=customtkinter.CTkFont(size=11))
            iter_label.grid(row=2, column=0, padx=0, pady=(5, 2), sticky="w")
            self.max_iter_entry = customtkinter.CTkEntry(self.params_frame, placeholder_text="100", width=260, height=30)
            self.max_iter_entry.grid(row=3, column=0, padx=0, pady=(0, 8), sticky="ew")
            
            # Number of runs
            runs_label = customtkinter.CTkLabel(self.params_frame, text="Number of Runs:", font=customtkinter.CTkFont(size=11))
            runs_label.grid(row=4, column=0, padx=0, pady=(5, 2), sticky="w")
            self.n_runs_entry = customtkinter.CTkEntry(self.params_frame, placeholder_text="5", width=260, height=30)
            self.n_runs_entry.grid(row=5, column=0, padx=0, pady=(0, 0), sticky="ew")
            
    def update_status(self, message, color=("gray50", "gray50")):
        """Update status label"""
        self.status_label.configure(text=message, text_color=color)
        self.sidebar_frame.update()
        
    def run_experiment(self):
        """Run the experiment in a separate thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.run_button.configure(state="disabled", text="⏳ Running...")
        self.update_status("Running comparison...", ("orange", "orange"))
        
        thread = threading.Thread(target=self._run_experiment_thread, daemon=True)
        thread.start()
        
    def _run_experiment_thread(self):
        """Thread worker for running experiments"""
        try:
            experiment = self.comparison_menu.get()
            runner = ComparisonRunner(seed=42)
            
            if "TSP" in experiment:
                cities_str = self.n_cities_entry.get()
                iter_str = self.max_iter_entry.get()
                runs_str = self.n_runs_entry.get()
                n_cities = int(cities_str) if cities_str else 20
                max_iter = int(iter_str) if iter_str else 100
                n_runs = int(runs_str) if runs_str else 5
                
                results = runner.run_tsp_comparison(n_cities, max_iter, n_runs)
            else:
                dim_str = self.dim_entry.get()
                iter_str = self.max_iter_entry.get()
                runs_str = self.n_runs_entry.get()
                dim = int(dim_str) if dim_str else 10
                max_iter = int(iter_str) if iter_str else 100
                n_runs = int(runs_str) if runs_str else 5
                
                # Determine problem and algorithms
                if "PSO vs HC" in experiment:
                    problem = "rastrigin"
                    algos = ["PSO", "HC"]
                elif "ABC vs GA" in experiment:
                    problem = "rastrigin"
                    algos = ["ABC", "GA"]
                elif "FA vs SA" in experiment:
                    problem = "ackley"
                    algos = ["FA", "SA"]
                elif "CS vs SA" in experiment:
                    problem = "ackley"
                    algos = ["CS", "SA"]
                
                results = runner.run_continuous_comparison(
                    problem, dim, max_iter, n_runs, algos
                )
            
            self.parent.after(0, self._update_ui_with_results, results)
            
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            self.parent.after(0, self._show_error, str(e))
        finally:
            self.parent.after(0, self._experiment_complete)
            
    def _update_ui_with_results(self, results):
        """Update UI with results"""
        self.metric_data = results
        self.show_metric_view(self.current_view)
        self.save_button.configure(state="normal")
        
    def _show_error(self, error_msg):
        """Show error message"""
        error_text = f"❌ ERROR\n\n{error_msg}"
        self.metric_data = {k: error_text for k in self.metric_data.keys()}
        self.show_metric_view(self.current_view)
        self.update_status("Error occurred", ("red", "red"))
        
    def _experiment_complete(self):
        """Clean up after experiment"""
        self.is_running = False
        self.run_button.configure(state="normal", text="▶ Run Comparison")
        self.update_status("Comparison complete ✓", ("green", "green"))
        
    def save_figure(self):
        """Save current figure"""
        data = self.metric_data.get(self.current_view)
        
        if isinstance(data, plt.Figure):
            filepath = tkinter.filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("All files", "*.*")
                ],
                title=f"Save {self.current_view.capitalize()} Plot"
            )
            if filepath:
                data.savefig(filepath, dpi=300, bbox_inches='tight')
                self.update_status(f"Saved to {os.path.basename(filepath)} ✓", ("green", "green"))
        else:
            self.update_status("No figure to save", ("orange", "orange"))
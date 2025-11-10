"""gui/visualization_tab.py - Single algorithm visualization tab"""

import customtkinter
import tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
from pathlib import Path

from .visualization_runner import VisualizationRunner

ALGO_SPECIFIC_PARAMS = {
    'PSO': {
        'n_particles': {'label': 'Number of Particles:', 'default': 30, 'type': int, 'min': 5, 'max': 100},
        'w': {'label': 'Inertia Weight (w):', 'default': 0.7298, 'type': float, 'min': 0.1, 'max': 1.5},
        'w_min': {'label': 'Min Inertia Weight:', 'default': 0.4, 'type': float, 'min': 0.1, 'max': 1.0},
        'w_max': {'label': 'Max Inertia Weight:', 'default': 0.9, 'type': float, 'min': 0.1, 'max': 1.5},
        'c1': {'label': 'Cognitive Coeff (c1):', 'default': 1.49618, 'type': float, 'min': 0.0, 'max': 4.0},
        'c2': {'label': 'Social Coeff (c2):', 'default': 1.49618, 'type': float, 'min': 0.0, 'max': 4.0},
        'v_max_ratio': {'label': 'Max Velocity Ratio:', 'default': 0.2, 'type': float, 'min': 0.05, 'max': 1.0},
    },
    'ABC': {
        'n_bees': {'label': 'Number of Bees:', 'default': 30, 'type': int, 'min': 5, 'max': 100},
        'limit': {'label': 'Abandonment Limit:', 'default': None, 'type': int, 'min': 10, 'max': 1000, 'optional': True},
        'modification_rate': {'label': 'Modification Rate:', 'default': 1.0, 'type': float, 'min': 0.1, 'max': 1.0},
    },
    'FA': {
        'n_fireflies': {'label': 'Number of Fireflies:', 'default': 25, 'type': int, 'min': 5, 'max': 100},
        'alpha': {'label': 'Alpha (randomization):', 'default': 0.5, 'type': float, 'min': 0.01, 'max': 2.0},
        'alpha_min': {'label': 'Min Alpha:', 'default': 0.01, 'type': float, 'min': 0.001, 'max': 0.5},
        'beta0': {'label': 'Beta0 (attractiveness):', 'default': 1.0, 'type': float, 'min': 0.1, 'max': 5.0},
        'gamma': {'label': 'Gamma (absorption):', 'default': 1.0, 'type': float, 'min': 0.01, 'max': 10.0},
    },
    'CS': {
        'n_nests': {'label': 'Number of Nests:', 'default': 25, 'type': int, 'min': 5, 'max': 100},
        'pa': {'label': 'Discovery Probability (pa):', 'default': 0.25, 'type': float, 'min': 0.0, 'max': 1.0},
        'beta': {'label': 'Levy Beta:', 'default': 1.5, 'type': float, 'min': 0.5, 'max': 2.5},
        'step_size_factor': {'label': 'Step Size Factor:', 'default': 0.01, 'type': float, 'min': 0.001, 'max': 0.5},
    }
}

PROBLEM_SPECIFIC_PARAMS = {
    'Sphere': {},
    'Rastrigin': {},
    'Rosenbrock': {},
    'Ackley': {},
}

class VisualizationTab:
    def __init__(self, parent):
        self.parent = parent
        self.is_running = False
        self.current_view = "convergence"
        self.metric_data = {}
        self.specific_param_entries = {}
        self.problem_param_entries = {}
        
        # Configure grid
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        
        self._create_sidebar()
        self._create_results_area()
        self._init_placeholder_text()
        self._update_specific_params_widgets()
        
    def _create_sidebar(self):
        """Create sidebar with controls"""
        self.sidebar_frame = customtkinter.CTkFrame(self.parent, width=280, corner_radius=8)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)
        self.sidebar_frame.grid_rowconfigure(8, weight=1)
        
        # Title
        title = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Swarm Algorithm Analysis",
            font=customtkinter.CTkFont(size=15, weight="bold")
        )
        title.grid(row=0, column=0, padx=15, pady=(15, 3), sticky="w")
        
        subtitle = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Visualize individual algorithm behavior",
            font=customtkinter.CTkFont(size=10),
            text_color=("gray60", "gray40")
        )
        subtitle.grid(row=1, column=0, padx=15, pady=(0, 10), sticky="w")
        
        # Separator
        sep1 = customtkinter.CTkFrame(self.sidebar_frame, height=1, fg_color=("gray70", "gray30"))
        sep1.grid(row=2, column=0, padx=15, pady=8, sticky="ew")
        
        # Algorithm selection
        algo_label = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Select Algorithm",
            font=customtkinter.CTkFont(size=12, weight="bold")
        )
        algo_label.grid(row=3, column=0, padx=15, pady=(5, 3), sticky="w")
        
        self.algo_menu = customtkinter.CTkOptionMenu(
            self.sidebar_frame,
            values=["PSO", "ABC", "FA", "CS"],
            width=250,
            height=32,
            font=customtkinter.CTkFont(size=11),
            command=self._update_specific_params_widgets
        )
        self.algo_menu.grid(row=4, column=0, padx=15, pady=(0, 10))
        
        # Problem selection
        problem_label = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Select Problem",
            font=customtkinter.CTkFont(size=12, weight="bold")
        )
        problem_label.grid(row=5, column=0, padx=15, pady=(5, 3), sticky="w")
        
        self.problem_menu = customtkinter.CTkOptionMenu(
            self.sidebar_frame,
            values=["Sphere", "Rastrigin", "Rosenbrock", "Ackley"],
            width=250,
            height=32,
            font=customtkinter.CTkFont(size=11),
            command=self._update_problem_params_widgets
        )
        self.problem_menu.grid(row=6, column=0, padx=15, pady=(0, 10))

        # Create Scrollable Frame for all parameters
        self.scrollable_params_frame = customtkinter.CTkScrollableFrame(
            self.sidebar_frame, 
            label_text="Parameters",
            label_font=customtkinter.CTkFont(size=12, weight="bold"),
            height=200
        )
        self.scrollable_params_frame.grid(row=7, column=0, padx=15, pady=8, sticky="nsew")

        # General Parameters Label
        params_label = customtkinter.CTkLabel(
            self.scrollable_params_frame, 
            text="General", 
            font=customtkinter.CTkFont(size=11, weight="bold")
        )
        params_label.pack(fill="x", padx=8, pady=(0, 3))

        # General Parameters Frame
        self.params_frame = customtkinter.CTkFrame(self.scrollable_params_frame, fg_color="transparent")
        self.params_frame.pack(fill="x", expand=True, padx=0, pady=(0, 10))
        
        # Dimension
        dim_label = customtkinter.CTkLabel(self.params_frame, text="Dimensions:", font=customtkinter.CTkFont(size=10))
        dim_label.grid(row=0, column=0, padx=0, pady=(3, 1), sticky="w")
        self.dim_entry = customtkinter.CTkEntry(self.params_frame, placeholder_text="10", width=230, height=28)
        self.dim_entry.grid(row=1, column=0, padx=0, pady=(0, 6), sticky="ew")
        
        # Max iterations
        iter_label = customtkinter.CTkLabel(self.params_frame, text="Max Iterations:", font=customtkinter.CTkFont(size=10))
        iter_label.grid(row=2, column=0, padx=0, pady=(3, 1), sticky="w")
        self.iter_entry = customtkinter.CTkEntry(self.params_frame, placeholder_text="100", width=230, height=28)
        self.iter_entry.grid(row=3, column=0, padx=0, pady=(0, 6), sticky="ew")
        
        # Number of runs
        runs_label = customtkinter.CTkLabel(self.params_frame, text="Number of Runs:", font=customtkinter.CTkFont(size=10))
        runs_label.grid(row=4, column=0, padx=0, pady=(3, 1), sticky="w")
        self.runs_entry = customtkinter.CTkEntry(self.params_frame, placeholder_text="10", width=230, height=28)
        self.runs_entry.grid(row=5, column=0, padx=0, pady=(0, 0), sticky="ew")

        # Algorithm-specific Parameters Label
        self.specific_params_label = customtkinter.CTkLabel(
            self.scrollable_params_frame, 
            text="Algorithm-Specific", 
            font=customtkinter.CTkFont(size=11, weight="bold")
        )
        self.specific_params_label.pack(fill="x", padx=8, pady=(8, 3))

        # Algorithm-specific Parameters Frame
        self.specific_params_frame = customtkinter.CTkFrame(self.scrollable_params_frame, fg_color="transparent")
        self.specific_params_frame.pack(fill="x", expand=True, padx=0, pady=0)

        # Problem-specific Parameters Label
        self.problem_params_label = customtkinter.CTkLabel(
            self.scrollable_params_frame, 
            text="Problem-Specific", 
            font=customtkinter.CTkFont(size=11, weight="bold")
        )
        
        # Problem-specific Parameters Frame
        self.problem_params_frame = customtkinter.CTkFrame(self.scrollable_params_frame, fg_color="transparent")

        # Separator
        sep2 = customtkinter.CTkFrame(self.sidebar_frame, height=1, fg_color=("gray70", "gray30"))
        sep2.grid(row=8, column=0, padx=15, pady=10, sticky="ew")
        
        # Run button
        self.run_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="▶ Run Analysis",
            command=self.run_experiment,
            width=250,
            height=36,
            font=customtkinter.CTkFont(size=13, weight="bold"),
            fg_color=("#2CC985", "#2FA572"),
            hover_color=("#28B574", "#298F64")
        )
        self.run_button.grid(row=9, column=0, padx=15, pady=8)
        
        # Save button
        self.save_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="💾 Save Figure",
            command=self.save_figure,
            width=250,
            height=32,
            font=customtkinter.CTkFont(size=11),
            state="disabled"
        )
        self.save_button.grid(row=10, column=0, padx=15, pady=(0, 8))
        
        # Status
        self.status_label = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Ready",
            font=customtkinter.CTkFont(size=10),
            text_color=("gray50", "gray50")
        )
        self.status_label.grid(row=11, column=0, padx=15, pady=(8, 15), sticky="s")
        
    def _update_specific_params_widgets(self, selected_algo=None):
        """Dynamically create widgets for algorithm-specific parameters."""
        if selected_algo is None:
            selected_algo = self.algo_menu.get()

        # Clear existing widgets
        for widget in self.specific_params_frame.winfo_children():
            widget.destroy()
        self.specific_param_entries = {}

        params = ALGO_SPECIFIC_PARAMS.get(selected_algo, {})
        
        if not params:
            self.specific_params_label.pack_forget()
            self.specific_params_frame.pack_forget()
            return
        
        self.specific_params_label.pack(fill="x", padx=8, pady=(8, 3))
        self.specific_params_frame.pack(fill="x", expand=True, padx=0, pady=(0, 10))
        
        for name, config in params.items():
            param_frame = customtkinter.CTkFrame(self.specific_params_frame, fg_color="transparent")
            param_frame.pack(fill="x", expand=True, padx=8, pady=(0, 6))
            
            label_text = config['label']
            if config.get('optional'):
                label_text += " (optional)"
            
            label = customtkinter.CTkLabel(param_frame, text=label_text, font=customtkinter.CTkFont(size=10))
            label.pack(anchor="w")
            
            placeholder = str(config['default']) if config['default'] is not None else "auto"
            entry = customtkinter.CTkEntry(param_frame, placeholder_text=placeholder, height=28)
            entry.pack(fill="x", expand=True)
            
            self.specific_param_entries[name] = entry

    def _update_problem_params_widgets(self, selected_problem=None):
        """Dynamically create widgets for problem-specific parameters."""
        if selected_problem is None:
            selected_problem = self.problem_menu.get()

        # Clear existing widgets
        for widget in self.problem_params_frame.winfo_children():
            widget.destroy()
        self.problem_param_entries = {}

        params = PROBLEM_SPECIFIC_PARAMS.get(selected_problem, {})
        
        if not params:
            self.problem_params_label.pack_forget()
            self.problem_params_frame.pack_forget()
            return
        
        self.problem_params_label.pack(fill="x", padx=8, pady=(8, 3))
        self.problem_params_frame.pack(fill="x", expand=True, padx=0, pady=0)
        
        for name, config in params.items():
            param_frame = customtkinter.CTkFrame(self.problem_params_frame, fg_color="transparent")
            param_frame.pack(fill="x", expand=True, padx=8, pady=(0, 6))
            
            label = customtkinter.CTkLabel(param_frame, text=config['label'], font=customtkinter.CTkFont(size=10))
            label.pack(anchor="w")
            
            entry = customtkinter.CTkEntry(param_frame, placeholder_text=str(config['default']), height=28)
            entry.pack(fill="x", expand=True)
            
            self.problem_param_entries[name] = entry

    def _create_results_area(self):
        """Create results display area"""
        self.results_frame = customtkinter.CTkFrame(self.parent, corner_radius=8)
        self.results_frame.grid(row=0, column=1, padx=(8, 0), pady=0, sticky="nsew")
        self.results_frame.grid_rowconfigure(2, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title = customtkinter.CTkLabel(
            self.results_frame,
            text="📊 Visualization Results",
            font=customtkinter.CTkFont(size=15, weight="bold")
        )
        title.grid(row=0, column=0, padx=15, pady=(15, 8), sticky="w")
        
        # Metric buttons
        self.metrics_button_frame = customtkinter.CTkFrame(self.results_frame, fg_color="transparent")
        self.metrics_button_frame.grid(row=1, column=0, padx=15, pady=(0, 8), sticky="ew")
        self.metrics_button_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        self.convergence_btn = customtkinter.CTkButton(
            self.metrics_button_frame, text="Convergence", 
            command=lambda: self.show_metric_view("convergence"),
            height=32,
            font=customtkinter.CTkFont(size=11)
        )
        self.convergence_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        
        self.performance_btn = customtkinter.CTkButton(
            self.metrics_button_frame, text="Performance",
            command=lambda: self.show_metric_view("performance"),
            height=32,
            font=customtkinter.CTkFont(size=11)
        )
        self.performance_btn.grid(row=0, column=1, padx=4, sticky="ew")
        
        self.sensitivity_btn = customtkinter.CTkButton(
            self.metrics_button_frame, text="Sensitivity",
            command=lambda: self.show_metric_view("sensitivity"),
            height=32,
            font=customtkinter.CTkFont(size=11)
        )
        self.sensitivity_btn.grid(row=0, column=2, padx=4, sticky="ew")
        
        self.landscape_btn = customtkinter.CTkButton(
            self.metrics_button_frame, text="3D Landscape",
            command=lambda: self.show_metric_view("landscape"),
            height=32,
            font=customtkinter.CTkFont(size=11)
        )
        self.landscape_btn.grid(row=0, column=3, padx=(4, 0), sticky="ew")
        
        # Content frame
        self.metric_content_frame = customtkinter.CTkFrame(self.results_frame)
        self.metric_content_frame.grid(row=2, column=0, padx=15, pady=(0, 15), sticky="nsew")
        self.metric_content_frame.grid_rowconfigure(0, weight=1)
        self.metric_content_frame.grid_columnconfigure(0, weight=1)
        
        # Display frame
        self.metric_display = customtkinter.CTkFrame(self.metric_content_frame, fg_color="transparent")
        self.metric_display.grid(row=0, column=0, sticky="nsew")
        self.metric_display.grid_rowconfigure(0, weight=1)
        self.metric_display.grid_columnconfigure(0, weight=1)
        
    def _init_placeholder_text(self):
        """Initialize placeholder text"""
        placeholder = "Run an analysis to see\ndetailed visualizations."
        self.metric_data = {
            'convergence': placeholder,
            'performance': placeholder,
            'sensitivity': placeholder,
            'landscape': placeholder
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
            "performance": self.performance_btn,
            "sensitivity": self.sensitivity_btn,
            "landscape": self.landscape_btn
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
                font=customtkinter.CTkFont(size=10, family="Courier"),
                wrap="word"
            )
            textbox.pack(fill="both", expand=True, padx=4, pady=4)
            textbox.insert("0.0", str(data))
            textbox.configure(state="disabled")
            
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
        self.update_status("Running analysis...", ("orange", "orange"))
        
        thread = threading.Thread(target=self._run_experiment_thread, daemon=True)
        thread.start()
        
    def _run_experiment_thread(self):
        """Thread worker for running experiments"""
        try:
            # Get parameters
            algorithm = self.algo_menu.get()
            problem = self.problem_menu.get().lower()
            dim_str = self.dim_entry.get()
            iter_str = self.iter_entry.get()
            runs_str = self.runs_entry.get()
            
            dim = int(dim_str) if dim_str else 10
            max_iter = int(iter_str) if iter_str else 100
            n_runs = int(runs_str) if runs_str else 10

            # Get algorithm-specific parameters
            algo_specific_params = {}
            for name, entry in self.specific_param_entries.items():
                value_str = entry.get()
                default_config = ALGO_SPECIFIC_PARAMS[algorithm][name]
                if value_str:
                    try:
                        algo_specific_params[name] = default_config['type'](value_str)
                    except ValueError:
                        pass  # Skip invalid values, use defaults
                else:
                    if default_config['default'] is not None:
                        algo_specific_params[name] = default_config['default']

            # Get problem-specific parameters
            problem_specific_params = {}
            for name, entry in self.problem_param_entries.items():
                value_str = entry.get()
                default_config = PROBLEM_SPECIFIC_PARAMS[problem.capitalize()][name]
                if value_str:
                    try:
                        problem_specific_params[name] = default_config['type'](value_str)
                    except ValueError:
                        pass
                else:
                    problem_specific_params[name] = default_config['default']

            # Run analysis
            runner = VisualizationRunner(seed=42)
            
            self.parent.after(0, self.update_status, f"Running {algorithm} on {problem}...", ("orange", "orange"))
            results = runner.run_visualization_analysis(
                algorithm, problem, dim, max_iter, n_runs, 
                algo_specific_params, problem_specific_params
            )
            
            # Update UI
            self.parent.after(0, self._update_ui_with_results, results, algorithm, problem)
            
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            self.parent.after(0, self._show_error, str(e))
        finally:
            self.parent.after(0, self._experiment_complete)
            
    def _update_ui_with_results(self, results, algorithm, problem):
        """Update UI with results"""
        # Close previous figures to free memory
        for fig in self.metric_data.values():
            if isinstance(fig, plt.Figure):
                plt.close(fig)
                
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
        self.run_button.configure(state="normal", text="▶ Run Analysis")
        self.update_status("Analysis complete ✓", ("green", "green"))
        
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
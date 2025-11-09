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
        self.param_entries = {}
        
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
        self.sidebar_frame.grid_rowconfigure(6, weight=1) # Make param frame expandable
        
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
        params_label.grid(row=5, column=0, padx=20, pady=(10, 5), sticky="w")
        
        self.params_frame = customtkinter.CTkScrollableFrame(self.sidebar_frame, label_text="Experiment Parameters")
        self.params_frame.grid(row=6, column=0, padx=20, pady=0, sticky="nsew")
        self.params_frame.grid_columnconfigure(0, weight=1)

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
        self.status_label.grid(row=10, column=0, padx=20, pady=(10, 20), sticky="s")
        
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

    def _add_param_entry(self, parent, row, key, label_text, placeholder, type_):
        label = customtkinter.CTkLabel(parent, text=label_text, font=customtkinter.CTkFont(size=11))
        label.grid(row=row, column=0, padx=0, pady=(5, 2), sticky="w")
        
        entry = customtkinter.CTkEntry(parent, placeholder_text=str(placeholder), width=260, height=30)
        entry.grid(row=row + 1, column=0, padx=0, pady=(0, 8), sticky="ew")
        
        self.param_entries[key] = (entry, type_)
        return row + 2

    def change_experiment(self, experiment: str):
        """Update parameter inputs based on selected experiment"""
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        self.param_entries = {}
        
        row = 0
        # General parameters
        if "TSP" in experiment:
            row = self._add_param_entry(self.params_frame, row, "n_cities", "Num Cities:", 20, int)
        else:
            row = self._add_param_entry(self.params_frame, row, "dim", "Dimensions:", 10, int)
        
        row = self._add_param_entry(self.params_frame, row, "max_iter", "Max Iterations:", 100, int)
        row = self._add_param_entry(self.params_frame, row, "n_runs", "Num Runs:", 10, int)

        # Separator
        sep = customtkinter.CTkFrame(self.params_frame, height=2, fg_color=("gray70", "gray30"))
        sep.grid(row=row, column=0, padx=0, pady=10, sticky="ew")
        row += 1

        # Algorithm-specific parameters
        if experiment == "ACO vs SA (TSP)":
            row = self._add_param_entry(self.params_frame, row, "ACO_n_ants", "ACO: Num Ants:", 50, int)
            row = self._add_param_entry(self.params_frame, row, "ACO_alpha", "ACO: Alpha:", 1.0, float)
            row = self._add_param_entry(self.params_frame, row, "ACO_beta", "ACO: Beta:", 2.0, float)
            row = self._add_param_entry(self.params_frame, row, "ACO_rho", "ACO: Rho:", 0.5, float)
            row = self._add_param_entry(self.params_frame, row, "SA_initial_temp", "SA: Initial Temp:", 1000, int)
            row = self._add_param_entry(self.params_frame, row, "SA_alpha", "SA: Cooling Rate (alpha):", 0.995, float)
        elif experiment == "PSO vs HC (Rastrigin)":
            row = self._add_param_entry(self.params_frame, row, "PSO_n_particles", "PSO: Num Particles:", 30, int)
            row = self._add_param_entry(self.params_frame, row, "PSO_w", "PSO: Inertia (w):", 0.5, float)
            row = self._add_param_entry(self.params_frame, row, "PSO_c1", "PSO: Cognitive (c1):", 1.5, float)
            row = self._add_param_entry(self.params_frame, row, "PSO_c2", "PSO: Social (c2):", 1.5, float)
            row = self._add_param_entry(self.params_frame, row, "HC_step_size", "HC: Step Size:", 0.1, float)
            row = self._add_param_entry(self.params_frame, row, "HC_random_restart", "HC: Random Restarts:", 5, int)
        elif experiment == "ABC vs GA (Rastrigin)":
            row = self._add_param_entry(self.params_frame, row, "ABC_n_bees", "ABC: Num Bees:", 30, int)
            row = self._add_param_entry(self.params_frame, row, "ABC_limit", "ABC: Limit:", 10, int)
            row = self._add_param_entry(self.params_frame, row, "GA_pop_size", "GA: Population Size:", 50, int)
            row = self._add_param_entry(self.params_frame, row, "GA_mutation_rate", "GA: Mutation Rate:", 0.01, float)
            row = self._add_param_entry(self.params_frame, row, "GA_crossover_rate", "GA: Crossover Rate:", 0.8, float)
        elif experiment == "FA vs SA (Ackley)":
            row = self._add_param_entry(self.params_frame, row, "FA_n_fireflies", "FA: Num Fireflies:", 25, int)
            row = self._add_param_entry(self.params_frame, row, "FA_alpha", "FA: Alpha:", 0.5, float)
            row = self._add_param_entry(self.params_frame, row, "FA_gamma", "FA: Gamma:", 0.97, float)
            row = self._add_param_entry(self.params_frame, row, "SA_initial_temp", "SA: Initial Temp:", 100, int)
            row = self._add_param_entry(self.params_frame, row, "SA_alpha", "SA: Cooling Rate (alpha):", 0.99, float)
        elif experiment == "CS vs SA (Ackley)":
            row = self._add_param_entry(self.params_frame, row, "CS_n_nests", "CS: Num Nests:", 25, int)
            row = self._add_param_entry(self.params_frame, row, "CS_pa", "CS: Abandon Prob (pa):", 0.25, float)
            row = self._add_param_entry(self.params_frame, row, "SA_initial_temp", "SA: Initial Temp:", 100, int)
            row = self._add_param_entry(self.params_frame, row, "SA_alpha", "SA: Cooling Rate (alpha):", 0.99, float)

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

            # Gather all parameters from the UI
            params = {}
            for key, (widget, type_) in self.param_entries.items():
                value_str = widget.get()
                if not value_str:
                    value_str = widget.cget("placeholder_text")
                params[key] = type_(value_str)

            # Separate general params from algo-specific params
            general_params = {k: v for k, v in params.items() if "_" not in k}
            algo_params = {}
            for key, value in params.items():
                if "_" in key:
                    algo_name, param_name = key.split("_", 1)
                    if algo_name not in algo_params:
                        algo_params[algo_name] = {}
                    algo_params[algo_name][param_name] = value

            if "TSP" in experiment:
                results = runner.run_tsp_comparison(
                    n_cities=general_params.get("n_cities", 20),
                    max_iter=general_params.get("max_iter", 100),
                    n_runs=general_params.get("n_runs", 10),
                    algo_params=algo_params
                )
            else:
                # Determine problem and algorithms from experiment name
                problem = "rastrigin" if "Rastrigin" in experiment else "ackley"
                algos = []
                if "PSO" in experiment: algos.append("PSO")
                if "HC" in experiment: algos.append("HC")
                if "ABC" in experiment: algos.append("ABC")
                if "GA" in experiment: algos.append("GA")
                if "FA" in experiment: algos.append("FA")
                if "CS" in experiment: algos.append("CS")
                if "SA" in experiment: algos.append("SA")

                results = runner.run_continuous_comparison(
                    problem=problem,
                    dim=general_params.get("dim", 10),
                    max_iter=general_params.get("max_iter", 100),
                    n_runs=general_params.get("n_runs", 10),
                    algos=algos,
                    algo_params=algo_params
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
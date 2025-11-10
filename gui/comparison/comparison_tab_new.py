"""gui/comparison_tab.py - Algorithm comparison tab"""

import customtkinter
import tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os

from .comparison_runner import ComparisonRunner


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
        self.sidebar_frame = customtkinter.CTkFrame(self.parent, width=280, corner_radius=8)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)
        self.sidebar_frame.grid_rowconfigure(10, weight=1)
        
        # Title
        title = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Algorithm Comparison",
            font=customtkinter.CTkFont(size=15, weight="bold")
        )
        title.grid(row=0, column=0, padx=15, pady=(15, 3), sticky="w")
        
        subtitle = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Compare algorithm pairs on benchmarks",
            font=customtkinter.CTkFont(size=10),
            text_color=("gray60", "gray40")
        )
        subtitle.grid(row=1, column=0, padx=15, pady=(0, 10), sticky="w")
        
        # Separator
        sep1 = customtkinter.CTkFrame(self.sidebar_frame, height=1, fg_color=("gray70", "gray30"))
        sep1.grid(row=2, column=0, padx=15, pady=8, sticky="ew")
        
        # Comparison selection
        comp_label = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Select Comparison",
            font=customtkinter.CTkFont(size=12, weight="bold")
        )
        comp_label.grid(row=3, column=0, padx=15, pady=(5, 3), sticky="w")
        
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
            width=250,
            height=32,
            font=customtkinter.CTkFont(size=11)
        )
        self.comparison_menu.grid(row=4, column=0, padx=15, pady=(0, 10))
        
        # Scrollable parameters frame
        params_label = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Parameters",
            font=customtkinter.CTkFont(size=12, weight="bold")
        )
        params_label.grid(row=5, column=0, padx=15, pady=(5, 8), sticky="w")
        
        self.params_scroll = customtkinter.CTkScrollableFrame(
            self.sidebar_frame, 
            height=300,
            fg_color="transparent"
        )
        self.params_scroll.grid(row=6, column=0, padx=15, pady=(0, 10), sticky="ew")
        
        # Separator
        sep2 = customtkinter.CTkFrame(self.sidebar_frame, height=1, fg_color=("gray70", "gray30"))
        sep2.grid(row=7, column=0, padx=15, pady=10, sticky="ew")
        
        # Run button
        self.run_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="▶ Run Comparison",
            command=self.run_experiment,
            width=250,
            height=36,
            font=customtkinter.CTkFont(size=13, weight="bold"),
            fg_color=("#2CC985", "#2FA572"),
            hover_color=("#28B574", "#298F64")
        )
        self.run_button.grid(row=8, column=0, padx=15, pady=8)
        
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
        self.save_button.grid(row=9, column=0, padx=15, pady=(0, 8))
        
        # Status
        self.status_label = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Ready",
            font=customtkinter.CTkFont(size=10),
            text_color=("gray50", "gray50")
        )
        self.status_label.grid(row=11, column=0, padx=15, pady=(8, 15), sticky="s")
        
        # Set default
        self.change_experiment(self.comparison_menu.get())
        
    def _create_results_area(self):
        """Create results display area"""
        self.results_frame = customtkinter.CTkFrame(self.parent, corner_radius=8)
        self.results_frame.grid(row=0, column=1, padx=(8, 0), pady=0, sticky="nsew")
        self.results_frame.grid_rowconfigure(2, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title = customtkinter.CTkLabel(
            self.results_frame,
            text="📈 Comparison Results",
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
        
        self.complexity_btn = customtkinter.CTkButton(
            self.metrics_button_frame, text="Complexity",
            command=lambda: self.show_metric_view("complexity"),
            height=32,
            font=customtkinter.CTkFont(size=11)
        )
        self.complexity_btn.grid(row=0, column=1, padx=4, sticky="ew")
        
        self.robustness_btn = customtkinter.CTkButton(
            self.metrics_button_frame, text="Robustness",
            command=lambda: self.show_metric_view("robustness"),
            height=32,
            font=customtkinter.CTkFont(size=11)
        )
        self.robustness_btn.grid(row=0, column=2, padx=4, sticky="ew")
        
        self.scalability_btn = customtkinter.CTkButton(
            self.metrics_button_frame, text="Scalability",
            command=lambda: self.show_metric_view("scalability"),
            height=32,
            font=customtkinter.CTkFont(size=11)
        )
        self.scalability_btn.grid(row=0, column=3, padx=(4, 0), sticky="ew")
        
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
                font=customtkinter.CTkFont(size=10, family="Courier"),
                wrap="word"
            )
            textbox.pack(fill="both", expand=True, padx=4, pady=4)
            textbox.insert("0.0", str(data))
            textbox.configure(state="disabled")
            
    def change_experiment(self, experiment: str):
        """Update parameter inputs based on selected experiment"""
        # Clear old widgets
        for widget in self.params_scroll.winfo_children():
            widget.destroy()
        
        self.param_entries = {}
        row_idx = 0
        
        if "TSP" in experiment:
            # TSP Common parameters
            self._add_param_entry("n_cities", "Number of Cities", "20", row_idx)
            row_idx += 2
            self._add_param_entry("max_iter", "Max Iterations", "100", row_idx)
            row_idx += 2
            self._add_param_entry("n_runs", "Number of Runs", "1", row_idx)
            row_idx += 2
            
            # Add separator
            sep = customtkinter.CTkFrame(self.params_scroll, height=2, fg_color=("gray70", "gray30"))
            sep.grid(row=row_idx, column=0, columnspan=2, padx=0, pady=10, sticky="ew")
            row_idx += 1
            
            # ACO parameters
            aco_label = customtkinter.CTkLabel(
                self.params_scroll, 
                text="ACO Parameters",
                font=customtkinter.CTkFont(size=11, weight="bold")
            )
            aco_label.grid(row=row_idx, column=0, columnspan=2, padx=0, pady=(5, 8), sticky="w")
            row_idx += 1
            
            self._add_param_entry("ACO_n_ants", "Number of Ants", "20", row_idx)
            row_idx += 2
            self._add_param_entry("ACO_alpha", "Alpha (pheromone)", "1.0", row_idx)
            row_idx += 2
            self._add_param_entry("ACO_beta", "Beta (heuristic)", "2.0", row_idx)
            row_idx += 2
            self._add_param_entry("ACO_rho", "Rho (evaporation)", "0.1", row_idx)
            row_idx += 2
            self._add_param_entry("ACO_phi", "Phi (local update)", "0.1", row_idx)
            row_idx += 2
            self._add_param_entry("ACO_q0", "Q0 (exploitation)", "0.9", row_idx)
            row_idx += 2
            
            # Add separator
            sep = customtkinter.CTkFrame(self.params_scroll, height=2, fg_color=("gray70", "gray30"))
            sep.grid(row=row_idx, column=0, columnspan=2, padx=0, pady=10, sticky="ew")
            row_idx += 1
            
            # SA parameters
            sa_label = customtkinter.CTkLabel(
                self.params_scroll, 
                text="SA Parameters",
                font=customtkinter.CTkFont(size=11, weight="bold")
            )
            sa_label.grid(row=row_idx, column=0, columnspan=2, padx=0, pady=(5, 8), sticky="w")
            row_idx += 1
            
            self._add_param_entry("SA_initial_temp", "Initial Temperature", "1000", row_idx)
            row_idx += 2
            self._add_param_entry("SA_final_temp", "Final Temperature", "0.001", row_idx)
            row_idx += 2
            self._add_param_entry("SA_alpha", "Alpha (cooling)", "0.995", row_idx)
            row_idx += 2
            self._add_param_entry("SA_cooling_schedule", "Cooling Schedule", "exponential", row_idx)
            row_idx += 2
            self._add_param_entry("SA_patience", "Patience", "2000", row_idx)
            row_idx += 2
            
        elif "PSO vs HC" in experiment:
            # Common parameters
            self._add_param_entry("dim", "Dimensions", "10", row_idx)
            row_idx += 2
            self._add_param_entry("max_iter", "Max Iterations", "100", row_idx)
            row_idx += 2
            self._add_param_entry("n_runs", "Number of Runs", "5", row_idx)
            row_idx += 2
            
            # Add separator
            sep = customtkinter.CTkFrame(self.params_scroll, height=2, fg_color=("gray70", "gray30"))
            sep.grid(row=row_idx, column=0, columnspan=2, padx=0, pady=10, sticky="ew")
            row_idx += 1
            
            # PSO parameters
            pso_label = customtkinter.CTkLabel(
                self.params_scroll, 
                text="PSO Parameters",
                font=customtkinter.CTkFont(size=11, weight="bold")
            )
            pso_label.grid(row=row_idx, column=0, columnspan=2, padx=0, pady=(5, 8), sticky="w")
            row_idx += 1
            
            self._add_param_entry("PSO_n_particles", "Number of Particles", "30", row_idx)
            row_idx += 2
            self._add_param_entry("PSO_w", "Inertia Weight", "0.7298", row_idx)
            row_idx += 2
            self._add_param_entry("PSO_c1", "Cognitive (c1)", "1.49618", row_idx)
            row_idx += 2
            self._add_param_entry("PSO_c2", "Social (c2)", "1.49618", row_idx)
            row_idx += 2
            self._add_param_entry("PSO_w_min", "Min Inertia", "0.4", row_idx)
            row_idx += 2
            self._add_param_entry("PSO_w_max", "Max Inertia", "0.9", row_idx)
            row_idx += 2
            self._add_param_entry("PSO_v_max_ratio", "Max Velocity Ratio", "0.2", row_idx)
            row_idx += 2
            
            # Add separator
            sep = customtkinter.CTkFrame(self.params_scroll, height=2, fg_color=("gray70", "gray30"))
            sep.grid(row=row_idx, column=0, columnspan=2, padx=0, pady=10, sticky="ew")
            row_idx += 1
            
            # HC parameters
            hc_label = customtkinter.CTkLabel(
                self.params_scroll, 
                text="Hill Climbing Parameters",
                font=customtkinter.CTkFont(size=11, weight="bold")
            )
            hc_label.grid(row=row_idx, column=0, columnspan=2, padx=0, pady=(5, 8), sticky="w")
            row_idx += 1
            
            self._add_param_entry("HC_step_size", "Step Size", "0.1", row_idx)
            row_idx += 2
            self._add_param_entry("HC_random_restart", "Random Restarts", "5", row_idx)
            row_idx += 2
            
        elif "ABC vs GA" in experiment:
            # Common parameters
            self._add_param_entry("dim", "Dimensions", "10", row_idx)
            row_idx += 2
            self._add_param_entry("max_iter", "Max Iterations", "100", row_idx)
            row_idx += 2
            self._add_param_entry("n_runs", "Number of Runs", "5", row_idx)
            row_idx += 2
            
            # Add separator
            sep = customtkinter.CTkFrame(self.params_scroll, height=2, fg_color=("gray70", "gray30"))
            sep.grid(row=row_idx, column=0, columnspan=2, padx=0, pady=10, sticky="ew")
            row_idx += 1
            
            # ABC parameters
            abc_label = customtkinter.CTkLabel(
                self.params_scroll, 
                text="ABC Parameters",
                font=customtkinter.CTkFont(size=11, weight="bold")
            )
            abc_label.grid(row=row_idx, column=0, columnspan=2, padx=0, pady=(5, 8), sticky="w")
            row_idx += 1
            
            self._add_param_entry("ABC_n_bees", "Number of Bees", "30", row_idx)
            row_idx += 2
            self._add_param_entry("ABC_limit", "Abandonment Limit", "auto", row_idx)
            row_idx += 2
            self._add_param_entry("ABC_modification_rate", "Modification Rate", "1.0", row_idx)
            row_idx += 2
            
            # Add separator
            sep = customtkinter.CTkFrame(self.params_scroll, height=2, fg_color=("gray70", "gray30"))
            sep.grid(row=row_idx, column=0, columnspan=2, padx=0, pady=10, sticky="ew")
            row_idx += 1
            
            # GA parameters
            ga_label = customtkinter.CTkLabel(
                self.params_scroll, 
                text="Genetic Algorithm Parameters",
                font=customtkinter.CTkFont(size=11, weight="bold")
            )
            ga_label.grid(row=row_idx, column=0, columnspan=2, padx=0, pady=(5, 8), sticky="w")
            row_idx += 1
            
            self._add_param_entry("GA_pop_size", "Population Size", "50", row_idx)
            row_idx += 2
            self._add_param_entry("GA_crossover_rate", "Crossover Rate", "0.8", row_idx)
            row_idx += 2
            self._add_param_entry("GA_mutation_rate", "Mutation Rate", "0.1", row_idx)
            row_idx += 2
            self._add_param_entry("GA_tournament_size", "Tournament Size", "3", row_idx)
            row_idx += 2
            self._add_param_entry("GA_elitism_ratio", "Elitism Ratio", "0.1", row_idx)
            row_idx += 2
            
        elif "FA vs SA" in experiment:
            # Common parameters
            self._add_param_entry("dim", "Dimensions", "10", row_idx)
            row_idx += 2
            self._add_param_entry("max_iter", "Max Iterations", "100", row_idx)
            row_idx += 2
            self._add_param_entry("n_runs", "Number of Runs", "5", row_idx)
            row_idx += 2
            
            # Add separator
            sep = customtkinter.CTkFrame(self.params_scroll, height=2, fg_color=("gray70", "gray30"))
            sep.grid(row=row_idx, column=0, columnspan=2, padx=0, pady=10, sticky="ew")
            row_idx += 1
            
            # FA parameters
            fa_label = customtkinter.CTkLabel(
                self.params_scroll, 
                text="Firefly Algorithm Parameters",
                font=customtkinter.CTkFont(size=11, weight="bold")
            )
            fa_label.grid(row=row_idx, column=0, columnspan=2, padx=0, pady=(5, 8), sticky="w")
            row_idx += 1
            
            self._add_param_entry("FA_n_fireflies", "Number of Fireflies", "25", row_idx)
            row_idx += 2
            self._add_param_entry("FA_alpha", "Alpha (randomization)", "0.5", row_idx)
            row_idx += 2
            self._add_param_entry("FA_alpha_min", "Min Alpha", "0.01", row_idx)
            row_idx += 2
            self._add_param_entry("FA_beta0", "Beta0 (attractiveness)", "1.0", row_idx)
            row_idx += 2
            self._add_param_entry("FA_gamma", "Gamma (absorption)", "1.0", row_idx)
            row_idx += 2
            
            # Add separator
            sep = customtkinter.CTkFrame(self.params_scroll, height=2, fg_color=("gray70", "gray30"))
            sep.grid(row=row_idx, column=0, columnspan=2, padx=0, pady=10, sticky="ew")
            row_idx += 1
            
            # SA parameters
            sa_label = customtkinter.CTkLabel(
                self.params_scroll, 
                text="SA Parameters",
                font=customtkinter.CTkFont(size=11, weight="bold")
            )
            sa_label.grid(row=row_idx, column=0, columnspan=2, padx=0, pady=(5, 8), sticky="w")
            row_idx += 1
            
            self._add_param_entry("SA_initial_temp", "Initial Temperature", "1000", row_idx)
            row_idx += 2
            self._add_param_entry("SA_final_temp", "Final Temperature", "0.001", row_idx)
            row_idx += 2
            self._add_param_entry("SA_alpha", "Alpha (cooling)", "0.98", row_idx)
            row_idx += 2
            self._add_param_entry("SA_cooling_schedule", "Cooling Schedule", "exponential", row_idx)
            row_idx += 2
            self._add_param_entry("SA_neighbor_std", "Neighbor Std", "0.3", row_idx)
            row_idx += 2
            self._add_param_entry("SA_inner_loops", "Inner Loops", "50", row_idx)
            row_idx += 2
            self._add_param_entry("SA_patience", "Patience", "1500", row_idx)
            row_idx += 2
            
        elif "CS vs SA" in experiment:
            # Common parameters
            self._add_param_entry("dim", "Dimensions", "10", row_idx)
            row_idx += 2
            self._add_param_entry("max_iter", "Max Iterations", "100", row_idx)
            row_idx += 2
            self._add_param_entry("n_runs", "Number of Runs", "5", row_idx)
            row_idx += 2
            
            # Add separator
            sep = customtkinter.CTkFrame(self.params_scroll, height=2, fg_color=("gray70", "gray30"))
            sep.grid(row=row_idx, column=0, columnspan=2, padx=0, pady=10, sticky="ew")
            row_idx += 1
            
            # CS parameters
            cs_label = customtkinter.CTkLabel(
                self.params_scroll, 
                text="Cuckoo Search Parameters",
                font=customtkinter.CTkFont(size=11, weight="bold")
            )
            cs_label.grid(row=row_idx, column=0, columnspan=2, padx=0, pady=(5, 8), sticky="w")
            row_idx += 1
            
            self._add_param_entry("CS_n_nests", "Number of Nests", "25", row_idx)
            row_idx += 2
            self._add_param_entry("CS_pa", "Discovery Probability", "0.25", row_idx)
            row_idx += 2
            self._add_param_entry("CS_beta", "Beta (Lévy)", "1.5", row_idx)
            row_idx += 2
            self._add_param_entry("CS_step_size_factor", "Step Size Factor", "0.01", row_idx)
            row_idx += 2
            
            # Add separator
            sep = customtkinter.CTkFrame(self.params_scroll, height=2, fg_color=("gray70", "gray30"))
            sep.grid(row=row_idx, column=0, columnspan=2, padx=0, pady=10, sticky="ew")
            row_idx += 1
            
            # SA parameters
            sa_label = customtkinter.CTkLabel(
                self.params_scroll, 
                text="SA Parameters",
                font=customtkinter.CTkFont(size=11, weight="bold")
            )
            sa_label.grid(row=row_idx, column=0, columnspan=2, padx=0, pady=(5, 8), sticky="w")
            row_idx += 1
            
            self._add_param_entry("SA_initial_temp", "Initial Temperature", "1000", row_idx)
            row_idx += 2
            self._add_param_entry("SA_final_temp", "Final Temperature", "0.001", row_idx)
            row_idx += 2
            self._add_param_entry("SA_alpha", "Alpha (cooling)", "0.98", row_idx)
            row_idx += 2
            self._add_param_entry("SA_cooling_schedule", "Cooling Schedule", "exponential", row_idx)
            row_idx += 2
            self._add_param_entry("SA_neighbor_std", "Neighbor Std", "0.3", row_idx)
            row_idx += 2
            self._add_param_entry("SA_inner_loops", "Inner Loops", "50", row_idx)
            row_idx += 2
            self._add_param_entry("SA_patience", "Patience", "1500", row_idx)
            row_idx += 2
    
    def _add_param_entry(self, key: str, label: str, default: str, row: int):
        """Add a parameter entry widget"""
        label_widget = customtkinter.CTkLabel(
            self.params_scroll, 
            text=f"{label}:",
            font=customtkinter.CTkFont(size=10)
        )
        label_widget.grid(row=row, column=0, padx=0, pady=(3, 1), sticky="w")
        
        entry = customtkinter.CTkEntry(
            self.params_scroll, 
            placeholder_text=default,
            width=250,
            height=28
        )
        entry.grid(row=row + 1, column=0, padx=0, pady=(0, 6), sticky="ew")
        entry.insert(0, default)
        
        self.param_entries[key] = entry
        
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
                n_cities = int(self.param_entries["n_cities"].get())
                max_iter = int(self.param_entries["max_iter"].get())
                n_runs = int(self.param_entries["n_runs"].get())
                
                # Parse ACO parameters
                aco_params = {
                    'n_ants': int(self.param_entries["ACO_n_ants"].get()),
                    'alpha': float(self.param_entries["ACO_alpha"].get()),
                    'beta': float(self.param_entries["ACO_beta"].get()),
                    'rho': float(self.param_entries["ACO_rho"].get()),
                    'phi': float(self.param_entries["ACO_phi"].get()),
                    'q0': float(self.param_entries["ACO_q0"].get()),
                }
                
                # Parse SA parameters
                sa_params = {
                    'initial_temp': int(self.param_entries["SA_initial_temp"].get()),
                    'final_temp': float(self.param_entries["SA_final_temp"].get()),
                    'alpha': float(self.param_entries["SA_alpha"].get()),
                    'cooling_schedule': self.param_entries["SA_cooling_schedule"].get(),
                    'patience': int(self.param_entries["SA_patience"].get()),
                }
                
                algo_params = {'ACO': aco_params, 'SA': sa_params}
                
                results = runner.run_tsp_comparison(n_cities, max_iter, n_runs, algo_params)
            else:
                dim = int(self.param_entries["dim"].get())
                max_iter = int(self.param_entries["max_iter"].get())
                n_runs = int(self.param_entries["n_runs"].get())
                
                # Determine problem and algorithms with their parameters
                algo_params = {}
                if "PSO vs HC" in experiment:
                    problem = "rastrigin"
                    algos = ["PSO", "HC"]
                    pso_params = {
                        'n_particles': int(self.param_entries['PSO_n_particles'].get()),
                        'w': float(self.param_entries['PSO_w'].get()),
                        'c1': float(self.param_entries['PSO_c1'].get()),
                        'c2': float(self.param_entries['PSO_c2'].get()),
                        'w_min': float(self.param_entries['PSO_w_min'].get()),
                        'w_max': float(self.param_entries['PSO_w_max'].get()),
                        'v_max_ratio': float(self.param_entries['PSO_v_max_ratio'].get()),
                    }
                    hc_params = {
                        'step_size': float(self.param_entries['HC_step_size'].get()),
                        'random_restart': int(self.param_entries['HC_random_restart'].get()),
                    }
                    algo_params = {'PSO': pso_params, 'HC': hc_params}

                elif "ABC vs GA" in experiment:
                    problem = "rastrigin"
                    algos = ["ABC", "GA"]
                    abc_limit = self.param_entries['ABC_limit'].get()
                    abc_params = {
                        'n_bees': int(self.param_entries['ABC_n_bees'].get()),
                        'limit': None if abc_limit == 'auto' else int(abc_limit),
                        'modification_rate': float(self.param_entries['ABC_modification_rate'].get()),
                    }
                    ga_params = {
                        'pop_size': int(self.param_entries['GA_pop_size'].get()),
                        'crossover_rate': float(self.param_entries['GA_crossover_rate'].get()),
                        'mutation_rate': float(self.param_entries['GA_mutation_rate'].get()),
                        'tournament_size': int(self.param_entries['GA_tournament_size'].get()),
                        'elitism_ratio': float(self.param_entries['GA_elitism_ratio'].get()),
                    }
                    algo_params = {'ABC': abc_params, 'GA': ga_params}

                elif "FA vs SA" in experiment:
                    problem = "ackley"
                    algos = ["FA", "SA"]
                    fa_params = {
                        'n_fireflies': int(self.param_entries['FA_n_fireflies'].get()),
                        'alpha': float(self.param_entries['FA_alpha'].get()),
                        'alpha_min': float(self.param_entries['FA_alpha_min'].get()),
                        'beta0': float(self.param_entries['FA_beta0'].get()),
                        'gamma': float(self.param_entries['FA_gamma'].get()),
                    }
                    sa_params = {
                        'initial_temp': int(self.param_entries['SA_initial_temp'].get()),
                        'final_temp': float(self.param_entries['SA_final_temp'].get()),
                        'alpha': float(self.param_entries['SA_alpha'].get()),
                        'cooling_schedule': self.param_entries['SA_cooling_schedule'].get(),
                        'neighbor_std': float(self.param_entries['SA_neighbor_std'].get()),
                        'inner_loops': int(self.param_entries['SA_inner_loops'].get()),
                        'patience': int(self.param_entries['SA_patience'].get()),
                    }
                    algo_params = {'FA': fa_params, 'SA': sa_params}

                elif "CS vs SA" in experiment:
                    problem = "ackley"
                    algos = ["CS", "SA"]
                    cs_params = {
                        'n_nests': int(self.param_entries['CS_n_nests'].get()),
                        'pa': float(self.param_entries['CS_pa'].get()),
                        'beta': float(self.param_entries['CS_beta'].get()),
                        'step_size_factor': float(self.param_entries['CS_step_size_factor'].get()),
                    }
                    sa_params = {
                        'initial_temp': int(self.param_entries['SA_initial_temp'].get()),
                        'final_temp': float(self.param_entries['SA_final_temp'].get()),
                        'alpha': float(self.param_entries['SA_alpha'].get()),
                        'cooling_schedule': self.param_entries['SA_cooling_schedule'].get(),
                        'neighbor_std': float(self.param_entries['SA_neighbor_std'].get()),
                        'inner_loops': int(self.param_entries['SA_inner_loops'].get()),
                        'patience': int(self.param_entries['SA_patience'].get()),
                    }
                    algo_params = {'CS': cs_params, 'SA': sa_params}
                
                results = runner.run_continuous_comparison(
                    problem, dim, max_iter, n_runs, algos, algo_params
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
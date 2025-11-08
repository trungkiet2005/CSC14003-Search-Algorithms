import tkinter
import customtkinter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import sys
import os
import threading
from pathlib import Path
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import algorithms
from algorithms.swarm.PSO import run_pso
from algorithms.swarm.ACO import run_aco
from algorithms.swarm.ABC import run_abc
from algorithms.swarm.FA import run_fa
from algorithms.swarm.CS import run_cs

from algorithms.traditional.simulated_annealing import run_simulated_annealing, run_simulated_annealing_tsp
from algorithms.traditional.genetic_algorithm import run_ga
from algorithms.traditional.hill_climbing import run_hill_climbing

# Import problems
from problems.continuous import get_problem
from problems.tsp import create_tsp_problem

# Import utilities
from utils.benchmark import BenchmarkRunner
from utils.visualize import (
    plot_convergence_comparison, plot_boxplot_comparison,
    plot_tsp_route, plot_complexity_comparison, plot_scalability_comparison
)

customtkinter.set_appearance_mode("white")
customtkinter.set_default_color_theme("blue")


class GuiExperimentRunner:
    """Manage and run experiments for the GUI"""

    def __init__(self, output_dir: str = "results", seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        self.seed = seed
        self.benchmark_runner = BenchmarkRunner(seed=seed, verbose=False)

    def run_continuous_experiment_data(self, problem_name: str, dim: int,
                                     algorithms: dict, n_runs: int = 30,
                                     max_iter: int = 100):
        """Run experiment on continuous optimization problem and return data."""
        problem_func, problem_info = get_problem(problem_name, dim)
        bounds = problem_info['bounds']

        algo_dict = {}
        for algo_name, (algo_func, algo_params) in algorithms.items():
            params = {
                'dim': dim,
                'bounds': bounds,
                'max_iter': max_iter,
                **algo_params
            }
            algo_dict[algo_name] = (algo_func, params)

        df, stats_list = self.benchmark_runner.compare_algorithms(
            algo_dict, problem_func, problem_name, dim, n_runs=n_runs
        )

        metrics = {s.algorithm_name: s.__dict__ for s in stats_list}
        return stats_list, metrics

    def plot_continuous_experiment(self, stats_list, problem_name, dim):
        """Plot continuous experiment results and return separate figures."""
        # 1. Convergence plot
        convergence_fig, ax1 = plt.subplots(figsize=(10, 7))
        histories = {}
        for stats in stats_list:
            if stats.results:
                result = stats.results[0]
                history = result.get('history') if isinstance(result, dict) else getattr(result, 'history', [])
                histories[stats.algorithm_name] = history

        plot_convergence_comparison(
            histories,
            title=f"Convergence: {problem_name.capitalize()} (dim={dim})",
            ax=ax1,
            log_scale=True
        )
        convergence_fig.tight_layout()

        # 2. Boxplot
        boxplot_fig, ax2 = plt.subplots(figsize=(10, 7))
        boxplot_data = {stats.algorithm_name: stats.all_fitnesses for stats in stats_list}
        plot_boxplot_comparison(
            boxplot_data,
            title=f"Robustness: {problem_name.capitalize()} (dim={dim})",
            ax=ax2
        )
        boxplot_fig.tight_layout()

        return {
            "convergence": convergence_fig,
            "robustness": boxplot_fig
        }

    def plot_complexity_experiment(self, stats_list, problem_name, dim):
        """Plot complexity experiment results."""
        fig = plot_complexity_comparison(
            stats_list,
            title=f"Complexity: {problem_name.capitalize()} (dim={dim})"
        )
        return fig

    def run_scalability_experiment(self, problem_name, algorithms, n_runs, max_iter, dims):
        """Run experiment across multiple dimensions for scalability analysis."""
        all_stats = {algo_name: {'dims': [], 'fitness': [], 'times': []} for algo_name in algorithms}

        for dim in dims:
            stats_list, _ = self.run_continuous_experiment_data(
                problem_name, dim, algorithms, n_runs, max_iter
            )
            for stats in stats_list:
                all_stats[stats.algorithm_name]['dims'].append(dim)
                all_stats[stats.algorithm_name]['fitness'].append(stats.best_fitness)
                all_stats[stats.algorithm_name]['times'].append(stats.mean_time)
        
        return all_stats

    def plot_scalability_experiment(self, scalability_data, problem_name):
        """Plot scalability experiment results."""
        fig = plot_scalability_comparison(
            scalability_data,
            title=f"Scalability Analysis: {problem_name.capitalize()}"
        )
        return fig

    def run_tsp_experiment_data(self, n_cities: int, algorithms: dict,
                               n_runs: int = 5, max_iter: int = 100):
        """Run experiment on TSP and return data."""
        tsp = create_tsp_problem(n_cities, seed=self.seed)
        cities = tsp['cities']
        distance_matrix = tsp['distance_matrix']

        results = {}
        for algo_name, (algo_func, algo_params) in algorithms.items():
            fitnesses = []
            times = []
            best_result = None
            best_distance = float('inf')

            for run in range(n_runs):
                
                start = time.time()

                if 'distance_matrix' in algo_func.__code__.co_varnames:
                    result = algo_func(distance_matrix, max_iter=max_iter,
                                     seed=self.seed + run, **algo_params)
                else:
                    result = algo_func(tsp['objective'], max_iter=max_iter,
                                     seed=self.seed + run, **algo_params)

                elapsed = time.time() - start
                distance = result['best_distance']
                fitnesses.append(distance)
                times.append(elapsed)

                if distance < best_distance:
                    best_distance = distance
                    best_result = result

            results[algo_name] = {
                'mean_distance': np.mean(fitnesses),
                'std_distance': np.std(fitnesses),
                'best_distance': np.min(fitnesses),
                'mean_time': np.mean(times),
                'best_result': best_result,
                'all_distances': fitnesses
            }
        
        return cities, results

    def plot_tsp_experiment(self, cities, results):
        """Plot TSP experiment results."""
        fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6))
        if len(results) == 1:
            axes = [axes]
        
        axes = np.ravel(axes)

        for ax, (algo_name, stats) in zip(axes, results.items()):
            result = stats['best_result']
            route = result['best_route']
            distance = result['best_distance']
            plot_tsp_route(
                cities, route, distance,
                title=f"{algo_name}\nDistance: {distance:.2f}",
                ax=ax
            )
        plt.tight_layout()

        return fig
    
    def plot_tsp_complexity(self, results):
        """Plot complexity comparison for TSP."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("TSP Computational Complexity", fontsize=16, fontweight='bold')
        
        algo_names = list(results.keys())
        mean_times = [results[name]['mean_time'] for name in algo_names]
        mean_distances = [results[name]['mean_distance'] for name in algo_names]
        
        colors = sns.color_palette("viridis", len(algo_names))
        
        # 1. Execution Time
        bars1 = ax1.bar(algo_names, mean_times, color=colors, alpha=0.8)
        ax1.set_title("Mean Execution Time", fontsize=14)
        ax1.set_ylabel("Time (seconds)", fontsize=12)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}s', 
                     va='bottom', ha='center', fontsize=10)
        
        # 2. Solution Quality
        bars2 = ax2.bar(algo_names, mean_distances, color=colors, alpha=0.8)
        ax2.set_title("Mean Solution Quality", fontsize=14)
        ax2.set_ylabel("Tour Distance", fontsize=12)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
        
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', 
                     va='bottom', ha='center', fontsize=10)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def plot_tsp_robustness(self, results):
        """Plot robustness (boxplot) for TSP."""
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        algo_names = list(results.keys())
        # Note: We need all_distances stored in results
        data = []
        labels = []
        for name in algo_names:
            if 'all_distances' in results[name]:
                data.append(results[name]['all_distances'])
                labels.append(name)
        
        if not data:
            # If no data, show text
            ax.text(0.5, 0.5, 'Run multiple times\nto see robustness analysis', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            return fig
        
        # Create boxplot
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                       showmeans=True, meanline=True,
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5),
                       medianprops=dict(color='red', linewidth=2),
                       meanprops=dict(color='blue', linewidth=2, linestyle='--'))
        
        colors = sns.color_palette("Set3", len(data))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title("TSP Robustness Analysis - Distance Distribution", 
                    fontsize=16, fontweight='bold')
        ax.set_ylabel("Tour Distance", fontsize=13)
        ax.set_xlabel("Algorithm", fontsize=13)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, label='Median'),
            Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Mean')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=10)
        
        plt.tight_layout()
        return fig

    def plot_tsp_scalability(self, algorithms, max_iter, city_counts):
        """Plot scalability analysis for TSP."""
        
        scalability_data = {}
        for algo_name in algorithms:
            scalability_data[algo_name] = {
                'cities': [],
                'distances': [],
                'times': []
            }
        
        for n_cities in city_counts:
            cities, results = self.run_tsp_experiment_data(
                n_cities, algorithms, n_runs=3, max_iter=max_iter
            )
            for algo_name, result in results.items():
                scalability_data[algo_name]['cities'].append(n_cities)
                scalability_data[algo_name]['distances'].append(result['mean_distance'])
                scalability_data[algo_name]['times'].append(result['mean_time'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle("TSP Scalability Analysis", fontsize=18, fontweight='bold')
        
        colors = sns.color_palette("husl", len(scalability_data))
        
        # 1. Solution Quality vs Problem Size
        for (algo_name, data), color in zip(scalability_data.items(), colors):
            ax1.plot(data['cities'], data['distances'], 
                    marker='o', linestyle='-', color=color, 
                    linewidth=2, markersize=8, label=algo_name)
        ax1.set_title("Solution Quality vs. Problem Size", fontsize=14)
        ax1.set_xlabel("Number of Cities", fontsize=12)
        ax1.set_ylabel("Mean Tour Distance", fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 2. Execution Time vs Problem Size
        for (algo_name, data), color in zip(scalability_data.items(), colors):
            ax2.plot(data['cities'], data['times'], 
                    marker='o', linestyle='-', color=color,
                    linewidth=2, markersize=8, label=algo_name)
        ax2.set_title("Execution Time vs. Problem Size", fontsize=14)
        ax2.set_xlabel("Number of Cities", fontsize=12)
        ax2.set_ylabel("Mean Time (seconds)", fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Search Algorithms Visualization & Analysis")
        self.geometry(f"{1600}x{850}")
        
        # Configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create sidebar frame
        self.sidebar_frame = customtkinter.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar_frame.grid_rowconfigure(10, weight=1)
        
        # Logo/Title
        self.logo_label = customtkinter.CTkLabel(
            self.sidebar_frame, 
            text="🔍 Algorithm Lab", 
            font=customtkinter.CTkFont(size=24, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 10))
        
        # Subtitle
        self.subtitle_label = customtkinter.CTkLabel(
            self.sidebar_frame, 
            text="Compare & Analyze Performance", 
            font=customtkinter.CTkFont(size=12),
            text_color=("gray60", "gray40")
        )
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 20))
        
        # Separator
        self.separator1 = customtkinter.CTkFrame(self.sidebar_frame, height=2, fg_color=("gray70", "gray30"))
        self.separator1.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        # Experiment selector label
        self.experiment_label = customtkinter.CTkLabel(
            self.sidebar_frame, 
            text="Select Experiment",
            font=customtkinter.CTkFont(size=14, weight="bold")
        )
        self.experiment_label.grid(row=3, column=0, padx=20, pady=(10, 5), sticky="w")
        
        # Experiment dropdown
        self.experiment_menu = customtkinter.CTkOptionMenu(
            self.sidebar_frame, 
            values=[
                "ACO vs SA (TSP)", 
                "PSO vs HC (Rastrigin)", 
                "ABC vs GA (Rastrigin)", 
                "FA vs SA (Ackley)", 
                "CS vs SA (Ackley)"
            ],
            command=self.change_experiment,
            width=240,
            height=35,
            font=customtkinter.CTkFont(size=13),
            dropdown_font=customtkinter.CTkFont(size=12)
        )
        self.experiment_menu.grid(row=4, column=0, padx=20, pady=(0, 15))
        
        # Parameters label
        self.params_label = customtkinter.CTkLabel(
            self.sidebar_frame, 
            text="Parameters",
            font=customtkinter.CTkFont(size=14, weight="bold")
        )
        self.params_label.grid(row=5, column=0, padx=20, pady=(10, 10), sticky="w")
        
        # Parameters frame
        self.params_frame = customtkinter.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.params_frame.grid(row=6, column=0, padx=20, pady=(0, 15), sticky="ew")
        
        # Separator
        self.separator2 = customtkinter.CTkFrame(self.sidebar_frame, height=2, fg_color=("gray70", "gray30"))
        self.separator2.grid(row=7, column=0, padx=20, pady=15, sticky="ew")
        
        # Run button
        self.run_button = customtkinter.CTkButton(
            self.sidebar_frame, 
            text="▶ Run Experiment", 
            command=self.run_experiment,
            width=240,
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
            width=240,
            height=35,
            font=customtkinter.CTkFont(size=13),
            state="disabled"
        )
        self.save_button.grid(row=9, column=0, padx=20, pady=(0, 10))
        
        # Status label at bottom
        self.status_label = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="Ready",
            font=customtkinter.CTkFont(size=11),
            text_color=("gray50", "gray50")
        )
        self.status_label.grid(row=11, column=0, padx=20, pady=(10, 20), sticky="s")

        # Main content is now the results frame, no separate visualization area
        self.main_fig = None # To store the main figure for saving
        self.is_running = False

        # Create results frame
        self.results_frame = customtkinter.CTkFrame(self, corner_radius=10)
        self.results_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.results_frame.grid_rowconfigure(2, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)

        self.results_title = customtkinter.CTkLabel(
            self.results_frame,
            text="📈 Performance Analysis",
            font=customtkinter.CTkFont(size=16, weight="bold")
        )
        self.results_title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        # Frame for the metric buttons
        self.metrics_button_frame = customtkinter.CTkFrame(self.results_frame, fg_color="transparent")
        self.metrics_button_frame.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.metrics_button_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Metric buttons
        self.convergence_button = customtkinter.CTkButton(
            self.metrics_button_frame, text="Convergence", command=lambda: self.show_metric_view("convergence")
        )
        self.convergence_button.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        self.complexity_button = customtkinter.CTkButton(
            self.metrics_button_frame, text="Complexity", command=lambda: self.show_metric_view("complexity")
        )
        self.complexity_button.grid(row=0, column=1, padx=5, sticky="ew")

        self.robustness_button = customtkinter.CTkButton(
            self.metrics_button_frame, text="Robustness", command=lambda: self.show_metric_view("robustness")
        )
        self.robustness_button.grid(row=0, column=2, padx=5, sticky="ew")

        self.scalability_button = customtkinter.CTkButton(
            self.metrics_button_frame, text="Scalability", command=lambda: self.show_metric_view("scalability")
        )
        self.scalability_button.grid(row=0, column=3, padx=(5, 0), sticky="ew")

        # Content frame for metrics
        self.metric_content_frame = customtkinter.CTkFrame(self.results_frame)
        self.metric_content_frame.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.metric_content_frame.grid_rowconfigure(0, weight=1)
        self.metric_content_frame.grid_columnconfigure(0, weight=1)

        # This frame will hold the plot or text
        self.metric_display = customtkinter.CTkFrame(self.metric_content_frame, fg_color="transparent")
        self.metric_display.grid(row=0, column=0, sticky="nsew")
        self.metric_display.grid_rowconfigure(0, weight=1)
        self.metric_display.grid_columnconfigure(0, weight=1)

        # Store metric data (will hold text or figure objects)
        self.metric_data = {}
        self.current_metric_view = "convergence"  # default view

        # Initialize with placeholder text
        self._init_placeholder_text()
        self.show_metric_view(self.current_metric_view)

        # Set default values
        self.change_experiment(self.experiment_menu.get())

    def show_metric_view(self, view_name: str):
        """Display the content for the selected metric view."""
        self.current_metric_view = view_name

        # Clear previous content
        for widget in self.metric_display.winfo_children():
            widget.destroy()

        # Highlight button
        buttons = {
            "convergence": self.convergence_button,
            "complexity": self.complexity_button,
            "robustness": self.robustness_button,
            "scalability": self.scalability_button,
        }
        for name, btn in buttons.items():
            if name == view_name:
                btn.configure(fg_color=customtkinter.ThemeManager.theme["CTkButton"]["hover_color"])
            else:
                btn.configure(fg_color=customtkinter.ThemeManager.theme["CTkButton"]["fg_color"])

        # Get the data for the view
        data = self.metric_data.get(view_name)

        if isinstance(data, plt.Figure):
            # If data is a matplotlib figure, display it
            canvas = FigureCanvasTkAgg(data, master=self.metric_display)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        else:
            # Otherwise, display it as text
            textbox = customtkinter.CTkTextbox(
                self.metric_display,
                font=customtkinter.CTkFont(size=11, family="Courier"),
                wrap="word"
            )
            textbox.pack(fill="both", expand=True, padx=5, pady=5)
            textbox.insert("0.0", str(data))
            textbox.configure(state="disabled")

    def _init_placeholder_text(self):
        """Initialize placeholder text for all metric views."""
        placeholder_text = "Run an experiment to see\ndetailed performance metrics."
        self.metric_data = {
            'convergence': placeholder_text,
            'complexity': placeholder_text,
            'robustness': placeholder_text,
            'scalability': placeholder_text,
        }
        self.result_texts = self.metric_data.copy()
        self.show_metric_view(self.current_metric_view)

    def update_status(self, message, color=("gray50", "gray50")):
        """Update status label with message and color."""
        self.status_label.configure(text=message, text_color=color)
        self.update()

    def run_experiment(self):
        """Run the selected experiment in a separate thread."""
        if self.is_running:
            return
            
        self.is_running = True
        self.run_button.configure(state="disabled", text="⏳ Running...")
        self.update_status("Running experiment...", ("orange", "orange"))
        
        # Run in separate thread to keep UI responsive
        thread = threading.Thread(target=self._run_experiment_thread, daemon=True)
        thread.start()

    def _run_experiment_thread(self):
        """Thread worker for running experiments."""
        try:
            experiment = self.experiment_menu.get()
            runner = GuiExperimentRunner(output_dir="results/gui_runs", seed=42)

            if "TSP" in experiment:
                n_cities_str = self.n_cities_entry.get()
                max_iter_str = self.max_iter_entry.get()
                base_cities = int(n_cities_str) if n_cities_str else int(self.n_cities_entry.cget("placeholder_text"))
                max_iter = int(max_iter_str) if max_iter_str else int(self.max_iter_entry.cget("placeholder_text"))
                
                tsp_algos = {
                    'ACO': (run_aco, {}),
                    'SA': (run_simulated_annealing_tsp, {})
                }
                cities, results = runner.run_tsp_experiment_data(base_cities, tsp_algos, n_runs=1, max_iter=max_iter)
                result_texts = self._format_tsp_results(results, base_cities, max_iter)
                
                # Package data for TSP plots
                ui_data = {
                    "cities": cities, 
                    "results": results, 
                    "algorithms": tsp_algos,
                    "max_iter": max_iter,
                    "n_cities": base_cities
                }
                self.after(0, self._update_ui_with_results, "tsp", ui_data, result_texts)
            else: # Continuous problems
                dim_str = self.dim_entry.get()
                max_iter_str = self.max_iter_entry.get()
                n_runs_str = self.n_runs_entry.get()
                base_dim = int(dim_str) if dim_str else int(self.dim_entry.cget("placeholder_text"))
                max_iter = int(max_iter_str) if max_iter_str else int(self.max_iter_entry.cget("placeholder_text"))
                n_runs = int(n_runs_str) if n_runs_str else int(self.n_runs_entry.cget("placeholder_text"))

                problem_name = ""
                algorithms = {}
                if experiment == "PSO vs HC (Rastrigin)":
                    problem_name = "rastrigin"
                    algorithms = {'PSO': (run_pso, {'n_particles': 30}), 'HC': (run_hill_climbing, {'step_size': 0.1, 'random_restart': 5})}
                elif experiment == "ABC vs GA (Rastrigin)":
                    problem_name = "rastrigin"
                    algorithms = {'ABC': (run_abc, {'n_bees': 30}), 'GA': (run_ga, {'pop_size': 50})}
                elif experiment == "FA vs SA (Ackley)":
                    problem_name = "ackley"
                    algorithms = {'FA': (run_fa, {'n_fireflies': 25}), 'SA': (run_simulated_annealing, {})}
                elif experiment == "CS vs SA (Ackley)":
                    problem_name = "ackley"
                    algorithms = {'CS': (run_cs, {'n_nests': 25}), 'SA': (run_simulated_annealing, {})}

                # --- Main experiment for user-specified dimension ---
                self.after(0, self.update_status, f"Running main experiment (dim={base_dim})...", ("orange", "orange"))
                stats_list, metrics = runner.run_continuous_experiment_data(problem_name, base_dim, algorithms, n_runs, max_iter)
                result_texts = self._format_continuous_results(metrics, base_dim, max_iter, n_runs)

                # --- Scalability analysis ---
                self.after(0, self.update_status, "Running scalability analysis...", ("orange", "orange"))
                scalability_dims = sorted(list(set([5, 10, 20, 30, 50])))
                
                scalability_data = runner.run_scalability_experiment(
                    problem_name, algorithms, n_runs, max_iter, scalability_dims
                )
                scalability_fig = runner.plot_scalability_experiment(scalability_data, problem_name)

                # --- Package data for UI update ---
                ui_data = {
                    "stats_list": stats_list,
                    "problem_name": problem_name,
                    "dim": base_dim,
                    "scalability_fig": scalability_fig
                }
                self.after(0, self._update_ui_with_results, "continuous", ui_data, result_texts)

        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            print(traceback_str) # Print full traceback to console
            error_msg = f"Error running experiment:\n{str(e)}\n\nCheck console for details."
            self.after(0, self._show_error, error_msg)
        finally:
            self.after(0, self._experiment_complete)

    def _format_tsp_results(self, metrics, n_cities, max_iter):
        """Format TSP experiment results for display."""
        aco_dist = metrics['ACO']['best_distance']
        sa_dist = metrics['SA']['best_distance']
        
        # Summary tab
        summary = f"{'='*40}\n"
        summary += f"  TSP EXPERIMENT RESULTS\n"
        summary += f"{'='*40}\n\n"
        summary += f"Problem Size: {n_cities} cities\n"
        summary += f"ACO Iterations: {max_iter}\n\n"
        summary += f"{'─'*40}\n"
        summary += f"BEST SOLUTIONS:\n"
        summary += f"{'─'*40}\n"
        summary += f"ACO: {aco_dist:.2f}\n"
        summary += f"SA:  {sa_dist:.2f}\n\n"
        
        improvement = ((sa_dist - aco_dist) / sa_dist * 100)
        if improvement > 0:
            summary += f"✓ ACO is {improvement:.1f}% better"
        else:
            summary += f"✓ SA is {abs(improvement):.1f}% better"
        
        # Convergence tab (simplified for TSP)
        convergence = f"{'='*40}\n"
        convergence += f"  CONVERGENCE ANALYSIS\n"
        convergence += f"{'='*40}\n\n"
        convergence += f"ACO Iterations: {max_iter}\n"
        convergence += f"Final Distance: {aco_dist:.2f}\n\n"
        convergence += f"SA used iterative refinement\n"
        convergence += f"Final Distance: {sa_dist:.2f}\n\n"
        convergence += f"Note: See convergence plot\nfor detailed analysis."
        
        # Robustness tab (single run for TSP)
        robustness = f"{'='*40}\n"
        robustness += f"  ROBUSTNESS ANALYSIS\n"
        robustness += f"{'='*40}\n\n"
        robustness += f"Single run performed.\n\n"
        robustness += f"For robustness analysis,\nrun multiple times with\ndifferent random seeds.\n\n"
        robustness += f"Recommendations:\n"
        robustness += f"• Run 10+ times\n"
        robustness += f"• Analyze std deviation\n"
        robustness += f"• Compare best/worst cases"
        
        # Complexity tab
        complexity = f"{'='*40}\n"
        complexity += f"  COMPUTATIONAL COMPLEXITY\n"
        complexity += f"{'='*40}\n\n"
        complexity += f"TIME COMPLEXITY:\n"
        complexity += f"{'─'*40}\n"
        complexity += f"ACO: O(m·n²·iter)\n"
        complexity += f"  m = ants, n = cities\n"
        complexity += f"  iter = iterations\n\n"
        complexity += f"SA:  O(iter·n)\n"
        complexity += f"  iter = iterations\n\n"
        complexity += f"SPACE COMPLEXITY:\n"
        complexity += f"{'─'*40}\n"
        complexity += f"ACO: O(n²) pheromone matrix\n"
        complexity += f"SA:  O(n) current solution\n\n"
        complexity += f"SCALABILITY:\n"
        complexity += f"{'─'*40}\n"
        complexity += f"Current: {n_cities} cities\n"
        complexity += f"ACO scales well up to\n100-200 cities.\n"
        complexity += f"SA handles larger problems\nbut may need more iterations."
        
        return {
            'summary': summary,
            'convergence': convergence,
            'robustness': robustness,
            'complexity': complexity
        }

    def _format_continuous_results(self, metrics, dim, max_iter, n_runs):
        """Format continuous optimization results for display."""
        # Extract algorithm names
        algo_names = list(metrics.keys())
        
        # Summary tab
        summary = f"{'='*40}\n"
        summary += f"  OPTIMIZATION RESULTS\n"
        summary += f"{'='*40}\n\n"
        summary += f"Dimensions: {dim}\n"
        summary += f"Iterations: {max_iter}\n"
        summary += f"Runs: {n_runs}\n\n"
        
        for name, stats in metrics.items():
            summary += f"{'─'*40}\n"
            summary += f"{name}:\n"
            summary += f"{'─'*40}\n"
            summary += f"Best:   {stats['best_fitness']:.6f}\n"
            summary += f"Mean:   {stats['mean_fitness']:.6f}\n"
            summary += f"Worst:  {stats['worst_fitness']:.6f}\n"
            summary += f"Time:   {stats['mean_time']:.4f}s\n\n"
        
        # Determine winner
        best_algo = min(metrics.items(), key=lambda x: x[1]['mean_fitness'])
        summary += f"✓ {best_algo[0]} achieved best\nmean fitness"
        
        # Convergence tab
        convergence = f"{'='*40}\n"
        convergence += f"  CONVERGENCE SPEED\n"
        convergence += f"{'='*40}\n\n"
        
        for name, stats in metrics.items():
            result = stats['results'][0]
            if isinstance(result, dict):
                history = result.get('history', [])
            else:
                history = getattr(result, 'history', [])
            if history:
                # Calculate convergence metrics
                initial = history[0]
                final = history[-1]
                improvement = initial - final
                
                # Find iteration where 90% improvement achieved
                target = initial - 0.9 * improvement
                conv_iter = None
                for i, val in enumerate(history):
                    if val <= target:
                        conv_iter = i
                        break
                
                convergence += f"{name}:\n"
                convergence += f"{'─'*40}\n"
                convergence += f"Initial:  {initial:.6f}\n"
                convergence += f"Final:    {final:.6f}\n"
                convergence += f"Improved: {improvement:.6f}\n"
                if conv_iter:
                    convergence += f"90% conv: iter {conv_iter}/{len(history)}\n"
                    convergence += f"          ({conv_iter/len(history)*100:.1f}%)\n"
                convergence += f"\n"
        
        convergence += f"See convergence plot for\ndetailed visualization."
        
        # Robustness tab
        robustness = f"{'='*40}\n"
        robustness += f"  ROBUSTNESS ANALYSIS\n"
        robustness += f"{'='*40}\n\n"
        robustness += f"Performance across {n_runs} runs:\n\n"
        
        for name, stats in metrics.items():
            std = stats['std_fitness']
            mean = stats['mean_fitness']
            cv = (std / abs(mean)) * 100 if mean != 0 else 0
            
            robustness += f"{name}:\n"
            robustness += f"{'─'*40}\n"
            robustness += f"Mean:     {mean:.6f}\n"
            robustness += f"Std Dev:  {std:.6f}\n"
            robustness += f"CV:       {cv:.2f}%\n"
            robustness += f"Range:    [{stats['best_fitness']:.6f},\n"
            robustness += f"           {stats['worst_fitness']:.6f}]\n"
            
            # Robustness rating
            if cv < 5:
                rating = "Excellent"
            elif cv < 15:
                rating = "Good"
            elif cv < 30:
                rating = "Moderate"
            else:
                rating = "Variable"
            robustness += f"Rating:   {rating}\n\n"
        
        # Complexity tab
        complexity = f"{'='*40}\n"
        complexity += f"  COMPUTATIONAL COMPLEXITY\n"
        complexity += f"{'='*40}\n\n"
        
        complexity += f"TIME ANALYSIS:\n"
        complexity += f"{'─'*40}\n"
        for name, stats in metrics.items():
            complexity += f"{name}:\n"
            complexity += f"  Mean:   {stats['mean_time']:.4f}s\n"
            complexity += f"  Std:    {stats['std_time']:.4f}s\n"
            complexity += f"  Total:  {stats['total_time']:.4f}s\n"
        
        complexity += f"\nTIME COMPLEXITY:\n"
        complexity += f"{'─'*40}\n"
        
        # Add theoretical complexity for each algorithm
        for name in algo_names:
            if 'PSO' in name:
                complexity += f"PSO: O(iter·pop·dim)\n"
            elif 'GA' in name:
                complexity += f"GA:  O(iter·pop·dim)\n"
            elif 'ABC' in name:
                complexity += f"ABC: O(iter·pop·dim)\n"
            elif 'FA' in name:
                complexity += f"FA:  O(iter·pop²·dim)\n"
            elif 'CS' in name:
                complexity += f"CS:  O(iter·pop·dim)\n"
            elif 'SA' in name:
                complexity += f"SA:  O(iter·dim)\n"
        
        complexity += f"\nSPACE COMPLEXITY:\n"
        complexity += f"{'─'*40}\n"
        complexity += f"All: O(pop·dim)\n"
        complexity += f"     for population storage\n\n"
        
        complexity += f"SCALABILITY:\n"
        complexity += f"{'─'*40}\n"
        complexity += f"Current: {dim}D problem\n"
        complexity += f"Tested:  {n_runs} runs\n\n"
        complexity += f"See Scalability tab for\ndetailed dimension analysis."
        
        return {
            'summary': summary,
            'convergence': convergence,
            'robustness': robustness,
            'complexity': complexity
        }

    def _update_ui_with_results(self, experiment_type, ui_data, result_texts):
        """Update UI with experiment results (called in main thread)."""
        runner = GuiExperimentRunner()

        # Store the formatted text results first
        self.result_texts = result_texts
        
        if experiment_type == "tsp":
            cities, results = ui_data["cities"], ui_data["results"]
            algorithms = ui_data.get("algorithms", {})
            max_iter = ui_data.get("max_iter", 100)
            n_cities = ui_data.get("n_cities", 20)
            
            # Convergence view shows the route plot
            tsp_fig = runner.plot_tsp_experiment(cities, results)
            self.metric_data['convergence'] = tsp_fig
            
            # Generate plots for other metrics
            complexity_fig = runner.plot_tsp_complexity(results)
            self.metric_data['complexity'] = complexity_fig
            
            robustness_fig = runner.plot_tsp_robustness(results)
            self.metric_data['robustness'] = robustness_fig
            
            # Scalability analysis (will take time)
            self.update_status("Generating scalability plot...", ("orange", "orange"))
            city_counts = [10, 15, 20, 25, 30]
            scalability_fig = runner.plot_tsp_scalability(algorithms, max_iter, city_counts)
            self.metric_data['scalability'] = scalability_fig
            
            # Enable all buttons for TSP
            self.scalability_button.configure(state="normal")
        else:  # continuous
            stats_list = ui_data["stats_list"]
            # Enable scalability button for continuous problems
            self.scalability_button.configure(state="normal")
            problem_name = ui_data["problem_name"]
            dim = ui_data["dim"]
            
            # Enable scalability button for continuous problems
            self.scalability_button.configure(state="normal")
            
            # Generate and assign plots
            conv_robust_figs = runner.plot_continuous_experiment(stats_list, problem_name, dim)
            self.metric_data['convergence'] = conv_robust_figs['convergence']
            self.metric_data['robustness'] = conv_robust_figs['robustness']
            
            complexity_fig = runner.plot_complexity_experiment(stats_list, problem_name, dim)
            self.metric_data['complexity'] = complexity_fig
            
            self.metric_data['scalability'] = ui_data["scalability_fig"]

        # Set main_fig to the convergence plot for the save button, as a default
        self.main_fig = self.metric_data['convergence']

        # Refresh the current view to show the new data
        self.show_metric_view(self.current_metric_view)

        # Re-enable the save button as a figure is now available
        self.save_button.configure(state="normal")

    def _show_error(self, error_msg):
        """Show error message (called in main thread)."""
        error_text = f"❌ ERROR\n\n{error_msg}"
        self.metric_data = {
            'convergence': error_text,
            'complexity': error_text,
            'robustness': error_text,
            'scalability': error_text,
        }
        self.show_metric_view(self.current_metric_view)
        self.update_status("Error occurred", ("red", "red"))

    def _experiment_complete(self):
        """Clean up after experiment completes (called in main thread)."""
        self.is_running = False
        self.run_button.configure(state="normal", text="▶ Run Experiment")
        self.update_status("Experiment complete ✓", ("green", "green"))

    def change_experiment(self, experiment: str):
        """Update parameter inputs based on selected experiment."""
        # Clear old parameter widgets
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        # Add new parameter widgets based on experiment
        if experiment == "ACO vs SA (TSP)":
            # Number of cities
            n_cities_label = customtkinter.CTkLabel(
                self.params_frame, 
                text="Number of Cities:",
                font=customtkinter.CTkFont(size=12)
            )
            n_cities_label.grid(row=0, column=0, padx=0, pady=(5, 2), sticky="w")
            
            self.n_cities_entry = customtkinter.CTkEntry(
                self.params_frame, 
                placeholder_text="20",
                width=240,
                height=32
            )
            self.n_cities_entry.grid(row=1, column=0, padx=0, pady=(0, 10), sticky="ew")
            
            # Max iterations
            max_iter_label = customtkinter.CTkLabel(
                self.params_frame, 
                text="Max Iterations (ACO):",
                font=customtkinter.CTkFont(size=12)
            )
            max_iter_label.grid(row=2, column=0, padx=0, pady=(5, 2), sticky="w")
            
            self.max_iter_entry = customtkinter.CTkEntry(
                self.params_frame, 
                placeholder_text="100",
                width=240,
                height=32
            )
            self.max_iter_entry.grid(row=3, column=0, padx=0, pady=(0, 10), sticky="ew")
            
        else:
            # Dimensions
            dim_label = customtkinter.CTkLabel(
                self.params_frame, 
                text="Dimensions:",
                font=customtkinter.CTkFont(size=12)
            )
            dim_label.grid(row=0, column=0, padx=0, pady=(5, 2), sticky="w")
            
            self.dim_entry = customtkinter.CTkEntry(
                self.params_frame, 
                placeholder_text="10",
                width=240,
                height=32
            )
            self.dim_entry.grid(row=1, column=0, padx=0, pady=(0, 10), sticky="ew")
            
            # Max iterations
            max_iter_label = customtkinter.CTkLabel(
                self.params_frame, 
                text="Max Iterations:",
                font=customtkinter.CTkFont(size=12)
            )
            max_iter_label.grid(row=2, column=0, padx=0, pady=(5, 2), sticky="w")
            
            self.max_iter_entry = customtkinter.CTkEntry(
                self.params_frame, 
                placeholder_text="100",
                width=240,
                height=32
            )
            self.max_iter_entry.grid(row=3, column=0, padx=0, pady=(0, 10), sticky="ew")
            
            # Number of runs
            n_runs_label = customtkinter.CTkLabel(
                self.params_frame, 
                text="Number of Runs:",
                font=customtkinter.CTkFont(size=12)
            )
            n_runs_label.grid(row=4, column=0, padx=0, pady=(5, 2), sticky="w")
            
            self.n_runs_entry = customtkinter.CTkEntry(
                self.params_frame, 
                placeholder_text="5",
                width=240,
                height=32
            )
            self.n_runs_entry.grid(row=5, column=0, padx=0, pady=(0, 10), sticky="ew")

    def save_figure(self):
        """Save the currently viewed figure to a file."""
        fig_to_save = None
        
        # Get the data for the current view
        data = self.metric_data.get(self.current_metric_view)
        
        # Check if it's a figure
        if isinstance(data, plt.Figure):
            fig_to_save = data
        # If not, fall back to main_fig (which is the convergence plot)
        elif isinstance(self.main_fig, plt.Figure):
             fig_to_save = self.main_fig

        if fig_to_save:
            filepath = tkinter.filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("All files", "*.*")
                ],
                title=f"Save {self.current_metric_view.capitalize()} Plot"
            )
            if filepath:
                fig_to_save.savefig(filepath, dpi=300, bbox_inches='tight')
                self.update_status(f"Saved to {os.path.basename(filepath)} ✓", ("green", "green"))
        else:
            self.update_status("No figure to save for current view", ("orange", "orange"))


if __name__ == "__main__":
    app = App()
    app.mainloop()

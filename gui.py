import tkinter
import customtkinter
import matplotlib.pyplot as plt
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
    plot_tsp_route
)

customtkinter.set_appearance_mode("dark")
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
        """Plot continuous experiment results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
        fig.suptitle(f"{problem_name.capitalize()} (dim={dim})", fontsize=16)

        # 1. Convergence plot
        histories = {}
        for stats in stats_list:
            if stats.results:
                result = stats.results[0]
                history = result.get('history') if isinstance(result, dict) else getattr(result, 'history', [])
                histories[stats.algorithm_name] = history

        plot_convergence_comparison(
            histories,
            title="Convergence Comparison",
            ax=ax1,
            log_scale=True
        )

        # 2. Boxplot
        boxplot_data = {stats.algorithm_name: stats.all_fitnesses for stats in stats_list}
        plot_boxplot_comparison(
            boxplot_data,
            title="Performance Distribution",
            ax=ax2
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def run_tsp_experiment_data(self, n_cities: int, algorithms: dict,
                               n_runs: int = 1, max_iter: int = 100):
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
                'best_result': best_result
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


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Search Algorithms Visualization & Analysis")
        self.geometry(f"{1600}x{850}")
        
        # Configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
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
            fg_color=("gray70", "gray30"),
            hover_color=("gray60", "gray40")
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

        # Create main visualization area
        self.main_frame = customtkinter.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Placeholder for visualization
        self.placeholder_label = customtkinter.CTkLabel(
            self.main_frame,
            text="📊\n\nSelect an experiment and click 'Run'\nto visualize algorithm performance",
            font=customtkinter.CTkFont(size=16),
            text_color=("gray50", "gray50")
        )
        self.placeholder_label.grid(row=0, column=0, padx=40, pady=40)

        self.fig = None
        self.canvas = None
        self.is_running = False

        # Create results frame with tabs
        self.results_frame = customtkinter.CTkFrame(self, corner_radius=10)
        self.results_frame.grid(row=0, column=2, padx=(0, 20), pady=20, sticky="nsew")
        self.results_frame.grid_rowconfigure(1, weight=1)
        
        self.results_title = customtkinter.CTkLabel(
            self.results_frame,
            text="📈 Performance Analysis",
            font=customtkinter.CTkFont(size=16, weight="bold")
        )
        self.results_title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        # Create tabview for different metrics
        self.tabview = customtkinter.CTkTabview(self.results_frame, width=400)
        self.tabview.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        
        # Add tabs
        self.tabview.add("Summary")
        self.tabview.add("Convergence")
        self.tabview.add("Robustness")
        self.tabview.add("Complexity")
        
        # Create textboxes for each tab
        self.summary_textbox = customtkinter.CTkTextbox(
            self.tabview.tab("Summary"),
            font=customtkinter.CTkFont(size=11, family="Courier"),
            wrap="word"
        )
        self.summary_textbox.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.convergence_textbox = customtkinter.CTkTextbox(
            self.tabview.tab("Convergence"),
            font=customtkinter.CTkFont(size=11, family="Courier"),
            wrap="word"
        )
        self.convergence_textbox.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.robustness_textbox = customtkinter.CTkTextbox(
            self.tabview.tab("Robustness"),
            font=customtkinter.CTkFont(size=11, family="Courier"),
            wrap="word"
        )
        self.robustness_textbox.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.complexity_textbox = customtkinter.CTkTextbox(
            self.tabview.tab("Complexity"),
            font=customtkinter.CTkFont(size=11, family="Courier"),
            wrap="word"
        )
        self.complexity_textbox.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Initialize with placeholder text
        self._init_placeholder_text()

        # Set default values
        self.change_experiment(self.experiment_menu.get())

    def _init_placeholder_text(self):
        """Initialize placeholder text in all tabs."""
        placeholder = "Run an experiment to see\ndetailed performance metrics."
        
        self.summary_textbox.insert("0.0", placeholder)
        self.convergence_textbox.insert("0.0", placeholder)
        self.robustness_textbox.insert("0.0", placeholder)
        self.complexity_textbox.insert("0.0", placeholder)

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

            if experiment == "ACO vs SA (TSP)":
                n_cities_str = self.n_cities_entry.get()
                max_iter_str = self.max_iter_entry.get()
                n_cities = int(n_cities_str) if n_cities_str else int(self.n_cities_entry.cget("placeholder_text"))
                max_iter = int(max_iter_str) if max_iter_str else int(self.max_iter_entry.cget("placeholder_text"))
                
                tsp_algos = {
                    'ACO': (run_aco, {}),
                    'SA': (run_simulated_annealing_tsp, {})
                }
                cities, results = runner.run_tsp_experiment_data(n_cities, tsp_algos, n_runs=1, max_iter=max_iter)
                result_texts = self._format_tsp_results(results, n_cities, max_iter)
                self.after(0, self._update_ui_with_results, "tsp", (cities, results), result_texts)
            else:
                dim_str = self.dim_entry.get()
                max_iter_str = self.max_iter_entry.get()
                n_runs_str = self.n_runs_entry.get()
                dim = int(dim_str) if dim_str else int(self.dim_entry.cget("placeholder_text"))
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

                stats_list, metrics = runner.run_continuous_experiment_data(problem_name, dim, algorithms, n_runs, max_iter)
                result_texts = self._format_continuous_results(metrics, dim, max_iter, n_runs)
                self.after(0, self._update_ui_with_results, "continuous", (stats_list, problem_name, dim), result_texts)

        except Exception as e:
            error_msg = f"Error running experiment:\n{str(e)}"
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
            history = stats['results'][0].get('history', [])
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
        complexity += f"Population-based methods\nscale linearly with dim.\n"
        complexity += f"FA has higher cost due to\npairwise comparisons."
        
        return {
            'summary': summary,
            'convergence': convergence,
            'robustness': robustness,
            'complexity': complexity
        }

    def _update_ui_with_results(self, experiment_type, data, result_texts):
        """Update UI with experiment results (called in main thread)."""
        runner = GuiExperimentRunner()
        if experiment_type == "tsp":
            cities, results = data
            self.fig = runner.plot_tsp_experiment(cities, results)
        else:  # continuous
            stats_list, problem_name, dim = data
            self.fig = runner.plot_continuous_experiment(stats_list, problem_name, dim)

        # Remove placeholder if it exists
        if self.placeholder_label.winfo_exists():
            self.placeholder_label.destroy()
            
        # Clear old canvas
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        # Update all textboxes
        self.summary_textbox.delete("0.0", "end")
        self.summary_textbox.insert("0.0", result_texts['summary'])
        
        self.convergence_textbox.delete("0.0", "end")
        self.convergence_textbox.insert("0.0", result_texts['convergence'])
        
        self.robustness_textbox.delete("0.0", "end")
        self.robustness_textbox.insert("0.0", result_texts['robustness'])
        
        self.complexity_textbox.delete("0.0", "end")
        self.complexity_textbox.insert("0.0", result_texts['complexity'])

        # Display new figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    def _show_error(self, error_msg):
        """Show error message (called in main thread)."""
        for textbox in [self.summary_textbox, self.convergence_textbox, 
                       self.robustness_textbox, self.complexity_textbox]:
            textbox.delete("0.0", "end")
            textbox.insert("0.0", f"❌ ERROR\n\n{error_msg}")
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
        """Save the current figure to a file."""
        if self.fig:
            filepath = tkinter.filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("All files", "*.*")
                ]
            )
            if filepath:
                self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
                self.update_status(f"Saved to {os.path.basename(filepath)} ✓", ("green", "green"))
        else:
            self.update_status("No figure to save", ("orange", "orange"))


if __name__ == "__main__":
    app = App()
    app.mainloop()

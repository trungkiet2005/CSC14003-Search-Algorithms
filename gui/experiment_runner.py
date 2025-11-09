"""gui/experiment_runner.py - Experiment runners for visualization and comparison"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import time
from typing import Dict, List, Tuple
import seaborn as sns

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
    plot_3d_surface, plot_contour, plot_tsp_route,
    plot_complexity_comparison, plot_scalability_comparison,
    plot_parameter_sensitivity
)


class VisualizationRunner:
    """Runner for individual algorithm visualization"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.benchmark_runner = BenchmarkRunner(seed=seed, verbose=False)
        
    def run_visualization_analysis(self, algorithm: str, problem: str, 
                                   dim: int, max_iter: int, n_runs: int) -> Dict:
        """
        Run comprehensive visualization analysis for a single algorithm.
        
        Returns dict with 4 required visualizations:
        - convergence: Convergence plot
        - performance: Comparative performance across runs
        - sensitivity: Parameter sensitivity analysis
        - landscape: 3D surface plot (for 2D problems)
        """
        # Get algorithm function and parameters
        algo_func, algo_params = self._get_algorithm(algorithm)
        problem_func, problem_info = get_problem(problem, dim)
        bounds = problem_info['bounds']
        
        # 1. CONVERGENCE ANALYSIS
        convergence_fig = self._create_convergence_plot(
            algo_func, algo_params, problem_func, bounds, dim, max_iter, algorithm, problem
        )
        
        # 2. PERFORMANCE COMPARISON (across multiple runs)
        performance_fig = self._create_performance_plot(
            algo_func, algo_params, problem_func, bounds, dim, max_iter, n_runs, algorithm, problem
        )
        
        # 3. PARAMETER SENSITIVITY ANALYSIS
        sensitivity_fig = self._create_sensitivity_plot(
            algorithm, algo_func, algo_params, problem_func, bounds, dim, max_iter, problem
        )
        
        # 4. 3D LANDSCAPE (for 2D visualization)
        landscape_fig = self._create_landscape_plot(
            algo_func, algo_params, problem_func, bounds, max_iter, algorithm, problem
        )
        
        return {
            'convergence': convergence_fig,
            'performance': performance_fig,
            'sensitivity': sensitivity_fig,
            'landscape': landscape_fig
        }
    
    def _get_algorithm(self, algorithm: str) -> Tuple:
        """Get algorithm function and default parameters"""
        algo_map = {
            'PSO': (run_pso, {'n_particles': 30}),
            'ACO': (run_aco, {}),
            'ABC': (run_abc, {'n_bees': 30}),
            'FA': (run_fa, {'n_fireflies': 25}),
            'CS': (run_cs, {'n_nests': 25})
        }
        return algo_map[algorithm]
    
    def _create_convergence_plot(self, algo_func, algo_params, problem_func, bounds, 
                                dim, max_iter, algorithm, problem):
        """Create convergence plot"""
        params = {
            'dim': dim,
            'bounds': bounds,
            'max_iter': max_iter,
            **algo_params
        }
        
        result = algo_func(objective_func=problem_func, seed=self.seed, **params)
        history = result['history'] if isinstance(result, dict) else result.history
        
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(history, linewidth=2.5, alpha=0.8, color='#3498db')
        ax.set_title(f"Convergence: {algorithm} on {problem.capitalize()} (dim={dim})", 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel("Iteration", fontsize=13)
        ax.set_ylabel("Fitness (log scale)", fontsize=13)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        return fig
    
    def _create_performance_plot(self, algo_func, algo_params, problem_func, bounds,
                                dim, max_iter, n_runs, algorithm, problem):
        """Create performance comparison across multiple runs"""
        params = {
            'dim': dim,
            'bounds': bounds,
            'max_iter': max_iter,
            **algo_params
        }
        
        all_histories = []
        best_fitnesses = []
        
        for run in range(n_runs):
            result = algo_func(objective_func=problem_func, seed=self.seed + run, **params)
            history = result['history'] if isinstance(result, dict) else result.history
            all_histories.append(history)
            best_fitnesses.append(history[-1])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Performance Analysis: {algorithm} on {problem.capitalize()}", 
                    fontsize=16, fontweight='bold')
        
        # Plot all runs
        for i, hist in enumerate(all_histories):
            ax1.plot(hist, alpha=0.5, linewidth=1.5)
        ax1.set_title(f"Convergence Across {n_runs} Runs", fontsize=14)
        ax1.set_xlabel("Iteration", fontsize=12)
        ax1.set_ylabel("Fitness (log scale)", fontsize=12)
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Boxplot of final fitnesses
        ax2.boxplot([best_fitnesses], labels=[algorithm], patch_artist=True,
                   boxprops=dict(facecolor='#3498db', alpha=0.7))
        ax2.set_title("Final Fitness Distribution", fontsize=14)
        ax2.set_ylabel("Final Fitness", fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def _create_sensitivity_plot(self, algorithm, algo_func, algo_params, problem_func,
                                bounds, dim, max_iter, problem):
        """Create parameter sensitivity analysis"""
        # Define parameter ranges based on algorithm
        param_ranges = {
            'PSO': ('n_particles', [10, 20, 30, 40, 50]),
            'ABC': ('n_bees', [10, 20, 30, 40, 50]),
            'FA': ('n_fireflies', [10, 15, 20, 25, 30]),
            'CS': ('n_nests', [10, 15, 20, 25, 30]),
            'ACO': ('n_ants', [10, 20, 30, 40, 50])
        }
        
        param_name, param_values = param_ranges.get(algorithm, ('n_particles', [10, 20, 30, 40, 50]))
        
        mean_fitness = []
        std_fitness = []
        
        for val in param_values:
            params = {
                'dim': dim,
                'bounds': bounds,
                'max_iter': max_iter,
                **algo_params
            }
            params[param_name] = val
            
            fitnesses = []
            for run in range(5):  # 5 runs per parameter value
                result = algo_func(objective_func=problem_func, seed=self.seed + run, **params)
                fitness = result['best_fitness'] if isinstance(result, dict) else result.best_fitness
                fitnesses.append(fitness)
            
            mean_fitness.append(np.mean(fitnesses))
            std_fitness.append(np.std(fitnesses))
        
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.errorbar(param_values, mean_fitness, yerr=std_fitness,
                   marker='o', markersize=8, linewidth=2, capsize=5, capthick=2,
                   alpha=0.8, color='#e74c3c', label='Mean ± Std')
        ax.set_title(f"Parameter Sensitivity: {algorithm} on {problem.capitalize()}", 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel(param_name.replace('_', ' ').title(), fontsize=13)
        ax.set_ylabel('Final Fitness', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        plt.tight_layout()
        return fig
    
    def _create_landscape_plot(self, algo_func, algo_params, problem_func, bounds,
                              max_iter, algorithm, problem):
        """Create 3D landscape plot (for 2D problems)"""
        # Run algorithm on 2D problem
        params = {
            'dim': 2,
            'bounds': bounds,
            'max_iter': max_iter,
            **algo_params
        }
        
        result = algo_func(objective_func=problem_func, seed=self.seed, **params)
        best_solution = result['best_solution'] if isinstance(result, dict) else result.best_solution
        
        # Create 3D surface plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), 
                                       subplot_kw={'projection': '3d'} if True else None)
        fig.suptitle(f"3D Landscape: {problem.capitalize()} Function", 
                    fontsize=16, fontweight='bold')
        
        # Prepare data
        lower, upper = bounds
        resolution = 50
        x = np.linspace(lower, upper, resolution)
        y = np.linspace(lower, upper, resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = problem_func([X[i, j], Y[i, j]])
        
        # 3D surface
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, linewidth=0)
        if len(best_solution) >= 2:
            z_best = problem_func(best_solution)
            ax1.scatter([best_solution[0]], [best_solution[1]], [z_best],
                       color='red', s=200, marker='*', edgecolors='black', linewidths=2)
        ax1.set_xlabel('X', fontsize=12)
        ax1.set_ylabel('Y', fontsize=12)
        ax1.set_zlabel('f(X, Y)', fontsize=12)
        ax1.set_title('3D Surface', fontsize=14)
        ax1.view_init(elev=30, azim=45)
        fig.colorbar(surf, ax=ax1, shrink=0.5)
        
        # Contour plot
        ax2 = fig.add_subplot(122)
        contourf = ax2.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
        ax2.contour(X, Y, Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
        if len(best_solution) >= 2:
            ax2.scatter([best_solution[0]], [best_solution[1]],
                       color='red', s=300, marker='*', edgecolors='black', linewidths=2,
                       label=f'{algorithm} Solution')
        ax2.set_xlabel('X', fontsize=12)
        ax2.set_ylabel('Y', fontsize=12)
        ax2.set_title('Contour Plot', fontsize=14)
        ax2.legend(fontsize=10)
        fig.colorbar(contourf, ax=ax2)
        
        plt.tight_layout()
        return fig


class ComparisonRunner:
    """Runner for algorithm comparison"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.benchmark_runner = BenchmarkRunner(seed=seed, verbose=False)
        
    def run_continuous_comparison(self, problem: str, dim: int, max_iter: int,
                                  n_runs: int, algos: List[str]) -> Dict:
        """Run comparison for continuous optimization"""
        problem_func, problem_info = get_problem(problem, dim)
        bounds = problem_info['bounds']
        
        # Map algorithm names to functions
        algo_map = {
            'PSO': (run_pso, {'n_particles': 30}),
            'HC': (run_hill_climbing, {'step_size': 0.1, 'random_restart': 5}),
            'ABC': (run_abc, {'n_bees': 30}),
            'GA': (run_ga, {'pop_size': 50}),
            'FA': (run_fa, {'n_fireflies': 25}),
            'SA': (run_simulated_annealing, {}),
            'CS': (run_cs, {'n_nests': 25})
        }
        
        algorithms = {name: algo_map[name] for name in algos}
        
        # Add parameters
        algo_dict = {}
        for algo_name, (algo_func, algo_params) in algorithms.items():
            params = {'dim': dim, 'bounds': bounds, 'max_iter': max_iter, **algo_params}
            algo_dict[algo_name] = (algo_func, params)
        
        # Run comparison
        df, stats_list = self.benchmark_runner.compare_algorithms(
            algo_dict, problem_func, problem, dim, n_runs=n_runs
        )
        
        # Create visualizations
        convergence_fig = self._plot_convergence(stats_list, problem, dim)
        complexity_fig = plot_complexity_comparison(stats_list, 
                                                    title=f"Complexity: {problem.capitalize()} (dim={dim})")
        robustness_fig = self._plot_robustness(stats_list, problem, dim)
        scalability_fig = self._plot_scalability(algorithms, problem, max_iter, n_runs)
        
        return {
            'convergence': convergence_fig,
            'complexity': complexity_fig,
            'robustness': robustness_fig,
            'scalability': scalability_fig
        }
    
    def run_tsp_comparison(self, n_cities: int, max_iter: int, n_runs: int) -> Dict:
        """Run comparison for TSP"""
        tsp = create_tsp_problem(n_cities, seed=self.seed)
        cities = tsp['cities']
        distance_matrix = tsp['distance_matrix']
        
        algorithms = {
            'ACO': (run_aco, {}),
            'SA': (run_simulated_annealing_tsp, {})
        }
        
        results = {}
        for algo_name, (algo_func, algo_params) in algorithms.items():
            fitnesses = []
            times = []
            best_result = None
            best_distance = float('inf')

            for run in range(n_runs):
                start = time.time()
                if 'distance_matrix' in algo_func.__code__.co_varnames:
                    result = algo_func(distance_matrix, max_iter=max_iter, seed=self.seed + run, **algo_params)
                else:
                    result = algo_func(tsp['objective'], max_iter=max_iter, seed=self.seed + run, **algo_params)
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
                'all_distances': fitnesses,
                'best_route': best_result['best_route']
            }
        
        # Create visualizations
        convergence_fig = self._plot_tsp_routes(cities, results)
        complexity_fig = self._plot_tsp_complexity(results)
        robustness_fig = self._plot_tsp_robustness(results)
        scalability_fig = self._plot_tsp_scalability(algorithms, max_iter)
        
        return {
            'convergence': convergence_fig,
            'complexity': complexity_fig,
            'robustness': robustness_fig,
            'scalability': scalability_fig
        }
    
    def _plot_convergence(self, stats_list, problem, dim):
        """Plot convergence comparison"""
        fig, ax = plt.subplots(figsize=(12, 7))
        histories = {}
        for stats in stats_list:
            if stats.results:
                result = stats.results[0]
                history = result.get('history') if isinstance(result, dict) else getattr(result, 'history', [])
                histories[stats.algorithm_name] = history
        
        plot_convergence_comparison(
            histories,
            title=f"Convergence: {problem.capitalize()} (dim={dim})",
            ax=ax,
            log_scale=True
        )
        return fig
    
    def _plot_robustness(self, stats_list, problem, dim):
        """Plot robustness comparison"""
        fig, ax = plt.subplots(figsize=(12, 7))
        boxplot_data = {stats.algorithm_name: stats.all_fitnesses for stats in stats_list}
        plot_boxplot_comparison(
            boxplot_data,
            title=f"Robustness: {problem.capitalize()} (dim={dim})",
            ax=ax
        )
        return fig
    
    def _plot_scalability(self, algorithms, problem, max_iter, n_runs):
        """Plot scalability analysis"""
        dims = [5, 10, 20, 30, 50]
        scalability_data = {}
        
        for algo_name in algorithms.keys():
            scalability_data[algo_name] = {'dims': [], 'fitness': [], 'times': []}
        
        for dim in dims:
            problem_func, problem_info = get_problem(problem, dim)
            bounds = problem_info['bounds']
            
            algo_dict = {}
            for algo_name, (algo_func, algo_params) in algorithms.items():
                params = {'dim': dim, 'bounds': bounds, 'max_iter': max_iter, **algo_params}
                algo_dict[algo_name] = (algo_func, params)
            
            _, stats_list = self.benchmark_runner.compare_algorithms(
                algo_dict, problem_func, problem, dim, n_runs=n_runs
            )
            
            for stats in stats_list:
                scalability_data[stats.algorithm_name]['dims'].append(dim)
                scalability_data[stats.algorithm_name]['fitness'].append(stats.best_fitness)
                scalability_data[stats.algorithm_name]['times'].append(stats.mean_time)
        
        fig = plot_scalability_comparison(scalability_data, 
                                         title=f"Scalability: {problem.capitalize()}")
        return fig
    
    def _plot_tsp_routes(self, cities, results):
        """Plot TSP routes"""
        fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6))
        if len(results) == 1:
            axes = [axes]
        
        for ax, (algo_name, data) in zip(axes, results.items()):
            plot_tsp_route(cities, data['best_route'], data['best_distance'],
                          title=f"{algo_name}\nDistance: {data['best_distance']:.2f}",
                          ax=ax)
        plt.tight_layout()
        return fig
    
    def _plot_tsp_complexity(self, results):
        """Plot TSP complexity"""
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
    
    def _plot_tsp_robustness(self, results):
        """Plot robustness (boxplot) for TSP."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        algo_names = list(results.keys())
        data = []
        labels = []
        for name in algo_names:
            if 'all_distances' in results[name]:
                data.append(results[name]['all_distances'])
                labels.append(name)
        
        if not data or any(len(d) < 2 for d in data):
            ax.text(0.5, 0.5, 'Run multiple times\nto see robustness analysis', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.axis('off')
            return fig
        
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
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
    
    def _plot_tsp_scalability(self, algorithms, max_iter):
        """Plot TSP scalability"""
        city_counts = [10, 15, 20, 25, 30]
        scalability_data = {name: {'cities': [], 'distances': [], 'times': []} 
                           for name in algorithms.keys()}
        
        for n_cities in city_counts:
            tsp = create_tsp_problem(n_cities, seed=self.seed)
            distance_matrix = tsp['distance_matrix']
            
            for algo_name, (algo_func, algo_params) in algorithms.items():
                start = time.time()
                if 'distance_matrix' in algo_func.__code__.co_varnames:
                    result = algo_func(distance_matrix, max_iter=max_iter, 
                                     seed=self.seed, **algo_params)
                else:
                    result = algo_func(tsp['objective'], max_iter=max_iter,
                                     seed=self.seed, **algo_params)
                elapsed = time.time() - start
                
                scalability_data[algo_name]['cities'].append(n_cities)
                scalability_data[algo_name]['distances'].append(result['best_distance'])
                scalability_data[algo_name]['times'].append(elapsed)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle("TSP Scalability Analysis", fontsize=18, fontweight='bold')
        
        colors = sns.color_palette("husl", len(scalability_data))
        
        for (algo_name, data), color in zip(scalability_data.items(), colors):
            ax1.plot(data['cities'], data['distances'], marker='o', linestyle='-',
                    color=color, linewidth=2, markersize=8, label=algo_name)
        ax1.set_title("Solution Quality vs. Problem Size", fontsize=14)
        ax1.set_xlabel("Number of Cities", fontsize=12)
        ax1.set_ylabel("Tour Distance", fontsize=12)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        for (algo_name, data), color in zip(scalability_data.items(), colors):
            ax2.plot(data['cities'], data['times'], marker='o', linestyle='-',
                    color=color, linewidth=2, markersize=8, label=algo_name)
        ax2.set_title("Execution Time vs. Problem Size", fontsize=14)
        ax2.set_xlabel("Number of Cities", fontsize=12)
        ax2.set_ylabel("Time (seconds)", fontsize=12)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        return fig
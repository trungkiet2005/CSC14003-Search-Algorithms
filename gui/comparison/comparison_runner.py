import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import time
from typing import Dict, List, Tuple
import seaborn as sns

# Import algorithms
from algorithms.swarm.PSO import run_pso
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

class ComparisonRunner:
    """Runner for algorithm comparison"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.benchmark_runner = BenchmarkRunner(seed=seed, verbose=False)
        
    def run_continuous_comparison(self, problem: str, dim: int, max_iter: int,
                                  n_runs: int, algos: List[str], algo_params: Dict = None) -> Dict:
        """Run comparison for continuous optimization"""
        if algo_params is None:
            algo_params = {}
            
        problem_func, problem_info = get_problem(problem, dim)
        bounds = problem_info['bounds']
        
        # Map algorithm names to functions
        func_map = {
            'PSO': run_pso, 'HC': run_hill_climbing, 'ABC': run_abc, 'GA': run_ga,
            'FA': run_fa, 'SA': run_simulated_annealing, 'CS': run_cs
        }
        
        # Build algorithm dictionary with user-provided parameters
        algo_dict = {}
        for algo_name in algos:
            algo_func = func_map[algo_name]
            user_params = algo_params.get(algo_name, {})
            full_params = {'dim': dim, 'bounds': bounds, 'max_iter': max_iter, **user_params}
            algo_dict[algo_name] = (algo_func, full_params)
        
        # Run comparison
        df, stats_list = self.benchmark_runner.compare_algorithms(
            algo_dict, problem_func, problem, dim, n_runs=n_runs
        )
        
        # Create visualizations
        convergence_fig = self._plot_convergence(stats_list, problem, dim)
        complexity_fig = plot_complexity_comparison(stats_list, 
                                                    title=f"Complexity: {problem.capitalize()} (dim={dim})")
        robustness_fig = self._plot_robustness(stats_list, problem, dim)
        scalability_fig = self._plot_scalability(algo_dict, problem, max_iter, n_runs)
        
        return {
            'convergence': convergence_fig,
            'complexity': complexity_fig,
            'robustness': robustness_fig,
            'scalability': scalability_fig
        }
    
    def run_tsp_comparison(self, n_cities: int, max_iter: int, n_runs: int = 1, algo_params: Dict = None) -> Dict:
        """Run comparison for TSP"""
        if algo_params is None:
            algo_params = {}
            
        tsp = create_tsp_problem(n_cities, seed=self.seed)
        cities = tsp['cities']
        distance_matrix = tsp['distance_matrix']
        
        from algorithms.swarm.ACO import run_aco
        
        algorithms = {
            'ACO': (run_aco, algo_params.get('ACO', {})),
            'SA': (run_simulated_annealing_tsp, algo_params.get('SA', {}))
        }
        
        results = {}
        for algo_name, (algo_func, user_params) in algorithms.items():
            fitnesses = []
            times = []
            best_result = None
            best_distance = float('inf')

            for run in range(n_runs):
                start = time.time()
                
                run_params = {'max_iter': max_iter, 'seed': self.seed + run, **user_params}

                if 'distance_matrix' in algo_func.__code__.co_varnames:
                    result = algo_func(distance_matrix, **run_params)
                else:
                    result = algo_func(tsp['objective'], **run_params)
                
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
    
    def _plot_scalability(self, algo_dict, problem, max_iter, n_runs):
        """Plot scalability analysis"""
        dims = [5, 10, 20, 30, 50]
        scalability_data = {}
        
        for algo_name in algo_dict.keys():
            scalability_data[algo_name] = {'dims': [], 'fitness': [], 'times': []}
        
        for dim in dims:
            problem_func, problem_info = get_problem(problem, dim)
            bounds = problem_info['bounds']
            
            test_algo_dict = {}
            for algo_name, (algo_func, base_params) in algo_dict.items():
                # Update only dim and bounds, keep other params
                params = {**base_params, 'dim': dim, 'bounds': bounds, 'max_iter': max_iter}
                test_algo_dict[algo_name] = (algo_func, params)
            
            _, stats_list = self.benchmark_runner.compare_algorithms(
                test_algo_dict, problem_func, problem, dim, n_runs=n_runs
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
            ax.text(0.5, 0.5, 'Run multiple times\n(increase number of runs in comparison tab)\nto see robustness analysis', 
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
            
            for algo_name, (algo_func, base_params) in algorithms.items():
                start = time.time()
                run_params = {**base_params, 'max_iter': max_iter, 'seed': self.seed}
                
                if 'distance_matrix' in algo_func.__code__.co_varnames:
                    result = algo_func(distance_matrix, **run_params)
                else:
                    result = algo_func(tsp['objective'], **run_params)

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
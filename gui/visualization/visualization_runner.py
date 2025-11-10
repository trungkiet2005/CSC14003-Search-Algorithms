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

class VisualizationRunner:
    """Runner for individual algorithm visualization"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.benchmark_runner = BenchmarkRunner(seed=seed, verbose=False)
        
    def run_visualization_analysis(self, algorithm: str, problem: str, 
                                   dim: int, max_iter: int, n_runs: int,
                                   algo_specific_params: Dict = None,
                                   problem_specific_params: Dict = None) -> Dict:
        """
        Run comprehensive visualization analysis for a single algorithm.
        
        Returns dict with 4 required visualizations:
        - convergence: Convergence ability plot
        - performance: Comparative performance across runs (boxplot + multi-run convergence)
        - sensitivity: Parameter sensitivity analysis
        - landscape: 3D surface plot showing objective function landscape
        """
        # Get algorithm function and parameters
        algo_func, algo_params = self._get_algorithm(algorithm, algo_specific_params)
        problem_func, problem_info = get_problem(problem, dim)
        bounds = problem_info['bounds']
        
        # 1. CONVERGENCE ABILITY - Single run to demonstrate convergence
        convergence_fig = self._create_convergence_plot(
            algo_func, algo_params, problem_func, bounds, dim, max_iter, algorithm, problem
        )
        
        # 2. COMPARATIVE PERFORMANCE - Multiple runs with boxplot
        performance_fig = self._create_performance_plot(
            algo_func, algo_params, problem_func, bounds, dim, max_iter, n_runs, algorithm, problem
        )
        
        # 3. PARAMETER SENSITIVITY ANALYSIS - Using existing function
        sensitivity_fig = self._create_sensitivity_plot(
            algorithm, algo_func, algo_params, problem_func, bounds, dim, max_iter, problem
        )
        
        # 4. 3D SURFACE PLOTS (ADVANCED) - Landscape visualization
        landscape_fig = self._create_landscape_plot(
            algo_func, algo_params, problem_func, bounds, max_iter, algorithm, problem
        )
        
        return {
            'convergence': convergence_fig,
            'performance': performance_fig,
            'sensitivity': sensitivity_fig,
            'landscape': landscape_fig
        }
    
    def _get_algorithm(self, algorithm: str, algo_specific_params: Dict = None) -> Tuple:
        """Get algorithm function and parameters, merging defaults with specific ones."""
        default_params = {
            'PSO': {'n_particles': 30, 'w': 0.7298, 'c1': 1.49618, 'c2': 1.49618},
            'ABC': {'n_bees': 30},
            'FA': {'n_fireflies': 25, 'alpha': 0.5, 'beta0': 1.0, 'gamma': 1.0},
            'CS': {'n_nests': 25, 'pa': 0.25, 'beta': 1.5}
        }
        
        func_map = {
            'PSO': run_pso,
            'ABC': run_abc,
            'FA': run_fa,
            'CS': run_cs
        }
        
        algo_func = func_map[algorithm]
        
        # Merge params
        params = default_params.get(algorithm, {}).copy()
        if algo_specific_params:
            params.update(algo_specific_params)
            
        return algo_func, params
    
    def _create_convergence_plot(self, algo_func, algo_params, problem_func, bounds, 
                                dim, max_iter, algorithm, problem):
        """
        1. CONVERGENCE ABILITY
        Demonstrates how the algorithm converges over iterations
        """
        params = {
            'dim': dim,
            'bounds': bounds,
            'max_iter': max_iter,
            **algo_params
        }
        
        # Run algorithm
        result = algo_func(objective_func=problem_func, seed=self.seed, **params)
        history = result['history'] if isinstance(result, dict) else result.history
        
        # Create convergence plot using existing function
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Use plot_convergence_comparison with single algorithm
        histories_dict = {algorithm: history}
        plot_convergence_comparison(
            histories_dict,
            title=f"Convergence Ability: {algorithm} on {problem.capitalize()} (dim={dim})",
            xlabel="Iteration",
            log_scale=True,
            ax=ax
        )
        
        # Add additional information
        final_fitness = history[-1]
        initial_fitness = history[0]
        improvement = ((initial_fitness - final_fitness) / initial_fitness) * 100
        
        textstr = f'Initial: {initial_fitness:.6f}\nFinal: {final_fitness:.6f}\nImprovement: {improvement:.2f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        return fig
    
    def _create_performance_plot(self, algo_func, algo_params, problem_func, bounds,
                                dim, max_iter, n_runs, algorithm, problem):
        """
        2. COMPARATIVE PERFORMANCE
        Shows performance across multiple runs - demonstrates robustness
        """
        params = {
            'dim': dim,
            'bounds': bounds,
            'max_iter': max_iter,
            **algo_params
        }
        
        all_histories = []
        best_fitnesses = []
        
        # Run multiple times
        for run in range(n_runs):
            result = algo_func(objective_func=problem_func, seed=self.seed + run, **params)
            history = result['history'] if isinstance(result, dict) else result.history
            all_histories.append(history)
            best_fitnesses.append(history[-1])
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Comparative Performance: {algorithm} on {problem.capitalize()} ({n_runs} runs)", 
                    fontsize=16, fontweight='bold')
        
        # Left: Multiple convergence curves
        colors = sns.color_palette("husl", n_runs)
        for i, (hist, color) in enumerate(zip(all_histories, colors)):
            ax1.plot(hist, alpha=0.6, linewidth=1.5, color=color, label=f'Run {i+1}')
        
        # Add mean convergence
        mean_history = np.mean(all_histories, axis=0)
        ax1.plot(mean_history, 'k--', linewidth=3, alpha=0.9, label='Mean')
        
        ax1.set_title(f"Convergence Across {n_runs} Runs", fontsize=14)
        ax1.set_xlabel("Iteration", fontsize=12)
        ax1.set_ylabel("Fitness (log scale)", fontsize=12)
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=9, loc='best', ncol=2)
        
        # Right: Boxplot using existing function
        boxplot_data = {algorithm: best_fitnesses}
        plot_boxplot_comparison(
            boxplot_data,
            title=f"Final Fitness Distribution",
            ylabel="Final Fitness Value",
            ax=ax2
        )
        
        # Add statistics to boxplot
        mean_fit = np.mean(best_fitnesses)
        std_fit = np.std(best_fitnesses)
        min_fit = np.min(best_fitnesses)
        max_fit = np.max(best_fitnesses)
        
        stats_text = f'Mean: {mean_fit:.6f}\nStd: {std_fit:.6f}\nMin: {min_fit:.6f}\nMax: {max_fit:.6f}'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        return fig
    
    def _create_sensitivity_plot(self, algorithm, algo_func, algo_params, problem_func,
                                bounds, dim, max_iter, problem):
        """
        3. PARAMETER SENSITIVITY ANALYSIS
        Tests different parameter values to show impact on performance
        """
        # Define parameter ranges based on algorithm
        param_ranges = {
            'PSO': ('n_particles', [10, 20, 30, 40, 50]),
            'ABC': ('n_bees', [10, 20, 30, 40, 50]),
            'FA': ('n_fireflies', [10, 15, 20, 25, 30]),
            'CS': ('n_nests', [10, 15, 20, 25, 30])
        }
        
        param_name, param_values = param_ranges.get(algorithm, ('n_particles', [10, 20, 30, 40, 50]))
        
        mean_fitness = []
        std_fitness = []
        
        # Test each parameter value
        for val in param_values:
            current_algo_params = algo_params.copy()
            current_algo_params[param_name] = val

            params = {
                'dim': dim,
                'bounds': bounds,
                'max_iter': max_iter,
                **current_algo_params
            }
            
            fitnesses = []
            for run in range(5):  # 5 runs per parameter value
                result = algo_func(objective_func=problem_func, seed=self.seed + run, **params)
                fitness = result['best_fitness'] if isinstance(result, dict) else result.best_fitness
                fitnesses.append(fitness)
            
            mean_fitness.append(np.mean(fitnesses))
            std_fitness.append(np.std(fitnesses))
        
        # Use existing plot_parameter_sensitivity function
        fig = plt.figure(figsize=(12, 7))
        plot_parameter_sensitivity(
            param_values,
            mean_fitness,
            std_fitness,
            param_name.replace('_', ' ').title(),
            title=f"Parameter Sensitivity Analysis: {algorithm} on {problem.capitalize()}"
        )
        
        # The function creates its own figure, so we need to get the current figure
        fig = plt.gcf()
        
        # Add best parameter annotation
        best_idx = np.argmin(mean_fitness)
        best_param = param_values[best_idx]
        best_fitness = mean_fitness[best_idx]
        
        ax = fig.gca()
        ax.axvline(x=best_param, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Best Parameter')
        ax.legend(fontsize=11)
        
        textstr = f'Optimal: {param_name}={best_param}\nFitness: {best_fitness:.6f}'
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
        ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        return fig
    
    def _create_landscape_plot(self, algo_func, algo_params, problem_func, bounds,
                              max_iter, algorithm, problem):
        """
        4. 3D SURFACE PLOTS (ADVANCED)
        Shows the objective function landscape with the algorithm's solution
        """
        # Run algorithm on 2D problem to get solution
        params_2d = {
            'dim': 2,
            'bounds': bounds,
            'max_iter': max_iter,
            **algo_params
        }
        
        result = algo_func(objective_func=problem_func, seed=self.seed, **params_2d)
        best_position = result['best_position'] if isinstance(result, dict) else result.best_position
        
        # Create figure with 3D surface and contour
        fig = plt.figure(figsize=(18, 7))
        fig.suptitle(f"3D Landscape: {algorithm} on {problem.capitalize()} Function", 
                    fontsize=16, fontweight='bold')
        
        # Prepare mesh grid
        lower, upper = bounds
        resolution = 50
        x = np.linspace(lower, upper, resolution)
        y = np.linspace(lower, upper, resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = problem_func([X[i, j], Y[i, j]])
        
        # Left: 3D Surface
        ax1 = fig.add_subplot(121, projection='3d')
        
        surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, 
                               linewidth=0, antialiased=True, edgecolor='none')
        
        # Mark best solution
        if len(best_position) >= 2:
            z_best = problem_func(best_position[:2])
            ax1.scatter([best_position[0]], [best_position[1]], [z_best],
                       color='red', s=300, marker='*', 
                       edgecolors='black', linewidths=2,
                       label=f'{algorithm} Solution', zorder=10)
            ax1.legend(fontsize=11, loc='upper left')
        
        ax1.set_xlabel('X', fontsize=12, labelpad=10)
        ax1.set_ylabel('Y', fontsize=12, labelpad=10)
        ax1.set_zlabel('f(X, Y)', fontsize=12, labelpad=10)
        ax1.set_title('3D Surface View', fontsize=14)
        ax1.view_init(elev=30, azim=45)
        
        # Add colorbar
        cbar1 = fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, pad=0.1)
        cbar1.set_label('Fitness Value', fontsize=11)
        
        # Right: Contour plot
        ax2 = fig.add_subplot(122)
        
        contourf = ax2.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
        contour = ax2.contour(X, Y, Z, levels=15, colors='black', 
                             alpha=0.3, linewidths=0.5)
        
        # Add colorbar
        cbar2 = plt.colorbar(contourf, ax=ax2, label='Fitness Value')
        
        # Mark best solution
        if len(best_position) >= 2:
            ax2.scatter([best_position[0]], [best_position[1]],
                       color='red', s=400, marker='*',
                       edgecolors='black', linewidths=2.5,
                       label=f'{algorithm} Solution', zorder=10)
            
            # Add crosshair
            ax2.axhline(y=best_position[1], color='red', linestyle='--', 
                       linewidth=1, alpha=0.5)
            ax2.axvline(x=best_position[0], color='red', linestyle='--', 
                       linewidth=1, alpha=0.5)
            
            # Add text annotation
            ax2.text(best_position[0], best_position[1], 
                    f'  ({best_position[0]:.3f}, {best_position[1]:.3f})',
                    fontsize=10, color='darkred', fontweight='bold')
        
        ax2.set_xlabel('X', fontsize=12)
        ax2.set_ylabel('Y', fontsize=12)
        ax2.set_title('Contour View', fontsize=14)
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_aspect('equal')
        
        plt.tight_layout()
        return fig
import numpy as np
import time
from typing import Dict, List, Tuple

# Import algorithms
from algorithms.swarm.PSO import run_pso
from algorithms.swarm.ABC import run_abc
from algorithms.swarm.FA import run_fa
from algorithms.swarm.CS import run_cs
from algorithms.traditional.simulated_annealing import run_simulated_annealing
from algorithms.traditional.genetic_algorithm import run_ga
from algorithms.traditional.hill_climbing import run_hill_climbing

# Import problems
from problems.continuous import get_problem

class VisualizationRunner:
    """Runner for individual algorithm visualization - computation only."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        
    def run_visualization_analysis(self, algorithm: str, problem: str, 
                                   dim: int, max_iter: int, n_runs: int,
                                   algo_specific_params: Dict = None,
                                   problem_specific_params: Dict = None,
                                   progress_callback=None) -> Dict:
        """
        Run comprehensive visualization analysis and return raw data for plotting.
        """
        # Get algorithm function and parameters
        algo_func, algo_params = self._get_algorithm(algorithm, algo_specific_params)
        problem_func, problem_info = get_problem(problem, dim)
        bounds = problem_info['bounds']
        
        if progress_callback: progress_callback("Calculating convergence...")
        convergence_data = self._get_convergence_data(
            algo_func, algo_params, problem_func, bounds, dim, max_iter
        )
        
        if progress_callback: progress_callback("Assessing performance...")
        performance_data = self._get_performance_data(
            algo_func, algo_params, problem_func, bounds, dim, max_iter, n_runs
        )
        
        if progress_callback: progress_callback("Analyzing sensitivity...")
        sensitivity_data = self._get_sensitivity_data(
            algorithm, algo_func, algo_params, problem_func, bounds, dim, max_iter
        )
        
        if progress_callback: progress_callback("Computing landscape...")
        landscape_data = self._get_landscape_data(
            algo_func, algo_params, problem_func, bounds, max_iter
        )
        
        return {
            'convergence': convergence_data,
            'performance': performance_data,
            'sensitivity': sensitivity_data,
            'landscape': landscape_data,
            'metadata': {
                'algorithm': algorithm,
                'problem': problem,
                'dim': dim,
                'n_runs': n_runs
            }
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
            'PSO': run_pso, 'ABC': run_abc, 'FA': run_fa, 'CS': run_cs
        }
        
        algo_func = func_map[algorithm]
        
        params = default_params.get(algorithm, {}).copy()
        if algo_specific_params:
            params.update(algo_specific_params)
            
        return algo_func, params
    
    def _get_convergence_data(self, algo_func, algo_params, problem_func, bounds, 
                              dim, max_iter):
        """1. CONVERGENCE ABILITY DATA"""
        params = {'dim': dim, 'bounds': bounds, 'max_iter': max_iter, **algo_params}
        result = algo_func(objective_func=problem_func, seed=self.seed, **params)
        history = result['history'] if isinstance(result, dict) else result.history
        return {'history': history}
    
    def _get_performance_data(self, algo_func, algo_params, problem_func, bounds,
                              dim, max_iter, n_runs):
        """2. COMPARATIVE PERFORMANCE DATA"""
        params = {'dim': dim, 'bounds': bounds, 'max_iter': max_iter, **algo_params}
        
        all_histories = []
        best_fitnesses = []
        
        for run in range(n_runs):
            result = algo_func(objective_func=problem_func, seed=self.seed + run, **params)
            history = result['history'] if isinstance(result, dict) else result.history
            all_histories.append(history)
            best_fitnesses.append(history[-1])
            
        return {'all_histories': all_histories, 'best_fitnesses': best_fitnesses}
    
    def _get_sensitivity_data(self, algorithm, algo_func, algo_params, problem_func,
                              bounds, dim, max_iter):
        """3. PARAMETER SENSITIVITY ANALYSIS DATA"""
        param_ranges = {
            'PSO': ('n_particles', [10, 20, 30, 40, 50]),
            'ABC': ('n_bees', [10, 20, 30, 40, 50]),
            'FA': ('n_fireflies', [10, 15, 20, 25, 30]),
            'CS': ('n_nests', [10, 15, 20, 25, 30])
        }
        
        param_name, param_values = param_ranges.get(algorithm, ('n_particles', [10, 20, 30, 40, 50]))
        
        mean_fitness = []
        std_fitness = []
        
        for val in param_values:
            current_algo_params = algo_params.copy()
            current_algo_params[param_name] = val
            params = {'dim': dim, 'bounds': bounds, 'max_iter': max_iter, **current_algo_params}
            
            fitnesses = []
            for run in range(5):  # 5 runs per parameter value
                result = algo_func(objective_func=problem_func, seed=self.seed + run, **params)
                fitness = result['best_fitness'] if isinstance(result, dict) else result.best_fitness
                fitnesses.append(fitness)
            
            mean_fitness.append(np.mean(fitnesses))
            std_fitness.append(np.std(fitnesses))
            
        return {
            'param_name': param_name,
            'param_values': param_values,
            'mean_fitness': mean_fitness,
            'std_fitness': std_fitness
        }
    
    def _get_landscape_data(self, algo_func, algo_params, problem_func, bounds,
                            max_iter):
        """4. 3D LANDSCAPE DATA"""
        params_2d = {'dim': 2, 'bounds': bounds, 'max_iter': max_iter, **algo_params}
        result = algo_func(objective_func=problem_func, seed=self.seed, **params_2d)
        best_position = result['best_position'] if isinstance(result, dict) else result.best_position
        
        lower, upper = bounds
        resolution = 50
        x = np.linspace(lower, upper, resolution)
        y = np.linspace(lower, upper, resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = problem_func([X[i, j], Y[i, j]])
                
        return {
            'best_position': best_position,
            'X': X, 'Y': Y, 'Z': Z
        }
import numpy as np
import time
from typing import Dict, List, Tuple

# Import algorithms
from algorithms.swarm.PSO import run_pso
from algorithms.swarm.ABC import run_abc
from algorithms.swarm.FA import run_fa
from algorithms.swarm.CS import run_cs

# Import problems
from problems.continuous import get_problem
from config.experiment_config import (
    PARAMETER_RANGES, DEFAULT_PSO_CONFIG, DEFAULT_ABC_CONFIG,
    DEFAULT_FA_CONFIG, DEFAULT_CS_CONFIG, ExperimentConfig
)

class VisualizationRunner:
    """Runner for individual algorithm visualization - computation only."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        
    def run_visualization_analysis(self, exp_config: ExperimentConfig,
                                   sensitivity_params: List[str] = None,
                                   progress_callback=None) -> Dict:
        """
        Run comprehensive visualization analysis and return raw data for plotting.
        """
        # Extract parameters from the experiment configuration
        problem_config = exp_config.problem
        algo_config = exp_config.algorithms[0]  # Visualization uses one algorithm
        
        algorithm = algo_config.name
        problem = problem_config.name
        dim = problem_config.dim
        max_iter = problem_config.max_iter
        n_runs = exp_config.n_runs
        algo_specific_params = algo_config.params

        # Get algorithm function and parameters
        algo_func, algo_params = self._get_algorithm(algorithm, algo_specific_params)
        problem_func, problem_info = get_problem(problem, dim)
        bounds = problem_info['bounds']
        
        if progress_callback: progress_callback("Calculating convergence...")
        convergence_data = self._get_convergence_data(
            algo_func, algo_params, problem_func, bounds, dim, max_iter
        )
        
        performance_data = self._get_performance_data(
            algo_func, algo_params, problem_func, bounds, dim, max_iter, n_runs,
            progress_callback
        )
        
        sensitivity_data = self._get_sensitivity_data(
            algorithm, algo_func, algo_params, problem_func, bounds, dim, max_iter, sensitivity_params,
            progress_callback
        )
        
        landscape_data = self._get_landscape_data(
            algo_func, algo_params, problem_func, bounds, max_iter,
            progress_callback
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
        default_configs = {
            'PSO': DEFAULT_PSO_CONFIG,
            'ABC': DEFAULT_ABC_CONFIG,
            'FA': DEFAULT_FA_CONFIG,
            'CS': DEFAULT_CS_CONFIG
        }
        
        func_map = {
            'PSO': run_pso, 'ABC': run_abc, 'FA': run_fa, 'CS': run_cs
        }
        
        algo_func = func_map[algorithm]
        
        # Use the params from the default config object
        params = default_configs.get(algorithm).params.copy()
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
                              dim, max_iter, n_runs, progress_callback=None):
        """2. COMPARATIVE PERFORMANCE DATA"""
        params = {'dim': dim, 'bounds': bounds, 'max_iter': max_iter, **algo_params}
        
        all_histories = []
        best_fitnesses = []
        
        for run in range(n_runs):
            if progress_callback:
                progress_callback(f"Assessing performance... (run {run + 1}/{n_runs})")
            result = algo_func(objective_func=problem_func, seed=self.seed + run, **params)
            history = result['history'] if isinstance(result, dict) else result.history
            all_histories.append(history)
            best_fitnesses.append(history[-1])
            
        return {'all_histories': all_histories, 'best_fitnesses': best_fitnesses}
    
    def _get_sensitivity_data(self, algorithm, algo_func, algo_params, problem_func,
                              bounds, dim, max_iter, sensitivity_params: List[str] = None,
                              progress_callback=None):
        """3. PARAMETER SENSITIVITY ANALYSIS DATA"""
        if not sensitivity_params:
            return {}

        results = {}
        
        total_params = len(sensitivity_params)
        for i, param_name in enumerate(sensitivity_params):
            param_range_info = PARAMETER_RANGES.get(algorithm, {}).get(param_name)
            
            if not param_range_info:
                continue

            param_values = None
            actual_param_name = param_name

            # Special handling for limit_factor
            if param_name == 'limit_factor' and algorithm == 'ABC':
                if isinstance(param_range_info, list):
                    n_bees = algo_params.get('n_bees', 30)
                    param_values = [int(f * dim * n_bees) for f in param_range_info]
                    actual_param_name = 'limit' # We are actually modifying the 'limit' param
            else:
                # Standard parameter handling
                if isinstance(param_range_info, tuple) and len(param_range_info) == 2:
                    param_values = np.linspace(param_range_info[0], param_range_info[1], 5)
                    if 'n_' in param_name or 'pop_size' in param_name:
                        param_values = np.round(param_values).astype(int)
                elif isinstance(param_range_info, list):
                    param_values = param_range_info
            
            if param_values is None:
                continue

            mean_fitness = []
            std_fitness = []

            total_values = len(param_values)
            for j, val in enumerate(param_values):
                if progress_callback:
                    progress_callback(
                        f"Analyzing sensitivity for '{param_name}' ({i+1}/{total_params}): "
                        f"value {j+1}/{total_values}"
                    )
                current_algo_params = algo_params.copy()
                current_algo_params[actual_param_name] = val
                params = {'dim': dim, 'bounds': bounds, 'max_iter': max_iter, **current_algo_params}
                
                fitnesses = []
                for run in range(5):  # 5 runs per parameter value
                    result = algo_func(objective_func=problem_func, seed=self.seed + run, **params)
                    fitness = result['best_fitness'] if isinstance(result, dict) else result.best_fitness
                    fitnesses.append(fitness)
                
                mean_fitness.append(np.mean(fitnesses))
                std_fitness.append(np.std(fitnesses))
            
            results[param_name] = {
                'param_values': param_values,
                'mean_fitness': mean_fitness,
                'std_fitness': std_fitness
            }
            
        return results
    
    def _get_landscape_data(self, algo_func, algo_params, problem_func, bounds,
                            max_iter, progress_callback=None):
        """4. 3D LANDSCAPE DATA"""
        if progress_callback: progress_callback("Computing landscape (running algorithm)...")
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
            if progress_callback:
                progress_callback(f"Computing landscape... (row {i + 1}/{resolution})")
            for j in range(resolution):
                Z[i, j] = problem_func([X[i, j], Y[i, j]])
                
        return {
            'best_position': best_position,
            'X': X, 'Y': Y, 'Z': Z
        }
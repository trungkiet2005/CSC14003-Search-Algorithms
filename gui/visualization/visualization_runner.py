import numpy as np
import time
from typing import Dict, List, Tuple

# Import algorithm classes
from algorithms.swarm.PSO import PSO
from algorithms.swarm.ABC import ABC
from algorithms.swarm.FA import FA
from algorithms.swarm.CS import CS
from algorithms.swarm.ACO import ACO

from algorithms.base import generate_initial_population, OptimizationResult

# Import problems
from problems.continuous import get_problem
from problems.tsp import create_tsp_problem
from config.experiment_config import (
    PARAMETER_RANGES, DEFAULT_PSO_CONFIG, DEFAULT_ABC_CONFIG,
    DEFAULT_FA_CONFIG, DEFAULT_CS_CONFIG, DEFAULT_ACO_CONFIG, ExperimentConfig
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
        if exp_config.problem.name == 'tsp':
            return self._run_tsp_visualization_analysis(exp_config, sensitivity_params, progress_callback)
            
        problem_config = exp_config.problem
        algo_config = exp_config.algorithms[0]
        
        algorithm_class, algo_params = self._get_algorithm(algo_config.name, algo_config.params)
        problem_func, problem_info = get_problem(problem_config.name, problem_config.dim)
        bounds = problem_info['bounds']

        pop_size = algo_params.get('population_size', 30)
        
        initial_pop = generate_initial_population(
            problem_config.dim, bounds, pop_size, self.seed, avoid_origin_radius=(bounds[1] - bounds[0]) * 0.5
        )
        
        if progress_callback: progress_callback("Calculating convergence...")
        convergence_data = self._get_convergence_data(
            algorithm_class, algo_params, problem_func, bounds, problem_config.dim, problem_config.max_iter,
            initial_population=initial_pop
        )
        
        performance_data = self._get_performance_data(
            algorithm_class, algo_params, problem_func, bounds, problem_config.dim, problem_config.max_iter, exp_config.n_runs,
            progress_callback
        )
        
        sensitivity_data = self._get_sensitivity_data(
            algo_config.name, algorithm_class, algo_params, problem_func, bounds, problem_config.dim, problem_config.max_iter, sensitivity_params,
            progress_callback
        )
        
        landscape_data = self._get_landscape_data(
            algorithm_class, algo_params, problem_func, bounds, problem_config.max_iter,
            progress_callback, pop_size=pop_size, avoid_radius=(bounds[1] - bounds[0]) * 0.5
        )
        
        return {
            'convergence': convergence_data,
            'performance': performance_data,
            'sensitivity': sensitivity_data,
            'landscape': landscape_data,
            'metadata': {
                'algorithm': algo_config.name,
                'problem': problem_config.name,
                'dim': problem_config.dim,
                'n_runs': exp_config.n_runs
            }
        }
    
    def _get_algorithm(self, algorithm: str, algo_specific_params: Dict = None) -> Tuple[type, Dict]:
        """Get algorithm class and parameters, merging defaults with specific ones."""
        default_configs = {
            'PSO': DEFAULT_PSO_CONFIG, 'ABC': DEFAULT_ABC_CONFIG, 'FA': DEFAULT_FA_CONFIG,
            'CS': DEFAULT_CS_CONFIG, 'ACO': DEFAULT_ACO_CONFIG
        }
        class_map = {
            'PSO': PSO, 'ABC': ABC, 'FA': FA, 'CS': CS, 'ACO': ACO
        }
        
        algo_class = class_map[algorithm]
        params = default_configs.get(algorithm).params.copy()
        if algo_specific_params:
            params.update(algo_specific_params)
            
        return algo_class, params
    
    def _get_convergence_data(self, algo_class, algo_params, problem_func, bounds, 
                              dim, max_iter, initial_population=None):
        instance = algo_class(seed=self.seed, **algo_params)
        result = instance.optimize(
            objective_func=problem_func, dim=dim, bounds=bounds, max_iter=max_iter,
            initial_population=initial_population
        )
        return {'history': result.history}
    
    def _get_performance_data(self, algo_class, algo_params, problem_func, bounds,
                              dim, max_iter, n_runs, progress_callback=None):
        all_histories, best_fitnesses = [], []
        for run in range(n_runs):
            if progress_callback:
                progress_callback(f"Assessing performance... (run {run + 1}/{n_runs})")
            instance = algo_class(seed=self.seed + run, **algo_params)
            result = instance.optimize(
                objective_func=problem_func, dim=dim, bounds=bounds, max_iter=max_iter
            )
            all_histories.append(result.history)
            best_fitnesses.append(result.best_fitness)
        return {'all_histories': all_histories, 'best_fitnesses': best_fitnesses}
    
    def _get_sensitivity_data(self, algorithm_name, algo_class, algo_params, problem_func,
                              bounds, dim, max_iter, sensitivity_params: List[str] = None,
                              progress_callback=None):
        if not sensitivity_params: return {}
        results = {}
        total_params = len(sensitivity_params)
        for i, param_name in enumerate(sensitivity_params):
            param_range_info = PARAMETER_RANGES.get(algorithm_name, {}).get(param_name)
            if not param_range_info: continue

            param_values, actual_param_name = None, param_name
            if param_name == 'limit_factor' and algorithm_name == 'ABC':
                if isinstance(param_range_info, list):
                    pop_size = algo_params.get('population_size', 30)
                    param_values = [int(f * dim * pop_size) for f in param_range_info]
                    actual_param_name = 'limit'
            elif isinstance(param_range_info, tuple) and len(param_range_info) == 2:
                param_values = np.linspace(param_range_info[0], param_range_info[1], 5)
                if param_name == 'population_size':
                    param_values = np.round(param_values).astype(int)
            elif isinstance(param_range_info, list):
                param_values = param_range_info
            
            if param_values is None: continue

            mean_fitness, std_fitness = [], []
            total_values = len(param_values)
            for j, val in enumerate(param_values):
                if progress_callback:
                    progress_callback(f"Analyzing sensitivity for '{param_name}' ({i+1}/{total_params}): value {j+1}/{total_values}")
                
                current_algo_params = {**algo_params, actual_param_name: val}
                fitnesses = []
                for run in range(5):
                    instance = algo_class(seed=self.seed + run, **current_algo_params)
                    result = instance.optimize(objective_func=problem_func, dim=dim, bounds=bounds, max_iter=max_iter)
                    fitnesses.append(result.best_fitness)
                
                mean_fitness.append(np.mean(fitnesses))
                std_fitness.append(np.std(fitnesses))
            
            results[param_name] = {'param_values': param_values, 'mean_fitness': mean_fitness, 'std_fitness': std_fitness}
        return results
    
    def _get_landscape_data(self, algo_class, algo_params, problem_func, bounds,
                            max_iter, progress_callback=None, pop_size=30, avoid_radius=0.0):
        if progress_callback: progress_callback("Computing landscape (running algorithm)...")
        
        initial_pop_2d = generate_initial_population(2, bounds, pop_size, self.seed, avoid_radius)
        instance = algo_class(seed=self.seed, **algo_params)
        result = instance.optimize(objective_func=problem_func, dim=2, bounds=bounds, max_iter=max_iter, initial_population=initial_pop_2d)
        
        resolution = 50
        x = np.linspace(bounds[0], bounds[1], resolution)
        y = np.linspace(bounds[0], bounds[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.array([problem_func([X[i, j], Y[i, j]]) for i in range(resolution) for j in range(resolution)]).reshape(resolution, resolution)
                
        return {'best_position': result.best_position, 'best_fitness': result.best_fitness, 'X': X, 'Y': Y, 'Z': Z}

    def _run_tsp_visualization_analysis(self, exp_config: ExperimentConfig,
                                      sensitivity_params: List[str] = None,
                                      progress_callback=None) -> Dict:
        problem_config = exp_config.problem
        algo_config = exp_config.algorithms[0]
        
        algo_class, algo_params = self._get_algorithm(algo_config.name, algo_config.params)
        tsp_problem = create_tsp_problem(n_cities=problem_config.dim, seed=self.seed)

        if progress_callback: progress_callback("Calculating convergence for TSP...")
        convergence_data = self._get_tsp_convergence_data(algo_class, algo_params, tsp_problem['distance_matrix'], problem_config.max_iter)
        
        performance_data = self._get_tsp_performance_data(algo_class, algo_params, problem_config.dim, problem_config.max_iter, exp_config.n_runs, progress_callback)
        
        sensitivity_data = self._get_tsp_sensitivity_data(algo_config.name, algo_class, algo_params, problem_config.dim, problem_config.max_iter, sensitivity_params, progress_callback)
        
        return {
            'convergence': convergence_data, 'performance': performance_data, 'sensitivity': sensitivity_data,
            'landscape': "Not applicable for TSP.",
            'metadata': {'algorithm': algo_config.name, 'problem': problem_config.name, 'dim': problem_config.dim, 'n_runs': exp_config.n_runs}
        }

    def _get_tsp_convergence_data(self, algo_class, algo_params, distance_matrix, max_iter):
        instance = algo_class(seed=self.seed, **algo_params)
        result = instance.optimize_tsp(distance_matrix=distance_matrix, max_iter=max_iter)
        return {'history': result.history}

    def _get_tsp_performance_data(self, algo_class, algo_params, n_cities, max_iter, n_runs, progress_callback=None):
        all_histories, best_fitnesses = [], []
        for run in range(n_runs):
            if progress_callback:
                progress_callback(f"Assessing TSP performance... (run {run + 1}/{n_runs})")
            tsp_problem = create_tsp_problem(n_cities=n_cities, seed=self.seed + run)
            instance = algo_class(seed=self.seed + run, **algo_params)
            result = instance.optimize_tsp(distance_matrix=tsp_problem['distance_matrix'], max_iter=max_iter)
            all_histories.append(result.history)
            best_fitnesses.append(result.best_fitness)
        return {'all_histories': all_histories, 'best_fitnesses': best_fitnesses}

    def _get_tsp_sensitivity_data(self, algorithm_name, algo_class, algo_params, n_cities, max_iter, sensitivity_params: List[str] = None, progress_callback=None):
        if not sensitivity_params: return {}
        results = {}
        total_params = len(sensitivity_params)
        for i, param_name in enumerate(sensitivity_params):
            param_range_info = PARAMETER_RANGES.get(algorithm_name, {}).get(param_name)
            if not param_range_info: continue

            param_values = None
            if isinstance(param_range_info, tuple) and len(param_range_info) == 2:
                param_values = np.linspace(param_range_info[0], param_range_info[1], 5)
                if param_name == 'population_size':
                    param_values = np.round(param_values).astype(int)
            elif isinstance(param_range_info, list):
                param_values = param_range_info
            
            if param_values is None: continue

            mean_fitness, std_fitness = [], []
            total_values = len(param_values)
            for j, val in enumerate(param_values):
                if progress_callback:
                    progress_callback(f"Analyzing TSP sensitivity for '{param_name}' ({i+1}/{total_params}): value {j+1}/{total_values}")
                
                current_algo_params = {**algo_params, param_name: val}
                fitnesses = []
                for run in range(5):
                    tsp_problem = create_tsp_problem(n_cities=n_cities, seed=self.seed + run)
                    instance = algo_class(seed=self.seed + run, **current_algo_params)
                    result = instance.optimize_tsp(distance_matrix=tsp_problem['distance_matrix'], max_iter=max_iter)
                    fitnesses.append(result.best_fitness)
                
                mean_fitness.append(np.mean(fitnesses))
                std_fitness.append(np.std(fitnesses))
            
            results[param_name] = {'param_values': param_values, 'mean_fitness': mean_fitness, 'std_fitness': std_fitness}
        return results
import numpy as np
import time
import tracemalloc
from typing import Dict, List, Tuple, Callable

# Import algorithm classes
from algorithms.swarm.PSO import PSO
from algorithms.swarm.ABC import ABC
from algorithms.swarm.FA import FA
from algorithms.swarm.CS import CS
from algorithms.swarm.ACO import ACO
from algorithms.traditional.simulated_annealing import SimulatedAnnealing, SimulatedAnnealingTSP
from algorithms.traditional.genetic_algorithm import GeneticAlgorithm
from algorithms.traditional.hill_climbing import HillClimbing

# Import problems
from problems.continuous import get_problem
from problems.tsp import create_tsp_problem

# Import utilities
from utils.benchmark import BenchmarkRunner
from config.experiment_config import ExperimentConfig

def _create_continuous_runner(algo_class: type, constructor_params: Dict) -> Callable:
    """Creates a runner function for a continuous optimizer class."""
    def runner(objective_func: Callable, seed: int, minimize: bool, initial_population: np.ndarray, **optimizer_params: Dict) -> Dict:
        instance = algo_class(seed=seed, **constructor_params)
        return instance.optimize(
            objective_func=objective_func,
            minimize=minimize,
            initial_population=initial_population,
            **optimizer_params
        )
    return runner

class ComparisonRunner:
    """Runner for algorithm comparison - computation only."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.benchmark_runner = BenchmarkRunner(seed=seed, verbose=False)
        
    def run_continuous_comparison(self, exp_config: ExperimentConfig, progress_callback=None) -> Dict:
        """Run comparison for continuous optimization and return raw data."""
        problem_config = exp_config.problem
        
        problem_func, problem_info = get_problem(problem_config.name, problem_config.dim)
        bounds = problem_info['bounds']
        
        class_map = {
            'PSO': PSO, 'HC': HillClimbing, 'ABC': ABC, 'GA': GeneticAlgorithm,
            'FA': FA, 'SA': SimulatedAnnealing, 'CS': CS
        }
        
        algo_dict = {}
        for algo_config in exp_config.algorithms:
            if algo_config.enabled and algo_config.name in class_map:
                constructor_params = algo_config.params
                optimizer_params = {
                    'dim': problem_config.dim,
                    'bounds': bounds,
                    'max_iter': problem_config.max_iter,
                }
                runner_func = _create_continuous_runner(class_map[algo_config.name], constructor_params)
                algo_dict[algo_config.name] = (runner_func, optimizer_params)

        if progress_callback: progress_callback("Running main benchmark...")
        _, stats_list = self.benchmark_runner.compare_algorithms(
            algo_dict, problem_func, problem_config.name, problem_config.dim, 
            n_runs=exp_config.n_runs, progress_callback=progress_callback
        )
        
        scalability_data = self._get_scalability_data(
            exp_config, progress_callback
        )
        
        return {
            'stats_list': stats_list,
            'scalability_data': scalability_data,
            'metadata': {'problem': problem_config.name, 'dim': problem_config.dim}
        }
    
    def run_tsp_comparison(self, n_cities: int, max_iter: int, n_runs: int = 1, 
                           algo_params: Dict = None, progress_callback=None) -> Dict:
        """Run comparison for TSP and return raw data."""
        if algo_params is None:
            algo_params = {}
            
        tsp = create_tsp_problem(n_cities, seed=self.seed)
        
        algorithms = {
            'ACO': (ACO, algo_params.get('ACO', {})),
            'SA': (SimulatedAnnealingTSP, algo_params.get('SA', {}))
        }
        
        results = {}
        total_algos = len(algorithms)
        for i, (algo_name, (algo_class, user_params)) in enumerate(algorithms.items()):
            fitnesses, times, mems, best_result, best_distance = [], [], [], None, float('inf')

            for run in range(n_runs):
                if progress_callback:
                    progress_callback(f"Running {algo_name} ({i+1}/{total_algos}): Run {run + 1}/{n_runs}")

                tracemalloc.start()
                start = time.perf_counter()
                
                instance = algo_class(seed=self.seed + run, **user_params)
                result = instance.optimize_tsp(tsp['distance_matrix'], max_iter=max_iter)
                
                _, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                distance = result.best_fitness
                fitnesses.append(distance)
                times.append(time.perf_counter() - start)
                mems.append(peak_mem / (1024 * 1024))

                if distance < best_distance:
                    best_distance = distance
                    best_result = result
            
            results[algo_name] = {
                'mean_distance': np.mean(fitnesses), 'std_distance': np.std(fitnesses),
                'best_distance': np.min(fitnesses), 'mean_time': np.mean(times),
                'mean_mem': np.mean(mems), 'std_mem': np.std(mems),
                'best_result': best_result, 'all_distances': fitnesses,
                'best_route': best_result.best_position
            }
        
        scalability_data = self._get_tsp_scalability_data(algorithms, max_iter, progress_callback)
        
        return {
            'main_results': results,
            'scalability_data': scalability_data,
            'metadata': {'cities': tsp['cities'], 'n_cities': n_cities}
        }
    
    def _get_scalability_data(self, exp_config: ExperimentConfig, progress_callback=None):
        """Get scalability analysis data."""
        dims = [5, 10, 20, 30, 50]
        
        class_map = {
            'PSO': PSO, 'HC': HillClimbing, 'ABC': ABC, 'GA': GeneticAlgorithm,
            'FA': FA, 'SA': SimulatedAnnealing, 'CS': CS
        }
        
        algo_names = [algo.name for algo in exp_config.algorithms if algo.enabled and algo.name in class_map]
        scalability_data = {name: {'dims': [], 'fitness': [], 'times': []} for name in algo_names}
        
        total_dims = len(dims)
        for i, dim in enumerate(dims):
            if progress_callback:
                progress_callback(f"Analyzing scalability... (dimension {dim}, {i+1}/{total_dims})")

            problem_func, problem_info = get_problem(exp_config.problem.name, dim)
            bounds = problem_info['bounds']
            
            algo_dict = {}
            for algo_config in exp_config.algorithms:
                if algo_config.enabled and algo_config.name in class_map:
                    constructor_params = algo_config.params
                    optimizer_params = {
                        'dim': dim,
                        'bounds': bounds,
                        'max_iter': exp_config.problem.max_iter,
                    }
                    runner_func = _create_continuous_runner(class_map[algo_config.name], constructor_params)
                    algo_dict[algo_config.name] = (runner_func, optimizer_params)

            _, stats_list = self.benchmark_runner.compare_algorithms(
                algo_dict, problem_func, exp_config.problem.name, dim, n_runs=exp_config.n_runs,
                progress_callback=lambda msg: progress_callback(f"Scalability (dim={dim}): {msg}") if progress_callback else None
            )
            
            for stats in stats_list:
                scalability_data[stats.algorithm_name]['dims'].append(dim)
                scalability_data[stats.algorithm_name]['fitness'].append(stats.best_fitness)
                scalability_data[stats.algorithm_name]['times'].append(stats.mean_time)
                
        return scalability_data

    def _get_tsp_scalability_data(self, algorithms: Dict, max_iter: int, progress_callback=None):
        """Get TSP scalability data."""
        city_counts = [10, 20, 30, 40, 50]
        scalability_data = {name: {'cities': [], 'distances': [], 'times': [], 'mems': []} for name in algorithms.keys()}
        
        total_cities = len(city_counts)
        for i, n_cities in enumerate(city_counts):
            if progress_callback:
                progress_callback(f"Analyzing TSP scalability... (cities: {n_cities}, {i+1}/{total_cities})")

            tsp = create_tsp_problem(n_cities, seed=self.seed)
            
            for algo_name, (algo_class, base_params) in algorithms.items():
                tracemalloc.start()
                start = time.perf_counter()
                
                instance = algo_class(seed=self.seed, **base_params)
                result = instance.optimize_tsp(tsp['distance_matrix'], max_iter=max_iter)
                
                _, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                scalability_data[algo_name]['cities'].append(n_cities)
                scalability_data[algo_name]['distances'].append(result.best_fitness)
                scalability_data[algo_name]['times'].append(time.perf_counter() - start)
                scalability_data[algo_name]['mems'].append(peak_mem / (1024 * 1024))
                
        return scalability_data
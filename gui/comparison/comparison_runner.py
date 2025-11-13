import numpy as np
import time
import tracemalloc
from typing import Dict, List, Tuple

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
from config.experiment_config import ExperimentConfig

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
        
        func_map = {
            'PSO': run_pso, 'HC': run_hill_climbing, 'ABC': run_abc, 'GA': run_ga,
            'FA': run_fa, 'SA': run_simulated_annealing, 'CS': run_cs
        }
        
        # Create the dictionary of algorithms to be compared from the ExperimentConfig
        algo_dict = {}
        for algo_config in exp_config.algorithms:
            if algo_config.enabled:
                params = {
                    'dim': problem_config.dim,
                    'bounds': bounds,
                    'max_iter': problem_config.max_iter,
                    **algo_config.params
                }
                algo_dict[algo_config.name] = (func_map[algo_config.name], params)

        if progress_callback: progress_callback("Running main benchmark...")
        _, stats_list = self.benchmark_runner.compare_algorithms(
            algo_dict, problem_func, problem_config.name, problem_config.dim, 
            n_runs=exp_config.n_runs, progress_callback=progress_callback
        )
        
        scalability_data = self._get_scalability_data(
            algo_dict, problem_config.name, problem_config.max_iter, exp_config.n_runs,
            progress_callback
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
        
        from algorithms.swarm.ACO import run_aco
        
        algorithms = {
            'ACO': (run_aco, algo_params.get('ACO', {})),
            'SA': (run_simulated_annealing_tsp, algo_params.get('SA', {}))
        }
        
        results = {}
        total_algos = len(algorithms)
        for i, (algo_name, (algo_func, user_params)) in enumerate(algorithms.items()):
            fitnesses, times, mems, best_result, best_distance = [], [], [], None, float('inf')

            for run in range(n_runs):
                if progress_callback:
                    progress_callback(f"Running {algo_name} ({i+1}/{total_algos}): Run {run + 1}/{n_runs}")

                tracemalloc.start()
                start = time.perf_counter()
                run_params = {'max_iter': max_iter, 'seed': self.seed + run, **user_params}
                
                # Both ACO and SA for TSP expect the distance matrix as the first argument.
                # The 'objective' function is used by continuous optimizers, not these discrete ones.
                if algo_name in ['SA', 'ACO']:
                    result = algo_func(tsp['distance_matrix'], **run_params)
                else: 
                    # Fallback for other potential TSP algos that might use the objective function
                    result = algo_func(tsp['objective'], **run_params)
                
                _, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                distance = result['best_distance']
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
                'best_route': best_result['best_route']
            }
        
        scalability_data = self._get_tsp_scalability_data(algorithms, max_iter, progress_callback)
        
        return {
            'main_results': results,
            'scalability_data': scalability_data,
            'metadata': {'cities': tsp['cities'], 'n_cities': n_cities}
        }
    
    def _get_scalability_data(self, algo_dict, problem, max_iter, n_runs, progress_callback=None):
        """Get scalability analysis data."""
        dims = [5, 10, 20, 30, 50]
        scalability_data = {name: {'dims': [], 'fitness': [], 'times': []} for name in algo_dict.keys()}
        
        total_dims = len(dims)
        for i, dim in enumerate(dims):
            if progress_callback:
                progress_callback(f"Analyzing scalability... (dimension {dim}, {i+1}/{total_dims})")

            problem_func, problem_info = get_problem(problem, dim)
            bounds = problem_info['bounds']
            
            test_algo_dict = {
                name: (func, {**params, 'dim': dim, 'bounds': bounds, 'max_iter': max_iter})
                for name, (func, params) in algo_dict.items()
            }
            
            # Pass a simplified callback to avoid nested progress messages
            _, stats_list = self.benchmark_runner.compare_algorithms(
                test_algo_dict, problem_func, problem, dim, n_runs=n_runs,
                progress_callback=lambda msg: progress_callback(f"Scalability (dim={dim}): {msg}") if progress_callback else None
            )
            
            for stats in stats_list:
                scalability_data[stats.algorithm_name]['dims'].append(dim)
                scalability_data[stats.algorithm_name]['fitness'].append(stats.best_fitness)
                scalability_data[stats.algorithm_name]['times'].append(stats.mean_time)
                
        return scalability_data

    def _get_tsp_scalability_data(self, algorithms, max_iter, progress_callback=None):
        """Get TSP scalability data."""
        city_counts = [10, 20, 30, 40, 50]
        scalability_data = {name: {'cities': [], 'distances': [], 'times': [], 'mems': []} for name in algorithms.keys()}
        
        total_cities = len(city_counts)
        for i, n_cities in enumerate(city_counts):
            if progress_callback:
                progress_callback(f"Analyzing TSP scalability... (cities: {n_cities}, {i+1}/{total_cities})")

            tsp = create_tsp_problem(n_cities, seed=self.seed)
            
            for algo_name, (algo_func, base_params) in algorithms.items():
                tracemalloc.start()
                start = time.perf_counter()
                run_params = {**base_params, 'max_iter': max_iter, 'seed': self.seed}
                
                if algo_name in ['SA', 'ACO']:
                    result = algo_func(tsp['distance_matrix'], **run_params)
                else:
                    result = algo_func(tsp['objective'], **run_params)
                
                _, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                scalability_data[algo_name]['cities'].append(n_cities)
                scalability_data[algo_name]['distances'].append(result['best_distance'])
                scalability_data[algo_name]['times'].append(time.perf_counter() - start)
                scalability_data[algo_name]['mems'].append(peak_mem / (1024 * 1024))
                
        return scalability_data
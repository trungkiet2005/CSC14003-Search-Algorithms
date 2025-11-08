"""utils/benchmark.py - Enhanced benchmark and comparison utilities"""

import time
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import json


@dataclass
class AlgorithmStats:
    """Statistics for algorithm performance"""
    algorithm_name: str
    problem_name: str
    dimension: int
    n_runs: int
    
    # Fitness statistics
    best_fitness: float
    worst_fitness: float
    mean_fitness: float
    median_fitness: float
    std_fitness: float
    q1_fitness: float
    q3_fitness: float
    
    # Time statistics
    mean_time: float
    std_time: float
    total_time: float
    
    # Convergence statistics
    mean_convergence_iter: Optional[float] = None
    convergence_rate: Optional[float] = None
    
    # Raw data
    all_fitnesses: List[float] = None
    all_times: List[float] = None
    results: List[Any] = None
    
    def to_dict(self):
        """Convert to dictionary"""
        d = asdict(self)
        # Remove raw data for cleaner output
        d.pop('all_fitnesses', None)
        d.pop('all_times', None)
        return d


class BenchmarkRunner:
    """Run comprehensive benchmarks on optimization algorithms"""
    
    def __init__(self, seed: int = 42, verbose: bool = True):
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
    
    def run_single_experiment(self, algorithm_func: Callable, 
                             objective_func: Callable,
                             n_runs: int = 30,
                             minimize: bool = True,
                             **kwargs) -> Dict[str, Any]:
        """
        Run algorithm multiple times and collect statistics
        
        Args:
            algorithm_func: Algorithm function to run
            objective_func: Objective function
            n_runs: Number of independent runs
            minimize: True for minimization
            **kwargs: Arguments for algorithm
            
        Returns:
            Dictionary with comprehensive statistics
        """
        results = []
        times = []
        best_fitnesses = []
        convergence_iters = []
        
        iterator = tqdm(range(n_runs), desc="Running") if self.verbose else range(n_runs)
        
        for run in iterator:
            # Generate unique seed for this run
            run_seed = self.rng.integers(0, 2**31 - 1)
            
            # Time execution
            start_time = time.perf_counter()
            result = algorithm_func(objective_func=objective_func, seed=run_seed, minimize=minimize, **kwargs)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            
            # Extract results
            if isinstance(result, dict):
                best_fitness = result['best_fitness']
                history = result.get('history', [])
                convergence_iter = result.get('convergence_iter', None)
            else:
                best_fitness = result.best_fitness
                history = result.history
                convergence_iter = result.convergence_iter
            
            results.append(result)
            times.append(execution_time)
            best_fitnesses.append(best_fitness)
            if convergence_iter is not None:
                convergence_iters.append(convergence_iter)
        
        # Calculate statistics
        best_fitnesses = np.array(best_fitnesses)
        times = np.array(times)
        
        stats = {
            'results': results,
            'best_fitness': np.min(best_fitnesses) if minimize else np.max(best_fitnesses),
            'worst_fitness': np.max(best_fitnesses) if minimize else np.min(best_fitnesses),
            'mean_fitness': np.mean(best_fitnesses),
            'median_fitness': np.median(best_fitnesses),
            'std_fitness': np.std(best_fitnesses),
            'q1_fitness': np.percentile(best_fitnesses, 25),
            'q3_fitness': np.percentile(best_fitnesses, 75),
            'iqr_fitness': np.percentile(best_fitnesses, 75) - np.percentile(best_fitnesses, 25),
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'total_time': np.sum(times),
            'all_fitnesses': best_fitnesses.tolist(),
            'all_times': times.tolist()
        }
        
        # Add convergence statistics if available
        if convergence_iters:
            stats['mean_convergence_iter'] = np.mean(convergence_iters)
            stats['convergence_rate'] = len(convergence_iters) / n_runs
        
        return stats
    
    def compare_algorithms(self, algorithms: Dict[str, Tuple[Callable, Dict]],
                          objective_func: Callable,
                          problem_name: str,
                          dimension: int,
                          n_runs: int = 30,
                          minimize: bool = True,
                          **common_kwargs) -> pd.DataFrame:
        """
        Compare multiple algorithms on the same problem
        
        Args:
            algorithms: Dict of {name: (algorithm_func, algorithm_kwargs)}
            objective_func: Objective function
            problem_name: Name of the problem
            dimension: Problem dimensionality
            n_runs: Number of runs per algorithm
            minimize: True for minimization
            **common_kwargs: Common parameters for all algorithms
            
        Returns:
            DataFrame with comparison results
        """
        all_stats = []
        
        for algo_name, (algo_func, algo_kwargs) in algorithms.items():
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Running {algo_name} on {problem_name} (dim={dimension})")
                print(f"{'='*60}")
            
            # Merge kwargs
            run_kwargs = {**common_kwargs, **algo_kwargs}
            
            # Run experiments
            stats = self.run_single_experiment(
                algo_func, objective_func, n_runs, minimize, **run_kwargs
            )
            
            # Create stats object
            algo_stats = AlgorithmStats(
                algorithm_name=algo_name,
                problem_name=problem_name,
                dimension=dimension,
                n_runs=n_runs,
                best_fitness=stats['best_fitness'],
                worst_fitness=stats['worst_fitness'],
                mean_fitness=stats['mean_fitness'],
                median_fitness=stats['median_fitness'],
                std_fitness=stats['std_fitness'],
                q1_fitness=stats['q1_fitness'],
                q3_fitness=stats['q3_fitness'],
                mean_time=stats['mean_time'],
                std_time=stats['std_time'],
                total_time=stats['total_time'],
                mean_convergence_iter=stats.get('mean_convergence_iter'),
                convergence_rate=stats.get('convergence_rate'),
                all_fitnesses=stats['all_fitnesses'],
                all_times=stats['all_times'],
                results=stats['results']
            )
            
            all_stats.append(algo_stats)
        
        # Convert to DataFrame
        df = pd.DataFrame([s.to_dict() for s in all_stats])
        
        # Sort by mean fitness
        df = df.sort_values('mean_fitness', ascending=minimize)
        
        return df, all_stats


class StatisticalTests:
    """Statistical significance tests for algorithm comparison"""
    
    @staticmethod
    def wilcoxon_test(data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
        """
        Wilcoxon signed-rank test (non-parametric)
        
        Returns:
            (statistic, p_value)
        """
        from scipy.stats import wilcoxon
        stat, p_value = wilcoxon(data1, data2)
        return stat, p_value
    
    @staticmethod
    def mannwhitneyu_test(data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
        """
        Mann-Whitney U test (non-parametric)
        
        Returns:
            (statistic, p_value)
        """
        from scipy.stats import mannwhitneyu
        stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
        return stat, p_value
    
    @staticmethod
    def friedman_test(*data_groups) -> Tuple[float, float]:
        """
        Friedman test for multiple related samples
        
        Returns:
            (statistic, p_value)
        """
        from scipy.stats import friedmanchisquare
        stat, p_value = friedmanchisquare(*data_groups)
        return stat, p_value


def convergence_speed_analysis(history: List[float], 
                               threshold_percentage: float = 0.99,
                               known_optimum: Optional[float] = None) -> Dict[str, Any]:
    """
    Analyze convergence speed of an algorithm
    
    Args:
        history: List of fitness values over iterations
        threshold_percentage: Percentage of improvement to consider converged
        known_optimum: Known optimal value (if available)
        
    Returns:
        Dictionary with convergence metrics
    """
    history = np.array(history)
    n_iters = len(history)
    
    # Calculate improvement
    initial_fitness = history[0]
    final_fitness = history[-1]
    total_improvement = abs(initial_fitness - final_fitness)
    
    # Find convergence point
    if known_optimum is not None:
        threshold = known_optimum + (1 - threshold_percentage) * abs(initial_fitness - known_optimum)
    else:
        threshold = initial_fitness - threshold_percentage * total_improvement
    
    converged_at = None
    for i, fitness in enumerate(history):
        if fitness <= threshold:
            converged_at = i
            break
    
    # Calculate convergence rate
    if converged_at is not None:
        convergence_rate = converged_at / n_iters
    else:
        convergence_rate = None
    
    # Calculate average improvement per iteration
    improvements = np.diff(history)
    avg_improvement = np.mean(np.abs(improvements))
    
    return {
        'total_iterations': n_iters,
        'converged_at': converged_at,
        'convergence_rate': convergence_rate,
        'initial_fitness': float(initial_fitness),
        'final_fitness': float(final_fitness),
        'total_improvement': float(total_improvement),
        'avg_improvement_per_iter': float(avg_improvement),
        'threshold': float(threshold)
    }


def export_results(stats_list: List[AlgorithmStats], 
                  filename: str = 'results.json'):
    """Export benchmark results to JSON"""
    results = [s.to_dict() for s in stats_list]
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(filename: str = 'results.json') -> List[Dict]:
    """Load benchmark results from JSON"""
    with open(filename, 'r') as f:
        return json.load(f)
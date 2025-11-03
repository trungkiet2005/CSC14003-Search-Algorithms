"""Benchmark helper functions for algorithm comparison"""

import time
import numpy as np


def timeit(func, *args, repeat=1, **kwargs):
    """Run func and measure average execution time over `repeat` runs.

    Returns (result, avg_time_seconds).
    """
    total = 0.0
    result = None
    for _ in range(repeat):
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        total += (t1 - t0)
    return result, total / repeat


def run_multiple_times(algorithm_func, n_runs=10, seed=None, minimize=True, **kwargs):
    """Run an algorithm multiple times and collect statistics.
    
    Args:
        algorithm_func: function to run
        n_runs: number of independent runs
        seed: base random seed for reproducibility
        minimize: True for minimization, False for maximization
        **kwargs: arguments to pass to algorithm_func
    
    Returns:
        dict with statistics
    """
    results = []
    times = []
    best_fitnesses = []
    
    # Create a random number generator for seeds
    seed_rng = np.random.default_rng(seed)
    
    for run in range(n_runs):
        # Generate a new seed for each run to ensure variability
        run_seed = seed_rng.integers(low=0, high=2**32 - 1)
        
        t0 = time.time()
        result = algorithm_func(seed=run_seed, **kwargs)
        t1 = time.time()
        
        results.append(result)
        times.append(t1 - t0)
        best_fitnesses.append(result['best_fitness'])
    
    best_fitnesses = np.array(best_fitnesses)
    
    return {
        'results': results,
        'best_fitness': np.min(best_fitnesses) if minimize else np.max(best_fitnesses),
        'worst_fitness': np.max(best_fitnesses) if minimize else np.min(best_fitnesses),
        'mean_fitness': np.mean(best_fitnesses),
        'std_fitness': np.std(best_fitnesses),
        'median_fitness': np.median(best_fitnesses),
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'total_time': np.sum(times),
        'all_times': times,
        'all_fitnesses': best_fitnesses.tolist()
    }


def compare_algorithms(algorithms_dict, problem_func, n_runs=10, **problem_kwargs):
    """Compare multiple algorithms on the same problem.
    
    Args:
        algorithms_dict: dict of {name: (func, kwargs)} for each algorithm
        problem_func: objective function
        n_runs: number of runs per algorithm
        **problem_kwargs: additional arguments (dim, bounds, seed, etc.)
    
    Returns:
        dict with comparison results for each algorithm
    """
    comparison = {}
    
    # Extract seed and minimize, as they are not for the algorithm functions
    seed = problem_kwargs.pop('seed', None)
    minimize = problem_kwargs.pop('minimize', True)
    
    for name, (algo_func, algo_kwargs) in algorithms_dict.items():
        print(f"Running {name}...")
        
        # Combine problem and algorithm kwargs
        run_kwargs = {**problem_kwargs, **algo_kwargs}
        run_kwargs['objective_func'] = problem_func
        
        stats = run_multiple_times(algo_func, n_runs=n_runs, seed=seed, minimize=minimize, **run_kwargs)
        comparison[name] = stats
    
    return comparison


def convergence_speed(history, threshold=None):
    """Compute convergence speed metrics.
    
    Args:
        history: list of fitness values over iterations
        threshold: fitness threshold for convergence (if None, uses 99% of best)
    
    Returns:
        dict with convergence metrics
    """
    history = np.array(history)
    best_fitness = history[-1]
    
    if threshold is None:
        # 99% of improvement from start to end
        improvement = history[0] - best_fitness
        threshold = history[0] - 0.99 * improvement
    
    # Find iteration where threshold is reached
    converged_at = None
    for i, fitness in enumerate(history):
        if fitness <= threshold:
            converged_at = i
            break
    
    return {
        'converged_at_iteration': converged_at,
        'total_iterations': len(history),
        'convergence_rate': converged_at / len(history) if converged_at else None,
        'final_fitness': best_fitness,
        'initial_fitness': history[0]
    }


def solution_quality_metrics(best_fitness, known_optimum=None):
    """Compute solution quality metrics.
    
    Args:
        best_fitness: fitness of best solution found
        known_optimum: known optimal fitness (if available)
    
    Returns:
        dict with quality metrics
    """
    metrics = {'best_fitness': best_fitness}
    
    if known_optimum is not None:
        gap = abs(best_fitness - known_optimum)
        relative_error = gap / abs(known_optimum) if known_optimum != 0 else gap
        
        metrics['optimality_gap'] = gap
        metrics['relative_error'] = relative_error
        metrics['percent_from_optimal'] = relative_error * 100
    
    return metrics


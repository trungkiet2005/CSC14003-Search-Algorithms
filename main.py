"""main.py - Main entry point for Search Algorithms Project

Usage:
    python main.py experiment --name pso_vs_ga --problem rastrigin --dim 10
    python main.py benchmark --all
    python main.py visualize --results results/
"""

import argparse
import sys
import os
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import algorithms
from algorithms.swarm.PSO import run_pso
from algorithms.swarm.ACO import run_aco
from algorithms.swarm.ABC import run_abc
from algorithms.swarm.FA import run_fa
from algorithms.swarm.CS import run_cs

from algorithms.traditional.hill_climbing import run_hill_climbing
from algorithms.traditional.simulated_annealing import run_simulated_annealing, run_simulated_annealing_tsp
from algorithms.traditional.genetic_algorithm import run_ga

# Import problems
from problems.continuous import get_problem, CONTINUOUS_PROBLEMS
from problems.tsp import create_tsp_problem

# Import utilities
from utils.benchmark import BenchmarkRunner, convergence_speed_analysis
from utils.visualize import (
    plot_convergence_comparison, plot_boxplot_comparison,
    plot_3d_surface, plot_contour, plot_tsp_route,
    ensure_figure_dir
)


class ExperimentRunner:
    """Manage and run experiments"""
    
    def __init__(self, output_dir: str = "results", seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        self.seed = seed
        self.benchmark_runner = BenchmarkRunner(seed=seed, verbose=True)
    
    def run_continuous_experiment(self, problem_name: str, dim: int,
                                 algorithms: dict, n_runs: int = 30,
                                 max_iter: int = 100):
        """Run experiment on continuous optimization problem"""
        print(f"\n{'='*80}")
        print(f"Experiment: {problem_name.upper()} (dimension={dim})")
        print(f"{'='*80}\n")
        
        # Get problem
        problem_func, problem_info = get_problem(problem_name, dim)
        bounds = problem_info['bounds']
        
        # Prepare algorithms
        algo_dict = {}
        for algo_name, (algo_func, algo_params) in algorithms.items():
            params = {
                'dim': dim,
                'bounds': bounds,
                'max_iter': max_iter,
                **algo_params
            }
            algo_dict[algo_name] = (algo_func, params)
        
        # Run comparison
        df, stats_list = self.benchmark_runner.compare_algorithms(
            algo_dict, problem_func, problem_name, dim, n_runs=n_runs
        )
        
        # Save results
        csv_path = self.output_dir / f"{problem_name}_dim{dim}_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"RESULTS SUMMARY - {problem_name.upper()}")
        print(f"{'='*80}\n")
        print(df[['algorithm_name', 'mean_fitness', 'std_fitness', 
                 'mean_time', 'convergence_rate']].to_string(index=False))
        
        # Visualizations
        self._create_visualizations(stats_list, problem_name, dim, problem_func, bounds)
        
        return df, stats_list
    
    def run_tsp_experiment(self, n_cities: int, algorithms: dict, 
                          n_runs: int = 30, max_iter: int = 100):
        """Run experiment on TSP"""
        print(f"\n{'='*80}")
        print(f"Experiment: TSP (n_cities={n_cities})")
        print(f"{'='*80}\n")
        
        # Create TSP problem
        tsp = create_tsp_problem(n_cities, seed=self.seed)
        cities = tsp['cities']
        distance_matrix = tsp['distance_matrix']
        
        results = {}
        
        for algo_name, (algo_func, algo_params) in algorithms.items():
            print(f"\nRunning {algo_name}...")
            
            fitnesses = []
            times = []
            best_result = None
            best_distance = float('inf')
            
            for run in range(n_runs):
                import time
                start = time.time()
                
                if 'distance_matrix' in algo_func.__code__.co_varnames:
                    result = algo_func(distance_matrix, max_iter=max_iter, 
                                     seed=self.seed + run, **algo_params)
                else:
                    result = algo_func(tsp['objective'], max_iter=max_iter,
                                     seed=self.seed + run, **algo_params)
                
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
                'best_result': best_result
            }
        
        # Print results
        print(f"\n{'='*80}")
        print(f"RESULTS SUMMARY - TSP ({n_cities} cities)")
        print(f"{'='*80}\n")
        
        for algo_name, stats in results.items():
            print(f"{algo_name:20s}: Mean={stats['mean_distance']:.2f} ± {stats['std_distance']:.2f}, "
                  f"Best={stats['best_distance']:.2f}, Time={stats['mean_time']:.3f}s")
        
        # Visualize best solutions
        self._visualize_tsp_results(results, cities, n_cities)
        
        return results
    
    def _create_visualizations(self, stats_list, problem_name, dim, 
                              problem_func, bounds):
        """Create comprehensive visualizations"""
        
        # 1. Convergence comparison
        histories = {}
        for stats in stats_list:
            result = stats.results[0]  # Use first run
            if isinstance(result, dict):
                histories[stats.algorithm_name] = result['history']
            else:
                histories[stats.algorithm_name] = result.history
        
        fig1_path = self.figures_dir / f"{problem_name}_dim{dim}_convergence.png"
        plot_convergence_comparison(
            histories,
            title=f"Convergence Comparison - {problem_name.upper()} (dim={dim})",
            save_path=fig1_path,
            log_scale=True
        )
        
        # 2. Boxplot comparison
        boxplot_data = {
            stats.algorithm_name: stats.all_fitnesses 
            for stats in stats_list
        }
        fig2_path = self.figures_dir / f"{problem_name}_dim{dim}_boxplot.png"
        plot_boxplot_comparison(
            boxplot_data,
            title=f"Performance Distribution - {problem_name.upper()} (dim={dim})",
            save_path=fig2_path
        )
        
        # 3. 3D surface (if 2D problem)
        if dim == 2:
            # Get best solution from best algorithm
            best_stats = min(stats_list, key=lambda s: s.mean_fitness)
            best_result = best_stats.results[0]
            best_pos = best_result['best_position'] if isinstance(best_result, dict) else best_result.best_position
            
            fig3_path = self.figures_dir / f"{problem_name}_3d_surface.png"
            plot_3d_surface(
                problem_func, bounds, resolution=50,
                title=f"{problem_name.upper()} Function Surface",
                save_path=fig3_path,
                best_point=best_pos
            )
            
            # 4. Contour plot
            fig4_path = self.figures_dir / f"{problem_name}_contour.png"
            plot_contour(
                problem_func, bounds, resolution=100,
                title=f"{problem_name.upper()} Contour Plot",
                save_path=fig4_path,
                best_point=best_pos
            )
    
    def _visualize_tsp_results(self, results, cities, n_cities):
        """Visualize TSP results"""
        n_algos = len(results)
        fig, axes = plt.subplots(1, n_algos, figsize=(7*n_algos, 6))
        
        if n_algos == 1:
            axes = [axes]
        
        for ax, (algo_name, stats) in zip(axes, results.items()):
            result = stats['best_result']
            route = result['best_route']
            distance = result['best_distance']
            
            plot_tsp_route(
                cities, route, distance,
                title=f"{algo_name}\nDistance: {distance:.2f}",
                ax=ax
            )
        
        plt.tight_layout()
        save_path = self.figures_dir / f"tsp_{n_cities}cities_solutions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nTSP visualizations saved to: {save_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Search Algorithms Project - Swarm Intelligence vs Traditional Methods"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Experiment command
    exp_parser = subparsers.add_parser('experiment', help='Run specific experiment')
    exp_parser.add_argument('--name', required=True, 
                            choices=['pso_vs_hc', 'abc_vs_ga', 'fa_vs_sa', 
                                    'cs_vs_sa', 'aco_vs_sa', 'all_swarm'],
                            help='Experiment name')
    exp_parser.add_argument('--problem', default='rastrigin',
                           choices=list(CONTINUOUS_PROBLEMS.keys()) + ['tsp'],
                           help='Problem to solve')
    exp_parser.add_argument('--dim', type=int, default=10,
                           help='Problem dimension (for continuous problems)')
    exp_parser.add_argument('--cities', type=int, default=20,
                           help='Number of cities (for TSP)')
    exp_parser.add_argument('--runs', type=int, default=30,
                           help='Number of independent runs')
    exp_parser.add_argument('--iter', type=int, default=100,
                           help='Maximum iterations')
    exp_parser.add_argument('--output', default='results',
                           help='Output directory')
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run comprehensive benchmarks')
    bench_parser.add_argument('--all', action='store_true',
                             help='Run all benchmarks')
    bench_parser.add_argument('--output', default='results',
                             help='Output directory')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize results')
    viz_parser.add_argument('--results-dir', default='results',
                           help='Results directory')
    
    args = parser.parse_args()
    
    if args.command == 'experiment':
        runner = ExperimentRunner(output_dir=args.output)
        
        # Define experiment configurations
        experiments = {
            'pso_vs_hc': {
                'algorithms': {
                    'PSO': (run_pso, {'n_particles': 30}),
                    'HC': (run_hill_climbing, {'step_size': 0.1, 'random_restart': 5})
                }
            },
            'abc_vs_ga': {
                'algorithms': {
                    'ABC': (run_abc, {'n_bees': 30}),
                    'GA': (run_ga, {'pop_size': 50})
                }
            },
            'fa_vs_sa': {
                'algorithms': {
                    'FA': (run_fa, {'n_fireflies': 25}),
                    'SA': (run_simulated_annealing, {})
                }
            },
            'cs_vs_sa': {
                'algorithms': {
                    'CS': (run_cs, {'n_nests': 25}),
                    'SA': (run_simulated_annealing, {})
                }
            },
            'aco_vs_sa': {
                'problem': 'tsp'
            },
            'all_swarm': {
                'algorithms': {
                    'PSO': (run_pso, {'n_particles': 30}),
                    'ABC': (run_abc, {'n_bees': 30}),
                    'FA': (run_fa, {'n_fireflies': 25}),
                    'CS': (run_cs, {'n_nests': 25}),
                    'GA': (run_ga, {'pop_size': 50}),
                    'SA': (run_simulated_annealing, {})
                }
            }
        }
        
        exp_config = experiments[args.name]
        
        if args.problem == 'tsp' or args.name == 'aco_vs_sa':
            # TSP experiment
            tsp_algos = {
                'ACO': (run_aco, {}),
                'SA': (run_simulated_annealing_tsp, {})
            }
            runner.run_tsp_experiment(
                args.cities, tsp_algos, args.runs, args.iter
            )
        else:
            # Continuous optimization experiment
            runner.run_continuous_experiment(
                args.problem, args.dim, exp_config['algorithms'],
                args.runs, args.iter
            )
    
    elif args.command == 'benchmark':
        print("Running comprehensive benchmarks...")
        runner = ExperimentRunner(output_dir=args.output)
        
        # Run on multiple problems and dimensions
        problems = ['sphere', 'rastrigin', 'rosenbrock', 'ackley']
        dimensions = [10, 20, 30]
        
        all_algos = {
            'PSO': (run_pso, {'n_particles': 30}),
            'ABC': (run_abc, {'n_bees': 30}),
            'FA': (run_fa, {'n_fireflies': 25}),
            'CS': (run_cs, {'n_nests': 25}),
            'GA': (run_ga, {'pop_size': 50}),
            'HC': (run_hill_climbing, {}),
            'SA': (run_simulated_annealing, {})
        }
        
        for problem in problems:
            for dim in dimensions:
                print(f"\n\n{'#'*80}")
                print(f"# {problem.upper()} - Dimension {dim}")
                print(f"{'#'*80}\n")
                
                runner.run_continuous_experiment(
                    problem, dim, all_algos, n_runs=30, max_iter=100
                )
    
    elif args.command == 'visualize':
        print(f"Visualizing results from: {args.results_dir}")
        # Add visualization code here
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
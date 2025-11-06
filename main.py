"""Main runner for Search Algorithms Project

Runs experiments comparing swarm intelligence and traditional algorithms
on continuous and discrete optimization problems.
"""

import numpy as np
import sys
import os
import argparse
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import problems
from problems.sphere import CONTINUOUS_PROBLEMS
from problems.tsp import create_tsp_problem
from problems.knapsack import create_knapsack_problem

# Import swarm algorithms
from algorithms.swarm.PSO import run_pso
from algorithms.swarm.ACO import run_aco
from algorithms.swarm.ABC import run_abc
from algorithms.swarm.FA import run_fa
from algorithms.swarm.CS import run_cs

# Import traditional algorithms
from algorithms.traditional.hill_climbing import run_hill_climbing
from algorithms.traditional.simulated_annealing import run_simulated_annealing, run_simulated_annealing_tsp
from algorithms.traditional.genetic_algorithm import run_ga
from algorithms.traditional.BFS import run_bfs_greedy_tsp
from algorithms.traditional.DFS import run_dfs_tsp
from algorithms.traditional.AStar import run_astar_tsp

# Import utilities
from utils.benchmark import run_multiple_times, compare_algorithms
from utils.visualize import (plot_convergence_comparison, plot_3d_surface, 
                             plot_contour, plot_boxplot_comparison, 
                             ensure_figure_dir, plot_tsp_route)


def run_comparison(algorithms, problem_name, problem_func, n_runs, minimize, seed, **kwargs):
    """Helper function to run a comparison, print results, and plot graphs."""
    figures_dir = ensure_figure_dir()
    
    print(f"\nTesting on {problem_name} function (dim={kwargs.get('dim', 'N/A')})")
    
    results = compare_algorithms(algorithms, problem_func, n_runs=n_runs, minimize=minimize, seed=seed)
    
    print("\nResults Summary:")
    print("-" * 80)
    print(f"{'Algorithm':<20} {'Best':<15} {'Mean±Std':<25} {'Time (s)':<15}")
    print("-" * 80)
    
    for name, stats in results.items():
        print(f"{name:<20} {stats['best_fitness']:<15.6f} "
              f"{stats['mean_fitness']:.6f}±{stats['std_fitness']:.6f}    "
              f"{stats['mean_time']:<15.4f}")
    
    histories = {name: stats['results'][0]['history'] for name, stats in results.items() if 'history' in stats['results'][0]}
    
    if histories:
        plot_convergence_comparison(
            histories,
            title=f"Convergence Comparison - {problem_name}",
            save_path=f"{figures_dir}/{problem_name.lower().replace(' ', '_')}_convergence.png",
            log_scale=True
        )

    boxplot_data = {name: stats['all_fitnesses'] for name, stats in results.items()}
    plot_boxplot_comparison(
        boxplot_data,
        title=f"Algorithm Performance Distribution - {problem_name}",
        ylabel="Final Fitness",
        save_path=f"{figures_dir}/{problem_name.lower().replace(' ', '_')}_boxplot.png"
    )
    
    return results

def run_aco_vs_sa_tsp(n_cities, max_iter):
    """Compares ACO and Simulated Annealing on the Traveling Salesman Problem."""
    print("\n" + "="*80)
    print("Experiment 1: ACO vs Simulated Annealing on TSP")
    print("="*80)
    
    figures_dir = ensure_figure_dir()
    tsp_problem = create_tsp_problem(n_cities=n_cities)
    cities = tsp_problem['cities']
    dist_matrix = tsp_problem['distance_matrix']
    
    print(f"\nTesting on TSP with {n_cities} cities")
    
    print("\nRunning ACO...")
    aco_result = run_aco(dist_matrix, n_ants=30, max_iter=max_iter, seed=42)
    print(f"ACO - Best distance: {aco_result['best_distance']:.2f}")
    
    print("\nRunning Simulated Annealing...")
    sa_result = run_simulated_annealing_tsp(dist_matrix, max_iter=max_iter * 100, seed=42)
    print(f"SA - Best distance: {sa_result['best_distance']:.2f}")

    # Create subplots for displaying TSP routes
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('TSP Solutions', fontsize=16)

    # Plot ACO TSP route
    plot_tsp_route(
        cities, aco_result['best_route'], aco_result['best_distance'],
        title=f"TSP Solution - ACO ({n_cities} cities)",
        save_path=f"{figures_dir}/tsp_aco_route.png",
        ax=axes[0]
    )
    
    # Plot SA TSP route
    plot_tsp_route(
        cities, sa_result['best_route'], sa_result['best_distance'],
        title=f"TSP Solution - SA ({n_cities} cities)",
        save_path=f"{figures_dir}/tsp_sa_route.png",
        ax=axes[1]
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # Plot convergence comparison
    histories = {
        'ACO': aco_result['history'],
        'SA': sa_result['history']
    }
    plot_convergence_comparison(
        histories,
        title=f"ACO vs SA Convergence on TSP ({n_cities} cities)",
        save_path=f"{figures_dir}/tsp_aco_sa_convergence.png",
        log_scale=True
    )

def run_pso_vs_ga_rastrigin(dim, max_iter, n_runs):
    """Compares PSO and GA on the Rastrigin function."""
    print("\n" + "="*80)
    print("Experiment 2: PSO vs GA on Rastrigin")
    print("="*80)
    
    problem = CONTINUOUS_PROBLEMS['rastrigin']
    algorithms = {
        'PSO': (run_pso, {'dim': dim, 'bounds': problem['bounds'], 'n_particles': 30, 'max_iter': max_iter}),
        'GA': (run_ga, {'dim': dim, 'bounds': problem['bounds'], 'pop_size': 50, 'max_iter': max_iter}),
    }
    run_comparison(algorithms, "Rastrigin (PSO vs GA)", problem['function'], n_runs, True, 42, dim=dim)

def run_abc_vs_ga_rastrigin(dim, max_iter, n_runs):
    """Compares ABC and GA on the Rastrigin function."""
    print("\n" + "="*80)
    print("Experiment 3: ABC vs GA on Rastrigin")
    print("="*80)
    
    problem = CONTINUOUS_PROBLEMS['rastrigin']
    algorithms = {
        'ABC': (run_abc, {'dim': dim, 'bounds': problem['bounds'], 'n_bees': 30, 'max_iter': max_iter}),
        'GA': (run_ga, {'dim': dim, 'bounds': problem['bounds'], 'pop_size': 50, 'max_iter': max_iter}),
    }
    run_comparison(algorithms, "Rastrigin (ABC vs GA)", problem['function'], n_runs, True, 42, dim=dim)

def run_fa_vs_sa_ackley(dim, max_iter, n_runs):
    """Compares FA and SA on the Ackley function."""
    print("\n" + "="*80)
    print("Experiment 4: FA vs SA on Ackley")
    print("="*80)
    
    problem = CONTINUOUS_PROBLEMS['ackley']
    n_pop = 25  # Population size for FA
    max_evals = n_pop * max_iter

    algorithms = {
        'FA': (run_fa, {'dim': dim, 'bounds': problem['bounds'], 'n_fireflies': n_pop, 'max_iter': max_iter}),
        'SA': (run_simulated_annealing, {'dim': dim, 'bounds': problem['bounds'], 'max_iter': max_evals, 'pop_size_equiv': n_pop}),
    }
    run_comparison(algorithms, "Ackley (FA vs SA)", problem['function'], n_runs, True, 42, dim=dim)

def run_cs_vs_sa_ackley(dim, max_iter, n_runs):
    """Compares CS and SA on the Ackley function."""
    print("\n" + "="*80)
    print("Experiment 5: CS vs SA on Ackley")
    print("="*80)
    
    problem = CONTINUOUS_PROBLEMS['ackley']
    n_pop = 25  # Population size for CS
    max_evals = n_pop * max_.gemini-agent-gen-0
    
    algorithms = {
        'CS': (run_cs, {'dim': dim, 'bounds': problem['bounds'], 'n_nests': n_pop, 'max_iter': max_iter}),
        'SA': (run_simulated_annealing, {'dim': dim, 'bounds': problem['bounds'], 'max_iter': max_evals, 'pop_size_equiv': n_pop}),
    }
    run_comparison(algorithms, "Ackley (CS vs SA)", problem['function'], n_runs, True, 42, dim=dim)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run comparison experiments for search algorithms.")
    subparsers = parser.add_subparsers(dest="experiment", help="Select the experiment to run.")

    # TSP experiment
    parser_tsp = subparsers.add_parser("aco_vs_sa", help="Run ACO vs SA on TSP.")
    parser_tsp.add_argument("--n_cities", type=int, default=20, help="Number of cities for TSP.")
    parser_tsp.add_argument("--max_iter", type=int, default=100, help="Maximum iterations for ACO.")

    # Rastrigin experiments
    parser_pso_ga = subparsers.add_parser("pso_vs_ga", help="Run PSO vs GA on Rastrigin.")
    parser_pso_ga.add_argument("--dim", type=int, default=10, help="Dimension of the problem.")
    parser_pso_ga.add_argument("--max_iter", type=int, default=100, help="Maximum iterations.")
    parser_pso_ga.add_argument("--n_runs", type=int, default=5, help="Number of runs.")

    parser_abc_ga = subparsers.add_parser("abc_vs_ga", help="Run ABC vs GA on Rastrigin.")
    parser_abc_ga.add_argument("--dim", type=int, default=10, help="Dimension of the problem.")
    parser_abc_ga.add_argument("--max_iter", type=int, default=100, help="Maximum iterations.")
    parser_abc_ga.add_argument("--n_runs", type=int, default=5, help="Number of runs.")

    # Ackley experiments
    parser_fa_sa = subparsers.add_parser("fa_vs_sa", help="Run FA vs SA on Ackley.")
    parser_fa_sa.add_argument("--dim", type=int, default=10, help="Dimension of the problem.")
    parser_fa_sa.add_argument("--max_iter", type=int, default=100, help="Maximum iterations.")
    parser_fa_sa.add_argument("--n_runs", type=int, default=5, help="Number of runs.")

    parser_cs_sa = subparsers.add_parser("cs_vs_sa", help="Run CS vs SA on Ackley.")
    parser_cs_sa.add_argument("--dim", type=int, default=10, help="Dimension of the problem.")
    parser_cs_sa.add_argument("--max_iter", type=int, default=100, help="Maximum iterations.")
    parser_cs_sa.add_argument("--n_runs", type=int, default=5, help="Number of runs.")

    args = parser.parse_args()

    if args.experiment == "aco_vs_sa":
        run_aco_vs_sa_tsp(n_cities=args.n_cities, max_iter=args.max_iter)
    elif args.experiment == "pso_vs_ga":
        run_pso_vs_ga_rastrigin(dim=args.dim, max_iter=args.max_iter, n_runs=args.n_runs)
    elif args.experiment == "abc_vs_ga":
        run_abc_vs_ga_rastrigin(dim=args.dim, max_iter=args.max_iter, n_runs=args.n_runs)
    elif args.experiment == "fa_vs_sa":
        run_fa_vs_sa_ackley(dim=args.dim, max_iter=args.max_iter, n_runs=args.n_runs)
    elif args.experiment == "cs_vs_sa":
        run_cs_vs_sa_ackley(dim=args.dim, max_iter=args.max_iter, n_runs=args.n_runs)
    else:
        parser.print_help()

    print("\n" + "="*80)
    print("Experiments completed!")
    print("Figures saved to: report/figures/")
    print("="*80)

if __name__ == "__main__":
    main()

"""Main runner for Search Algorithms Project

Runs experiments comparing swarm intelligence and traditional algorithms
on continuous and discrete optimization problems.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import problems
from problems.sphere import sphere, rastrigin, rosenbrock, ackley, CONTINUOUS_PROBLEMS
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
from algorithms.traditional.simulated_annealing import run_simulated_annealing
from algorithms.traditional.genetic_algorithm import run_ga
from algorithms.traditional.BFS import run_bfs_greedy_tsp
from algorithms.traditional.DFS import run_dfs_tsp
from algorithms.traditional.AStar import run_astar_tsp

# Import utilities
from utils.benchmark import run_multiple_times, compare_algorithms
from utils.visualize import (plot_convergence_comparison, plot_3d_surface, 
                             plot_contour, plot_boxplot_comparison, 
                             ensure_figure_dir, plot_tsp_route)


def experiment_continuous_optimization(dim=10, max_iter=100, n_runs=5):
    """Run experiments on continuous optimization problems."""
    print("\n" + "="*80)
    print("EXPERIMENT 1: CONTINUOUS OPTIMIZATION")
    print("="*80)
    
    figures_dir = ensure_figure_dir()
    
    # Test on Sphere function
    problem = CONTINUOUS_PROBLEMS['sphere']
    func = problem['function']
    bounds = problem['bounds']
    
    print(f"\nTesting on Sphere function (dim={dim})")
    print(f"Bounds: {bounds}, Target: {problem['optimum']}")
    
    # Define algorithms
    algorithms = {
        'PSO': (run_pso, {'dim': dim, 'bounds': bounds, 'n_particles': 30, 'max_iter': max_iter}),
        'ABC': (run_abc, {'dim': dim, 'bounds': bounds, 'n_bees': 30, 'max_iter': max_iter}),
        'FA': (run_fa, {'dim': dim, 'bounds': bounds, 'n_fireflies': 25, 'max_iter': max_iter}),
        'CS': (run_cs, {'dim': dim, 'bounds': bounds, 'n_nests': 25, 'max_iter': max_iter}),
        'GA': (run_ga, {'dim': dim, 'bounds': bounds, 'pop_size': 50, 'max_iter': max_iter}),
        'SA': (run_simulated_annealing, {'dim': dim, 'bounds': bounds, 'max_iter': max_iter*10}),
        'Hill Climbing': (run_hill_climbing, {'dim': dim, 'bounds': bounds, 'max_iter': max_iter}),
    }
    
    # Run comparison
    results = compare_algorithms(algorithms, func, n_runs=n_runs, minimize=True, seed=42)
    
    # Print results
    print("\nResults Summary:")
    print("-" * 80)
    print(f"{'Algorithm':<20} {'Best':<15} {'Mean±Std':<25} {'Time (s)':<15}")
    print("-" * 80)
    
    for name, stats in results.items():
        print(f"{name:<20} {stats['best_fitness']:<15.6f} "
              f"{stats['mean_fitness']:.6f}±{stats['std_fitness']:.6f}    "
              f"{stats['mean_time']:<15.4f}")
    
    # Plot convergence comparison
    histories = {}
    for name, stats in results.items():
        # Use first run's history
        if 'history' in stats['results'][0]:
            histories[name] = stats['results'][0]['history']
    
    if histories:
        plot_convergence_comparison(
            histories, 
            title=f"Convergence Comparison - Sphere Function (dim={dim})",
            save_path=f"{figures_dir}/sphere_convergence.png",
            log_scale=True
        )
    
    # Plot 3D surface for 2D case
    if dim == 2:
        best_result = min(results.items(), key=lambda x: x[1]['best_fitness'])
        best_name, best_stats = best_result
        best_pos = best_stats['results'][0]['best_position']
        
        plot_3d_surface(
            func, bounds, resolution=50,
            title="Sphere Function - 3D Surface",
            save_path=f"{figures_dir}/sphere_3d.png",
            best_point=best_pos
        )
        
        plot_contour(
            func, bounds, resolution=100,
            title="Sphere Function - Contour Plot",
            save_path=f"{figures_dir}/sphere_contour.png",
            best_point=best_pos
        )
    
    # Boxplot comparison
    boxplot_data = {name: stats['all_fitnesses'] for name, stats in results.items()}
    plot_boxplot_comparison(
        boxplot_data,
        title=f"Algorithm Performance Distribution - Sphere (dim={dim})",
        ylabel="Final Fitness",
        save_path=f"{figures_dir}/sphere_boxplot.png"
    )
    
    return results


def experiment_tsp(n_cities=20, max_iter=100):
    """Run experiments on TSP."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: TRAVELING SALESMAN PROBLEM (TSP)")
    print("="*80)
    
    figures_dir = ensure_figure_dir()
    
    # Create TSP instance
    tsp_problem = create_tsp_problem(n_cities=n_cities, seed=42)
    cities = tsp_problem['cities']
    dist_matrix = tsp_problem['distance_matrix']
    
    print(f"\nTesting on TSP with {n_cities} cities")
    
    # Run ACO (best for TSP)
    print("\nRunning ACO...")
    aco_result = run_aco(dist_matrix, n_ants=30, max_iter=max_iter, seed=42)
    print(f"ACO - Best distance: {aco_result['best_distance']:.2f}")
    
    # Run Greedy BFS
    print("Running Greedy BFS...")
    bfs_result = run_bfs_greedy_tsp(dist_matrix)
    print(f"Greedy BFS - Best distance: {bfs_result['best_distance']:.2f}")
    
    # Run A*  (if cities <= 12)
    if n_cities <= 12:
        print("Running A*...")
        astar_result = run_astar_tsp(dist_matrix, max_nodes=10000)
        print(f"A* - Best distance: {astar_result['best_distance']:.2f}")
    
    # Visualize best route
    plot_tsp_route(
        cities, aco_result['best_route'], aco_result['best_distance'],
        title=f"TSP Solution - ACO ({n_cities} cities)",
        save_path=f"{figures_dir}/tsp_aco_route.png"
    )
    
    # Plot convergence
    if 'history' in aco_result:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(aco_result['history'], linewidth=2)
        plt.title(f"ACO Convergence - TSP ({n_cities} cities)")
        plt.xlabel("Iteration")
        plt.ylabel("Best Distance")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/tsp_aco_convergence.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    return {'ACO': aco_result, 'Greedy_BFS': bfs_result}


def experiment_rastrigin(dim=10, max_iter=100, n_runs=3):
    """Test on multimodal Rastrigin function."""
    print("\n" + "="*80)
    print("EXPERIMENT 3: RASTRIGIN FUNCTION (Multimodal)")
    print("="*80)
    
    figures_dir = ensure_figure_dir()
    
    problem = CONTINUOUS_PROBLEMS['rastrigin']
    func = problem['function']
    bounds = problem['bounds']
    
    print(f"\nTesting on Rastrigin function (dim={dim})")
    
    # Test swarm algorithms
    algorithms = {
        'PSO': (run_pso, {'dim': dim, 'bounds': bounds, 'n_particles': 40, 'max_iter': max_iter}),
        'ABC': (run_abc, {'dim': dim, 'bounds': bounds, 'n_bees': 40, 'max_iter': max_iter}),
        'FA': (run_fa, {'dim': dim, 'bounds': bounds, 'n_fireflies': 30, 'max_iter': max_iter}),
        'GA': (run_ga, {'dim': dim, 'bounds': bounds, 'pop_size': 50, 'max_iter': max_iter}),
    }
    
    results = compare_algorithms(algorithms, func, n_runs=n_runs, minimize=True, seed=42)
    
    print("\nResults:")
    for name, stats in results.items():
        print(f"{name}: Best={stats['best_fitness']:.4f}, Mean={stats['mean_fitness']:.4f}")
    
    # Plot convergence
    histories = {name: stats['results'][0]['history'] 
                for name, stats in results.items() if 'history' in stats['results'][0]}
    
    if histories:
        plot_convergence_comparison(
            histories,
            title=f"Rastrigin Function Comparison (dim={dim})",
            save_path=f"{figures_dir}/rastrigin_convergence.png"
        )
    
    return results


def quick_demo():
    """Quick demo showing basic functionality."""
    print("\n" + "="*80)
    print("QUICK DEMO - Testing PSO on Sphere Function")
    print("="*80)
    
    # Simple 2D Sphere function test
    result = run_pso(
        objective_func=sphere,
        dim=2,
        bounds=(-10, 10),
        n_particles=20,
        max_iter=50,
        seed=42
    )
    
    print(f"\nBest position: {result['best_position']}")
    print(f"Best fitness: {result['best_fitness']:.6f}")
    print(f"Expected optimum: [0, 0] with fitness 0")
    
    # Visualize
    from utils.visualize import plot_history
    plot_history(result['history'], title="PSO on 2D Sphere Function")


def main():
    """Main entry point."""
    print("="*80)
    print("PROJECT 01 - SEARCH ALGORITHMS")
    print("Swarm Intelligence vs Traditional Optimization")
    print("="*80)
    
    # Choose what to run
    print("\nSelect experiment:")
    print("1. Quick Demo (PSO on Sphere 2D)")
    print("2. Full Continuous Optimization (Sphere, Rastrigin, etc.)")
    print("3. TSP Experiment")
    print("4. All Experiments")
    
    choice = input("\nEnter choice (1-4) or press Enter for Quick Demo: ").strip()
    
    if choice == "" or choice == "1":
        quick_demo()
    
    elif choice == "2":
        experiment_continuous_optimization(dim=10, max_iter=100, n_runs=5)
        experiment_rastrigin(dim=10, max_iter=100, n_runs=3)
    
    elif choice == "3":
        experiment_tsp(n_cities=20, max_iter=100)
    
    elif choice == "4":
        print("\nRunning all experiments...")
        experiment_continuous_optimization(dim=10, max_iter=100, n_runs=5)
        experiment_tsp(n_cities=15, max_iter=100)
        experiment_rastrigin(dim=10, max_iter=100, n_runs=3)
    
    else:
        print("Invalid choice. Running quick demo...")
        quick_demo()
    
    print("\n" + "="*80)
    print("Experiments completed!")
    print("Figures saved to: report/figures/")
    print("="*80)


if __name__ == "__main__":
    main()


"""Demo script - Auto-run experiments without user input

Chạy các experiments cơ bản và tạo visualizations.
"""

import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problems.sphere import sphere, rastrigin, CONTINUOUS_PROBLEMS
from problems.tsp import create_tsp_problem

from algorithms.swarm.PSO import run_pso
from algorithms.swarm.ACO import run_aco
from algorithms.swarm.ABC import run_abc
from algorithms.swarm.FA import run_fa
from algorithms.swarm.CS import run_cs

from algorithms.traditional.hill_climbing import run_hill_climbing
from algorithms.traditional.simulated_annealing import run_simulated_annealing
from algorithms.traditional.genetic_algorithm import run_ga

from utils.benchmark import compare_algorithms
from utils.visualize import (plot_convergence_comparison, plot_3d_surface,
                             plot_boxplot_comparison, ensure_figure_dir,
                             plot_tsp_route)

print("="*80)
print("AUTO DEMO - SEARCH ALGORITHMS PROJECT")
print("="*80)

figures_dir = ensure_figure_dir()

# ============================================================================
# DEMO 1: Compare algorithms on Sphere function (10D)
# ============================================================================
print("\n[DEMO 1] Comparing algorithms on Sphere function (10D)...")

dim = 10
bounds = (-100, 100)
max_iter = 50

algorithms = {
    'PSO': (run_pso, {'dim': dim, 'bounds': bounds, 'n_particles': 30, 'max_iter': max_iter}),
    'ABC': (run_abc, {'dim': dim, 'bounds': bounds, 'n_bees': 30, 'max_iter': max_iter}),
    'FA': (run_fa, {'dim': dim, 'bounds': bounds, 'n_fireflies': 25, 'max_iter': max_iter}),
    'GA': (run_ga, {'dim': dim, 'bounds': bounds, 'pop_size': 40, 'max_iter': max_iter}),
    'SA': (run_simulated_annealing, {'dim': dim, 'bounds': bounds, 'max_iter': max_iter*5}),
}

results = compare_algorithms(algorithms, sphere, n_runs=3, minimize=True, seed=42)

print("\nResults:")
print("-" * 70)
print(f"{'Algorithm':<15} {'Best':<12} {'Mean±Std':<20} {'Time(s)':<10}")
print("-" * 70)
for name, stats in results.items():
    print(f"{name:<15} {stats['best_fitness']:<12.6f} "
          f"{stats['mean_fitness']:.4f}±{stats['std_fitness']:.4f}    "
          f"{stats['mean_time']:<10.4f}")

# Plot convergence
histories = {name: stats['results'][0]['history'] 
            for name, stats in results.items() if 'history' in stats['results'][0]}

plot_convergence_comparison(
    histories,
    title="Convergence Comparison - Sphere Function (10D)",
    save_path=f"{figures_dir}/demo_sphere_convergence.png",
    log_scale=True
)
print(f"✓ Saved: {figures_dir}/demo_sphere_convergence.png")

# Boxplot
boxplot_data = {name: stats['all_fitnesses'] for name, stats in results.items()}
plot_boxplot_comparison(
    boxplot_data,
    title="Performance Distribution - Sphere (10D)",
    save_path=f"{figures_dir}/demo_sphere_boxplot.png"
)
print(f"✓ Saved: {figures_dir}/demo_sphere_boxplot.png")


# ============================================================================
# DEMO 2: 3D Visualization on 2D Sphere
# ============================================================================
print("\n[DEMO 2] Creating 3D visualization for 2D Sphere function...")

result_2d = run_pso(sphere, dim=2, bounds=(-10, 10), n_particles=20, max_iter=30, seed=42)

plot_3d_surface(
    sphere,
    bounds=(-10, 10),
    resolution=50,
    title="Sphere Function - 3D Surface",
    save_path=f"{figures_dir}/demo_sphere_3d.png",
    best_point=result_2d['best_position']
)
print(f"✓ Saved: {figures_dir}/demo_sphere_3d.png")
print(f"  Best solution: {result_2d['best_position']}, Fitness: {result_2d['best_fitness']:.6f}")


# ============================================================================
# DEMO 3: TSP with ACO
# ============================================================================
print("\n[DEMO 3] Solving TSP with ACO (15 cities)...")

tsp = create_tsp_problem(n_cities=15, seed=42)
aco_result = run_aco(tsp['distance_matrix'], n_ants=20, max_iter=50, seed=42)

print(f"Best tour distance: {aco_result['best_distance']:.2f}")

plot_tsp_route(
    tsp['cities'],
    aco_result['best_route'],
    aco_result['best_distance'],
    title="TSP Solution - ACO (15 cities)",
    save_path=f"{figures_dir}/demo_tsp_route.png"
)
print(f"✓ Saved: {figures_dir}/demo_tsp_route.png")

# ACO convergence
plt.figure(figsize=(10, 6))
plt.plot(aco_result['history'], linewidth=2, color='blue')
plt.title("ACO Convergence on TSP (15 cities)")
plt.xlabel("Iteration")
plt.ylabel("Best Distance")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{figures_dir}/demo_tsp_convergence.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {figures_dir}/demo_tsp_convergence.png")


# ============================================================================
# DEMO 4: Multimodal function (Rastrigin)
# ============================================================================
print("\n[DEMO 4] Testing on multimodal Rastrigin function (5D)...")

rastrigin_results = {}
for algo_name in ['PSO', 'ABC', 'FA']:
    algo_func, kwargs = algorithms[algo_name]
    kwargs['objective_func'] = rastrigin
    kwargs['bounds'] = (-5.12, 5.12)
    kwargs['dim'] = 5
    
    result = algo_func(**kwargs)
    rastrigin_results[algo_name] = result
    print(f"{algo_name}: {result['best_fitness']:.4f}")

# Plot comparison
rastrigin_histories = {name: res['history'] for name, res in rastrigin_results.items()}
plot_convergence_comparison(
    rastrigin_histories,
    title="Convergence on Rastrigin (Multimodal)",
    save_path=f"{figures_dir}/demo_rastrigin_convergence.png"
)
print(f"✓ Saved: {figures_dir}/demo_rastrigin_convergence.png")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("DEMO COMPLETED!")
print("="*80)
print(f"\nAll figures saved to: {figures_dir}/")
print("\nGenerated files:")
print("  - demo_sphere_convergence.png")
print("  - demo_sphere_boxplot.png")
print("  - demo_sphere_3d.png")
print("  - demo_tsp_route.png")
print("  - demo_tsp_convergence.png")
print("  - demo_rastrigin_convergence.png")
print("\nTo run full experiments with more options, use: python main.py")
print("="*80)

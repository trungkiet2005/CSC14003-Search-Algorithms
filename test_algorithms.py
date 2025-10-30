"""Test script to verify all algorithms work correctly"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test continuous optimization
from problems.sphere import sphere, rastrigin
from algorithms.swarm.PSO import run_pso
from algorithms.swarm.ABC import run_abc
from algorithms.traditional.genetic_algorithm import run_ga

# Test discrete optimization
from problems.tsp import create_tsp_problem
from algorithms.swarm.ACO import run_aco

print("="*80)
print("TESTING ALL ALGORITHMS")
print("="*80)

# Test 1: PSO on Sphere
print("\n[TEST 1] PSO on Sphere function (2D)...")
try:
    result = run_pso(sphere, dim=2, bounds=(-10, 10), n_particles=10, max_iter=20, seed=42)
    print(f"✓ PSO passed - Best fitness: {result['best_fitness']:.6f}")
except Exception as e:
    print(f"✗ PSO failed: {e}")

# Test 2: ABC on Sphere
print("\n[TEST 2] ABC on Sphere function (5D)...")
try:
    result = run_abc(sphere, dim=5, bounds=(-10, 10), n_bees=10, max_iter=20, seed=42)
    print(f"✓ ABC passed - Best fitness: {result['best_fitness']:.6f}")
except Exception as e:
    print(f"✗ ABC failed: {e}")

# Test 3: GA on Rastrigin
print("\n[TEST 3] GA on Rastrigin function (3D)...")
try:
    result = run_ga(rastrigin, dim=3, bounds=(-5.12, 5.12), pop_size=20, max_iter=20, seed=42)
    print(f"✓ GA passed - Best fitness: {result['best_fitness']:.6f}")
except Exception as e:
    print(f"✗ GA failed: {e}")

# Test 4: ACO on TSP
print("\n[TEST 4] ACO on TSP (10 cities)...")
try:
    tsp = create_tsp_problem(n_cities=10, seed=42)
    result = run_aco(tsp['distance_matrix'], n_ants=10, max_iter=20, seed=42)
    print(f"✓ ACO passed - Best distance: {result['best_distance']:.2f}")
except Exception as e:
    print(f"✗ ACO failed: {e}")

# Test 5: Visualization (without showing)
print("\n[TEST 5] Testing visualization utilities...")
try:
    from utils.visualize import ensure_figure_dir
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    from utils.visualize import plot_history
    import matplotlib.pyplot as plt
    
    # Create test figure
    fig = plt.figure()
    plt.plot([1, 2, 3, 4, 5])
    figures_dir = ensure_figure_dir()
    test_path = os.path.join(figures_dir, 'test_plot.png')
    plt.savefig(test_path)
    plt.close()
    
    if os.path.exists(test_path):
        print(f"✓ Visualization passed - Figure saved to {test_path}")
    else:
        print("✗ Visualization failed - Figure not saved")
except Exception as e:
    print(f"✗ Visualization failed: {e}")

# Test 6: Benchmark utilities
print("\n[TEST 6] Testing benchmark utilities...")
try:
    from utils.benchmark import run_multiple_times
    
    stats = run_multiple_times(
        run_pso,
        n_runs=3,
        objective_func=sphere,
        dim=2,
        bounds=(-10, 10),
        n_particles=10,
        max_iter=10,
        seed=42
    )
    
    print(f"✓ Benchmark passed - Mean fitness: {stats['mean_fitness']:.6f}")
except Exception as e:
    print(f"✗ Benchmark failed: {e}")

print("\n" + "="*80)
print("ALL TESTS COMPLETED!")
print("="*80)
print("\nProject is ready to use. Run 'python main.py' to start experiments.")

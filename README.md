# Project 01 - Search Algorithms

Repository scaffold for metaheuristics and local search experiments.

Structure

- algorithms/: implementations and stubs for ACO, PSO, ABC, FA, CS, hill_climbing, simulated_annealing, genetic_algorithm
- problems/: benchmark problems (sphere, tsp)
- utils/: helpers for visualization and benchmarking
- main.py: starter entrypoint
- report/figures/: place figures for the report

How to use

1. Implement algorithm functions in `algorithms/` (each file exposes a run\_\* function).
2. Add or adapt problems in `problems/`.
3. Use `utils/visualize.py` and `utils/benchmark.py` to plot and measure runs.
4. Edit `main.py` to orchestrate experiments.

Minimum requirements

- Python 3.8+
- Optional: matplotlib for plotting

"""algorithms/traditional/simulated_annealing.py - Improved Simulated Annealing

Reference: Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). 
Optimization by simulated annealing. Science, 220(4598), 671-680.
"""

import numpy as np
from typing import Callable, Tuple, Optional
from ..base import LocalSearchOptimizer, OptimizationResult, run_with_timing


class SimulatedAnnealing(LocalSearchOptimizer):
    """
    Continuous Simulated Annealing (SA)
    
    Based on:
        - Kirkpatrick et al., Science, 1983
        - Corana et al., ACM Transactions on Mathematical Software, 1987
    
    Supports temperature-dependent step size, multiple cooling schedules, and early stopping.
    """

    def __init__(self,
                 initial_temp: float = 1000,
                 final_temp: float = 1e-3,
                 alpha: float = 0.98,
                 cooling_schedule: str = 'exponential',
                 neighbor_std: float = 0.3,
                 inner_loops: int = 50,
                 patience: int = 1500,
                 seed: Optional[int] = None):
        super().__init__(seed=seed)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.cooling_schedule = cooling_schedule
        self.neighbor_std = neighbor_std
        self.inner_loops = inner_loops
        self.patience = patience
        self.name = "SA-Continuous"

    @run_with_timing
    def optimize(self,
                 objective_func: Callable,
                 dim: int,
                 bounds: Tuple[float, float],
                 max_iter: int = 2500,
                 minimize: bool = True,
                 **kwargs) -> OptimizationResult:
        """
        Run continuous simulated annealing optimization.

        Args:
            objective_func: Callable objective function.
            dim: Problem dimensionality.
            bounds: (lower, upper) tuple defining variable range.
            max_iter: Max total iterations.
            minimize: True for minimization, False for maximization.
        """
        lower, upper = bounds
        scale = upper - lower

        # Initial solution
        current_solution = self._random_position(dim, lower, upper)
        current_fitness = objective_func(current_solution)
        best_solution = current_solution.copy()
        best_fitness = current_fitness

        # Initialize temperature
        temp = self.initial_temp

        # Tracking
        history = [best_fitness]
        acceptance_history = []
        no_improve = 0
        iteration = 0

        # --- Main Annealing Loop ---
        while temp > self.final_temp and iteration < max_iter and no_improve < self.patience:
            for _ in range(self.inner_loops):
                # Generate neighbor (temperature-dependent perturbation)
                neighbor_solution = self._generate_neighbor_solution(current_solution, scale, lower, upper, temp)
                neighbor_fitness = objective_func(neighbor_solution)

                # Energy difference
                delta_e = neighbor_fitness - current_fitness if minimize else current_fitness - neighbor_fitness

                # Acceptance check
                if delta_e < 0:
                    current_solution = neighbor_solution
                    current_fitness = neighbor_fitness
                    acceptance_history.append(1)
                else:
                    # Metropolis criterion
                    acceptance_prob = np.exp(-min(delta_e / temp, 700))  # avoid overflow
                    if self.rng.random() < acceptance_prob:
                        current_solution = neighbor_solution
                        current_fitness = neighbor_fitness
                        acceptance_history.append(1)
                    else:
                        acceptance_history.append(0)

                # Update best
                if (minimize and current_fitness < best_fitness) or (not minimize and current_fitness > best_fitness):
                    best_fitness = current_fitness
                    best_solution = current_solution.copy()
                    no_improve = 0
                else:
                    no_improve += 1

                iteration += 1
                if iteration >= max_iter or no_improve >= self.patience:
                    break

            # Cool down after inner loop
            temp = self._cool_down(temp, iteration, max_iter)
            if iteration % 50 == 0:
                history.append(best_fitness)

        history.append(best_fitness)

        return OptimizationResult(
            best_position=best_solution,
            best_fitness=best_fitness,
            history=history,
            final_temperature=temp,
            acceptance_rate=np.mean(acceptance_history) if acceptance_history else 0
        )

    # ------------------------------------------------------------
    # --- Helper Methods ---
    # ------------------------------------------------------------
    def _generate_neighbor_solution(self, current: np.ndarray, scale: float,
                                   lower: float, upper: float, temp: float) -> np.ndarray:
        """
        Generate a neighbor solution using temperature-dependent Gaussian perturbation.
        Step size shrinks with temperature (Corana et al., 1987).
        """
        sigma = self.neighbor_std * scale * (temp / self.initial_temp)
        perturbation = self.rng.normal(0, sigma, len(current))
        neighbor = current + perturbation
        return self._clip_bounds(neighbor, lower, upper)

    def _cool_down(self, temp: float, iteration: int, max_iter: int) -> float:
        """Apply chosen cooling schedule."""
        if self.cooling_schedule == 'exponential':
            return temp * self.alpha
        elif self.cooling_schedule == 'linear':
            return self.initial_temp - (self.initial_temp - self.final_temp) * iteration / max_iter
        elif self.cooling_schedule == 'logarithmic':
            return self.initial_temp / (1 + np.log1p(iteration))
        else:
            return temp * self.alpha


class SimulatedAnnealingTSP:
    """Simulated Annealing for TSP (following Kirkpatrick et al., 1983)"""
    
    def __init__(self, 
                 initial_temp: float = 1000,
                 final_temp: float = 1e-3,
                 alpha: float = 0.995,
                 inner_loops: Optional[int] = None,
                 cooling_schedule: str = "exponential",
                 patience: int = 2000,
                 seed: Optional[int] = None):
        """
        Args:
            initial_temp: Starting temperature.
            final_temp: Final temperature threshold.
            alpha: Cooling rate for exponential schedule.
            inner_loops: Number of moves per temperature (default = 10 * n_cities).
            cooling_schedule: 'exponential', 'linear', or 'logarithmic'.
            patience: Stop early if no improvement after these many iterations.
            seed: Random seed.
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.inner_loops = inner_loops
        self.cooling_schedule = cooling_schedule
        self.patience = patience
        self.rng = np.random.default_rng(seed)
        self.name = "SA-TSP"
    
    @run_with_timing
    def optimize_tsp(self, distance_matrix: np.ndarray,
                    max_iter: int = 20000) -> OptimizationResult:
        n_cities = len(distance_matrix)
        if self.inner_loops is None:
            self.inner_loops = 10 * n_cities  # Kirkpatrick-style equilibrium

        # --- Initialization ---
        current_route = list(self.rng.permutation(n_cities))
        current_distance = self._calculate_route_distance(current_route, distance_matrix)

        best_route = current_route.copy()
        best_distance = current_distance

        temp = self.initial_temp
        iteration = 0
        no_improve = 0
        history = [best_distance]

        # --- Annealing Loop ---
        while temp > self.final_temp and iteration < max_iter and no_improve < self.patience:
            for _ in range(self.inner_loops):
                # Generate neighbor (temperature-dependent 2-opt)
                neighbor_route = self._two_opt_swap(current_route, temp)
                neighbor_distance = self._calculate_route_distance(neighbor_route, distance_matrix)
                delta_e = neighbor_distance - current_distance

                # Metropolis criterion
                if delta_e < 0:
                    accept = True
                else:
                    # clamp exponent to avoid overflow
                    expo = min(delta_e / temp, 700.0)
                    accept = (self.rng.random() < np.exp(-expo))
                if accept:
                    current_route = neighbor_route
                    current_distance = neighbor_distance
                    if current_distance < best_distance:
                        best_distance = current_distance
                        best_route = current_route.copy()
                        no_improve = 0
                else:
                    no_improve += 1

                iteration += 1
                if iteration % 200 == 0:
                    history.append(best_distance)

                if iteration >= max_iter or no_improve >= self.patience:
                    break

            # Cool down temperature
            temp = self._cool_down(temp, iteration, max_iter)

        history.append(best_distance)

        return OptimizationResult(
            best_position=best_route,
            best_fitness=best_distance,
            history=history,
            best_route=best_route,
            best_distance=best_distance,
            final_temperature=temp
        )

    # -----------------------------
    # --- Helper Methods ---
    # -----------------------------
    def _calculate_route_distance(self, route: list, distance_matrix: np.ndarray) -> float:
        """Calculate total route distance"""
        distance = 0.0
        for i in range(len(route)):
            distance += distance_matrix[route[i], route[(i + 1) % len(route)]]
        return distance

    def _two_opt_swap(self, route: list, temp: float) -> list:
        """Perform a temperature-dependent 2-opt swap"""
        n = len(route)
        new_route = route.copy()

        # At high temp: longer segments → more exploration
        # At low temp: shorter swaps → fine-tuning
        # Determine the max segment length based on temperature.
        # It shrinks from n/2 down to a minimum of 2.
        max_len = int(2 + (temp / self.initial_temp) * (n / 2 - 2))

        # Ensure max_len is at least 2
        if max_len <= 2:
            seg_len = 2
        else:
            # Choose a segment length up to max_len
            seg_len = self.rng.integers(2, max_len + 1)  # high is exclusive, so +1

        # Choose a starting point for the segment
        i = self.rng.integers(0, n - seg_len + 1)
        j = i + seg_len

        # Reverse the segment
        new_route[i:j] = list(reversed(new_route[i:j]))
        return new_route

    def _cool_down(self, temp: float, iteration: int, max_iter: int) -> float:
        """Apply chosen cooling schedule"""
        if self.cooling_schedule == "exponential":
            return temp * self.alpha
        elif self.cooling_schedule == "linear":
            return self.initial_temp - (self.initial_temp - self.final_temp) * (iteration / max_iter)
        elif self.cooling_schedule == "logarithmic":
            return self.initial_temp / (1 + np.log1p(iteration))
        else:
            return temp * self.alpha


# Convenience functions
def run_simulated_annealing(
    objective_func: Callable,
    dim: int,
    bounds: Tuple[float, float],
    max_iter: int = 2500,
    initial_temp: float = 1000,
    final_temp: float = 1e-3,
    alpha: float = 0.98,
    cooling_schedule: str = 'exponential',
    neighbor_std: float = 0.3,
    inner_loops: int = 50,
    patience: int = 1500,
    minimize: bool = True,
    seed: Optional[int] = None,
    **kwargs
) -> dict:
    """
    Run Simulated Annealing (SA) for continuous optimization problems.
    Supports temperature-dependent perturbations and flexible cooling schedules.

    Args:
        objective_func : Objective function to optimize (min or max).
        dim            : Problem dimensionality.
        bounds         : (lower, upper) bounds for each dimension.
        max_iter       : Maximum number of total iterations.
        initial_temp   : Starting temperature.
        final_temp     : Final temperature (stop criterion).
        alpha          : Cooling rate for exponential schedule.
        cooling_schedule : 'exponential', 'linear', or 'logarithmic'.
        neighbor_std   : Base step size (scaled by temperature).
        inner_loops    : Number of neighbor evaluations per temperature.
        patience       : Early stopping threshold (no improvement limit).
        minimize       : True for minimization, False for maximization.
        seed           : Random seed.

    Returns:
        dict: Dictionary containing best solution, best fitness, history, etc.
    """
    sa = SimulatedAnnealing(
        initial_temp=initial_temp,
        final_temp=final_temp,
        alpha=alpha,
        cooling_schedule=cooling_schedule,
        neighbor_std=neighbor_std,
        inner_loops=inner_loops,
        patience=patience,
        seed=seed
    )
    result = sa.optimize(
        objective_func,
        dim,
        bounds,
        max_iter=max_iter,
        minimize=minimize
    )
    return result.to_dict()


def run_simulated_annealing_tsp(
    distance_matrix: np.ndarray,
    max_iter: int = 20000,
    initial_temp: float = 1000,
    final_temp: float = 1e-3,
    alpha: float = 0.995,
    inner_loops: Optional[int] = None,
    cooling_schedule: str = "exponential",
    patience: int = 2000,
    seed: Optional[int] = None
) -> dict:
    """
    Run Simulated Annealing (SA) for the Traveling Salesman Problem (TSP).
    
    Args:
        distance_matrix : 2D numpy array of distances between cities
        max_iter         : Maximum total number of move attempts
        initial_temp     : Starting temperature
        final_temp       : Final (minimum) temperature
        alpha            : Cooling factor for exponential schedule
        inner_loops      : Number of moves per temperature step (default = 10 * n_cities)
        cooling_schedule : 'exponential', 'linear', or 'logarithmic'
        patience         : Stop early if no improvement for this many iterations
        seed             : Random seed

    Returns:
        dict: Dictionary with best route, distance, and history.
    """
    sa_tsp = SimulatedAnnealingTSP(
        initial_temp=initial_temp,
        final_temp=final_temp,
        alpha=alpha,
        inner_loops=inner_loops,
        cooling_schedule=cooling_schedule,
        patience=patience,
        seed=seed
    )
    result = sa_tsp.optimize_tsp(distance_matrix, max_iter)
    return result.to_dict()

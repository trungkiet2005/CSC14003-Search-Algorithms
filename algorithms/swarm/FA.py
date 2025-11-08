"""algorithms/swarm/fa.py - Improved Firefly Algorithm

Reference: Yang, X. S. (2009). Firefly algorithms for multimodal optimization.
In Stochastic algorithms: foundations and applications (pp. 169-178).
"""

import numpy as np
from typing import Callable, Tuple, Optional
from ..base import PopulationBasedOptimizer, OptimizationResult, run_with_timing


class FA(PopulationBasedOptimizer):
    """Firefly Algorithm
    
    Fireflies are attracted to brighter fireflies. Brightness is determined
    by the objective function value.
    
    Attributes:
        n_fireflies: Number of fireflies
        alpha: Randomization parameter (exploration)
        beta0: Attractiveness at distance r=0
        gamma: Light absorption coefficient
        alpha_decay: Decay rate for alpha (adaptive randomization)
    """
    
    def __init__(self, n_fireflies: int = 25, 
                 alpha: float = 0.5, alpha_min: float = 0.01,
                 beta0: float = 1.0, gamma: float = 1.0,
                 seed: Optional[int] = None):
        super().__init__(population_size=n_fireflies, seed=seed)
        self.n_fireflies = n_fireflies
        self.alpha_initial = alpha
        self.alpha_min = alpha_min
        self.beta0 = beta0
        self.gamma = gamma
        self.name = "FA"
    
    @run_with_timing
    def optimize(self, objective_func: Callable, 
                dim: int, bounds: Tuple[float, float],
                max_iter: int = 100, minimize: bool = True,
                **kwargs) -> OptimizationResult:
        """
        Run Firefly Algorithm optimization
        
        Args:
            objective_func: Objective function to optimize
            dim: Problem dimensionality
            bounds: (lower, upper) bounds for each dimension
            max_iter: Maximum number of iterations
            minimize: True for minimization, False for maximization
            
        Returns:
            OptimizationResult with best solution and history
        """
        lower, upper = bounds
        
        # Initialize firefly positions
        fireflies = self._initialize_population(dim, lower, upper)
        intensity = self._evaluate_population(fireflies, objective_func)
        
        # For minimization, lower objective values = higher brightness
        # For maximization, higher objective values = higher brightness
        if minimize:
            light_intensity = -intensity
        else:
            light_intensity = intensity.copy()
        
        # Initialize best solution
        best_idx = np.argmax(light_intensity)
        best_position = fireflies[best_idx].copy()
        best_fitness = intensity[best_idx]
        history = [best_fitness]
        convergence_iter = None
        
        # Calculate search space scale for step size normalization
        scale = upper - lower
        
        for iteration in range(max_iter):
            # Adaptive randomization parameter (decreases over time)
            alpha_t = self.alpha_min + (self.alpha_initial - self.alpha_min) * \
                     np.exp(-iteration / max_iter)
            
            # Store original positions for distance calculations
            fireflies_old = fireflies.copy()
            
            # Move fireflies
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    # If firefly j is brighter than firefly i
                    if light_intensity[j] > light_intensity[i]:
                        # Calculate Euclidean distance
                        r = np.linalg.norm(fireflies_old[i] - fireflies_old[j])
                        
                        # Normalize distance by search space
                        r_normalized = r / scale
                        
                        # Calculate attractiveness: β(r) = β0 * exp(-γ * r²)
                        beta = self.beta0 * np.exp(-self.gamma * r_normalized ** 2)
                        
                        # Move firefly i towards j
                        # x_i = x_i + β * (x_j - x_i) + α * (rand - 0.5)
                        attraction = beta * (fireflies_old[j] - fireflies_old[i])
                        randomization = alpha_t * scale * (self.rng.random(dim) - 0.5)
                        
                        fireflies[i] += attraction + randomization
                        
                        # Apply boundary constraints
                        fireflies[i] = self._clip_bounds(fireflies[i], lower, upper)
            
            # Evaluate new positions
            intensity = self._evaluate_population(fireflies, objective_func)
            
            # Update light intensity
            if minimize:
                light_intensity = -intensity
            else:
                light_intensity = intensity.copy()
            
            # Update best solution
            current_best_idx = np.argmax(light_intensity)
            if minimize:
                if intensity[current_best_idx] < best_fitness:
                    best_position = fireflies[current_best_idx].copy()
                    best_fitness = intensity[current_best_idx]
            else:
                if intensity[current_best_idx] > best_fitness:
                    best_position = fireflies[current_best_idx].copy()
                    best_fitness = intensity[current_best_idx]
            
            history.append(best_fitness)
            
            # Check convergence
            if convergence_iter is None:
                convergence_iter = self._check_convergence(history)
        
        return OptimizationResult(
            best_position=best_position,
            best_fitness=best_fitness,
            history=history,
            convergence_iter=convergence_iter,
            final_fireflies=fireflies,
            final_intensities=intensity
        )


def run_fa(objective_func: Callable, dim: int, bounds: Tuple[float, float],
          n_fireflies: int = 25, max_iter: int = 100,
          alpha: float = 0.5, beta0: float = 1.0, gamma: float = 1.0,
          minimize: bool = True, seed: Optional[int] = None) -> dict:
    """
    Convenience function to run Firefly Algorithm
    
    Returns dictionary for backward compatibility
    """
    fa = FA(n_fireflies=n_fireflies, alpha=alpha, beta0=beta0, 
            gamma=gamma, seed=seed)
    result = fa.optimize(objective_func, dim, bounds, max_iter, minimize)
    return result.to_dict()
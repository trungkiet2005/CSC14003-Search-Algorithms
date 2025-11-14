import numpy as np
from typing import Callable, Tuple, Dict, Any
from dataclasses import dataclass


class ContinuousProblem:
    """Base class for continuous optimization problems"""
    
    def __init__(self, dim: int):
        self.dim = dim
    
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate the function"""
        return self.evaluate(x)
    
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the function at point x"""
        raise NotImplementedError


class Sphere(ContinuousProblem):
    """Sphere function: f(x) = sum(x_i^2)
    
    Properties:
        - Unimodal
        - Convex
        - Separable
        - Global minimum: f(0,...,0) = 0
        - Search domain: typically [-100, 100]^d
    """
    
    def evaluate(self, x: np.ndarray) -> float:
        x = np.asarray(x)
        return np.sum(x ** 2)


class Rastrigin(ContinuousProblem):
    """Rastrigin function
    
    f(x) = 10*d + sum(x_i^2 - 10*cos(2*pi*x_i))
    
    Properties:
        - Highly multimodal with many local minima
        - Non-convex
        - Separable
        - Global minimum: f(0,...,0) = 0
        - Search domain: typically [-5.12, 5.12]^d
    """
    
    def evaluate(self, x: np.ndarray) -> float:
        x = np.asarray(x)
        d = len(x)
        return 10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


class Rosenbrock(ContinuousProblem):
    """Rosenbrock function (Valley-shaped)
    
    f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    
    Properties:
        - Unimodal
        - Non-convex
        - Non-separable
        - Valley-shaped, challenging for optimization
        - Global minimum: f(1,...,1) = 0
        - Search domain: typically [-5, 10]^d
    """
    
    def evaluate(self, x: np.ndarray) -> float:
        x = np.asarray(x)
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


class Ackley(ContinuousProblem):
    """Ackley function
    
    f(x) = -20*exp(-0.2*sqrt(sum(x_i^2)/d)) 
           - exp(sum(cos(2*pi*x_i))/d) + 20 + e
    
    Properties:
        - Highly multimodal
        - Non-convex
        - Non-separable
        - Global minimum: f(0,...,0) = 0
        - Search domain: typically [-32.768, 32.768]^d
    """
    
    def evaluate(self, x: np.ndarray) -> float:
        x = np.asarray(x)
        d = len(x)
        
        sum_sq = np.sum(x ** 2)
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        
        term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / d))
        term2 = -np.exp(sum_cos / d)
        
        return term1 + term2 + 20 + np.e

# Problem registry
CONTINUOUS_PROBLEMS = {
    'sphere': {
        'class': Sphere,
        'bounds': (-100, 100),
        'optimum': 0.0,
        'optimum_position': 'zeros',
        'description': 'Simple unimodal convex function'
    },
    'rastrigin': {
        'class': Rastrigin,
        'bounds': (-5.12, 5.12),
        'optimum': 0.0,
        'optimum_position': 'zeros',
        'description': 'Highly multimodal function'
    },
    'rosenbrock': {
        'class': Rosenbrock,
        'bounds': (-5, 10),
        'optimum': 0.0,
        'optimum_position': 'ones',
        'description': 'Valley-shaped function'
    },
    'ackley': {
        'class': Ackley,
        'bounds': (-32.768, 32.768),
        'optimum': 0.0,
        'optimum_position': 'zeros',
        'description': 'Multimodal function with exponential terms'
    }
}


def get_problem(name: str, dim: int = 10) -> Tuple[Callable, Dict[str, Any]]:
    """
    Get a continuous optimization problem by name
    
    Args:
        name: Problem name
        dim: Problem dimensionality
        
    Returns:
        Tuple of (objective_function, problem_info)
    """
    if name not in CONTINUOUS_PROBLEMS:
        raise ValueError(f"Unknown problem: {name}")
    
    spec = CONTINUOUS_PROBLEMS[name]
    problem = spec['class'](dim)
    
    # Get optimum position
    if spec['optimum_position'] == 'zeros':
        optimum_pos = np.zeros(dim)
    elif spec['optimum_position'] == 'ones':
        optimum_pos = np.ones(dim)
    elif isinstance(spec['optimum_position'], (int, float)):
        optimum_pos = np.full(dim, spec['optimum_position'])
    else:
        optimum_pos = None
    
    info = {
        'name': name,
        'dim': dim,
        'bounds': spec['bounds'],
        'optimum': spec['optimum'],
        'optimum_position': optimum_pos,
        'description': spec['description'],
        'function': problem
    }
    
    return problem, info
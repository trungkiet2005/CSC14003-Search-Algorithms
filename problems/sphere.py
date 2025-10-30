"""Continuous optimization benchmark problems

Implements Sphere, Rastrigin, Rosenbrock, and Ackley functions.
All functions are minimization problems.
"""

import numpy as np


def sphere(x):
    """Sphere function: f(x) = sum(x_i^2)
    
    Global minimum: f(0,...,0) = 0
    Search domain: typically [-100, 100]^d
    
    Args:
        x: numpy array or list of coordinates
    
    Returns:
        float: function value
    """
    x = np.asarray(x)
    return np.sum(x ** 2)


def rastrigin(x):
    """Rastrigin function: f(x) = 10*d + sum(x_i^2 - 10*cos(2*pi*x_i))
    
    Highly multimodal function with many local minima.
    Global minimum: f(0,...,0) = 0
    Search domain: typically [-5.12, 5.12]^d
    
    Args:
        x: numpy array or list of coordinates
    
    Returns:
        float: function value
    """
    x = np.asarray(x)
    d = len(x)
    return 10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x):
    """Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    
    Valley-shaped function, challenging for optimization.
    Global minimum: f(1,...,1) = 0
    Search domain: typically [-5, 10]^d or [-2.048, 2.048]^d
    
    Args:
        x: numpy array or list of coordinates
    
    Returns:
        float: function value
    """
    x = np.asarray(x)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def ackley(x):
    """Ackley function: complex multimodal function
    
    f(x) = -20*exp(-0.2*sqrt(sum(x_i^2)/d)) - exp(sum(cos(2*pi*x_i))/d) + 20 + e
    
    Global minimum: f(0,...,0) = 0
    Search domain: typically [-32.768, 32.768]^d
    
    Args:
        x: numpy array or list of coordinates
    
    Returns:
        float: function value
    """
    x = np.asarray(x)
    d = len(x)
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)
    
    return term1 + term2 + 20 + np.e


# Problem configurations for easy access
CONTINUOUS_PROBLEMS = {
    'sphere': {
        'function': sphere,
        'bounds': (-100, 100),
        'optimum': 0.0,
        'optimum_position': None  # will be zeros
    },
    'rastrigin': {
        'function': rastrigin,
        'bounds': (-5.12, 5.12),
        'optimum': 0.0,
        'optimum_position': None
    },
    'rosenbrock': {
        'function': rosenbrock,
        'bounds': (-5, 10),
        'optimum': 0.0,
        'optimum_position': None  # will be ones
    },
    'ackley': {
        'function': ackley,
        'bounds': (-32.768, 32.768),
        'optimum': 0.0,
        'optimum_position': None
    }
}

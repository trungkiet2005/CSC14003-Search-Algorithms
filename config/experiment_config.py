from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import json


@dataclass
class AlgorithmConfig:
    """Configuration for an algorithm"""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class ProblemConfig:
    """Configuration for a problem"""
    name: str
    dim: int = 10
    bounds: Tuple[float, float] = (-10, 10)
    optimum: float = 0.0
    max_iter: int = 100


@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    name: str
    problem: ProblemConfig
    algorithms: List[AlgorithmConfig]
    n_runs: int = 30
    seed: int = 42
    output_dir: str = "results"


# ==================== DEFAULT ALGORITHM CONFIGURATIONS ====================

DEFAULT_PSO_CONFIG = AlgorithmConfig(
    name="PSO",
    params={
        "population_size": 30,
        "w": 0.7298,
        "c1": 1.49618,
        "c2": 1.49618,
        "v_max_ratio": 0.2
    }
)

DEFAULT_ACO_CONFIG = AlgorithmConfig(
    name="ACO",
    params={
        "population_size": 20,
        "alpha": 1.0,
        "beta": 2.0,
        "evaporation_rate": 0.1,
        "phi": 0.1,
        "q0": 0.9,
    }
)

DEFAULT_ABC_CONFIG = AlgorithmConfig(
    name="ABC",
    params={
        "population_size": 30,
        "limit": None,  # Will be set as dim * population_size
        "modification_rate": 1.0
    }
)

DEFAULT_FA_CONFIG = AlgorithmConfig(
    name="FA",
    params={
        "population_size": 25,
        "alpha": 0.5,
        "beta0": 1.0,
        "gamma": 1.0
    }
)

DEFAULT_CS_CONFIG = AlgorithmConfig(
    name="CS",
    params={
        "population_size": 25,
        "pa": 0.25,
        "beta": 1.5,
        "step_size_factor": 0.01
    }
)

DEFAULT_GA_CONFIG = AlgorithmConfig(
    name="GA",
    params={
        "population_size": 50,
        "crossover_rate": 0.8,
        "mutation_rate": 0.1,
        "tournament_size": 3,
        "elitism_ratio": 0.1
    }
)

DEFAULT_HC_CONFIG = AlgorithmConfig(
    name="HC",
    params={
        "step_size": 0.1
    }
)

DEFAULT_SA_CONFIG = AlgorithmConfig(
    name="SA",
    params={
        "initial_temp": 1000,
        "final_temp": 1e-3,
        "alpha": 0.99,
        "cooling_schedule": "exponential",
        "neighbor_std": 0.5
    }
)


# ==================== DEFAULT PROBLEM CONFIGURATIONS ====================

SPHERE_CONFIG = ProblemConfig(
    name="sphere",
    dim=10,
    bounds=(-100, 100),
    optimum=0.0,
    max_iter=100
)

RASTRIGIN_CONFIG = ProblemConfig(
    name="rastrigin",
    dim=10,
    bounds=(-5.12, 5.12),
    optimum=0.0,
    max_iter=100
)

ROSENBROCK_CONFIG = ProblemConfig(
    name="rosenbrock",
    dim=10,
    bounds=(-5, 10),
    optimum=0.0,
    max_iter=150
)

ACKLEY_CONFIG = ProblemConfig(
    name="ackley",
    dim=10,
    bounds=(-32.768, 32.768),
    optimum=0.0,
    max_iter=100
)


# ==================== PREDEFINED EXPERIMENTS ====================

SWARM_VS_TRADITIONAL = ExperimentConfig(
    name="swarm_vs_traditional",
    problem=RASTRIGIN_CONFIG,
    algorithms=[
        DEFAULT_PSO_CONFIG,
        DEFAULT_ABC_CONFIG,
        DEFAULT_FA_CONFIG,
        DEFAULT_CS_CONFIG,
        DEFAULT_GA_CONFIG,
        DEFAULT_SA_CONFIG
    ],
    n_runs=30,
    seed=42
)

PSO_PARAMETER_SWEEP = {
    "name": "pso_parameter_sensitivity",
    "algorithm": "PSO",
    "problem": RASTRIGIN_CONFIG,
    "parameter": "w",  # Inertia weight
    "values": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "n_runs": 10
}

SCALABILITY_TEST = {
    "name": "scalability_analysis",
    "algorithm": "PSO",
    "problem": "sphere",
    "dimensions": [5, 10, 20, 30, 50],
    "n_runs": 10,
    "max_iter": 100
}


# ==================== CONFIGURATION UTILITIES ====================

def load_config_from_json(filepath: str) -> ExperimentConfig:
    """Load experiment configuration from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Parse problem config
    problem = ProblemConfig(**data['problem'])
    
    # Parse algorithm configs
    algorithms = [AlgorithmConfig(**algo) for algo in data['algorithms']]
    
    # Create experiment config
    config = ExperimentConfig(
        name=data['name'],
        problem=problem,
        algorithms=algorithms,
        n_runs=data.get('n_runs', 30),
        seed=data.get('seed', 42),
        output_dir=data.get('output_dir', 'results')
    )
    
    return config


def save_config_to_json(config: ExperimentConfig, filepath: str):
    """Save experiment configuration to JSON file"""
    data = {
        'name': config.name,
        'problem': {
            'name': config.problem.name,
            'dim': config.problem.dim,
            'bounds': config.problem.bounds,
            'optimum': config.problem.optimum,
            'max_iter': config.problem.max_iter
        },
        'algorithms': [
            {
                'name': algo.name,
                'params': algo.params,
                'enabled': algo.enabled
            }
            for algo in config.algorithms
        ],
        'n_runs': config.n_runs,
        'seed': config.seed,
        'output_dir': config.output_dir
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def get_default_config(experiment_name: str) -> ExperimentConfig:
    """Get a predefined experiment configuration by name"""
    configs = {
        'swarm_vs_traditional': SWARM_VS_TRADITIONAL,
    }
    
    if experiment_name not in configs:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    
    return configs[experiment_name]


# ==================== RECOMMENDED PARAMETER RANGES ====================

PARAMETER_RANGES = {
    'PSO': {
        'population_size': (20, 50),
        'w': (0.4, 0.9),
        'c1': (1.0, 2.5),
        'c2': (1.0, 2.5)
    },
    'ACO': {
        'population_size': (10, 50),
        'alpha': (0.5, 2.0),
        'beta': (1.0, 5.0),
        'evaporation_rate': (0.01, 0.5),
        'phi': (0.01, 0.5),
        'q0': (0.5, 0.99)
    },
    'ABC': {
        'population_size': (20, 50),
        'limit_factor': [0.5, 1.0, 1.5, 2.0]  # Factor to multiply by dim * population_size
    },
    'FA': {
        'population_size': (15, 40),
        'alpha': (0.2, 1.0),
        'beta0': (0.5, 2.0),
        'gamma': (0.01, 2.0)
    },
    'CS': {
        'population_size': (15, 40),
        'pa': (0.1, 0.4),
        'beta': (1.0, 2.0),
        'step_size_factor': (0.001, 0.1)
    },
    'GA': {
        'population_size': (30, 100),
        'crossover_rate': (0.6, 0.95),
        'mutation_rate': (0.01, 0.2),
        'elitism_ratio': (0.05, 0.2)
    },
    'SA': {
        'initial_temp': (100, 10000),
        'final_temp': (1e-5, 0.1),
        'alpha': (0.85, 0.99)
    }
}


# ==================== UI CONFIGURATION ====================

ALGORITHM_UI_CONFIG = {
    'PSO': {
        'population_size': {'label': 'Population Size', 'default': DEFAULT_PSO_CONFIG.params['population_size'], 'type': int, 'min': 5, 'max': 100},
        'w': {'label': 'Inertia Weight', 'default': DEFAULT_PSO_CONFIG.params['w'], 'type': float, 'min': 0.1, 'max': 1.5},
        'c1': {'label': 'Cognitive Coefficient', 'default': DEFAULT_PSO_CONFIG.params['c1'], 'type': float, 'min': 0.0, 'max': 4.0},
        'c2': {'label': 'Social Coefficient', 'default': DEFAULT_PSO_CONFIG.params['c2'], 'type': float, 'min': 0.0, 'max': 4.0},
        'v_max_ratio': {'label': 'Velocity Limit', 'default': DEFAULT_PSO_CONFIG.params['v_max_ratio'], 'type': float, 'min': 0.05, 'max': 1.0},
    },
    'ACO': {
        'population_size': {'label': 'Population Size', 'default': DEFAULT_ACO_CONFIG.params['population_size'], 'type': int, 'min': 5, 'max': 100},
        'alpha': {'label': 'Alpha (Pheromone Importance)', 'default': DEFAULT_ACO_CONFIG.params['alpha'], 'type': float, 'min': 0.1, 'max': 5.0},
        'beta': {'label': 'Beta (Heuristic Importance)', 'default': DEFAULT_ACO_CONFIG.params['beta'], 'type': float, 'min': 0.1, 'max': 5.0},
        'evaporation_rate': {'label': 'Evaporation Rate', 'default': DEFAULT_ACO_CONFIG.params['evaporation_rate'], 'type': float, 'min': 0.01, 'max': 1.0},
        'phi': {'label': 'Local Update Rate (phi)', 'default': DEFAULT_ACO_CONFIG.params['phi'], 'type': float, 'min': 0.01, 'max': 1.0},
        'q0': {'label': 'Pseudorandom Proportional (q0)', 'default': DEFAULT_ACO_CONFIG.params['q0'], 'type': float, 'min': 0.0, 'max': 1.0},
    },
    'ABC': {
        'population_size': {'label': 'Population Size', 'default': DEFAULT_ABC_CONFIG.params['population_size'], 'type': int, 'min': 5, 'max': 100},
        'limit': {'label': 'Scout Limit', 'default': None, 'type': int, 'min': 10, 'max': 1000, 'optional': True},
        'modification_rate': {'label': 'Modification Rate', 'default': DEFAULT_ABC_CONFIG.params['modification_rate'], 'type': float, 'min': 0.1, 'max': 1.0},
    },
    'FA': {
        'population_size': {'label': 'Population Size', 'default': DEFAULT_FA_CONFIG.params['population_size'], 'type': int, 'min': 5, 'max': 100},
        'alpha': {'label': 'Randomness', 'default': DEFAULT_FA_CONFIG.params['alpha'], 'type': float, 'min': 0.01, 'max': 2.0},
        'beta0': {'label': 'Attractiveness', 'default': DEFAULT_FA_CONFIG.params['beta0'], 'type': float, 'min': 0.1, 'max': 5.0},
        'gamma': {'label': 'Absorption', 'default': DEFAULT_FA_CONFIG.params['gamma'], 'type': float, 'min': 0.01, 'max': 10.0},
    },
    'CS': {
        'population_size': {'label': 'Population Size', 'default': DEFAULT_CS_CONFIG.params['population_size'], 'type': int, 'min': 5, 'max': 100},
        'pa': {'label': 'Discovery Rate', 'default': DEFAULT_CS_CONFIG.params['pa'], 'type': float, 'min': 0.0, 'max': 1.0},
        'beta': {'label': 'Lévy Parameter', 'default': DEFAULT_CS_CONFIG.params['beta'], 'type': float, 'min': 0.5, 'max': 2.5},
        'step_size_factor': {'label': 'Step Scale', 'default': DEFAULT_CS_CONFIG.params['step_size_factor'], 'type': float, 'min': 0.001, 'max': 0.5},
    }
}


def validate_parameters(algorithm: str, params: Dict[str, Any]) -> bool:
    """Validate algorithm parameters are within recommended ranges"""
    if algorithm not in PARAMETER_RANGES:
        return True  # Unknown algorithm, skip validation
    
    ranges = PARAMETER_RANGES[algorithm]
    
    for param_name, value in params.items():
        if param_name in ranges:
            if isinstance(ranges[param_name], tuple):
                min_val, max_val = ranges[param_name]
                if not (min_val <= value <= max_val):
                    print(f"Warning: {algorithm}.{param_name}={value} "
                          f"outside recommended range [{min_val}, {max_val}]")
                    return False
    
    return True


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example 1: Create custom experiment
    custom_experiment = ExperimentConfig(
        name="my_experiment",
        problem=RASTRIGIN_CONFIG,
        algorithms=[DEFAULT_PSO_CONFIG, DEFAULT_GA_CONFIG],
        n_runs=20,
        seed=123
    )
    
    # Save to JSON
    save_config_to_json(custom_experiment, "my_config.json")
    print("Configuration saved to my_config.json")
    
    # Example 2: Load from JSON
    loaded_config = load_config_from_json("my_config.json")
    print(f"Loaded experiment: {loaded_config.name}")
    
    # Example 3: Validate parameters
    is_valid = validate_parameters("PSO", DEFAULT_PSO_CONFIG.params)
    print(f"PSO parameters valid: {is_valid}")
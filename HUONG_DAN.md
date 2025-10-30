# HƯỚNG DẪN SỬ DỤNG PROJECT

## 🚀 Bắt đầu nhanh

### 1. Cài đặt môi trường

```bash
# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Chạy thử nghiệm

**Option A: Demo tự động (không cần nhập input)**

```bash
python run_demo.py
```

Script này sẽ:

- So sánh 5 thuật toán trên Sphere function (10D)
- Tạo visualization 3D cho hàm Sphere (2D)
- Giải TSP với ACO (15 cities)
- Test trên hàm Rastrigin (multimodal)
- Tạo tất cả hình ảnh vào `report/figures/`

**Option B: Chạy main.py với menu**

```bash
python main.py
```

Chọn các option:

- 1: Quick Demo (PSO trên Sphere 2D)
- 2: Full experiments continuous optimization
- 3: TSP experiment
- 4: Tất cả experiments

**Option C: Chạy test để verify code**

```bash
python test_algorithms.py
```

---

## 📚 Cách sử dụng từng module

### 1. Chạy một thuật toán đơn lẻ

```python
from algorithms.swarm.PSO import run_pso
from problems.sphere import sphere

# Chạy PSO
result = run_pso(
    objective_func=sphere,
    dim=10,                    # Số chiều
    bounds=(-100, 100),        # Giới hạn search space
    n_particles=30,            # Số particles
    max_iter=100,              # Số iterations
    seed=42                    # Random seed
)

print(f"Best fitness: {result['best_fitness']}")
print(f"Best position: {result['best_position']}")
print(f"History: {result['history']}")  # Fitness mỗi iteration
```

### 2. So sánh nhiều thuật toán

```python
from utils.benchmark import compare_algorithms
from algorithms.swarm.PSO import run_pso
from algorithms.swarm.ABC import run_abc
from problems.sphere import sphere

algorithms = {
    'PSO': (run_pso, {'dim': 10, 'bounds': (-100, 100), 'n_particles': 30}),
    'ABC': (run_abc, {'dim': 10, 'bounds': (-100, 100), 'n_bees': 30})
}

results = compare_algorithms(
    algorithms,
    sphere,
    n_runs=10,              # Chạy mỗi thuật toán 10 lần
    max_iter=100,
    minimize=True,
    seed=42
)

# In kết quả
for name, stats in results.items():
    print(f"{name}:")
    print(f"  Best: {stats['best_fitness']:.6f}")
    print(f"  Mean: {stats['mean_fitness']:.6f} ± {stats['std_fitness']:.6f}")
    print(f"  Time: {stats['mean_time']:.4f}s")
```

### 3. Visualization

```python
from utils.visualize import (
    plot_convergence_comparison,
    plot_3d_surface,
    plot_contour,
    plot_boxplot_comparison
)

# So sánh convergence
histories = {
    'PSO': pso_result['history'],
    'ABC': abc_result['history']
}
plot_convergence_comparison(
    histories,
    title="Convergence Comparison",
    save_path="figures/comparison.png",
    log_scale=True  # Dùng log scale cho y-axis
)

# 3D surface (chỉ cho 2D functions)
plot_3d_surface(
    sphere,
    bounds=(-10, 10),
    resolution=50,
    best_point=result['best_position'],
    save_path="figures/sphere_3d.png"
)

# Contour plot với particles
plot_contour(
    sphere,
    bounds=(-10, 10),
    best_point=result['best_position'],
    particle_positions=result['positions'],  # Nếu có
    save_path="figures/contour.png"
)

# Boxplot so sánh performance
boxplot_data = {
    'PSO': [0.01, 0.02, 0.015, 0.018, 0.012],
    'ABC': [0.03, 0.025, 0.028, 0.032, 0.029]
}
plot_boxplot_comparison(
    boxplot_data,
    title="Performance Distribution",
    save_path="figures/boxplot.png"
)
```

### 4. TSP với ACO

```python
from problems.tsp import create_tsp_problem
from algorithms.swarm.ACO import run_aco
from utils.visualize import plot_tsp_route

# Tạo TSP instance
tsp = create_tsp_problem(n_cities=20, seed=42)

# Chạy ACO
result = run_aco(
    tsp['distance_matrix'],
    n_ants=30,
    max_iter=100,
    alpha=1.0,          # Pheromone importance
    beta=2.0,           # Heuristic importance
    evaporation_rate=0.5,
    Q=100,
    seed=42
)

print(f"Best distance: {result['best_distance']:.2f}")
print(f"Best route: {result['best_route']}")

# Visualize
plot_tsp_route(
    tsp['cities'],
    result['best_route'],
    result['best_distance'],
    save_path="figures/tsp_solution.png"
)
```

---

## 🎯 Các thuật toán và parameters

### Swarm Intelligence Algorithms

**1. PSO (Particle Swarm Optimization)**

```python
run_pso(
    objective_func,
    dim,
    bounds,
    n_particles=30,    # Số particles
    max_iter=100,
    w=0.7,            # Inertia weight
    c1=1.5,           # Cognitive parameter
    c2=1.5,           # Social parameter
    minimize=True,
    seed=None
)
```

**2. ACO (Ant Colony Optimization)** - Cho TSP

```python
run_aco(
    distance_matrix,
    n_ants=20,
    max_iter=100,
    alpha=1.0,        # Pheromone importance
    beta=2.0,         # Heuristic importance
    evaporation_rate=0.5,
    Q=100,
    seed=None
)
```

**3. ABC (Artificial Bee Colony)**

```python
run_abc(
    objective_func,
    dim,
    bounds,
    n_bees=30,        # Số employed bees
    max_iter=100,
    limit=50,         # Abandonment limit
    minimize=True,
    seed=None
)
```

**4. FA (Firefly Algorithm)**

```python
run_fa(
    objective_func,
    dim,
    bounds,
    n_fireflies=25,
    max_iter=100,
    alpha=0.5,        # Randomization
    beta0=1.0,        # Attractiveness
    gamma=1.0,        # Absorption coefficient
    minimize=True,
    seed=None
)
```

**5. CS (Cuckoo Search)**

```python
run_cs(
    objective_func,
    dim,
    bounds,
    n_nests=25,
    max_iter=100,
    pa=0.25,          # Discovery rate
    beta=1.5,         # Lévy flight parameter
    minimize=True,
    seed=None
)
```

### Traditional Algorithms

**6. Hill Climbing**

```python
run_hill_climbing(
    objective_func,
    dim,
    bounds,
    max_iter=100,
    step_size=0.1,
    minimize=True,
    seed=None
)
```

**7. Simulated Annealing**

```python
run_simulated_annealing(
    objective_func,
    dim,
    bounds,
    max_iter=1000,
    initial_temp=100,
    cooling_rate=0.95,
    minimize=True,
    seed=None
)
```

**8. Genetic Algorithm**

```python
run_ga(
    objective_func,
    dim,
    bounds,
    pop_size=50,
    max_iter=100,
    crossover_rate=0.8,
    mutation_rate=0.1,
    minimize=True,
    seed=None
)
```

---

## 📊 Continuous Optimization Problems

Tất cả có sẵn trong `problems.sphere`:

```python
from problems.sphere import sphere, rastrigin, rosenbrock, ackley

# Sphere: f(x) = sum(x_i^2), bounds=(-100, 100), optimum=0
# Rastrigin: multimodal, bounds=(-5.12, 5.12), optimum=0
# Rosenbrock: valley-shaped, bounds=(-5, 10), optimum=0
# Ackley: multimodal, bounds=(-32.768, 32.768), optimum=0

result = run_pso(rastrigin, dim=10, bounds=(-5.12, 5.12), max_iter=100)
```

---

## 🎓 Tips cho báo cáo

### 1. Chạy nhiều lần để có kết quả ổn định

```python
from utils.benchmark import run_multiple_times

stats = run_multiple_times(
    run_pso,
    n_runs=30,  # Chạy 30 lần
    objective_func=sphere,
    dim=10,
    bounds=(-100, 100),
    max_iter=100,
    seed=None  # Mỗi lần khác seed
)

print(f"Mean: {stats['mean_fitness']:.6f}")
print(f"Std: {stats['std_fitness']:.6f}")
print(f"Best: {stats['best_fitness']:.6f}")
print(f"Worst: {stats['worst_fitness']:.6f}")
```

### 2. So sánh theo nhiều tiêu chí

- **Convergence speed**: Xem plot convergence
- **Solution quality**: Best/Mean/Std fitness
- **Computational time**: Mean time
- **Robustness**: Std càng nhỏ càng tốt

### 3. Test với nhiều dimensions

```python
for dim in [5, 10, 20, 30]:
    result = run_pso(sphere, dim=dim, bounds=(-100, 100), max_iter=100)
    print(f"Dim {dim}: {result['best_fitness']:.6f}")
```

### 4. Parameter sensitivity analysis

```python
# Test different n_particles
for n_particles in [10, 20, 30, 40, 50]:
    result = run_pso(sphere, dim=10, bounds=(-100, 100),
                    n_particles=n_particles, max_iter=100)
    print(f"n_particles={n_particles}: {result['best_fitness']:.6f}")
```

---

## 🐛 Troubleshooting

**Problem: Import errors**

```bash
# Solution: Cài lại dependencies
pip install --upgrade numpy matplotlib seaborn
```

**Problem: BFS/DFS/A\* quá chậm với TSP**

- Dùng với n_cities <= 10 cho BFS/DFS
- Dùng với n_cities <= 12 cho A\*
- Với TSP lớn hơn, dùng ACO

**Problem: Figures không hiển thị**

- Trong script, dùng `matplotlib.use('Agg')` ở đầu file
- Hoặc dùng `save_path` parameter thay vì `plt.show()`

**Problem: Algorithm không hội tụ**

- Tăng `max_iter`
- Điều chỉnh parameters (population size, learning rates, etc.)
- Thử seed khác nhau

---

## 📧 Support

Nếu có vấn đề, check:

1. README.md
2. Code examples trong main.py và run_demo.py
3. Docstrings trong từng function

Good luck với project! 🚀

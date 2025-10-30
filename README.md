# Project 01 - Search Algorithms

**Đồ án môn Cơ sở Trí tuệ Nhân tạo (CSC14003)**  
**VNUHCM - University of Science**

Dự án so sánh và phân tích các thuật toán **Swarm Intelligence** (Trí tuệ bầy đàn) với các thuật toán tối ưu truyền thống trên các bài toán tối ưu hóa liên tục và rời rạc.

---

## 📋 Tổng quan

Project này triển khai và so sánh:

### Thuật toán Swarm Intelligence (5 thuật toán)

- **PSO** (Particle Swarm Optimization) - Tối ưu hóa bầy đàn hạt
- **ACO** (Ant Colony Optimization) - Thuật toán kiến đàn
- **ABC** (Artificial Bee Colony) - Thuật toán ong nhân tạo
- **FA** (Firefly Algorithm) - Thuật toán đom đóm
- **CS** (Cuckoo Search) - Tìm kiếm đỗ quyên

### Thuật toán truyền thống (6 thuật toán)

- **Hill Climbing** - Leo đồi (steepest ascent)
- **Simulated Annealing** - Luyện kim mô phỏng
- **Genetic Algorithm** - Thuật toán di truyền
- **BFS** (Breadth-First Search) - Tìm kiếm theo chiều rộng
- **DFS** (Depth-First Search) - Tìm kiếm theo chiều sâu
- **A\*** - Thuật toán A\* với heuristic

### Bài toán thử nghiệm

**Continuous Optimization:**

- Sphere Function
- Rastrigin Function (multimodal)
- Rosenbrock Function
- Ackley Function

**Discrete Optimization:**

- Traveling Salesman Problem (TSP)
- Knapsack Problem (0/1)

---

## 🚀 Cài đặt và Sử dụng

### Yêu cầu

- Python 3.8+
- NumPy
- Matplotlib
- Seaborn (optional)

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Chạy experiments

**1. Quick Demo (PSO trên Sphere 2D):**

```bash
python main.py
# Chọn option 1 hoặc nhấn Enter
```

**2. Chạy thí nghiệm đầy đủ:**

```bash
python main.py
# Chọn option 4 để chạy tất cả experiments
```

**3. Chạy từng experiment riêng lẻ:**

```python
from main import experiment_continuous_optimization, experiment_tsp

# Continuous optimization
experiment_continuous_optimization(dim=10, max_iter=100, n_runs=5)

# TSP
experiment_tsp(n_cities=20, max_iter=100)
```

---

## 📁 Cấu trúc Project

```
PROJECT_01_SEARCH_ALGORITHMS/
│
├── algorithms/
│   ├── swarm/
│   │   ├── PSO.py          # Particle Swarm Optimization
│   │   ├── ACO.py          # Ant Colony Optimization
│   │   ├── ABC.py          # Artificial Bee Colony
│   │   ├── FA.py           # Firefly Algorithm
│   │   └── CS.py           # Cuckoo Search
│   │
│   └── traditional/
│       ├── hill_climbing.py
│       ├── simulated_annealing.py
│       ├── genetic_algorithm.py
│       ├── BFS.py
│       ├── DFS.py
│       └── AStar.py
│
├── problems/
│   ├── sphere.py           # Continuous functions (sphere, rastrigin, rosenbrock, ackley)
│   ├── tsp.py              # TSP problem generator
│   └── knapsack.py         # Knapsack problem
│
├── utils/
│   ├── benchmark.py        # Performance measurement tools
│   └── visualize.py        # Plotting and visualization
│
├── report/
│   └── figures/            # Generated figures saved here
│
├── main.py                 # Main experiment runner
├── requirements.txt
└── README.md
```

---

## 💡 Ví dụ sử dụng

### Example 1: Chạy PSO trên Sphere Function

```python
from algorithms.swarm.PSO import run_pso
from problems.sphere import sphere

result = run_pso(
    objective_func=sphere,
    dim=10,
    bounds=(-100, 100),
    n_particles=30,
    max_iter=100,
    seed=42
)

print(f"Best fitness: {result['best_fitness']}")
print(f"Best position: {result['best_position']}")
```

### Example 2: So sánh nhiều thuật toán

```python
from utils.benchmark import compare_algorithms
from algorithms.swarm.PSO import run_pso
from algorithms.swarm.ABC import run_abc
from problems.sphere import sphere

algorithms = {
    'PSO': (run_pso, {'dim': 10, 'bounds': (-100, 100)}),
    'ABC': (run_abc, {'dim': 10, 'bounds': (-100, 100)})
}

results = compare_algorithms(
    algorithms,
    sphere,
    n_runs=10,
    minimize=True,
    seed=42
)

for name, stats in results.items():
    print(f"{name}: {stats['mean_fitness']:.6f} ± {stats['std_fitness']:.6f}")
```

### Example 3: Visualize kết quả

```python
from utils.visualize import plot_convergence_comparison, plot_3d_surface

# Plot convergence
histories = {
    'PSO': pso_result['history'],
    'ABC': abc_result['history']
}
plot_convergence_comparison(histories, save_path='figures/comparison.png')

# Plot 3D surface (for 2D problems)
plot_3d_surface(
    sphere,
    bounds=(-10, 10),
    best_point=result['best_position'],
    save_path='figures/sphere_3d.png'
)
```

### Example 4: TSP với ACO

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
    max_iter=100
)

# Visualize
plot_tsp_route(
    tsp['cities'],
    result['best_route'],
    result['best_distance'],
    save_path='figures/tsp_solution.png'
)
```

---

## 📊 Kết quả và Visualizations

Tất cả hình ảnh được tự động lưu vào `report/figures/`:

- **Convergence curves**: So sánh tốc độ hội tụ
- **3D surface plots**: Mặt phẳng hàm mục tiêu
- **Contour plots**: Đường đồng mức với vị trí particles
- **Boxplots**: Phân bố hiệu suất qua nhiều lần chạy
- **TSP routes**: Visualization đường đi TSP

---

## 🔬 Metrics đánh giá

Project đo lường các metrics sau:

1. **Convergence Speed**: Tốc độ hội tụ
2. **Solution Quality**: Chất lượng nghiệm (best, mean, std)
3. **Computation Time**: Thời gian tính toán
4. **Robustness**: Độ ổn định qua nhiều lần chạy
5. **Scalability**: Khả năng mở rộng với kích thước bài toán

---

## 📝 Tham khảo

1. Dorigo, M., et al. (2007). Ant colony optimization. _IEEE Computational Intelligence Magazine_.
2. Wang, D., et al. (2018). Particle swarm optimization algorithm: an overview. _Soft Computing_.
3. Karaboga, D., & Basturk, B. (2007). Artificial bee colony (ABC) algorithm. _Journal of Global Optimization_.
4. Yang, X. S., & He, X. (2013). Firefly algorithm: recent advances and applications. _International Journal of Swarm Intelligence_.
5. Yang, X. S., & Deb, S. (2014). Cuckoo search: recent advances and applications. _Neural Computing and Applications_.

---

## 👥 Nhóm thực hiện

- **Sinh viên 1**: [23122014] - [Hoàng Minh Trung]
- **Sinh viên 2**: [23122015] - [Nguyễn Gia Bảo]
- **Sinh viên 3**: [23122021] - [Bùi Duy Bảo]
- **Sinh viên 3**: [23122039] - [Huỳnh Trung Kiệt]

**Repository**: https://github.com/trungkiet2005/CSC14003-Search-Algorithms

---

## 📄 License

Dự án cho mục đích học tập - VNUHCM University of Science

---

## 🆘 Hỗ trợ

Nếu gặp vấn đề:

1. Kiểm tra đã cài đặt dependencies chưa: `pip install -r requirements.txt`
2. Đảm bảo Python >= 3.8
3. Xem ví dụ trong `main.py`

**Lưu ý**: Các thuật toán BFS/DFS/A\* chỉ phù hợp với TSP có số thành phố nhỏ (< 12 cities) do độ phức tạp exponential.

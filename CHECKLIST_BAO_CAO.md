# CHECKLIST BÁO CÁO ĐỒ ÁN

## ✅ Yêu cầu đã hoàn thành

### 1. Implementation (40%)

#### Swarm Intelligence Algorithms (5/5) ✓

- [x] **PSO** - Particle Swarm Optimization (`algorithms/swarm/PSO.py`)
- [x] **ACO** - Ant Colony Optimization (`algorithms/swarm/ACO.py`)
- [x] **ABC** - Artificial Bee Colony (`algorithms/swarm/ABC.py`)
- [x] **FA** - Firefly Algorithm (`algorithms/swarm/FA.py`)
- [x] **CS** - Cuckoo Search (`algorithms/swarm/CS.py`)

**Tính năng:**

- Chỉ dùng NumPy (không dùng scipy, scikit-learn)
- Có docstring đầy đủ
- Parameters có thể configure
- Trả về best solution + history cho plotting

#### Traditional Algorithms (6/6) ✓

- [x] **Hill Climbing** - Steepest ascent (`algorithms/traditional/hill_climbing.py`)
- [x] **Simulated Annealing** (`algorithms/traditional/simulated_annealing.py`)
- [x] **Genetic Algorithm** (`algorithms/traditional/genetic_algorithm.py`)
- [x] **BFS** - Breadth-First Search (`algorithms/traditional/BFS.py`)
- [x] **DFS** - Depth-First Search (`algorithms/traditional/DFS.py`)
- [x] **A\*** - A\* Search (`algorithms/traditional/AStar.py`)

#### Test Problems ✓

**Continuous Optimization:**

- [x] Sphere function
- [x] Rastrigin function (multimodal)
- [x] Rosenbrock function
- [x] Ackley function

**Discrete Optimization:**

- [x] Traveling Salesman Problem (TSP)
- [x] Knapsack Problem (0/1)

---

### 2. Visualization (Đã hoàn thành) ✓

File: `utils/visualize.py`

- [x] Convergence curves (fitness vs iteration)
- [x] 3D surface plots cho continuous functions
- [x] Contour plots với particle positions
- [x] Boxplots cho performance comparison
- [x] Bar charts cho metrics comparison
- [x] TSP route visualization
- [x] Parameter sensitivity plots (có thể tạo)

**Thư viện:** Matplotlib, Seaborn

---

### 3. Benchmark & Comparison Metrics ✓

File: `utils/benchmark.py`

- [x] Convergence speed measurement
- [x] Computational complexity (time & space)
- [x] Robustness (multiple runs)
- [x] Scalability testing
- [x] Solution quality metrics

---

### 4. Code Quality ✓

- [x] Modular structure
- [x] Well-documented (docstrings)
- [x] Python best practices
- [x] Type hints (có thể thêm)
- [x] README với usage examples
- [x] requirements.txt

---

### 5. Demo & Testing ✓

- [x] `test_algorithms.py` - Test tất cả algorithms
- [x] `run_demo.py` - Auto demo không cần input
- [x] `main.py` - Full experiment runner với menu
- [x] Tất cả tests passed

---

## 📊 Nội dung cần có trong BÁO CÁO

### Phần 1: Thông tin nhóm

- [ ] Danh sách thành viên (MSSV, họ tên)
- [ ] Bảng phân công công việc
- [ ] Tự đánh giá tỷ lệ hoàn thành

### Phần 2: Mô tả thuật toán (Chi tiết từng thuật toán)

Cho mỗi thuật toán cần có:

- [ ] Công thức toán học
- [ ] Pseudo-code hoặc flowchart
- [ ] Giải thích từng bước
- [ ] Ưu điểm / nhược điểm
- [ ] Độ phức tạp thời gian/không gian
- [ ] Ứng dụng thực tế

**Swarm Algorithms:**

- [ ] PSO - Công thức update velocity & position
- [ ] ACO - Công thức pheromone update
- [ ] ABC - Employed/Onlooker/Scout bee phases
- [ ] FA - Công thức attractiveness & movement
- [ ] CS - Lévy flights công thức

**Traditional Algorithms:**

- [ ] Hill Climbing - Local search strategy
- [ ] Simulated Annealing - Cooling schedule
- [ ] Genetic Algorithm - Selection, crossover, mutation
- [ ] BFS/DFS/A\* - Search strategies

### Phần 3: Thiết kế & Implementation

- [ ] Kiến trúc tổng thể (diagram)
- [ ] Class/function design
- [ ] Data structures sử dụng
- [ ] Quyết định thiết kế quan trọng
- [ ] Cách handle continuous vs discrete problems

### Phần 4: Test Cases & Experiments

**Continuous Optimization:**

- [ ] Sphere function results (nhiều dimensions: 5, 10, 20, 30)
- [ ] Rastrigin results (multimodal challenge)
- [ ] Rosenbrock results (valley problem)
- [ ] Ackley results

**Discrete Optimization:**

- [ ] TSP results (10, 15, 20 cities)
- [ ] Knapsack results (nếu test)

**Cho mỗi test case:**

- [ ] Problem description & configuration
- [ ] Algorithm parameters used
- [ ] Results table (best, mean, std, time)
- [ ] Convergence plots
- [ ] Statistical analysis (t-test, ANOVA nếu cần)

### Phần 5: Kết quả & Phân tích

**Comparison Tables:**

- [ ] Performance comparison (all algorithms on each problem)
- [ ] Convergence speed comparison
- [ ] Time complexity comparison
- [ ] Robustness comparison (std across runs)

**Visualizations cần có:**

- [ ] Convergence curves comparison (1 figure cho mỗi problem)
- [ ] 3D surface plots (cho 2D problems)
- [ ] Boxplots (performance distribution)
- [ ] Bar charts (metric comparisons)
- [ ] TSP route visualizations

**Analysis:**

- [ ] Tại sao thuật toán X tốt hơn Y trên problem Z?
- [ ] Ảnh hưởng của parameters
- [ ] Scaling behavior với problem size
- [ ] Trade-off giữa quality và time

### Phần 6: Kết luận

- [ ] Tổng kết kết quả
- [ ] Insights học được
- [ ] Điểm mạnh/yếu của từng approach
- [ ] Khuyến nghị sử dụng
- [ ] Hướng phát triển

### Phần 7: References (APA format)

Đã có trong problems.txt:

- [ ] Dorigo et al. (2007) - ACO
- [ ] Wang et al. (2018) - PSO
- [ ] Karaboga & Basturk (2007) - ABC
- [ ] Yang & He (2013) - FA
- [ ] Yang & Deb (2014) - CS

---

## 🎬 Demo Video Requirements (20%)

- [ ] Ít nhất 5 phút
- [ ] Giới thiệu project structure
- [ ] Demo chạy algorithms
- [ ] Show convergence visualization
- [ ] Giải thích kết quả
- [ ] Upload lên YouTube/Google Drive
- [ ] Link trong báo cáo

**Gợi ý script:**

1. Giới thiệu (30s)
2. Code structure overview (1 min)
3. Chạy test_algorithms.py (1 min)
4. Chạy run_demo.py và show figures (2 min)
5. Giải thích kết quả trên 1-2 test cases (1.5 min)
6. Kết luận (30s)

---

## 📝 Format Báo cáo

- [ ] Tiếng Việt
- [ ] Font chữ dễ đọc (Times New Roman 13pt hoặc Arial 11pt)
- [ ] Có header/footer với số trang
- [ ] Figures có caption và được reference trong text
- [ ] Tables có caption
- [ ] Code snippets có syntax highlighting
- [ ] Minimum 25 pages
- [ ] Export to PDF (well-formatted)
- [ ] Không có hình bị cắt do page break

---

## 📦 Nộp bài (Submission)

File `Group_XX.zip` chứa:

- [ ] `report.pdf` - Báo cáo đầy đủ
- [ ] Source code (cả folder project)
- [ ] `requirements.txt`
- [ ] `README.md` với hướng dẫn chạy
- [ ] Test cases (hoặc link Google Drive nếu >25MB)
- [ ] Link demo video trong báo cáo

**Repository GitHub:**

- [ ] Push code lên GitHub
- [ ] Link: https://github.com/trungkiet2005/CSC14003-Search-Algorithms
- [ ] README có badge, screenshots
- [ ] .gitignore phù hợp

---

## 🎯 Tips để đạt điểm cao

### Code (40%):

✓ Đã implement đầy đủ 11 algorithms
✓ Code clean, modular, documented
✓ README chi tiết với examples
✓ Có tests và demo scripts

### Report (40%):

- Cần viết chi tiết phần mô tả thuật toán (công thức, pseudo-code)
- Phân tích kết quả sâu sắc (không chỉ list numbers)
- Có insights và so sánh có ý nghĩa
- Format đẹp, figures chất lượng cao

### Demo (20%):

- Video rõ ràng, có giọng nói
- Show được highlights của project
- Giải thích technical details
- Thời lượng vừa phải (5-10 phút)

---

## 🚀 Commands để chạy cho báo cáo

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python test_algorithms.py

# Run demo (tạo figures)
python run_demo.py

# Run full experiments với menu
python main.py
```

**Figures được tạo tự động vào:** `report/figures/`

---

## 📊 Suggested Experiments cho báo cáo

### Experiment 1: Algorithm comparison on Sphere (10D)

```bash
# Trong Python
from main import experiment_continuous_optimization
experiment_continuous_optimization(dim=10, max_iter=100, n_runs=30)
```

### Experiment 2: Scalability test

```python
for dim in [5, 10, 20, 30, 50]:
    # Run PSO and measure time
    ...
```

### Experiment 3: Parameter sensitivity (PSO)

```python
for n_particles in [10, 20, 30, 40, 50]:
    # Run and compare
    ...
```

### Experiment 4: TSP comparison

```bash
from main import experiment_tsp
experiment_tsp(n_cities=20, max_iter=100)
```

### Experiment 5: Multimodal optimization

```bash
from main import experiment_rastrigin
experiment_rastrigin(dim=10, max_iter=100, n_runs=30)
```

---

**Lưu ý cuối cùng:**

- Chạy mỗi experiment nhiều lần (n_runs=30) để có statistical significance
- Lưu tất cả figures với quality cao (dpi=300)
- Ghi lại parameters sử dụng cho mỗi experiment
- Có thể thêm more problems (knapsack, graph coloring) nếu muốn extra points

**Good luck! 🎓**

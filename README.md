# Project 01 - Search Algorithms

**Đồ án môn Cơ sở Trí tuệ Nhân tạo (CSC14003)**  
**VNUHCM - University of Science**

Đồ án so sánh và phân tích các thuật toán **Swarm Intelligence** (Trí tuệ bầy đàn) với các thuật toán tối ưu truyền thống trên các bài toán tối ưu hóa liên tục và rời rạc.

---

## Tổng quan

Project này triển khai và so sánh:

### Thuật toán Swarm Intelligence (5 thuật toán)

- **PSO** (Particle Swarm Optimization) - Tối ưu hóa bầy đàn hạt
- **ACO** (Ant Colony Optimization) - Thuật toán kiến đàn
- **ABC** (Artificial Bee Colony) - Thuật toán ong nhân tạo
- **FA** (Firefly Algorithm) - Thuật toán đom đóm
- **CS** (Cuckoo Search) - Tìm kiếm đỗ quyên

### Thuật toán truyền thống (3 thuật toán)

- **Hill Climbing** - Leo đồi (steepest ascent)
- **Simulated Annealing** - Luyện kim mô phỏng
- **Genetic Algorithm** - Thuật toán di truyền

### Bài toán thử nghiệm

**Continuous Optimization:**

- Sphere Function
- Rastrigin Function
- Rosenbrock Function
- Ackley Function

**Discrete Optimization:**

- Traveling Salesman Problem (TSP)

---

## Cài đặt và Sử dụng

### Yêu cầu

- Python 3.12.12
- PyQt6
- NumPy
- Matplotlib
- Seaborn

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Chạy ứng dụng GUI

Để chạy giao diện đồ họa của ứng dụng, sử dụng lệnh sau:

```bash
python main.py
```

Giao diện cung cấp hai tab chính:
1.  **Algorithm Visualization**: Phân tích và trực quan hóa hành vi của một thuật toán bất kì trên một bài toán.
2.  **Algorithm Comparison**: So sánh hiệu suất của hai thuật toán đối đầu trên cùng một bài toán.

---

## Cấu trúc Project

```
CSC14003-Search-Algorithms/
│
├── algorithms/
│   ├── base.py
│   ├── swarm/
│   │   ├── ABC.py
│   │   ├── ACO.py
│   │   ├── CS.py
│   │   ├── FA.py
│   │   └── PSO.py
│   └── traditional/
│       ├── genetic_algorithm.py
│       ├── hill_climbing.py
│       └── simulated_annealing.py
│
├── config/
│   └── experiment_config.py
│
├── gui/
│   ├── __init__.py
│   ├── app.py
│   ├── comparison/
│   │   ├── comparison_runner.py
│   │   └── comparison_tab.py
│   └── visualization/
│       ├── visualization_runner.py
│       └── visualization_tab.py
│
├── problems/
│   ├── continuous.py
│   └── tsp.py
│
├── results/
│   └── ...
│
├── utils/
│   ├── benchmark.py
│   └── visualize.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Kết quả và Visualizations

Nếu nhấn `Export All` trên giao diện thì tất cả hình ảnh được tự động lưu vào `results/figures/`.

- **Convergence curves**: So sánh tốc độ hội tụ
- **3D surface plots**: Mặt phẳng hàm mục tiêu
- **Contour plots**: Đường đồng mức với vị trí particles
- **Boxplots**: Phân bố hiệu suất qua nhiều lần chạy
- **TSP routes**: Visualization đường đi TSP

---

## Metrics đánh giá

Project đo lường các metrics sau:

1. **Convergence Speed**: Tốc độ hội tụ
2. **Solution Quality**: Chất lượng nghiệm
3. **Computation Time**: Thời gian tính toán
4. **Robustness**: Độ ổn định qua nhiều lần chạy
5. **Scalability**: Khả năng mở rộng với kích thước bài toán

---

## Tham khảo

1. Dorigo, M., et al. (2007). Ant colony optimization. _IEEE Computational Intelligence Magazine_.
2. Wang, D., et al. (2018). Particle swarm optimization algorithm: an overview. _Soft Computing_.
3. Karaboga, D., & Basturk, B. (2007). Artificial bee colony (ABC) algorithm. _Journal of Global Optimization_.
4. Yang, X. S., & He, X. (2013). Firefly algorithm: recent advances and applications. _International Journal of Swarm Intelligence_.
5. Yang, X. S., & Deb, S. (2014). Cuckoo search: recent advances and applications. _Neural Computing and Applications_.

---

## Nhóm thực hiện

- **Sinh viên 1**: 23122014 - Hoàng Minh Trung
- **Sinh viên 2**: 23122015 - Nguyễn Gia Bảo
- **Sinh viên 3**: 23122021 - Bùi Duy Bảo
- **Sinh viên 3**: 23122039 - Huỳnh Trung Kiệt

**Repository**: https://github.com/trungkiet2005/CSC14003-Search-Algorithms

---

## License

Đồ án phục vụ cho mục đích học tập - VNUHCM University of Science

---

## Hỗ trợ

Nếu gặp vấn đề:

1. Kiểm tra đã cài đặt dependencies chưa: `pip install -r requirements.txt`
2. Đảm bảo đúng phiên bản Python
3. Chạy ứng dụng bằng `python main.py`
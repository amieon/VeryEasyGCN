# VeryEasyGCN — 从零实现的稀疏图卷积网络（C++）

用纯 C++（不依赖任何深度学习框架）从零实现的两层图卷积网络（GCN）。项目的重点不在"实现一个 GCN"——而在把它当**系统问题**来做：自己写稀疏算子、量它的性能、用数值梯度检验证明反向传播正确、并把核心 kernel 做多核并行。

核心亮点：

- **真正高效的 CSR-SpMM kernel**：稀疏邻接矩阵 × 稠密特征，按非零做顺序访存的 AXPY，比"把稀疏矩阵当稠密做 GEMM"在稀疏图上快约 **80×**（开销随非零数 nnz 增长，而非 n²）。
- **扁平连续内存**：`DenseMatrix` 用单块 row-major 缓冲；稀疏/稠密是编译期已知的静态类型，无运行时切换开销。
- **数值梯度检验**：解析梯度 vs 中心差分，最大相对误差 **3e-9**，证明前向/反向实现正确。
- **完整训练栈**：softmax + 交叉熵、masked 训练/验证/测试、AdamW、Dropout，在 Cora 节点分类上达到与原论文同量级的准确率。
- **OpenMP 行级并行 SpMM**：并行结果与串行逐位一致。

------

## 项目结构

```
.
├── Tensor.h        # 核心：DenseMatrix（扁平 row-major）+ CSR（真正的 SpMM kernel）
├── Func.h          # 激活/损失、softmax、交叉熵、初始化、Dropout、图归一化
├── GCNLayer.h      # GCN 层：前向 / 反向 / 参数更新（可选 ReLU / Identity 激活）
├── Optim.h         # AdamW 优化器
├── Dataset.h       # Cora 加载器 + 合成数据集（SBM / Cora-like）
├── main.cpp        # teacher–student 训练（验证训练链路）
├── train_cora.cpp  # Cora 节点分类主程序
├── gradcheck.cpp   # 数值梯度检验
├── bench_spmm.cpp  # SpMM 性能基准（新 vs 旧 vs 稠密）
├── bench_omp.cpp   # OpenMP 多核扩展性基准
└── Matrix.h        # 旧实现（已退役，仅 bench_spmm 留作对照基线）
```

------

## 架构设计

GCN 的数据流里，归一化邻接矩阵 Â 恒为稀疏、节点特征 X 与权重 W 恒为稠密——这是编译期就确定的事实。因此项目把存储拆成两个静态类型，而非用一个会在运行时按密度来回切换的矩阵类：

- **`DenseMatrix<T>`**：单块连续 `std::vector<T>`，row-major。所有稠密 GEMM（`matmul` / `matmul_AtB` / `matmul_ABt`）走裸指针、按行连续访存。
- **`CSR<T>`**：标准压缩稀疏行（`row_ptr` / `col_idx` / `values`），核心方法是 `spmm`。

### 核心算子：CSR-SpMM

`C = Â · B`（Â 稀疏 m×k，B 稠密 k×n）的实现是教科书写法——遍历每个非零，对 B 的对应稠密行做一次 AXPY 累加到 C 的行上，全程顺序访存：

```cpp
for (int i = 0; i < rows; ++i) {
    T* crow = C.row_ptr(i);
    for (int idx = row_ptr[i]; idx < row_ptr[i+1]; ++idx) {
        const T a = values[idx];
        const T* brow = B.row_ptr(col_idx[idx]);
        for (int j = 0; j < N; ++j)
            crow[j] += a * brow[j];        // AXPY
    }
}
```

每行只写自己的 C 行、互不依赖，所以最外层行循环可直接用 `#pragma omp parallel for` 并行，无数据竞争。

------

## 数学推导

### 前向传播

每一层：

$$ H^{(l+1)} = \sigma!\left(\tilde{A}, H^{(l)} W^{(l)}\right),\qquad \tilde{A} = \hat{D}^{-\frac12}(A+I)\hat{D}^{-\frac12} $$

其中 $H^{(0)}=X$，$\sigma$ 为 ReLU（末层取恒等以输出 logits）。实现上拆成两步：先稠密 GEMM 算 $XW$，再用 SpMM 左乘稀疏 $\tilde{A}$。

### 反向传播

设 $Z = \tilde{A}(XW)$，$M = XW$。给定上层回传的 $\partial \mathcal{L}/\partial H$：

$$ \delta_Z = \frac{\partial \mathcal{L}}{\partial H}\odot \sigma'(Z) $$ $$ \frac{\partial \mathcal{L}}{\partial M} = \tilde{A}^{\top}\delta_Z \quad(\text{SpMM}),\qquad \frac{\partial \mathcal{L}}{\partial W} = X^{\top}!\left(\tilde{A}^{\top}\delta_Z\right),\qquad \frac{\partial \mathcal{L}}{\partial X} = \left(\tilde{A}^{\top}\delta_Z\right) W^{\top} $$

（对称归一化下 $\tilde{A}^{\top}=\tilde{A}$，代码仍构造转置以保持对一般图的正确性。）

### softmax + 交叉熵的合并梯度

末层输出 logits，经行 softmax 得 $P$，masked 交叉熵对 logits 的梯度有简洁形式：

$$ \frac{\partial \mathcal{L}}{\partial Z_{\text{logits}}} = \frac{1}{|\text{mask}|},(P - \text{onehot}(y)),\quad \text{仅训练节点非零} $$

------

## 性能基准（`bench_spmm.cpp`）

单线程，`-O2`。Â 为 n×n 稀疏，X 为 n×64 稠密。（具体数字依机器而定，以下为一次实测。）

**新 CSR-SpMM vs 旧实现的 set/get 累加：**

| n    | density | 旧 (ms) | 新 (ms) | 加速 |
| ---- | ------- | ------- | ------- | ---- |
| 64   | 0.05    | 0.157   | 0.0071  | ~22× |
| 256  | 0.05    | 1.363   | 0.118   | ~12× |

**新 CSR-SpMM vs 把 Â 当稠密做 GEMM：**

| n    | density | spmm (ms) | dense (ms) | 加速 |
| ---- | ------- | --------- | ---------- | ---- |
| 512  | 0.01    | 0.12      | 10.4       | ~87× |
| 1024 | 0.01    | 0.42      | 33.2       | ~79× |
| 2048 | 0.05    | 8.34      | 133.8      | ~16× |

关键结论：SpMM 开销随 nnz 增长、稠密 GEMM 随 n² 增长，所以**图越稀疏优势越大**。说明：此处稠密 baseline 是朴素三重循环而非优化过的 BLAS。

### 多核扩展性（`bench_omp.cpp`）

`-fopenmp` 编译后，行级并行的 SpMM 在多核机器上可进一步加速；并行结果与串行逐位一致。需要注意 SpMM 是**访存受限（memory-bound）**的——每个非零只做一次乘加却要拉取一整行——所以并行效率受内存带宽而非核数限制，通常前几个线程接近线性、之后饱和。

|threads |   time(ms) |  speedup | efficiency|
| ---- | ------- | --------- | ---------- |
|      1 |    18.5059 |    1.00x |       100%|
|      2 |    10.8692 |    1.70x |        85%|
|      4 |     6.2577 |    2.96x |        74%|
|      8 |     4.6131 |    4.01x |        50%|



## 正确性：数值梯度检验（`gradcheck.cpp`）

用 `double` + 中心差分对照解析梯度：

```
W1  max relative error = 3.411e-09
W2  max relative error = 3.657e-09
-> PASS (反向传播正确)
```

------

## 训练与结果

### Cora 节点分类（`train_cora.cpp`）

两层 GCN（hidden=16），AdamW（`lr=0.01, wd=5e-4`）+ Dropout 0.5，transductive 全图前向、仅在 train 掩码上回传。

下载原始 LINQS 格式的 `cora.content` 与 `cora.cites`（GitHub 多有镜像）后：

```bash
g++ train_cora.cpp -o train -std=c++17 -O2
./train cora.content cora.cites
```

不带参数则回退到合成数据集（无需下载即可验证管线）。预期测试准确率约 **78–82%**，与 GCN 原论文（81.5%）同量级。

### 优化器对照（合成 Cora-like 数据，可复现）

Cora 的特征是高维稀疏词袋、行归一化后量级极小，导致梯度过小、纯 SGD 几乎不动——这是 GCN 普遍使用 Adam 的原因。在同款 regime 的合成集上可清楚复现：

| 优化器 | 学习率 | 200 epoch 后 train_loss | 测试准确率 |
| ------ | ------ | ----------------------- | ---------- |
| SGD    | 0.2    | 1.946 → 1.945（爬不动） | 0.29       |
| AdamW  | 0.01   | 1.946 → 0.033           | 0.99       |

（合成集干净可分，故达 ~99%；真实 Cora 噪声更大，约 80%。）

------

## 编译与运行

环境：C++17，g++ / clang++。

```bash
# 训练（Cora 或合成回退）
g++ train_cora.cpp -o train     -std=c++17 -O2
g++ train_cora.cpp -o train     -std=c++17 -O2 -fopenmp   # 启用并行 SpMM
OMP_NUM_THREADS=8 ./train cora.content cora.cites

# 梯度检验
g++ gradcheck.cpp  -o gradcheck -std=c++17 -O2 && ./gradcheck

# 性能基准
g++ bench_spmm.cpp -o bench     -std=c++17 -O2 && ./bench
g++ bench_omp.cpp  -o bench_omp -std=c++17 -O2 -fopenmp && ./bench_omp
```

> 注意：需 **C++17**（`Tensor.h` / `Dataset.h` 使用了结构化绑定等特性）。

------

## 性能优化:三个层次

SpMM / GEMM 在三个层次上做了优化,并用 benchmark 量化每一层的贡献。

### 1. 算法层 — CSR 稀疏
只遍历非零,开销 O(nnz) 而非 O(n²)。详见上方 SpMM vs 稠密 GEMM 对照。

### 2. 指令级 — AVX2 SIMD
内层 AXPY / 点积用 AVX2 向量化(float 一次 8 个、double 一次 4 个),
通过 `-DUSE_AVX` 开关可对照标量版。

控制变量(锁定单线程),SpMM on 8000×8000, density=0.005, N=128:

| 版本      | 单线程时间 |      加速 |
| --------- | ---------: | --------: |
| 标量      |   15.55 ms |     1.00× |
| AVX2 SIMD |    4.62 ms | **3.37×** |

正确性由 `gradcheck` 保证:SIMD 版与标量版数值等价(double 最大相对误差 3.4e-9)。

### 3. 线程级 — OpenMP
行级并行(`#pragma omp parallel for`),每行独占写出,无数据竞争,
并行结果与串行逐位一致。

| 线程数 | 时间 (ms) |  加速 | 并行效率 |
| -----: | --------: | ----: | -------: |
|      1 |      4.62 | 1.00× |     100% |
|      2 |      3.04 | 1.52× |      76% |
|      4 |      2.02 | 2.28× |      57% |
|      8 |      1.63 | 2.83× |      35% |

**端到端(标量单线程 → SIMD + 8 线程):约 9.5×。**

### 关键观察:优化计算会把瓶颈推向内存带宽

并行效率随线程数下降(8 线程仅 35%),说明 SpMM 在此配置下是
**memory-bound** 的——多线程很快撞到内存带宽天花板。

更有意思的是对照标量版:标量版 8 线程效率 51%,SIMD 版只有 35%。
原因是 **SIMD 让单线程计算变快后,内存带宽更早成为瓶颈**,于是并行
扩展性反而变差。这是 HPC 里典型的"优化一个瓶颈,下一个瓶颈随即暴露"。

此外,SpMM 的 memory-bound 程度取决于稠密侧宽度 N:N 越大计算密度
越高,SIMD 收益越明显(本例 N=128 时 SIMD 仍有 3.37× 提升)。

> 说明:数字为单机实测(CPU 型号: 12th Gen Intel(R) Core(TM) i7-12700H , GCC 8.1.0),仅供说明趋势;
> 不同硬件绝对值会变,但"效率随线程下降""SIMD 后瓶颈转移"的规律稳定。

---



## 实现要点

- **存储**：稠密用扁平 row-major 缓冲；稀疏用 CSR。两者为静态类型，无运行时密度切换。
- **算子**：`matmul`（ikj 顺序，cache 友好）、`matmul_AtB` / `matmul_ABt`（避免显式转置）、`CSR::spmm`（AXPY kernel）、`CSR::transpose`。
- **训练**：Xavier 初始化、AdamW、Inverted Dropout（缓存 mask 供反向）、masked 损失与准确率。
- **可复现**：全局固定种子的 RNG。

## 局限与后续方向

- 当前为全图（full-batch）transductive 训练，未做 mini-batch / 邻居采样。
- SpMM 仅 CPU + OpenMP，未做 SIMD 显式向量化或 CUDA。
- 模型为标准两层 GCN，未实现 GAT / GraphSAGE 等变体。

后续可扩展：CUDA SpMM、邻居采样的 mini-batch 训练、更多 GNN 变体。

## 参考

- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks* (ICLR 2017)
- Cora 数据集（LINQS）

## 作者

聂宇航
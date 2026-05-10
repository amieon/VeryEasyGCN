# 基于 CSR 稀疏矩阵的图卷积神经网络（GCN）实现

## 项目简介

本项目使用 C++ 从零实现了一个简化版图卷积神经网络（Graph Convolutional Network, GCN）框架，重点完成了：

- 基于 CSR（Compressed Sparse Row）格式的稀疏矩阵实现
- 稀疏矩阵与稠密矩阵的自动切换
- GCN 前向传播与反向传播
- ReLU 激活函数
- MSE 损失函数
- 基于梯度下降的参数更新
- 图邻接矩阵归一化

该项目主要用于学习：

- 图神经网络（GNN / GCN）基本原理
- 稀疏矩阵存储与运算
- 神经网络反向传播机制
- C++ 模板编程
- 深度学习底层实现思想

------

# 项目结构

```text
.
├── main.cpp          # 主函数，训练流程入口
├── Matrix.h          # 稀疏/稠密矩阵类实现（CSR核心）
├── Graph.h           # 图结构定义
├── GCNLayer.h        # GCN层实现
├── Func.h            # 激活函数、损失函数、图归一化等工具函数
└── 综合实验--图卷积神经网络.docx
```

------

# 核心功能说明

## 1. CSR 稀疏矩阵实现

项目中的 `Matrix` 类支持两种模式：

```cpp
enum class Mode {
    SPARSE,
    DENSE
};
```

其中：

- `SPARSE`：使用 CSR 格式存储
- `DENSE`：使用二维 `vector` 存储

CSR 结构：

```cpp
template<typename T>
struct CSRMatrix {
    int rows, cols;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<T> values;
};
```

特点：

- 节省稀疏矩阵内存
- 提高矩阵乘法效率
- 支持动态插入与删除元素
- 支持 CSR 与 Dense 自动切换

------

## 2. 稀疏/稠密自动切换

矩阵会根据密度自动切换存储模式：

```cpp
void checkModeSwitch()
```

切换逻辑：

- 密度较低 → 使用 CSR
- 密度较高 → 使用 Dense

这样可以兼顾：

- 稀疏矩阵的存储效率
- 稠密矩阵的计算效率

------

## 3. GCN 层实现

GCN 的核心公式：

genui{"math_block_widget_always_prefetch_v2":{"content":"H = \sigma(\hat{A} X W)"}}

其中：

- `X`：节点特征矩阵
- `W`：可训练权重矩阵
- `Ĥ`：归一化邻接矩阵
- `σ`：ReLU 激活函数

前向传播：

```cpp
Z = adjHat * (X * weight);
return outputH = relu(Z);
```

反向传播实现：

- ReLU 梯度计算
- 权重梯度计算
- 输入梯度回传

------

## 4. 邻接矩阵归一化

项目中实现了标准 GCN 的邻接矩阵归一化：

genui{"math_block_widget_always_prefetch_v2":{"content":"\hat{A} = D^{-\frac{1}{2}} (A + I) D^{-\frac{1}{2}}"}}

作用：

- 避免节点度数差异过大
- 提高训练稳定性
- 防止特征尺度爆炸

实现函数：

```cpp
normalizeAdjacency()
```

------

## 5. 激活函数

项目实现了：

### ReLU

genui{"math_block_widget_always_prefetch_v2":{"content":"f(x)=\max(0,x)"}}

对应函数：

```cpp
relu()
reluDerivative()
```

------

## 6. 损失函数

使用均方误差（MSE）：

genui{"math_block_widget_always_prefetch_v2":{"content":"MSE = \frac{1}{nc}\sum_{i=1}^{n}\sum_{j=1}^{c}(y_{ij}-\hat{y}_{ij})^2"}}

实现函数：

```cpp
MSELoss()
LossGrad()
```

------

# 训练流程

主程序训练过程：

```text
1. 初始化权重矩阵
2. 随机生成图结构
3. 随机生成节点特征
4. 前向传播
5. 计算损失
6. 反向传播
7. 更新参数
8. 重复训练
```

训练代码入口：

```cpp
int main()
```

主要超参数：

```cpp
int max_epoch = 200;
float learning_rate = 0.00005;
```

------

# 编译与运行

## 环境要求

- C++11 及以上
- g++ / clang++ / MSVC

推荐：

- GCC 11+
- CLion / VSCode / Visual Studio

------

## Linux / MacOS

编译：

```bash
g++ main.cpp -o gcn -std=c++11
```

运行：

```bash
./gcn
```

------

## Windows（MinGW）

```bash
g++ main.cpp -o gcn.exe -std=c++11
```

运行：

```bash
gcn.exe
```

------

# 示例输出

```text
epoch 0 : loss = 0.284321
epoch 10 : loss = 0.197542
epoch 20 : loss = 0.154873
...
```

随着训练进行，Loss 会逐渐下降。

------

# 项目特点

## 优点

### 1. 从零实现

未依赖任何深度学习框架：

- 无 PyTorch
- 无 TensorFlow
- 无 Eigen

更适合理解底层原理。

------

### 2. 支持 CSR 稀疏矩阵

相比普通二维数组：

- 更节省空间
- 更适合图结构数据
- 更符合真实 GNN 场景

------

### 3. 自动存储模式切换

实现了：

- Sparse ↔ Dense 动态切换
- 自动根据矩阵密度优化

------

### 4. 包含完整训练流程

实现了：

- Forward
- Backward
- Gradient
- Weight Update

具备完整神经网络训练结构。

------

# 项目不足

当前项目仍属于教学级实现，存在一些限制：

- 使用随机生成数据，而非真实图数据集
- 未实现 Batch 训练
- 未实现 GPU 加速
- 未实现 Adam 优化器
- 未实现多层复杂 GNN 结构
- CSR 运算仍有进一步优化空间

------

# 可扩展方向

后续可以继续扩展：

## 图神经网络方向

- GAT（Graph Attention Network）
- GraphSAGE
- GIN
- RGCN

## 工程优化方向

- OpenMP 并行化
- CUDA GPU 加速
- SIMD 优化
- 更高效 CSR SpMM

## 深度学习方向

- Adam / RMSProp
- Dropout
- BatchNorm
- 残差连接

------

# 学习收获

通过本实验，可以深入理解：

- 图卷积的数学原理
- 稀疏矩阵存储结构
- 神经网络反向传播
- 深度学习框架底层机制
- C++ 模板与矩阵运算实现

相比直接调用深度学习框架，本项目更加注重：

- 底层实现
- 数据结构设计
- 数学推导理解
- 算法实现能力

------

# 参考资料

## GCN 论文

Thomas Kipf & Max Welling:

> Semi-Supervised Classification with Graph Convolutional Networks

------

## 推荐学习资料

- CS231n
- Stanford GNN Course
- PyTorch Geometric
- Deep Learning（Ian Goodfellow）
- 图神经网络：基础与前沿

------

# License

本项目仅用于学习与课程实验。
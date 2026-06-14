#pragma once
#include "Tensor.h"
#include <random>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include <tuple>

// 全局可复现 RNG
inline std::mt19937& global_rng() { static std::mt19937 gen(42); return gen; }
inline void seed_rng(unsigned s) { global_rng().seed(s); }

// ---- 激活 / 损失（稠密）----
template<typename T>
DenseMatrix<T> relu(const DenseMatrix<T>& m) {
    DenseMatrix<T> out(m.rows, m.cols);
    for (size_t i = 0; i < m.data.size(); ++i)
        out.data[i] = m.data[i] > T(0) ? m.data[i] : T(0);
    return out;
}
template<typename T>
DenseMatrix<T> relu_deriv(const DenseMatrix<T>& m) {
    DenseMatrix<T> out(m.rows, m.cols);
    for (size_t i = 0; i < m.data.size(); ++i)
        out.data[i] = m.data[i] > T(0) ? T(1) : T(0);
    return out;
}
template<typename T>
T MSELoss(const DenseMatrix<T>& pred, const DenseMatrix<T>& label) {
    assert(pred.rows == label.rows && pred.cols == label.cols);
    T loss = T(0);
    for (size_t i = 0; i < pred.data.size(); ++i) {
        T d = pred.data[i] - label.data[i];
        loss += d * d;
    }
    return loss / T(pred.data.size());
}
template<typename T>
DenseMatrix<T> LossGrad(const DenseMatrix<T>& pred, const DenseMatrix<T>& label) {
    assert(pred.rows == label.rows && pred.cols == label.cols);
    DenseMatrix<T> g(pred.rows, pred.cols);
    T scale = T(2) / T(pred.data.size());  // d(MSE)/d(pred)
    for (size_t i = 0; i < pred.data.size(); ++i)
        g.data[i] = scale * (pred.data[i] - label.data[i]);
    return g;
}

// ---- 初始化 ----
template<typename T>
void init_xavier(DenseMatrix<T>& A) {
    T limit = std::sqrt(T(6) / T(A.rows + A.cols));
    std::uniform_real_distribution<T> dist(-limit, limit);
    for (auto& x : A.data) x = dist(global_rng());
}
template<typename T>
void init_gaussian(DenseMatrix<T>& A, T mean = T(0), T stddev = T(1)) {
    std::normal_distribution<T> dist(mean, stddev);
    for (auto& x : A.data) x = dist(global_rng());
}

// ---- 图：随机对称边 + 对称归一化邻接（直接构造 CSR）----
inline std::vector<std::pair<int,int>> random_symmetric_edges(int n, double p) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (dist(global_rng()) < p) edges.emplace_back(i, j);
    return edges;
}

// Â = D^{-1/2} (A + I) D^{-1/2}，返回 CSR（行内列升序）
template<typename T>
CSR<T> build_normalized_adj(int n, const std::vector<std::pair<int,int>>& edges) {
    std::vector<std::vector<int>> nbr(n);
    for (int i = 0; i < n; ++i) nbr[i].push_back(i);  // 自环
    for (auto& pr : edges) { nbr[pr.first].push_back(pr.second); nbr[pr.second].push_back(pr.first); }

    std::vector<T> inv_sqrt_deg(n);
    for (int i = 0; i < n; ++i) {
        std::sort(nbr[i].begin(), nbr[i].end());
        nbr[i].erase(std::unique(nbr[i].begin(), nbr[i].end()), nbr[i].end());
        inv_sqrt_deg[i] = T(1) / std::sqrt(T(nbr[i].size()));
    }

    std::vector<std::tuple<int,int,T>> trips;
    for (int i = 0; i < n; ++i)
        for (int j : nbr[i])
            trips.emplace_back(i, j, inv_sqrt_deg[i] * inv_sqrt_deg[j]);
    return CSR<T>::from_sorted_triplets(n, n, trips);
}

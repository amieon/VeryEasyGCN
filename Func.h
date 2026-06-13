#pragma once
#include "Matrix.h"
#include <random>
#include <cmath>
#include <cassert>


// 全局 RNG：默认固定种子，保证结果可复现。需要随机性时调用 seed_rng(...)。
inline std::mt19937& global_rng() {
    static std::mt19937 gen(42);
    return gen;
}
inline void seed_rng(unsigned s) { global_rng().seed(s); }

// 对称归一化邻接矩阵： Ĥ = D^{-1/2} (A + I) D^{-1/2}
template<typename T>
Matrix<T> normalizeAdjacency(const Matrix<T>& A) {
    int n = A.rows;
    Matrix<T> A_hat = A;
    for (int i = 0; i < n; ++i)
        A_hat.set(i, i, 1);  // 加自环

    std::vector<T> deg(n, T{});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            deg[i] += A_hat.get(i, j);

    Matrix<T> D_inv_sqrt(n, n);
    for (int i = 0; i < n; ++i)
        D_inv_sqrt.set(i, i, T(1) / std::sqrt(deg[i] + T(1e-8)));  // 避免除零

    return D_inv_sqrt * A_hat * D_inv_sqrt;
}

template<typename T>
Matrix<T> reluDerivative(const Matrix<T>& m) {
    Matrix<T> out(m.rows, m.cols);
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            out.set(i, j, (m.get(i, j) > 0) ? T(1) : T(0));
    return out;
}

template<typename T>
Matrix<T> relu(const Matrix<T>& m) {
    Matrix<T> out(m.rows, m.cols);
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            out.set(i, j, (m.get(i, j) > 0) ? m.get(i, j) : T(0));
    return out;
}

template<typename T>
Matrix<T> MSELoss(const Matrix<T>& pred, const Matrix<T>& label) {
    assert(pred.rows == label.rows && pred.cols == label.cols &&
           "MSELoss: pred / label 形状不一致");
    T loss = T(0);
    int n = pred.rows, c = pred.cols;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < c; ++j) {
            T diff = pred.get(i, j) - label.get(i, j);
            loss += diff * diff;
        }
    Matrix<T> result(1, 1);
    result.set(0, 0, loss / T(n * c));
    return result;
}

template<typename T>
Matrix<T> LossGrad(const Matrix<T>& pred, const Matrix<T>& label) {
    assert(pred.rows == label.rows && pred.cols == label.cols &&
           "LossGrad: pred / label 形状不一致");
    int n = pred.rows, c = pred.cols;
    Matrix<T> grad(n, c);
    // d(MSE)/d(pred) = 2/(n*c) * (pred - label)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < c; ++j)
            grad.set(i, j, T(2) * (pred.get(i, j) - label.get(i, j)) / T(n * c));
    return grad;
}

// Glorot / Xavier 均匀初始化，均值为 0
template<typename T>
void init_xavier(Matrix<T>& A) {
    T limit = std::sqrt(T(6) / T(A.rows + A.cols));
    std::uniform_real_distribution<T> dist(-limit, limit);
    for (int i = 0; i < A.rows; ++i)
        for (int j = 0; j < A.cols; ++j)
            A.set(i, j, dist(global_rng()));
}

// 标准正态特征，便于产生有效信号
template<typename T>
void init_gaussian(Matrix<T>& A, T mean = T(0), T stddev = T(1)) {
    std::normal_distribution<T> dist(mean, stddev);
    for (int i = 0; i < A.rows; ++i)
        for (int j = 0; j < A.cols; ++j)
            A.set(i, j, dist(global_rng()));
}

// 生成对称的 Erdős–Rényi 随机无向图邻接矩阵（无自环，自环在归一化时加）
template<typename T>
void make_symmetric_adj(Matrix<T>& A, double p = 0.3) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < A.rows; ++i)
        for (int j = i + 1; j < A.cols; ++j) {
            T e = (dist(global_rng()) < p) ? T(1) : T(0);
            A.set(i, j, e);
            A.set(j, i, e);
        }
}

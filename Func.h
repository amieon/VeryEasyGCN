#pragma once
#include "Matrix.h"
#include <random>
#include <cmath>
template<typename T>
Matrix<T> normalizeAdjacency(const Matrix<T>& A){
    int n = A.rows;
    Matrix<T> A_hat = A;
    for(int i = 0;i < n; ++i)
        A_hat.set(i, i, 1);
    // 计算度矩阵 D
    std::vector<T> deg(n, 0.0f);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            deg[i] += A_hat.get(i, j);

    // 构造 D^(-1/2)
    Matrix<T> D_inv_sqrt(n, n);
    for (int i = 0; i < n; ++i)
        D_inv_sqrt.set(i, i, 1.0f / std::sqrt(deg[i] + 1e-8f)); // 避免除零

    // 计算 Ĥ = D^(-1/2) * A_hat * D^(-1/2)
    return D_inv_sqrt * A_hat * D_inv_sqrt;
}

template<typename T>
Matrix<T> reluDerivative(const Matrix<T>& m){
    Matrix<T> out(m.rows, m.cols);
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            out.set(i,j,(m.get(i,j) > 0) ? 1.0f : 0.0f);
    return out;
}

template<typename T>
Matrix<T> relu(const Matrix<T>& m){
    Matrix<T> out(m.rows, m.cols);
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            out.set(i,j,(m.get(i,j) > 0) ? m.get(i,j) : 0.0f);
    return out;
}

template<typename T>
Matrix<T> MSELoss(const Matrix<T>& pred, const Matrix<T>& label) {
    T loss = static_cast<T>(0);
    int n = pred.rows;
    int c = pred.cols;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < c; ++j) {
            T diff = pred.get(i, j) - label.get(i, j);
            loss += diff * diff;
        }
    }

    Matrix<T> result(1, 1);
    result.set(0, 0, loss / static_cast<T>(n * c));
    return result;
}

template<typename T>
Matrix<T> LossGrad(const Matrix<T>& pred, const Matrix<T>& label) {
    int n = pred.rows;
    int c = pred.cols;
    Matrix<T> grad(n, c);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < c; ++j)
            grad.set(i, j, (pred.get(i,j) - label.get(i,j)) / n);

    return grad;
}
template<typename T>
void random_inti_Matrix(Matrix<T>& A){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(0.0f, 1.0f);

    for (int i = 0; i < A.rows; ++i)
        for (int j = 0; j < A.cols; ++j)
            A.set(i,j,dist(gen));
}

template<typename T>
void read_Graph(Matrix<T>& A){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 10);

    for (int i = 0; i < A.rows; ++i)
        for (int j = 0; j < A.cols; ++j)
            A.set(i,j,T(dist(gen)%2));
}
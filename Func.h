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
void read_Graph(Matrix<T>& A){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 10);

    for (int i = 0; i < A.rows; ++i)
        for (int j = 0; j < A.cols; ++j)
            A.set(i,j,T(dist(gen)%2));
}
#pragma once
#include "Matrix.h"
#include "Func.h"

template<typename T>
class GCNLayer {
public:
    Matrix<T> weight;      // 参数 W
    Matrix<T> gradWeight;  // W 的梯度
    Matrix<T> adjHat;      // 归一化邻接矩阵

    // 前向缓存（用于反向传播）
    Matrix<T> inputX;      // 输入 X
    Matrix<T> Z;           // 激活前 Z = Ĥ X W
    Matrix<T> outputH;     // 激活后 H = σ(Z)


    GCNLayer(const Matrix<T>& W_init, const Matrix<T>& adj)
            : weight(W_init),
              gradWeight(W_init.rows, W_init.cols),
              adjHat(normalizeAdjacency(adj)),
              inputX(),
              Z(),
              outputH()
    {}


    Matrix<T> forward(const Matrix<T>& X) {
        inputX = X;
        Z = adjHat * (X * weight);  // Z = Ĥ X W
        return outputH = relu(Z);
    }

    // 反向传播：给定上层传下来的 dL/dH，返回 dL/dX
    Matrix<T> backward(const Matrix<T>& dLoss_dH) {
        Matrix<T> dZ = dLoss_dH % reluDerivative(Z);
        Matrix<T> dZ_adj = adjHat.transpose() * dZ;
        gradWeight = inputX.transpose() * dZ_adj;

        Matrix<T> dX = dZ_adj * weight.transpose();

        return dX;
    }


    void updateWeights(T lr) {
        weight = weight - gradWeight * lr;
    }

};

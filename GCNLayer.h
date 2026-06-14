#pragma once
#include "Tensor.h"
#include "Func.h"

template<typename T>
class GCNLayer {
public:
    DenseMatrix<T> weight;      // 参数 W
    DenseMatrix<T> gradWeight;  // dL/dW
    CSR<T> adjHat;              // 归一化邻接 Â（稀疏）
    CSR<T> adjHatT;             // Â^T（反向用；对称图下 == Â）

    // 前向缓存
    DenseMatrix<T> inputX, Z, outputH;

    GCNLayer() = default;
    explicit GCNLayer(const DenseMatrix<T>& W)
        : weight(W), gradWeight(W.rows, W.cols) {}

    void set_adj(const CSR<T>& a) { adjHat = a; adjHatT = a.transpose(); }

    // 前向: Z = Â (X W),  H = relu(Z)
    DenseMatrix<T> forward(const DenseMatrix<T>& X) {
        inputX = X;
        DenseMatrix<T> XW = matmul(X, weight);  // 稠密 GEMM
        Z = adjHat.spmm(XW);                    // SpMM：稀疏 Â × 稠密
        outputH = relu(Z);
        return outputH;
    }

    // 反向: 输入 dL/dH，返回 dL/dX
    DenseMatrix<T> backward(const DenseMatrix<T>& dH) {
        DenseMatrix<T> dZ = hadamard(dH, relu_deriv(Z));  // ⊙ σ'(Z)
        DenseMatrix<T> dM = adjHatT.spmm(dZ);             // dM = Â^T dZ
        gradWeight = matmul_AtB(inputX, dM);              // dW = X^T dM
        DenseMatrix<T> dX = matmul_ABt(dM, weight);       // dX = dM W^T
        return dX;
    }

    void updateWeights(T lr) { axpy_update(weight, gradWeight, lr); }
};

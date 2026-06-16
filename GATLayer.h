#pragma once
#include "Tensor.h"
#include "Func.h"

template<typename T>
class GATLayer {
public:
    enum class Activation { Relu, Identity };  // 末层用 Identity 输出 logits

    DenseMatrix<T> weight;
    DenseMatrix<T> gradWeight;
    CSR<T> adjHat, adjHatT;
    CSR<T> e;
    DenseMatrix<T> alpha_src,alpha_dst;
    DenseMatrix<T> inputX, Z, outputH;
    Activation act = Activation::Relu;

    GATLayer() = default;
    explicit GATLayer(const DenseMatrix<T>& W, Activation a = Activation::Relu)
            : weight(W), gradWeight(W.rows, W.cols),
              alpha_src(W.cols, 1), alpha_dst(W.cols, 1),   // 形状自己算，不靠外面
              act(a)
    {
        init_xavier(alpha_src);   // 初始化也在里面做
        init_xavier(alpha_dst);
    }

    void set_adj(const CSR<T>& a) { adjHat = a; adjHatT = a.transpose(); e = adjHat; }

    DenseMatrix<T> forward(const DenseMatrix<T>& X) {
        inputX = X;
        Z = matmul(X, weight);

        DenseMatrix<T> tmp_src = matmul(Z,alpha_src);
        DenseMatrix<T> tmp_dst = matmul(Z,alpha_dst);

        for (int i = 0; i < adjHat.rows; ++i) {
            for (int idx = adjHat.row_ptr[i]; idx < adjHat.row_ptr[i+1]; ++idx) {
                e.values[idx] = tmp_dst(adjHat.col_idx[idx],0) + tmp_src(i,0);
                if(e.values[idx] < 0) e.values[idx] *= 0.02;
            }
        }
        e = sparse_softmax(e);

        Z = e.spmm(Z);
        outputH = (act == Activation::Relu) ? relu(Z) : Z;
        return outputH;
    }

    DenseMatrix<T> backward(const DenseMatrix<T>& dH) {
        DenseMatrix<T> dZ = (act == Activation::Relu) ? hadamard(dH, relu_deriv(Z)) : dH;
        DenseMatrix<T> dM = adjHatT.spmm(dZ);        // dM = Â^T dZ
        gradWeight = matmul_AtB(inputX, dM);         // dW = X^T dM
        DenseMatrix<T> dX = matmul_ABt(dM, weight);  // dX = dM W^T
        return dX;
    }

    void updateWeights(T lr) { axpy_update(weight, gradWeight, lr); }
};

#include "GCNLayer.h"
#include "Func.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

// 用 double + 中心差分验证反向传播：解析梯度 vs 数值梯度
using T = double;

int main() {
    seed_rng(1);
    const int n = 5, F_in = 3, F_hidden = 4, F_out = 2;

    auto edges = random_symmetric_edges(n, 0.5);
    CSR<T> Ahat = build_normalized_adj<T>(n, edges);

    DenseMatrix<T> X(n, F_in), Y(n, F_out);
    init_gaussian(X); init_gaussian(Y);

    DenseMatrix<T> W1(F_in, F_hidden), W2(F_hidden, F_out);
    init_xavier(W1); init_xavier(W2);

    GCNLayer<T> L1(W1), L2(W2);
    L1.set_adj(Ahat); L2.set_adj(Ahat);

    auto loss_fn = [&]() {
        DenseMatrix<T> h = L1.forward(X);
        DenseMatrix<T> out = L2.forward(h);
        return MSELoss(out, Y);
    };

    // 解析梯度
    loss_fn();
    DenseMatrix<T> h = L1.outputH, out = L2.outputH;
    DenseMatrix<T> dOut = LossGrad(out, Y);
    L1.backward(L2.backward(dOut));
    DenseMatrix<T> gW1 = L1.gradWeight, gW2 = L2.gradWeight;

    // 数值梯度（中心差分）
    const T eps = 1e-6;
    auto numeric_grad = [&](DenseMatrix<T>& W) {
        DenseMatrix<T> g(W.rows, W.cols);
        for (size_t k = 0; k < W.data.size(); ++k) {
            T orig = W.data[k];
            W.data[k] = orig + eps; T lp = loss_fn();
            W.data[k] = orig - eps; T lm = loss_fn();
            W.data[k] = orig;
            g.data[k] = (lp - lm) / (2 * eps);
        }
        return g;
    };
    DenseMatrix<T> nW1 = numeric_grad(L1.weight);
    DenseMatrix<T> nW2 = numeric_grad(L2.weight);

    auto max_rel_err = [](const DenseMatrix<T>& a, const DenseMatrix<T>& b) {
        T m = 0;
        for (size_t k = 0; k < a.data.size(); ++k) {
            T num = std::fabs(a.data[k] - b.data[k]);
            T den = std::max((T)1e-12, std::max(std::fabs(a.data[k]), std::fabs(b.data[k])));
            m = std::max(m, num / den);
        }
        return m;
    };

    printf("gradient check (analytic vs numeric, central diff, double)\n");
    printf("  W1  max relative error = %.3e\n", max_rel_err(gW1, nW1));
    printf("  W2  max relative error = %.3e\n", max_rel_err(gW2, nW2));
    printf("  -> %s\n", (max_rel_err(gW1,nW1) < 1e-5 && max_rel_err(gW2,nW2) < 1e-5)
                        ? "PASS (反向传播正确)" : "FAIL");
    return 0;
}

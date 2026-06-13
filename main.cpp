#include "GCNLayer.h"
#include "Func.h"
#include <cstdio>
#include <vector>

using namespace std;

// ---------------------------------------------------------------------------
// teacher–student 健全性检验
//   固定一个 "教师" 2 层 GCN，对每个图生成标签 Y = teacher(Â, X)。
//   学生网络从随机初始化开始拟合。因为目标在同一架构下是可表示的，
//   只要前向 / 反向 / 参数更新都正确，loss 就应当稳定下降并逼近 0。
// ---------------------------------------------------------------------------

int main() {
    const int N        = 16;   // 数据集中的图数量
    const int n        = 12;   // 每个图的节点数
    const int F_in     = 8;    // 输入特征维度
    const int F_hidden = 16;   // 隐藏维度
    const int F_out    = 4;    // 输出维度
    const int max_epoch     = 300;
    const float learning_rate = 0.05f;

    seed_rng(42);

    // ---- 教师权重（固定，用来生成标签）----
    Matrix<float> tW1(F_in, F_hidden), tW2(F_hidden, F_out);
    init_xavier(tW1); init_xavier(tW2);

    // ---- 数据集：每个图有自己的邻接、特征，以及教师生成的标签 ----
    vector<Matrix<float>> Ahat(N), X(N), Y(N);
    {
        Matrix<float> dummy(n, n);  // 仅用于构造教师层
        GCNLayer<float> teacher1(tW1, dummy), teacher2(tW2, dummy);
        for (int i = 0; i < N; ++i) {
            Matrix<float> A(n, n);
            make_symmetric_adj(A, 0.3);
            Ahat[i] = normalizeAdjacency(A);

            X[i] = Matrix<float>(n, F_in);
            init_gaussian(X[i]);

            teacher1.adjHat = Ahat[i];
            teacher2.adjHat = Ahat[i];
            Matrix<float> h = teacher1.forward(X[i]);
            Y[i] = teacher2.forward(h);
        }
    }

    // ---- 学生网络（随机初始化，待训练）----
    Matrix<float> sW1(F_in, F_hidden), sW2(F_hidden, F_out);
    init_xavier(sW1); init_xavier(sW2);
    Matrix<float> dummy(n, n);
    GCNLayer<float> layer1(sW1, dummy), layer2(sW2, dummy);

    vector<float> loss_curve;
    loss_curve.reserve(max_epoch);

    for (int epoch = 0; epoch < max_epoch; ++epoch) {
        float epoch_loss = 0.0f;
        for (int i = 0; i < N; ++i) {
            layer1.adjHat = Ahat[i];
            layer2.adjHat = Ahat[i];

            Matrix<float> out = layer1.forward(X[i]);
            out = layer2.forward(out);

            epoch_loss += MSELoss(out, Y[i]).get(0, 0);

            Matrix<float> dOut = LossGrad(out, Y[i]);
            layer1.backward(layer2.backward(dOut));
            layer1.updateWeights(learning_rate);
            layer2.updateWeights(learning_rate);
        }
        loss_curve.push_back(epoch_loss / N);  // 每个 epoch 一个平均 loss
    }

    // ---- 正确地打印每个 epoch 的平均 loss ----
    printf("epoch | avg MSE loss\n");
    for (int epoch = 0; epoch < max_epoch; ++epoch)
        if (epoch % 20 == 0 || epoch == max_epoch - 1)
            printf("%5d | %.6f\n", epoch, loss_curve[epoch]);

    return 0;
}

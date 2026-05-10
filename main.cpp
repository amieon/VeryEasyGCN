#include "GCNLayer.h"
#include "Func.h"
using namespace std;
int main(){
    int max_epoch = 200;
    float learning_rate = 0.00005;
    Matrix<float> W_init1(10,64);
    Matrix<float> W_init2(64,1);
    random_inti_Matrix(W_init1),random_inti_Matrix(W_init2);
    Matrix<float> adj(6, 6);
    GCNLayer<float> layer1(W_init1, adj);
    GCNLayer<float> layer2(W_init2, adj);
    vector<Matrix<float>> adjs(100,Matrix<float>(6,6)),X(100,Matrix<float>(6,10)),Y(100,Matrix<float>(6,10));
    for(int i=0;i<100;++i) {
        read_Graph(adjs[i]);
        random_inti_Matrix(X[i]);
        random_inti_Matrix(Y[i]);
    }

    vector<float> losses;
    for (int epoch = 0; epoch < max_epoch; ++epoch) {
        for(int i=0;i<100;++i) {
            layer2.adjHat = layer1.adjHat = normalizeAdjacency(adjs[i]);

            Matrix<float> out = layer1.forward(X[i]);
            out = layer2.forward(out);

            Matrix<float> loss = MSELoss(out, Y[i]);
            losses.push_back(loss.get(0, 0));
            Matrix<float> dLoss_dOut = LossGrad(out, Y[i]);

            layer1.backward(layer2.backward(dLoss_dOut));
            layer1.updateWeights(learning_rate);
            layer2.updateWeights(learning_rate);
        }
    }
    for (int epoch = 0; epoch < max_epoch; ++epoch){
        if((epoch)%10 == 0)
            printf("epoch %d : loss = %f\n",epoch,losses[epoch]);
    }
    return 0;
}
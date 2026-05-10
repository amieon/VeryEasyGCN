#pragma once
#include "Matrix.h"
#include <random>
#include <cmath>
template<typename T>

template<typename T>
void read_Graph(Matrix<T>& A){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 10);

    for (int i = 0; i < A.rows; ++i)
        for (int j = 0; j < A.cols; ++j)
            A.set(i,j,T(dist(gen)%2));
}
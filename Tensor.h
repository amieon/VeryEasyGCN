#pragma once
#include <vector>
#include <cassert>
#include <cstddef>
#include <utility>

template<typename T>
struct DenseMatrix {
    int rows = 0, cols = 0;
    std::vector<T> data;  // row-major, 连续存储

    DenseMatrix() = default;
    DenseMatrix(int r, int c) : rows(r), cols(c), data((size_t)r * c, T{}) {}

    inline T&       operator()(int i, int j)       { return data[(size_t)i * cols + j]; }
    inline const T& operator()(int i, int j) const { return data[(size_t)i * cols + j]; }

    inline T*       row_ptr(int i)       { return &data[(size_t)i * cols]; }
    inline const T* row_ptr(int i) const { return &data[(size_t)i * cols]; }
};


// 稠密运算（自由函数，便于内联与后续并行/向量化）
// C = A * B   (ikj 顺序：对 B 的行做连续 AXPY，cache 友好)
template<typename T>
DenseMatrix<T> matmul(const DenseMatrix<T>& A, const DenseMatrix<T>& B) {
    assert(A.cols == B.rows);
    const int M = A.rows, K = A.cols, N = B.cols;
    DenseMatrix<T> C(M, N);
    for (int i = 0; i < M; ++i) {
        T* crow = C.row_ptr(i);
        const T* arow = A.row_ptr(i);
        for (int k = 0; k < K; ++k) {
            const T a = arow[k];
            const T* brow = B.row_ptr(k);
            for (int j = 0; j < N; ++j)
                crow[j] += a * brow[j];
        }
    }
    return C;
}

// C = A^T * B   (A: M×P, 结果 P×N；用于 dW = X^T dM)
template<typename T>
DenseMatrix<T> matmul_AtB(const DenseMatrix<T>& A, const DenseMatrix<T>& B) {
    assert(A.rows == B.rows);
    const int M = A.rows, P = A.cols, N = B.cols;
    DenseMatrix<T> C(P, N);
    for (int k = 0; k < M; ++k) {
        const T* arow = A.row_ptr(k);   // A[k][*]
        const T* brow = B.row_ptr(k);   // B[k][*]
        for (int i = 0; i < P; ++i) {
            const T a = arow[i];
            T* crow = C.row_ptr(i);
            for (int j = 0; j < N; ++j)
                crow[j] += a * brow[j];
        }
    }
    return C;
}

// C = A * B^T   (A: M×K, B: N×K, 结果 M×N；用于 dX = dM W^T)
template<typename T>
DenseMatrix<T> matmul_ABt(const DenseMatrix<T>& A, const DenseMatrix<T>& B) {
    assert(A.cols == B.cols);
    const int M = A.rows, K = A.cols, N = B.rows;
    DenseMatrix<T> C(M, N);
    for (int i = 0; i < M; ++i) {
        const T* arow = A.row_ptr(i);
        T* crow = C.row_ptr(i);
        for (int j = 0; j < N; ++j) {
            const T* brow = B.row_ptr(j);
            T s = T{};
            for (int l = 0; l < K; ++l)
                s += arow[l] * brow[l];
            crow[j] = s;
        }
    }
    return C;
}

// 逐元素 Hadamard 积
template<typename T>
DenseMatrix<T> hadamard(const DenseMatrix<T>& A, const DenseMatrix<T>& B) {
    assert(A.rows == B.rows && A.cols == B.cols);
    DenseMatrix<T> C(A.rows, A.cols);
    for (size_t i = 0; i < A.data.size(); ++i)
        C.data[i] = A.data[i] * B.data[i];
    return C;
}

// A <- A - lr * G   (原地更新，避免重新分配)
template<typename T>
void axpy_update(DenseMatrix<T>& A, const DenseMatrix<T>& G, T lr) {
    assert(A.rows == G.rows && A.cols == G.cols);
    for (size_t i = 0; i < A.data.size(); ++i)
        A.data[i] -= lr * G.data[i];
}


// CSR 稀疏矩阵
template<typename T>
struct CSR {
    int rows = 0, cols = 0;
    std::vector<int> row_ptr;   // 长度 rows+1
    std::vector<int> col_idx;   // 长度 nnz
    std::vector<T>   values;    // 长度 nnz

    CSR() = default;
    CSR(int r, int c) : rows(r), cols(c), row_ptr(r + 1, 0) {}

    int nnz() const { return (int)values.size(); }
    double density() const { return rows && cols ? (double)nnz() / ((double)rows * cols) : 0.0; }

    // 核心算子：C = this(稀疏 m×k) * B(稠密 k×n) -> C(稠密 m×n)
    // 每个非零做一次对稠密行的 AXPY，全程顺序访存，无 set/get 开销。
    DenseMatrix<T> spmm(const DenseMatrix<T>& B) const {
        assert(cols == B.rows);
        const int N = B.cols;
        DenseMatrix<T> C(rows, N);
        for (int i = 0; i < rows; ++i) {
            T* crow = C.row_ptr(i);
            for (int idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx) {
                const T a = values[idx];
                const T* brow = B.row_ptr(col_idx[idx]);
                for (int j = 0; j < N; ++j)
                    crow[j] += a * brow[j];
            }
        }
        return C;
    }

    // 转置：构造 A^T 的 CSR（用于反向传播；对称图下 A^T == A，但保持通用正确）
    CSR<T> transpose() const {
        CSR<T> Tm(cols, rows);
        Tm.row_ptr.assign(cols + 1, 0);
        for (int idx = 0; idx < nnz(); ++idx)
            Tm.row_ptr[col_idx[idx] + 1]++;
        for (int i = 0; i < cols; ++i)
            Tm.row_ptr[i + 1] += Tm.row_ptr[i];
        Tm.col_idx.resize(nnz());
        Tm.values.resize(nnz());
        std::vector<int> cursor(Tm.row_ptr.begin(), Tm.row_ptr.end() - 1);
        for (int i = 0; i < rows; ++i)
            for (int idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx) {
                int c = col_idx[idx];
                int pos = cursor[c]++;
                Tm.col_idx[pos] = i;
                Tm.values[pos] = values[idx];
            }
        return Tm;
    }

    // 转成稠密（仅供 benchmark 对照用）
    DenseMatrix<T> to_dense() const {
        DenseMatrix<T> D(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int idx = row_ptr[i]; idx < row_ptr[i + 1]; ++idx)
                D(i, col_idx[idx]) = values[idx];
        return D;
    }

    // 从已排序的 (i, j, val) 三元组构造（同一行内 j 递增）
    static CSR<T> from_sorted_triplets(int r, int c,
                                       const std::vector<std::tuple<int,int,T>>& t) {
        CSR<T> M(r, c);
        M.col_idx.reserve(t.size());
        M.values.reserve(t.size());
        int cur = 0;
        for (auto& [i, j, v] : t) {
            while (cur < i) M.row_ptr[++cur] = (int)M.values.size();
            M.col_idx.push_back(j);
            M.values.push_back(v);
        }
        while (cur < r) M.row_ptr[++cur] = (int)M.values.size();
        return M;
    }
};

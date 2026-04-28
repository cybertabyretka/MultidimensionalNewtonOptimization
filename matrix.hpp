#pragma once

#include "exceptions/vector_matrix_exceptions.hpp"

template <typename T>
class Matrix {
    size_t rows_, cols_;
    std::vector<T> data_;

    size_t idx(size_t r, size_t c) const { return r * cols_ + c; }

    void check_bounds(size_t r, size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Matrix index out of bounds");
        }
    }

public:
    Matrix(size_t r = 0, size_t c = 0, T val = T{}) : rows_(r), cols_(c), data_(r * c, val) {}

    size_t rows() const noexcept { return rows_; }
    size_t cols() const noexcept { return cols_; }

    const T& at(size_t r, size_t c) const { check_bounds(r, c); return data_[idx(r, c)]; }
    T& at(size_t r, size_t c) { check_bounds(r, c); return data_[idx(r, c)]; }

    static Matrix identity(size_t n) {
        Matrix I(n, n, T{});
        for (size_t i = 0; i < n; ++i) I.at(i, i) = T{1};
        return I;
    }

    Matrix operator+(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw DimensionMismatchError("Matrix addition dimension mismatch");
        }
        Matrix res(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) res.data_[i] = data_[i] + other.data_[i];
        return res;
    }

    Matrix operator-(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw DimensionMismatchError("Matrix subtraction dimension mismatch");
        }
        Matrix res(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) res.data_[i] = data_[i] - other.data_[i];
        return res;
    }

    Matrix operator*(T scalar) const {
        Matrix res(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) res.data_[i] = data_[i] * scalar;
        return res;
    }

    friend Matrix operator*(T scalar, const Matrix& m) {
        return m * scalar;
    }

    Matrix operator*(const Matrix& other) const {
        if (cols_ != other.rows_) {
            throw DimensionMismatchError("Matrix multiplication dimension mismatch");
        }
        Matrix res(rows_, other.cols_, T{});
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t k = 0; k < cols_; ++k) {
                const T aik = at(i, k);
                for (size_t j = 0; j < other.cols_; ++j) {
                    res.at(i, j) += aik * other.at(k, j);
                }
            }
        }
        return res;
    }

    Vector<T> operator*(const Vector<T>& v) const {
        if (cols_ != v.size()) {
            throw DimensionMismatchError("Matrix-Vector multiplication dimension mismatch");
        }
        Vector<T> res(rows_, T{});
        for (size_t i = 0; i < rows_; ++i) {
            T sum{};
            for (size_t j = 0; j < cols_; ++j) sum += at(i, j) * v[j];
            res[i] = sum;
        }
        return res;
    }

    Matrix transpose() const {
        Matrix res(cols_, rows_, T{});
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                res.at(j, i) = at(i, j);
            }
        }
        return res;
    }

    bool is_symmetric(double tol = 1e-12) const {
        if (rows_ != cols_) return false;
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = i + 1; j < cols_; ++j) {
                if (std::abs(at(i, j) - at(j, i)) > static_cast<T>(tol)) {
                    return false;
                }
            }
        }
        return true;
    }

    bool is_positive_definite(double tol = 1e-12) const {
        if (rows_ != cols_) return false;
        if (!is_symmetric(1e-10)) return false;

        const size_t n = rows_;
        using RealType = std::conditional_t<
            std::is_compound_v<T>, 
            typename T::value_type, 
            std::common_type_t<T, double>>;
        Matrix<RealType> L(n, n, RealType{0});

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                double sum = static_cast<double>(at(i, j));
                for (size_t k = 0; k < j; ++k) {
                    sum -= L.at(i, k) * L.at(j, k);
                }

                if (i == j) {
                    if (sum <= tol) return false;
                    L.at(i, j) = std::sqrt(sum);
                } else {
                    if (std::abs(L.at(j, j)) <= tol) return false;
                    L.at(i, j) = sum / L.at(j, j);
                }
            }
        }
        return true;
    }

    static Vector<T> solve(const Matrix& A, const Vector<T>& b) {
        if (A.rows_ != A.cols_) {
            throw DimensionMismatchError("Matrix must be square for solve");
        }
        if (A.rows_ != b.size()) {
            throw DimensionMismatchError("Matrix rows must match vector size");
        }

        const size_t n = A.rows_;
        const double eps = 1e-12;

        for (size_t col = 0; col < n; ++col) {
            size_t pivot_row = col;
            double max_val = std::abs(static_cast<double>(A.at(col, col)));

            for (size_t row = col + 1; row < n; ++row) {
                const double candidate = std::abs(static_cast<double>(A.at(row, col)));
                if (candidate > max_val) {
                    max_val = candidate;
                    pivot_row = row;
                }
            }

            if (max_val < eps) {
                throw SingularMatrixError("Zero pivot detected, matrix is singular");
            }

            if (pivot_row != col) {
                for (size_t j = 0; j < n; ++j) std::swap(A.at(col, j), A.at(pivot_row, j));
                std::swap(b[col], b[pivot_row]);
            }

            const T pivot = A.at(col, col);
            for (size_t j = 0; j < n; ++j) {
                if (j != col) A.at(col, j) /= pivot;
            }
            b[col] /= pivot;
            A.at(col, col) = T{1};

            for (size_t row = 0; row < n; ++row) {
                if (row == col) continue;
                const T factor = A.at(row, col);
                if (factor == T{}) continue;
                for (size_t j = 0; j < n; ++j) {
                    if (j != col) A.at(row, j) -= factor * A.at(col, j);
                }
                b[row] -= factor * b[col];
                A.at(row, col) = T{};
            }
        }
        return b;
    }
};
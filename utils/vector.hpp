#pragma once

#include <cmath>
#include <initializer_list>
#include <numeric>
#include <string>
#include <ostream>
#include <type_traits>
#include <vector>

#include "exceptions/vector_matrix_exceptions.hpp"
#include "exceptions/numerical_exceptions.hpp"

template <typename T>
class Vector {
    static_assert(
        std::is_arithmetic_v<std::remove_cv_t<T>> &&
        !std::is_same_v<std::remove_cv_t<T>, bool>,
        "Vector<T>: T must be a real or integer numeric type, excluding complex/bool"
    );

    std::vector<T> data_;

    void check_index(size_t idx) const {
        if (idx >= data_.size()) {
            throw std::out_of_range("Vector index out of range: " + std::to_string(idx));
        }
    }

    void check_dimension(const Vector& other, const std::string& op) const {
        if (data_.size() != other.size()) {
            throw DimensionMismatchError("Vectors must have same size for " + op);
        }
    }

public:
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    explicit Vector(size_t n = 0, T val = T{}) : data_(n, val) {}
    explicit Vector(std::initializer_list<T> list) : data_(list) {}

    size_t size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return data_.empty(); }

    iterator begin() noexcept { return data_.begin(); }
    iterator end() noexcept { return data_.end(); }
    const_iterator begin() const noexcept { return data_.begin(); }
    const_iterator end() const noexcept { return data_.end(); }
    const_iterator cbegin() const noexcept { return data_.cbegin(); }
    const_iterator cend() const noexcept { return data_.cend(); }

    const T& at(size_t idx) const { check_index(idx); return data_[idx]; }
    T& at(size_t idx) { check_index(idx); return data_[idx]; }

    const T& operator[](size_t idx) const { return data_[idx]; }
    T& operator[](size_t idx) { return data_[idx]; }

    bool operator==(const Vector& other) const noexcept { return data_ == other.data_; }
    bool operator!=(const Vector& other) const noexcept { return !(*this == other); }

    bool equals(const Vector& other, double tol = 1e-9) const {
        if (size() != other.size()) return false;
        for (size_t i = 0; i < size(); ++i) {
            if (std::abs(data_[i] - other[i]) > static_cast<T>(tol)) {
                return false;
            }
        }
        return true;
    }

    Vector operator+(const Vector& other) const {
        check_dimension(other, "addition");
        Vector res(size());
        for (size_t i = 0; i < size(); ++i) res[i] = data_[i] + other[i];
        return res;
    }

    Vector operator-(const Vector& other) const {
        check_dimension(other, "subtraction");
        Vector res(size());
        for (size_t i = 0; i < size(); ++i) res[i] = data_[i] - other[i];
        return res;
    }

    Vector operator-() const {
        Vector res(size());
        for (size_t i = 0; i < size(); ++i) res[i] = -data_[i];
        return res;
    }

    Vector operator*(T scalar) const {
        Vector res(size());
        for (size_t i = 0; i < size(); ++i) res[i] = data_[i] * scalar;
        return res;
    }

    Vector operator/(T scalar) const {
        if (scalar == T{}) {
            throw NumericalError("Division by zero in vector");
        }
        Vector res(size());
        for (size_t i = 0; i < size(); ++i) res[i] = data_[i] / scalar;
        return res;
    }

    friend Vector operator*(T scalar, const Vector& v) {
        return v * scalar;
    }

    friend std::ostream& operator<<(std::ostream& os, const Vector& v) {
        os << "[";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i != 0) os << ", ";
            os << v[i];
        }
        os << "]";
        return os;
    }

    T dot(const Vector& other) const {
        check_dimension(other, "dot product");
        return std::inner_product(data_.begin(), data_.end(), other.data_.begin(), T{});
    }

    auto norm() const {
        using ResultType = std::common_type_t<T, double>;
        return std::sqrt(static_cast<ResultType>(dot(*this)));
    }
};
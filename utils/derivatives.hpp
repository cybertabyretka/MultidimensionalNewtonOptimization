#pragma once

#include <optional>
#include <functional>
#include <type_traits>
#include <utility>

#include "utils/matrix.hpp"
#include "utils/vector.hpp"

template <typename T>
class Gradient {
    static_assert(
        std::is_arithmetic_v<std::remove_cv_t<T>> &&
        !std::is_same_v<std::remove_cv_t<T>, bool>,
        "Gradient<T>: T must be a real or integer numeric type, excluding complex/bool"
    );

    using Func = std::function<Vector<T>(const Vector<T>&)>;
    Func evaluator_;
    mutable std::optional<Vector<T>> cached_result_;
    mutable std::optional<Vector<T>> last_point_;

public:
    explicit Gradient(Func f) : evaluator_(std::move(f)) {}

    Vector<T> evaluate(const Vector<T>& x) const {
        if (last_point_ && last_point_->equals(x)) {
            return *cached_result_;
        }
        cached_result_ = evaluator_(x);
        last_point_ = x;
        return *cached_result_;
    }

    bool is_computed() const noexcept { return static_cast<bool>(cached_result_); }
    void clear_cache() { cached_result_.reset(); last_point_.reset(); }
};

template <typename T>
class Hessian {
    static_assert(
        std::is_arithmetic_v<std::remove_cv_t<T>> &&
        !std::is_same_v<std::remove_cv_t<T>, bool>,
        "Hessian<T>: T must be a real or integer numeric type, excluding complex/bool"
    );

    using Func = std::function<Matrix<T>(const Vector<T>&)>;
    Func evaluator_;
    mutable std::optional<Matrix<T>> cached_result_;
    mutable std::optional<Vector<T>> last_point_;

public:
    explicit Hessian(Func f) : evaluator_(std::move(f)) {}

    Matrix<T> evaluate(const Vector<T>& x) const {
        if (last_point_ && last_point_->equals(x)) {
            return *cached_result_;
        }
        cached_result_ = evaluator_(x);
        last_point_ = x;
        return *cached_result_;
    }

    bool is_computed() const noexcept { return static_cast<bool>(cached_result_); }
    void clear_cache() { cached_result_.reset(); last_point_.reset(); }
};
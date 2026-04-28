#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "newton_configs.hpp"
#include "newton_optimizer.hpp"

#include "utils/vector.hpp"
#include "utils/matrix.hpp"
#include "utils/derivatives.hpp"
#include "utils/grid_generator.hpp"

#include "exceptions/numerical_exceptions.hpp"
#include "exceptions/vector_matrix_exceptions.hpp"
#include "exceptions/optimization_exceptions.hpp"
#include "exceptions/intervals_exceptions.hpp"

bool NewtonOptimizer::is_duplicate(
    const std::vector<OptimizedPoint>& list,
    const Vector<double>& x,
    double tol
) {
    for (const auto& p : list) {
        if (p.point.equals(x, tol)) return true;
    }
    return false;
}

void NewtonOptimizer::validate_config() const {
    if (!config_.problem.objective || !config_.problem.gradient || !config_.problem.hessian) {
        throw OptimizationError("Objective, gradient and hessian must be provided");
    }
}

Vector<double> NewtonOptimizer::regularized_newton_direction(
    const Vector<double>& x,
    const Vector<double>& g
) const {
    Matrix<double> H = hessian_.evaluate(x);
    if (H.rows() != H.cols() || H.rows() != x.size()) {
        throw DimensionMismatchError("Hessian dimension mismatch");
    }

    Matrix<double> H_reg = H;
    double lambda = 0.0;

    for (size_t attempt = 0; attempt < 30; ++attempt) {
        if (attempt > 0) {
            lambda = (lambda == 0.0)
                ? config_.numeric.initial_regularization
                : lambda * config_.numeric.regularization_growth;

            if (lambda > config_.numeric.max_regularization) {
                break;
            }

            H_reg = H;
            for (size_t i = 0; i < H_reg.rows(); ++i) {
                H_reg.at(i, i) += lambda;
            }
        }

        if (!H_reg.is_positive_definite()) {
            continue;
        }

        Vector<double> d = Matrix<double>::solve(H_reg, g * -1.0);
        if (g.dot(d) < 0.0) {
            return d;
        }
    }
    return g * -1.0;
}

NewtonResult NewtonOptimizer::solve_from_start(const Vector<double>& start) const {
    validate_config();
    if (start.empty()) {
        throw DimensionMismatchError("Start point is empty");
    }

    Vector<double> x = start;
    double fx = config_.problem.objective(x);
    bool converged = false;
    size_t iter = 0;

    for (; iter < config_.numeric.max_iter; ++iter) {
        const Vector<double> g = gradient_.evaluate(x);
        const double gnorm = g.norm();
        if (gnorm < config_.numeric.grad_tol) {
            converged = true;
            fx = config_.problem.objective(x);
            return {x, fx, gnorm, converged, iter};
        }

        Vector<double> d = regularized_newton_direction(x, g);
        double dir_deriv = g.dot(d);
        if (dir_deriv >= 0.0) {
            d = g * -1.0;
            dir_deriv = -gnorm * gnorm;
        }

        double alpha = 1.0;
        bool accepted = false;
        Vector<double> candidate;
        double fc = fx;

        while (alpha >= config_.numeric.min_alpha) {
            candidate = x + d * alpha;
            try {
                fc = config_.problem.objective(candidate);
            } catch (const std::exception&) {
                alpha *= config_.numeric.backtracking_beta;
                continue;
            }

            if (fc <= fx + config_.numeric.armijo_c1 * alpha * dir_deriv) {
                accepted = true;
                break;
            }
            alpha *= config_.numeric.backtracking_beta;
        }

        if (!accepted) {
            throw ConvergenceError("Line search failed to find a decreasing step");
        }

        if ((candidate - x).norm() < config_.numeric.step_tol) {
            x = candidate;
            fx = fc;
            const double final_gnorm = gradient_.evaluate(x).norm();
            converged = final_gnorm < config_.numeric.grad_tol;
            return {x, fx, final_gnorm, converged, iter + 1};
        }

        x = candidate;
        fx = fc;

        for (double v : x) {
            if (std::isnan(v) || std::isinf(v)) {
                throw NumericalError("Divergence or overflow in Newton iteration");
            }
        }
    }

    const Vector<double> gfinal = gradient_.evaluate(x);
    const double gnorm = gfinal.norm();
    converged = gnorm < config_.numeric.grad_tol;
    return {x, fx, gnorm, converged, iter};
}

bool NewtonOptimizer::is_minimum_point(const Vector<double>& x) const {
    Matrix<double> H = hessian_.evaluate(x);
    return H.is_positive_definite();
}

NewtonOptimizer::NewtonOptimizer(NewtonOptimizerConfig config)
    : config_(std::move(config)),
      gradient_(config_.problem.gradient),
      hessian_(config_.problem.hessian) {
    validate_config();
}

const std::vector<OptimizedPoint>& NewtonOptimizer::get_stationary_points() const {
    return stationary_points_; 
}
const std::vector<OptimizedPoint>& NewtonOptimizer::get_minimum_points() const {
    return minimum_points_; 
}

void NewtonOptimizer::clear_results() {
    stationary_points_.clear();
    minimum_points_.clear();
}

NewtonResult NewtonOptimizer::optimize(const Vector<double>& start_point) const {
    return solve_from_start(start_point);
}

void NewtonOptimizer::optimize(const std::vector<Vector<double>>& start_points) {
    clear_results();

    for (const auto& start : start_points) {
        try {
            NewtonResult res = solve_from_start(start);
            if (res.gradient_norm <= config_.numeric.stationarity_tol) {
                if (!is_duplicate(stationary_points_, res.point, config_.numeric.duplicate_tol)) {
                    stationary_points_.push_back({res.point, res.value});
                    if (is_minimum_point(res.point)) {
                        minimum_points_.push_back({res.point, res.value});
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[NewtonOptimizer] Start point failed: " << e.what() << '\n';
        }
    }

    auto cmp = [](const OptimizedPoint& a, const OptimizedPoint& b) {
        return a.value < b.value;
    };
    std::sort(stationary_points_.begin(), stationary_points_.end(), cmp);
    std::sort(minimum_points_.begin(), minimum_points_.end(), cmp);
}
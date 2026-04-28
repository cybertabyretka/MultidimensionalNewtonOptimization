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

#include "vector.hpp"
#include "matrix.hpp"
#include "derivatives.hpp"

#include "exceptions/numerical_exceptions.hpp"
#include "exceptions/vector_matrix_exceptions.hpp"
#include "exceptions/optimization_exceptions.hpp"

struct NewtonFunctionSet {
    std::function<double(const Vector<double>&)> objective;
    std::function<Vector<double>(const Vector<double>&)> gradient;
    std::function<Matrix<double>(const Vector<double>&)> hessian;
};

struct NewtonSearchConfig {
    Vector<double> lower_bound;
    Vector<double> upper_bound;
    size_t grid_resolution = 5;
};

struct NewtonNumericConfig {
    size_t max_iter = 50;
    double grad_tol = 1e-6;
    double step_tol = 1e-8;
    double stationarity_tol = 1e-6;
    double duplicate_tol = 1e-4;

    double armijo_c1 = 1e-4;
    double backtracking_beta = 0.5;
    double min_alpha = 1e-12;

    double initial_regularization = 1e-8;
    double regularization_growth = 10.0;
    double max_regularization = 1e8;
};

struct NewtonOptimizerConfig {
    NewtonFunctionSet problem;
    NewtonSearchConfig search;
    NewtonNumericConfig numeric;
};

struct OptimizedPoint {
    Vector<double> point;
    double value{};
};

struct NewtonResult {
    Vector<double> point;
    double value{};
    double gradient_norm{};
    bool converged{false};
    size_t iterations{0};
};

class GridGenerator {
public:
    std::vector<Vector<double>> generate(const Vector<double>& lower,
                                         const Vector<double>& upper,
                                         size_t resolution) const {
        if (lower.size() != upper.size()) {
            throw DimensionMismatchError("Bounds dimension mismatch");
        }
        if (lower.empty()) {
            throw DimensionMismatchError("Bounds vector is empty");
        }
        if (resolution < 2) {
            throw InvalidIntervalError("Grid resolution must be at least 2");
        }

        const size_t n = lower.size();
        for (size_t i = 0; i < n; ++i) {
            if (lower[i] >= upper[i]) {
                throw InvalidIntervalError("Lower bound must be strictly less than upper bound");
            }
        }

        std::vector<Vector<double>> points;
        std::vector<size_t> index(n, 0);
        std::vector<double> step(n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            step[i] = (upper[i] - lower[i]) / static_cast<double>(resolution - 1);
        }

        while (true) {
            Vector<double> pt(n);
            for (size_t d = 0; d < n; ++d) {
                pt[d] = lower[d] + static_cast<double>(index[d]) * step[d];
            }
            points.push_back(pt);

            size_t pos = n;
            while (pos > 0) {
                --pos;
                ++index[pos];
                if (index[pos] < resolution) break;
                index[pos] = 0;
            }
            if (pos == 0 && index[0] == 0) break;
        }

        return points;
    }
};

class NewtonOptimizer {
    NewtonOptimizerConfig config_;
    Gradient<double> gradient_;
    Hessian<double> hessian_;

    std::vector<OptimizedPoint> stationary_points_;
    std::vector<OptimizedPoint> minimum_points_;

    static bool is_duplicate(const std::vector<OptimizedPoint>& list,
                             const Vector<double>& x,
                             double tol) {
        for (const auto& p : list) {
            if (p.point.equals(x, tol)) return true;
        }
        return false;
    }

    void validate_config() const {
        if (!config_.problem.objective || !config_.problem.gradient || !config_.problem.hessian) {
            throw OptimizationError("Objective, gradient and hessian must be provided");
        }
    }

    Vector<double> regularized_newton_direction(const Vector<double>& x,
                                                const Vector<double>& g) const {
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

    NewtonResult solve_from_start(const Vector<double>& start) const {
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

    bool is_minimum_point(const Vector<double>& x) const {
        Matrix<double> H = hessian_.evaluate(x);
        return H.is_positive_definite();
    }

public:
    explicit NewtonOptimizer(NewtonOptimizerConfig config)
        : config_(std::move(config)),
          gradient_(config_.problem.gradient),
          hessian_(config_.problem.hessian) {
        validate_config();
    }

    const std::vector<OptimizedPoint>& get_stationary_points() const { return stationary_points_; }
    const std::vector<OptimizedPoint>& get_minimum_points() const { return minimum_points_; }

    void clear_results() {
        stationary_points_.clear();
        minimum_points_.clear();
    }

    NewtonResult optimize(const Vector<double>& start_point) const {
        return solve_from_start(start_point);
    }

    void optimize(const std::vector<Vector<double>>& start_points) {
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
};

int main() {
    auto f = [](const Vector<double>& x) -> double {
        const double x1 = x[0], x2 = x[1];
        return std::pow(x1, 4) + std::pow(x2, 4) - 4.0 * x1 * x2 + 1.0;
    };

    auto grad_func = [](const Vector<double>& x) -> Vector<double> {
        const double x1 = x[0], x2 = x[1];
        return Vector<double>{4.0 * std::pow(x1, 3) - 4.0 * x2,
                              4.0 * std::pow(x2, 3) - 4.0 * x1};
    };

    auto hess_func = [](const Vector<double>& x) -> Matrix<double> {
        const double x1 = x[0], x2 = x[1];
        Matrix<double> H(2, 2, 0.0);
        H.at(0, 0) = 12.0 * std::pow(x1, 2); H.at(0, 1) = -4.0;
        H.at(1, 0) = -4.0;                   H.at(1, 1) = 12.0 * std::pow(x2, 2);
        return H;
    };

    NewtonOptimizerConfig cfg;
    cfg.problem.objective = f;
    cfg.problem.gradient = grad_func;
    cfg.problem.hessian = hess_func;
    cfg.search.lower_bound = Vector<double>{-2.0, -2.0};
    cfg.search.upper_bound = Vector<double>{2.0, 2.0};
    cfg.search.grid_resolution = 4;
    cfg.numeric.max_iter = 50;
    cfg.numeric.grad_tol = 1e-6;
    cfg.numeric.step_tol = 1e-10;
    cfg.numeric.stationarity_tol = 1e-6;

    GridGenerator grid;
    std::vector<Vector<double>> starts = grid.generate(
        cfg.search.lower_bound,
        cfg.search.upper_bound,
        cfg.search.grid_resolution
    );

    NewtonOptimizer optimizer(cfg);
    optimizer.optimize(starts);

    std::cout << std::fixed << std::setprecision(10);

    std::cout << "Found stationary points:\n";
    for (const auto& p : optimizer.get_stationary_points()) {
        std::cout << "  (" << p.point[0] << ", " << p.point[1] << ") -> f = " << p.value << '\n';
    }

    std::cout << "\nMinimum points:\n";
    for (const auto& p : optimizer.get_minimum_points()) {
        std::cout << "  (" << p.point[0] << ", " << p.point[1] << ") -> f = " << p.value << '\n';
    }
    return 0;
}
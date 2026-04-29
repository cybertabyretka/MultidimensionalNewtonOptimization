#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "newton_configs.hpp"
#include "utils/vector.hpp"
#include "utils/derivatives.hpp"

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

class NewtonOptimizer {
    NewtonOptimizerConfig config_;
    std::function<double(const Vector<double>&)> objective_;

    std::vector<OptimizedPoint> stationary_points_;
    std::vector<OptimizedPoint> minimum_points_;

    static bool is_duplicate(
        const std::vector<OptimizedPoint>& list,
        const Vector<double>& x,
        double tol
    );

    void validate_config() const;

    Vector<double> regularized_newton_direction(
        const Vector<double>& x,
        const Vector<double>& g,
        bool logs
    ) const;

    NewtonResult solve_from_start(
        const Vector<double>& start,
        bool logs
    ) const;

    bool is_minimum_point(const Vector<double>& x) const;

public:
    explicit NewtonOptimizer(NewtonOptimizerConfig config);

    const std::vector<OptimizedPoint>& get_stationary_points() const;
    const std::vector<OptimizedPoint>& get_minimum_points() const;

    void clear_results();

    NewtonResult optimize(const Vector<double>& start_point, bool logs = false) const;

    void optimize(const std::vector<Vector<double>>& start_points, bool logs = false);
};
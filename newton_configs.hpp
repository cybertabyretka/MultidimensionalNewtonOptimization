#pragma once

#include <functional>

#include "utils/vector.hpp"
#include "utils/matrix.hpp"

struct NewtonFunctionSet {
    std::function<double(const Vector<double>&)> objective;
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

    double gradient_step = 1e-6;
    double hessian_step = 1e-4;
};

struct NewtonOptimizerConfig {
    NewtonFunctionSet problem;
    NewtonSearchConfig search;
    NewtonNumericConfig numeric;
};
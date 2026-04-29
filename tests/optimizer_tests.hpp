#pragma once

#include <cmath>
#include <numbers>
#include <stdexcept>

#include "newton_optimizer.hpp"
#include "newton_configs.hpp"

#include "utils/vector.hpp"
#include "utils/matrix.hpp"
#include "utils/derivatives.hpp"

#include "tests/utils.hpp"

#include "exceptions/intervals_exceptions.hpp"
#include "exceptions/optimization_exceptions.hpp"

// TEST 5: Optimizer - Simple quadratic function (штатная ситуация)
// Функция: f(x) = x1^2 + x2^2, минимум в (0, 0), значение 0
void test_optimizer_quadratic() {
    try {
        auto f = [](const Vector<double>& x) -> double {
            return x[0] * x[0] + x[1] * x[1];
        };

        NewtonOptimizerConfig cfg;
        cfg.problem.objective = f;
        cfg.search.lower_bound = Vector<double>{-5.0, -5.0};
        cfg.search.upper_bound = Vector<double>{5.0, 5.0};
        cfg.search.grid_resolution = 3;
        cfg.numeric.max_iter = 50;
        cfg.numeric.grad_tol = 1e-6;
        cfg.numeric.gradient_step = 1e-6;
        cfg.numeric.hessian_step = 1e-4;

        NewtonOptimizer optimizer(cfg);
        
        // Запуск из одной стартовой точки
        NewtonResult result = optimizer.optimize(Vector<double>{2.0, 3.0}, false);
        
        // Проверка сходимости
        if (!result.converged)
            throw std::logic_error("Optimizer did not converge for simple quadratic");
        
        // Проверка того, что найденная точка близка к (0, 0)
        if (!result.point.equals(Vector<double>{0.0, 0.0}, 1e-4))
            throw std::logic_error("Solution not close to (0,0)");
        
        // Проверка значения функции в найденной точке
        double f_at_solution = f(result.point);
        if (!double_equals(f_at_solution, 0.0, 1e-6))
            throw std::logic_error("Function value at solution not close to 0");
    } catch (const std::exception& e) {
        print_test_failed("Optimizer_Quadratic", e.what());
    }
}

// TEST 6: Optimizer - Rosenbrock function (сложная штатная ситуация)
// Классическая функция: f(x1,x2) = (1-x1)^2 + 100*(x2-x1^2)^2
// Минимум в (1, 1), значение 0.
void test_optimizer_rosenbrock() {
    try {
        auto f = [](const Vector<double>& x) -> double {
            const double x1 = x[0], x2 = x[1];
            return (1.0 - x1) * (1.0 - x1) + 100.0 * (x2 - x1 * x1) * (x2 - x1 * x1);
        };

        NewtonOptimizerConfig cfg;
        cfg.problem.objective = f;
        cfg.search.lower_bound = Vector<double>{-2.0, -2.0};
        cfg.search.upper_bound = Vector<double>{2.0, 2.0};
        cfg.search.grid_resolution = 5;
        cfg.numeric.max_iter = 100;
        cfg.numeric.grad_tol = 1e-6;
        cfg.numeric.gradient_step = 1e-6;
        cfg.numeric.hessian_step = 1e-4;

        NewtonOptimizer optimizer(cfg);
        std::vector<Vector<double>> starts;
        starts.push_back(Vector<double>{-1.0, -1.0});
        starts.push_back(Vector<double>{0.0, 0.0});
        starts.push_back(Vector<double>{1.5, 1.5});
        
        optimizer.optimize(starts, false);
        
        // Проверка, что оптимизатор нашёл хотя бы одну точку
        const auto& min_points = optimizer.get_minimum_points();
        if (min_points.empty())
            throw std::logic_error("No minimum points found for Rosenbrock");
        
        // Проверка, что лучшая точка близка к (1, 1)
        const auto& best = min_points[0];
        if (!best.point.equals(Vector<double>{1.0, 1.0}, 5e-2))
            throw std::logic_error("Best solution not close to (1, 1)");
        
        // Проверка, что значение функции приемлемо
        if (best.value > 1e-3)
            throw std::logic_error("Function value at best point too high");
    } catch (const std::exception& e) {
        print_test_failed("Optimizer_Rosenbrock", e.what());
    }
}

// TEST 7: Optimizer - Saddle point function (крайняя ситуация)
// Функция: f(x1,x2) = (x1^2-1)^2 + (x2^2-1)^2 + 0.5*x1*x2
// Минимумы близко к (±1, ±1), седловые точки в других местах
void test_optimizer_saddle_point() {
    try {
        auto f = [](const Vector<double>& x) -> double {
            const double x1 = x[0], x2 = x[1];
            const double term1 = (x1 * x1 - 1.0) * (x1 * x1 - 1.0);
            const double term2 = (x2 * x2 - 1.0) * (x2 * x2 - 1.0);
            return term1 + term2 + 0.5 * x1 * x2;
        };

        NewtonOptimizerConfig cfg;
        cfg.problem.objective = f;
        cfg.search.lower_bound = Vector<double>{-2.0, -2.0};
        cfg.search.upper_bound = Vector<double>{2.0, 2.0};
        cfg.search.grid_resolution = 5;
        cfg.numeric.max_iter = 50;
        cfg.numeric.grad_tol = 1e-6;
        cfg.numeric.stationarity_tol = 1e-6;
        cfg.numeric.gradient_step = 1e-6;
        cfg.numeric.hessian_step = 1e-4;

        NewtonOptimizer optimizer(cfg);
        std::vector<Vector<double>> starts;
        starts.push_back(Vector<double>{0.5, 0.5});
        starts.push_back(Vector<double>{1.5, 1.5});
        starts.push_back(Vector<double>{-1.5, 1.5});
        
        optimizer.optimize(starts, false);
        
        // Проверка, что найдены минимальные точки
        const auto& min_points = optimizer.get_minimum_points();
        if (min_points.empty())
            throw std::logic_error("No minimum points found");
        
        // Проверка, что лучшая точка имеет приемлемое значение (близко к минимумам)
        // Минимумы должны быть близко к (±1, ±1)
        const auto& best = min_points[0];
        const bool x1_valid = (best.point[0] > 0.5 && best.point[0] < 1.5) ||
                              (best.point[0] < -0.5 && best.point[0] > -1.5);
        const bool x2_valid = (best.point[1] > 0.5 && best.point[1] < 1.5) ||
                              (best.point[1] < -0.5 && best.point[1] > -1.5);
        if (!(x1_valid && x2_valid))
            throw std::logic_error("Best solution not in expected region");
    } catch (const std::exception& e) {
        print_test_failed("Optimizer_SaddlePoint", e.what());
    }
}

// TEST 8: Optimizer - Multiple local minima (крайняя ситуация)
// Функция: f(x) = sin(pi*x1) * sin(pi*x2), несколько локальных минимумов и максимумов
void test_optimizer_multiple_minima() {
    try {
        const double PI = std::numbers::pi;
        
        auto f = [PI](const Vector<double>& x) -> double {
            return std::sin(PI * x[0]) * std::sin(PI * x[1]);
        };

        NewtonOptimizerConfig cfg;
        cfg.problem.objective = f;
        cfg.search.lower_bound = Vector<double>{-2.0, -2.0};
        cfg.search.upper_bound = Vector<double>{2.0, 2.0};
        cfg.search.grid_resolution = 7;
        cfg.numeric.max_iter = 50;
        cfg.numeric.grad_tol = 1e-6;
        cfg.numeric.gradient_step = 1e-6;
        cfg.numeric.hessian_step = 1e-4;

        NewtonOptimizer optimizer(cfg);
        std::vector<Vector<double>> starts;
        // Различные стартовые точки
        starts.push_back(Vector<double>{0.3, 0.3});
        starts.push_back(Vector<double>{-0.3, 0.3});
        starts.push_back(Vector<double>{1.2, 1.2});
        
        optimizer.optimize(starts, false);
        
        const auto& min_points = optimizer.get_minimum_points();
        
        // Проверка, что найдено несколько различных минимумов или хотя бы один
        if (min_points.empty())
            throw std::logic_error("No minimum points found");

        if (min_points[0].value > -0.5)
            throw std::logic_error("Minimum point should have negative value");
    } catch (const std::exception& e) {
        print_test_failed("Optimizer_MultipleMinima", e.what());
    }
}

// TEST 9: Numerical gradient - normal, edge and exceptional cases
void test_numerical_gradient() {
    try {
        auto f = [](const Vector<double>& x) -> double {
            return x[0] * x[0] + 3.0 * x[1] * x[1];
        };
        // Штатный случай
        {
            Vector<double> x{1.5, -2.0};
            Vector<double> g = numerical_gradient(f, x, 1e-6);
            Vector<double> expected{3.0, -12.0};

            if (!g.equals(expected, 1e-4))
                throw std::logic_error("Numerical gradient incorrect for quadratic function");
        }
        // Крайний случай: очень маленькие значения
        {
            Vector<double> x{1e-12, -1e-12};
            Vector<double> g = numerical_gradient(f, x, 1e-6);
            Vector<double> expected{2e-12, -6e-12};
            if (!g.equals(expected, 1e-8))
                throw std::logic_error("Numerical gradient inaccurate near zero");
        }
        // Внештатный случай: objective бросает исключение
        {
            auto throwing_f = [](const Vector<double>& x) -> double {
                if (x[0] > 0.4) {
                    throw ObjectiveEvaluationError("objective failed in gradient test");
                }
                return x[0] * x[0] + x[1] * x[1];
            };
            try {
                (void)numerical_gradient(throwing_f, Vector<double>{0.5, 0.0}, 1e-6);
                throw std::logic_error("numerical_gradient should have propagated exception");
            } catch (const ObjectiveEvaluationError&) {}
        }
    } catch (const std::exception& e) {
        print_test_failed("Numerical_Gradient", e.what());
    }
}

// TEST 10: Numerical hessian - normal, edge and exceptional cases
void test_numerical_hessian() {
    try {
        auto f = [](const Vector<double>& x) -> double {
            return x[0] * x[0] + 3.0 * x[0] * x[1] + 2.0 * x[1] * x[1];
        };
        // Штатный случай
        {
            Vector<double> x{1.0, -1.0};
            Matrix<double> H = numerical_hessian(f, x, 1e-4);

            Matrix<double> expected(2, 2, 0.0);
            expected.at(0, 0) = 2.0;
            expected.at(0, 1) = 3.0;
            expected.at(1, 0) = 3.0;
            expected.at(1, 1) = 4.0;

            for (size_t i = 0; i < 2; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    if (!double_equals(H.at(i, j), expected.at(i, j), 1e-3))
                        throw std::logic_error("Numerical hessian incorrect for quadratic function");
                }
            }
        }
        // Крайний случай: плоская функция около нуля
        {
            auto flat_f = [](const Vector<double>& x) -> double {
                return x[0] * x[0] * x[0] * x[0] + x[1] * x[1] * x[1] * x[1];
            };

            Matrix<double> H = numerical_hessian(flat_f, Vector<double>{0.0, 0.0}, 1e-4);
            for (size_t i = 0; i < 2; ++i) {
                for (size_t j = 0; j < 2; ++j) {
                    if (std::abs(H.at(i, j)) > 1e-6)
                        throw std::logic_error("Numerical hessian should be near zero at the flat point");
                }
            }
        }
        // Внештатный случай: objective бросает исключение
        {
            auto throwing_f = [](const Vector<double>& x) -> double {
                if (x[0] > 0.1) {
                    throw ObjectiveEvaluationError("objective failed in hessian test");
                }
                return x[0] * x[0] + x[1] * x[1];
            };
            try {
                (void)numerical_hessian(throwing_f, Vector<double>{0.2, 0.0}, 1e-4);
                throw std::logic_error("numerical_hessian should have propagated exception");
            } catch (const ObjectiveEvaluationError&) {}
        }
    } catch (const std::exception& e) {
        print_test_failed("Numerical_Hessian", e.what());
    }
}
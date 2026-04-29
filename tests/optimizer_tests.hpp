#pragma once

#include <numbers>

#include "newton_optimizer.hpp"
#include "newton_configs.hpp"

#include "utils/vector.hpp"
#include "utils/matrix.hpp"

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
        
        auto grad = [](const Vector<double>& x) -> Vector<double> {
            return Vector<double>{2.0 * x[0], 2.0 * x[1]};
        };
        
        auto hess = [](const Vector<double>& x) -> Matrix<double> {
            Matrix<double> H(2, 2, 0.0);
            H.at(0, 0) = 2.0; H.at(0, 1) = 0.0;
            H.at(1, 0) = 0.0; H.at(1, 1) = 2.0;
            return H;
        };
        
        NewtonOptimizerConfig cfg;
        cfg.problem.objective = f;
        cfg.problem.gradient = grad;
        cfg.problem.hessian = hess;
        cfg.search.lower_bound = Vector<double>{-5.0, -5.0};
        cfg.search.upper_bound = Vector<double>{5.0, 5.0};
        cfg.search.grid_resolution = 3;
        cfg.numeric.max_iter = 50;
        cfg.numeric.grad_tol = 1e-6;
        
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
// Минимум в (1, 1), значение 0. Это крайне нетривиальная функция.
void test_optimizer_rosenbrock() {
    try {
        auto f = [](const Vector<double>& x) -> double {
            const double x1 = x[0], x2 = x[1];
            return (1.0 - x1) * (1.0 - x1) + 100.0 * (x2 - x1 * x1) * (x2 - x1 * x1);
        };
        
        auto grad = [](const Vector<double>& x) -> Vector<double> {
            const double x1 = x[0], x2 = x[1];
            return Vector<double>{
                -2.0 * (1.0 - x1) - 400.0 * x1 * (x2 - x1 * x1),
                200.0 * (x2 - x1 * x1)
            };
        };
        
        auto hess = [](const Vector<double>& x) -> Matrix<double> {
            const double x1 = x[0], x2 = x[1];
            Matrix<double> H(2, 2, 0.0);
            H.at(0, 0) = 2.0 - 400.0 * x2 + 1200.0 * x1 * x1; H.at(0, 1) = -400.0 * x1;
            H.at(1, 0) = -400.0 * x1;                         H.at(1, 1) = 200.0;
            return H;
        };
        
        NewtonOptimizerConfig cfg;
        cfg.problem.objective = f;
        cfg.problem.gradient = grad;
        cfg.problem.hessian = hess;
        cfg.search.lower_bound = Vector<double>{-2.0, -2.0};
        cfg.search.upper_bound = Vector<double>{2.0, 2.0};
        cfg.search.grid_resolution = 5;
        cfg.numeric.max_iter = 100;
        cfg.numeric.grad_tol = 1e-6;
        
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
        if (!best.point.equals(Vector<double>{1.0, 1.0}, 1e-2))
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
            double x1 = x[0], x2 = x[1];
            double term1 = (x1*x1 - 1.0) * (x1*x1 - 1.0);
            double term2 = (x2*x2 - 1.0) * (x2*x2 - 1.0);
            return term1 + term2 + 0.5 * x1 * x2;
        };
        
        auto grad = [](const Vector<double>& x) -> Vector<double> {
            double x1 = x[0], x2 = x[1];
            double df1 = 4.0 * x1 * (x1*x1 - 1.0) + 0.5 * x2;
            double df2 = 4.0 * x2 * (x2*x2 - 1.0) + 0.5 * x1;
            return Vector<double>{df1, df2};
        };
        
        auto hess = [](const Vector<double>& x) -> Matrix<double> {
            double x1 = x[0], x2 = x[1];
            Matrix<double> H(2, 2, 0.0);
            H.at(0, 0) = 12.0 * x1 * x1 - 4.0 + 0.0; H.at(0, 1) = 0.5;
            H.at(1, 0) = 0.5;                        H.at(1, 1) = 12.0 * x2 * x2 - 4.0;
            return H;
        };
        
        NewtonOptimizerConfig cfg;
        cfg.problem.objective = f;
        cfg.problem.gradient = grad;
        cfg.problem.hessian = hess;
        cfg.search.lower_bound = Vector<double>{-2.0, -2.0};
        cfg.search.upper_bound = Vector<double>{2.0, 2.0};
        cfg.search.grid_resolution = 5;
        cfg.numeric.max_iter = 50;
        cfg.numeric.grad_tol = 1e-6;
        cfg.numeric.stationarity_tol = 1e-6;
        
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
        bool x1_valid = (best.point[0] > 0.5 && best.point[0] < 1.5) || 
                        (best.point[0] < -0.5 && best.point[0] > -1.5);
        bool x2_valid = (best.point[1] > 0.5 && best.point[1] < 1.5) || 
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
        
        auto grad = [PI](const Vector<double>& x) -> Vector<double> {
            return Vector<double>{
                PI * std::cos(PI * x[0]) * std::sin(PI * x[1]),
                PI * std::sin(PI * x[0]) * std::cos(PI * x[1])
            };
        };
        
        auto hess = [PI](const Vector<double>& x) -> Matrix<double> {
            Matrix<double> H(2, 2, 0.0);
            H.at(0, 0) = -PI * PI * std::sin(PI * x[0]) * std::sin(PI * x[1]);
            H.at(0, 1) = PI * PI * std::cos(PI * x[0]) * std::cos(PI * x[1]);
            H.at(1, 0) = PI * PI * std::cos(PI * x[0]) * std::cos(PI * x[1]);
            H.at(1, 1) = -PI * PI * std::sin(PI * x[0]) * std::sin(PI * x[1]);
            return H;
        };
        
        NewtonOptimizerConfig cfg;
        cfg.problem.objective = f;
        cfg.problem.gradient = grad;
        cfg.problem.hessian = hess;
        cfg.search.lower_bound = Vector<double>{-2.0, -2.0};
        cfg.search.upper_bound = Vector<double>{2.0, 2.0};
        cfg.search.grid_resolution = 7;
        cfg.numeric.max_iter = 50;
        cfg.numeric.grad_tol = 1e-6;
        
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
        
        // Проверка, что значение функции в минимуме отрицательное
        if (min_points[0].value > 0.0)
            throw std::logic_error("Minimum point should have negative value");
    } catch (const std::exception& e) {
        print_test_failed("Optimizer_MultipleMinima", e.what());
    }
}

// TEST 9: Optimizer - Missing required functions (внештатная ситуация)
// Проверка, что оптимизатор корректно обрабатывает отсутствующие функции
void test_optimizer_missing_functions() {
    try {
        NewtonOptimizerConfig cfg;
        // Намеренно не устанавливаем objective, gradient, hessian
        cfg.problem.objective = nullptr;
        cfg.problem.gradient = nullptr;
        cfg.problem.hessian = nullptr;
        
        try {
            NewtonOptimizer optimizer(cfg);
            throw std::logic_error("Should have thrown InputOptimizationError");
        } catch (const InputOptimizationError&) {}
        
        // Также проверим случай, когда только некоторые функции отсутствуют
        cfg.problem.objective = [](const Vector<double>&) { return 0.0; };
        cfg.problem.gradient = nullptr;
        cfg.problem.hessian = nullptr;
        
        try {
            NewtonOptimizer optimizer2(cfg);
            throw std::logic_error("Should have thrown InputOptimizationError");
        } catch (const InputOptimizationError&) {}
    } catch (const std::exception& e) {
        print_test_failed("Optimizer_MissingFunctions", e.what());
    }
}

// TEST 10: Optimizer - Incompatible gradient dimension (внештатная ситуация)
// Градиент возвращает вектор неправильного размера
void test_optimizer_incompatible_dimensions() {
    try {
        auto f = [](const Vector<double>& x) -> double {
            return x[0] * x[0] + x[1] * x[1];
        };
        
        // Градиент возвращает вектор размера 1, а должен размера 2
        auto bad_grad = [](const Vector<double>& x) -> Vector<double> {
            return Vector<double>{2.0 * x[0]};  // Неправильный размер!
        };
        
        auto hess = [](const Vector<double>& x) -> Matrix<double> {
            Matrix<double> H(2, 2, 0.0);
            H.at(0, 0) = 2.0;
            H.at(1, 1) = 2.0;
            return H;
        };
        
        NewtonOptimizerConfig cfg;
        cfg.problem.objective = f;
        cfg.problem.gradient = bad_grad;
        cfg.problem.hessian = hess;
        cfg.search.lower_bound = Vector<double>{-2.0, -2.0};
        cfg.search.upper_bound = Vector<double>{2.0, 2.0};
        cfg.search.grid_resolution = 2;
        cfg.numeric.max_iter = 5;
        
        NewtonOptimizer optimizer(cfg);
        
        try {
            NewtonResult result = optimizer.optimize(Vector<double>{1.0, 1.0}, false);
            throw std::logic_error("Should have thrown DimensionMismatchError");
        } catch (const DimensionMismatchError& e) {}
    } catch (const std::exception& e) {
        print_test_failed("Optimizer_IncompatibleDimensions", e.what());
    }
}
#include <cmath>

#include "newton_configs.hpp"
#include "newton_optimizer.hpp"

#include "utils/grid_generator.hpp"
#include "utils/vector.hpp"
#include "utils/matrix.hpp"

int main() {
    auto f = [](const Vector<double>& x) -> double {
        const double x1 = x[0], x2 = x[1];
        return std::pow((x2 - std::pow(x1, 2)), 2) + 100 * std::pow((1 - x1), 2);
    };

    auto grad_func = [](const Vector<double>& x) -> Vector<double> {
        const double x1 = x[0], x2 = x[1];
        return Vector<double>{
            4.0 * std::pow(x1, 3) - 4.0 * x1 * x2 + 200.0 * x1 - 200.0,
            2.0 * (x2 - std::pow(x1, 2))
        };
    };

    auto hess_func = [](const Vector<double>& x) -> Matrix<double> {
        const double x1 = x[0], x2 = x[1];
        Matrix<double> H(2, 2, 0.0);
        H.at(0, 0) = 12.0 * std::pow(x1, 2) - 4 * x2 + 200; H.at(0, 1) = -4.0 * x1;
        H.at(1, 0) = -4.0 * x1;                             H.at(1, 1) = 2.0;
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
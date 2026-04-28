#pragma once

#include <stdexcept>

class OptimizationError : public std::runtime_error {
public:
    explicit OptimizationError(const std::string& message) 
        : std::runtime_error("Optimization Error: " + message) {}
};

class ConvergenceError : public OptimizationError {
public:
    explicit ConvergenceError(const std::string& detail)
        : OptimizationError("Convergence Error: " + detail) {}
};
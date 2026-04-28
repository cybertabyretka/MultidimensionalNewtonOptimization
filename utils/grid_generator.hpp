#pragma once

#include <vector>

#include "utils/vector.hpp"

class GridGenerator {
public:
    std::vector<Vector<double>> generate(
        const Vector<double>& lower,
        const Vector<double>& upper,
        size_t resolution
    ) const;
};
#pragma once

#include "utils/matrix.hpp"

#include "tests/utils.hpp"

#include "exceptions/vector_matrix_exceptions.hpp"

// TEST 3: Matrix - Normal case (штатная ситуация)
// Проверяет базовую функциональность: создание, транспонирование, симметричность, identity
void test_matrix_normal() {
    try {
        // Создание матрицы 2x3
        Matrix<double> m(2, 3, 0.0);
        m.at(0, 0) = 1.0; m.at(0, 1) = 2.0; m.at(0, 2) = 3.0;
        m.at(1, 0) = 4.0; m.at(1, 1) = 5.0; m.at(1, 2) = 6.0;
        
        // Проверка размеров
        if (m.rows() != 2 || m.cols() != 3)
            throw std::logic_error("Matrix dimensions incorrect");
        
        // Проверка доступа
        if (m.at(0, 0) != 1.0 || m.at(1, 2) != 6.0)
            throw std::logic_error("Element access failed");
        
        // Проверка транспонирования: 2x3 -> 3x2
        Matrix<double> mt = m.transpose();
        if (mt.rows() != 3 || mt.cols() != 2)
            throw std::logic_error("Transpose dimensions incorrect");
        if (mt.at(0, 0) != 1.0 || mt.at(1, 0) != 2.0 || mt.at(2, 1) != 6.0)
            throw std::logic_error("Transpose values incorrect");
        
        // Проверка identity матрицы
        Matrix<double> I = Matrix<double>::identity(3);
        if (I.rows() != 3 || I.cols() != 3)
            throw std::logic_error("Identity matrix dimensions incorrect");
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                double expected = (i == j) ? 1.0 : 0.0;
                if (I.at(i, j) != expected)
                    throw std::logic_error("Identity matrix values incorrect");
            }
        }
        
        // Проверка is_symmetric на симметричной матрице
        Matrix<double> sym(2, 2);
        sym.at(0, 0) = 2.0; sym.at(0, 1) = 3.0;
        sym.at(1, 0) = 3.0; sym.at(1, 1) = 4.0;
        if (!sym.is_symmetric())
            throw std::logic_error("Symmetric matrix not detected");
        
        // Проверка is_symmetric на несимметричной матрице
        Matrix<double> nonsym(2, 2);
        nonsym.at(0, 0) = 1.0; nonsym.at(0, 1) = 2.0;
        nonsym.at(1, 0) = 3.0; nonsym.at(1, 1) = 4.0;
        if (nonsym.is_symmetric())
            throw std::logic_error("Non-symmetric matrix falsely reported as symmetric");
        
        // Проверка скалярного умножения матрицы: 2x3 матрица * 2.0
        Matrix<double> scaled = m * 2.0;
        if (scaled.at(0, 0) != 2.0 || scaled.at(1, 2) != 12.0)
            throw std::logic_error("Matrix scalar multiplication failed");
        
        // Проверка сложения матриц
        Matrix<double> m2(2, 3, 1.0);
        Matrix<double> sum = m + m2;
        if (sum.at(0, 0) != 2.0 || sum.at(1, 2) != 7.0)
            throw std::logic_error("Matrix addition failed");
    } catch (const std::exception& e) {
        print_test_failed("Matrix_Normal", e.what());
    }
}

// TEST 4: Matrix - Edge and abnormal cases (крайняя и внештатная ситуация)
// Проверяет матричные операции и обработку ошибок
void test_matrix_edge_cases() {
    try {
        // Случай 1: Несовместимые размеры при умножении матриц
        Matrix<double> m1(2, 3, 1.0);  // 2x3
        Matrix<double> m2(2, 2, 1.0);  // 2x2 (несовместима с 2x3)
        try {
            m1 * m2;  // Должно выбросить исключение
            throw std::logic_error("Should have thrown DimensionMismatchError for matrix mult");
        } catch (const DimensionMismatchError&) {}
        
        // Случай 2: Несовместимые размеры при сложении матриц
        try {
            m1 + m2;  // m1 is 2x3, m2 is 2x2 - несовместимо
            throw std::logic_error("Should have thrown DimensionMismatchError for addition");
        } catch (const DimensionMismatchError&) {}
        
        // Случай 3: Корректное матричное умножение 2x3 * 3x2 = 2x2
        Matrix<double> m3(3, 2, 1.0);  // 3x2
        Matrix<double> result = m1 * m3;  // 2x3 * 3x2 = 2x2
        if (result.rows() != 2 || result.cols() != 2)
            throw std::logic_error("Matrix product dimensions incorrect");
        // Каждый элемент = 1*1 + 1*1 + 1*1 = 3
        if (!double_equals(result.at(0, 0), 3.0) || !double_equals(result.at(1, 1), 3.0))
            throw std::logic_error("Matrix product values incorrect");
        
        // Случай 4: Умножение матрицы на вектор
        Matrix<double> m4(2, 3);
        m4.at(0, 0) = 1.0; m4.at(0, 1) = 2.0; m4.at(0, 2) = 3.0;
        m4.at(1, 0) = 4.0; m4.at(1, 1) = 5.0; m4.at(1, 2) = 6.0;
        Vector<double> v{1.0, 2.0, 3.0};
        Vector<double> res = m4 * v;
        if (res.size() != 2)
            throw std::logic_error("Matrix-vector product size incorrect");
        // [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
        if (!double_equals(res[0], 14.0) || !double_equals(res[1], 32.0))
            throw std::logic_error("Matrix-vector product values incorrect");
        
        // Случай 5: Несовместимые размеры при умножении матрицы на вектор
        Vector<double> v2{1.0, 2.0};  // Размер 2, но матрица 2x3
        try {
            m4 * v2;  // Должно выбросить исключение
            throw std::logic_error("Should have thrown DimensionMismatchError for matrix-vector mult");
        } catch (const DimensionMismatchError&) {}
        
        // Случай 6: Несимметричная матрица не является квадратной
        if (m1.is_symmetric())
            throw std::logic_error("Non-square matrix should not be symmetric");
    } catch (const std::exception& e) {
        print_test_failed("Matrix_EdgeCases", e.what());
    }
}
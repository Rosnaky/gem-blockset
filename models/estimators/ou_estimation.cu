/*
* Ornstein-Uhlenbeck estimation
* Using AR(1) discretization
*/
#pragma once


#include "composites.h"
#include "primitives.h"

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void ou_estimation(const T* input, int n, T* speed, T* equilibrium, T* volatility_sq, int window) {
    const int m = n - 1;
    const T* x = input;
    const T* y = input + 1;

    T* alpha = new T[m];
    T* beta = new T[m];
    T* temp = new T[m];
    T* residuals = new T[m];

    rolling_ols<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(x, y, alpha, beta, m, window);

    elementwise_ln<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(beta, speed, m);
    scale<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(speed, speed, m, -1.0);

    T* one_minus_beta = new T[m];
    offset<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(beta, one_minus_beta, m, -1.0);
    scale<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(one_minus_beta, one_minus_beta, m, -1.0);
    elementwise_divide<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(alpha, one_minus_beta, equilibrium, m);

    elementwise_multiply<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(beta, x, temp, m);
    elementwise_add<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(temp, alpha, temp, m);
    elementwise_subtract<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(y, temp, residuals, m);
    rolling_variance<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(residuals, volatility_sq, m, window);

    T* beta_sq = new T[m];
    elementwise_multiply<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(beta, beta, beta_sq, m);
    T* one_minus_beta_sq = new T[m];
    offset<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(beta_sq, one_minus_beta_sq, m, -1.0);
    scale<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(one_minus_beta_sq, one_minus_beta_sq, m, -1.0);

    T* scale_factor = new T[m];
    elementwise_multiply<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(speed, speed, scale_factor, m);
    scale<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(scale_factor, scale_factor, m, 2.0);
    elementwise_divide<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(scale_factor, one_minus_beta_sq, scale_factor, m);
    elementwise_multiply<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(volatility_sq, scale_factor, volatility_sq, m);

    delete[] alpha;
    delete[] beta;
    delete[] temp;
    delete[] residuals;
    delete[] one_minus_beta;
    delete[] beta_sq;
    delete[] one_minus_beta_sq;
    delete[] scale_factor;
}
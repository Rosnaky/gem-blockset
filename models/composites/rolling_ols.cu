/*
* Ordinary least squares
*/

#pragma once

#include "elementwise_divide.cu"
#include "elementwise_multiply.cu"
#include "elementwise_subtract.cu"
#include "rolling_covariance.cu"
#include "rolling_variance.cu"
#include "simple_moving_average.cu"

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void rolling_ols(const T* input_x, const T* input_y, T* output_alpha, T* output_beta, int n, int window) {
    T* variance = new T[n];
    T* covariance = new T[n];
    T* sma_x = new T[n];
    T* sma_y = new T[n];
    T* temp = new T[n];

    rolling_variance<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(input_x, variance, n, window);
    rolling_covariance<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(input_x, input_y, covariance, n, window);
    elementwise_divide<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(covariance, variance, output_beta, n);

    simple_moving_average<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(input_x, sma_x, n, window);
    simple_moving_average<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(input_y, sma_y, n, window);

    elementwise_multiply<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(output_beta, sma_x, temp, n);
    elementwise_subtract<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(sma_y, temp, output_alpha, n);

    delete[] variance;
    delete[] covariance;
    delete[] sma_x;
    delete[] sma_y;
    delete[] temp;
}

#pragma once

#include "simple_moving_average.cu"
#include "rolling_variance.cu"
#include "elementwise_sqrt.cu"
#include "elementwise_subtract.cu"
#include "elementwise_divide.cu"

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void rolling_z_score(const T* input, T* output, int n, int window) {
    T* temp = new T[n];

    rolling_variance<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(input, temp, n, window);
    elementwise_sqrt<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(temp, temp, n);
    
    simple_moving_average<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(input, output, n, window);
    elementwise_subtract<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(input, output, output, n);
    elementwise_divide<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(output, temp, output, n);

    delete[] temp;
}
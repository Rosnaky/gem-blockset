#pragma once

#include "elementwise_constant_binary.cuh"

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void scale(const T* input, T* output, int n, double scalar) {
    elementwise_constant_binary<T, Scale, BLOCK_SIZE, ELEMENTS_PER_THREAD>
        (input, scalar, output, n, Scale{});
}

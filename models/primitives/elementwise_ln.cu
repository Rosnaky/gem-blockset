#pragma once

#include "elementwise_constant_binary.cuh"
#include "elementwise_unary.cuh"

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void elementwise_ln(const T* input, T* output, int n) {
    elementwise_unary<T, Log, BLOCK_SIZE, ELEMENTS_PER_THREAD>(input, output, n, Log{});
}

#pragma once

#include "abstract_primitives.h"

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void power(const T* input, T* output, int n, double exponent) {
    elementwise_constant_binary<T, Power, BLOCK_SIZE, ELEMENTS_PER_THREAD>
        (input, exponent, output, n, Power{});
}

#pragma once

#include "abstract_primitives.h"

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void offset(const T* input, T* output, int n, double offset) {
    elementwise_constant_binary<T, Offset, BLOCK_SIZE, ELEMENTS_PER_THREAD>
        (input, offset, output, n, Offset{});
}

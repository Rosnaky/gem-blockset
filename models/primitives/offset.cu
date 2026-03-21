#pragma once

#include "elementwise_constant_binary.cuh"

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void offset(const T* input, T* output, int n, double offset) {
    const int TILE_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int NUM_BLOCKS = (n + TILE_SIZE - 1) / TILE_SIZE;

    T* d_input;
    T* d_output;
    cudaMalloc(&d_input, n * sizeof(T));
    cudaMalloc(&d_output, n * sizeof(T));

    cudaMemcpy(d_input, input, n * sizeof(T), cudaMemcpyHostToDevice);

    elementwise_constant_binary<T, Offset, BLOCK_SIZE, ELEMENTS_PER_THREAD>
        (d_input, offset, d_output, n, Offset{});

    cudaMemcpy(output, d_output, n * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

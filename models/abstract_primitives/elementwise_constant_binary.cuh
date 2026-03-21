#pragma once

struct Offset { template<typename T> __device__ T operator()(T a, T b) { return a + b; } };
struct Scale { template<typename T> __device__ T operator()(T a, T b) { return a * b; } };
struct Power { template<typename T> __device__ T operator()(T a, T b) { return pow(a, b); } };

template <typename T, typename Func, int ELEMENTS_PER_THREAD>
__global__ void elementwise_constant_binary_kernel(const T* __restrict__ a, T b, T* output, int n, Func f) {
    int gid = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = gid + i;
        if (idx >= n) return;
        output[idx] = f(a[idx], b);
    }
}

template <typename T, typename Func, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void elementwise_constant_binary(const T* a, double b, T* output, int n, Func f) {
    const int TILE_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int NUM_BLOCKS = (n + TILE_SIZE - 1) / TILE_SIZE;

    T* d_a;
    T* d_output;
    cudaMalloc(&d_a, n * sizeof(T));
    cudaMalloc(&d_output, n * sizeof(T));

    cudaMemcpy(d_a, a, n * sizeof(T), cudaMemcpyHostToDevice);
    
    elementwise_constant_binary_kernel<T, Func, ELEMENTS_PER_THREAD>
        <<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, b, d_output, n, f);

    cudaMemcpy(output, d_output, n * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_output);
}

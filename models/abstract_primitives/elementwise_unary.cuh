#pragma once

struct Sqrt { template<typename T> __device__ T operator()(T a) { return static_cast<T>(sqrt((double)a)); } };

template <typename T, typename Func, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
__global__ void elementwise_unary_kernel(const T* __restrict__ input, T* output, int n, Func f) {
    int gid = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = gid + i;
        if (idx >= n) return;
        output[idx] = f(input[idx]);
    }
}

template <typename T, typename Func, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void elementwise_unary(const T* input, T* output, int n, Func f) {
    const int TILE_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int NUM_BLOCKS = (n + TILE_SIZE - 1) / TILE_SIZE;

    T* d_input;
    T* d_output;
    
    cudaMalloc(&d_input, n * sizeof(T));
    cudaMalloc(&d_output, n * sizeof(T));

    cudaMemcpy(d_input, input, n * sizeof(T), cudaMemcpyHostToDevice);
    
    elementwise_unary_kernel<T, Func, ELEMENTS_PER_THREAD>
        <<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output, n, f);

    cudaMemcpy(output, d_output, n * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

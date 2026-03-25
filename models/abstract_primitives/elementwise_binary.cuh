#pragma once

struct Add { template<typename T> __device__ T operator()(T a, T b) { return a + b; } };
struct Sub { template<typename T> __device__ T operator()(T a, T b) { return a - b; } };
struct Mul { template<typename T> __device__ T operator()(T a, T b) { return a * b; } };
struct Div {
    template<typename T> __device__ T operator()(T a, T b) {
        double v = static_cast<double>(a);
        double d = static_cast<double>(b);
        if (isnan(v) || isnan(d)) return static_cast<T>(nanf(""));
        if (d == 0.0) {
            if (v == 0.0) return static_cast<T>(nanf(""));
            return v > 0.0 ? static_cast<T>(INFINITY) : static_cast<T>(-INFINITY);
        }
        return static_cast<T>(v / d);
    }
};

template <typename T, typename Func, int ELEMENTS_PER_THREAD>
__global__ void elementwise_binary_kernel(const T* __restrict__ a, const T* __restrict__ b, T* output, int n, Func f) {
    int gid = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = gid + i;
        if (idx >= n) return;
        output[idx] = f(a[idx], b[idx]);
    }
}

template <typename T, typename Func, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void elementwise_binary(const T* a, const T* b, T* output, int n, Func f) {
    const int TILE_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int NUM_BLOCKS = (n + TILE_SIZE - 1) / TILE_SIZE;

    T* d_a;
    T* d_b;
    T* d_output;
    cudaMalloc(&d_a, n * sizeof(T));
    cudaMalloc(&d_b, n * sizeof(T));
    cudaMalloc(&d_output, n * sizeof(T));

    cudaMemcpy(d_a, a, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(T), cudaMemcpyHostToDevice);

    elementwise_binary_kernel<T, Func, ELEMENTS_PER_THREAD>
        <<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_output, n, f);

    cudaMemcpy(output, d_output, n * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_output);
}

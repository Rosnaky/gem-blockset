
template <typename T, int ELEMENTS_PER_THREAD>
__global__ void scale_kernel(const T* __restrict__ input, T* output, int n, double scalar) {
    int gid = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = gid + i;
        if (idx >= n) return;
        output[idx] = static_cast<T>(static_cast<double>(input[idx]) * scalar);
    }
}

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void scale(const T* input, T* output, int n, double scalar) {
    const int TILE_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int NUM_BLOCKS = (n + TILE_SIZE - 1) / TILE_SIZE;

    T* d_input;
    T* d_output;
    cudaMalloc(&d_input, n * sizeof(T));
    cudaMalloc(&d_output, n * sizeof(T));

    cudaMemcpy(d_input, input, n * sizeof(T), cudaMemcpyHostToDevice);

    scale_kernel<T, ELEMENTS_PER_THREAD><<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output, n, scalar);

    cudaMemcpy(output, d_output, n * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

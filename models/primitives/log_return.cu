
template <typename T, int ELEMENTS_PER_THREAD>
__global__ void log_return_kernel(const T* __restrict__ input, T* __restrict__ output, int n) {
    int gid = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;

    if (gid >= n) return;

    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        if (gid + i >= n) return;
        if (gid + i == 0) {
            output[gid + i] = nan("");
            continue;
        }
        output[gid+i] = (T)log((double)input[gid+i]/(double)input[gid+i-1]);
    }
}

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void log_return(const T* input, T* output, int n) {
    const int TILE_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int NUM_BLOCKS = (n + TILE_SIZE - 1) / TILE_SIZE;

    T* d_input;
    T* d_output;

    cudaMalloc(&d_input, n * sizeof(T));
    cudaMalloc(&d_output, n * sizeof(T));

    cudaMemcpy(d_input, input, n * sizeof(T), cudaMemcpyHostToDevice);

    log_return_kernel<T, ELEMENTS_PER_THREAD>
        <<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output, n);

    cudaMemcpy(output, d_output, n * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

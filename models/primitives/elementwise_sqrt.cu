template <typename T>
__global__ void elementwise_sqrt_kernel(const T* __restrict__ input, T* output, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    output[gid] = sqrt(input[gid]);
}

template <typename T, int BLOCK_SIZE = 256>
void elementwise_sqrt(const T* input, T* output, int n) {
    const int NUM_BLOCKS = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    T* d_input;
    T* d_output;
    cudaMalloc(&d_input, n * sizeof(T));
    cudaMalloc(&d_output, n * sizeof(T));

    cudaMemcpy(d_input, input, n * sizeof(T), cudaMemcpyHostToDevice);

    elementwise_sqrt_kernel<T><<<NUM_BLOCKS, BLOCK_SIZE>>>(d_input, d_output, n);

    cudaMemcpy(output, d_output, n * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

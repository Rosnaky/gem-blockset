
template <typename T>
__global__ void exponential_moving_average_kernel(const T* __restrict__  input, T* __restrict__ output, int num_symbols, int n, T alpha) {

    int symbol = blockIdx.x * blockDim.x + threadIdx.x;
    if (symbol >= num_symbols) return;
    
    const T* in_row = input + symbol * n;
    T* out_row = output + symbol * n;

    out_row[0] = in_row[0];
    for (int i = 1; i < n; i++) {
        out_row[i] = (double)alpha * (double)in_row[i] + (1.0 - (double)alpha) * (double)out_row[i-1];
    }
}

template <typename T, int BLOCK_SIZE>
void exponential_moving_average(const T* input, T* output, int num_symbols, int n, T alpha) {
    const int NUM_BLOCKS = (num_symbols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    T* d_input;
    T* d_output;

    cudaMalloc(&d_input, num_symbols * n * sizeof(T));
    cudaMalloc(&d_output, num_symbols * n * sizeof(T));
    cudaMemcpy(d_input, input, num_symbols * n * sizeof(T), cudaMemcpyHostToDevice);

    exponential_moving_average_kernel<T><<<NUM_BLOCKS, BLOCK_SIZE>>>(
        d_input, d_output, num_symbols, n, alpha
    );

    cudaMemcpy(output, d_output, num_symbols * n * sizeof(T), cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);

    return;
}

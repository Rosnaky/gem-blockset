
template <typename T, int BLOCK_SIZE, int ELEMENTS_PER_THREAD>
__global__ void simple_rolling_mean_kernel(const T* __restrict__  input, T* output, int n, int window) {
    
    extern __shared__ char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);

    const int tile_size = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int tile_start = blockIdx.x * tile_size;
    const int halo = window - 1;
    const int shared_len = tile_size + halo;

    for (int i = threadIdx.x; i < shared_len; i += BLOCK_SIZE) {
        int gid = tile_start - halo + i;
        smem[i] = (gid >= 0 && gid < n) ? input[gid] : T(0);
    }

    __syncthreads();

    T running_sum = T(0);
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int thread_idx = threadIdx.x * ELEMENTS_PER_THREAD + i;
        int global_idx = tile_start + thread_idx;

        if (global_idx >= n) return;

        if (global_idx < window - 1) {
            output[global_idx] = nan("");
            continue;
        }

        int smem_pos = thread_idx + halo;

        if (i == 0 || global_idx == window - 1) {
            T sum = T(0);
            int smem_start = thread_idx;
            for (int j = 0; j < window; j++) {
                sum += smem[smem_start + j];
            }
            running_sum = sum;
        }
        else {
            running_sum += smem[smem_pos] - smem[smem_pos - window];
        }

        output[global_idx] = running_sum / static_cast<T>(window);
    }
}

template <typename T, int BLOCK_SIZE, int ELEMENTS_PER_THREAD>
void simple_rolling_mean(const T* input, T* output, int n, int window) {
    const int tile_size = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int NUM_BLOCKS = (n + tile_size - 1) / tile_size;

    const int SHARED_BYTES = (tile_size + window - 1) * sizeof(T);

    T* d_input;
    T* d_output;
    cudaMalloc(&d_input, n * sizeof(T));
    cudaMalloc(&d_output, n * sizeof(T));
    
    cudaMemcpy(d_input, input, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, n * sizeof(T));

    simple_rolling_mean_kernel<T, BLOCK_SIZE, ELEMENTS_PER_THREAD><<<NUM_BLOCKS, BLOCK_SIZE, SHARED_BYTES>>>(d_input, d_output, n, window);

    cudaMemcpy(output, d_output, n * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return;
}

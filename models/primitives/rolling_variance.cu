#pragma once

template <typename T, int ELEMENTS_PER_THREAD>
__global__ void rolling_variance_kernel(const T* __restrict__ input, T* output, int n, int window) {
    
    extern __shared__ char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);
    
    const int BLOCK_SIZE = blockDim.x;
    const int tile_size = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int tile_start = blockIdx.x * tile_size;
    const int halo = window - 1;
    const int shared_len = tile_size + halo;

    for (int i = threadIdx.x; i < shared_len; i += BLOCK_SIZE) {
        int gid = tile_start - halo + i;
        smem[i] = (gid >= 0 && gid < n) ? input[gid] : T(0);
    }

    __syncthreads();

    double running_sum = 0.0;
    double running_sq_sum = 0.0;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int thread_idx = threadIdx.x * ELEMENTS_PER_THREAD + i;
        int global_idx = tile_start + thread_idx;

        if (global_idx >= n) return;

        if (global_idx < window - 1) {
            output[global_idx] = nanf("");
            continue;
        }

        int smem_pos = thread_idx + halo;

        if (i == 0 || global_idx == window - 1) {
            double sum = 0.0;
            double sq_sum = 0.0;
            int smem_start = thread_idx;
            for (int j = 0; j < window; j++) {
                double val = static_cast<double>(smem[smem_start + j]);
                sum += val;
                sq_sum += val * val;
            }
            running_sum = sum;
            running_sq_sum = sq_sum;
        }
        else {
            double entering = static_cast<double>(smem[smem_pos]);
            double leaving = static_cast<double>(smem[smem_pos - window]);
            running_sum += entering - leaving;
            running_sq_sum += entering * entering - leaving * leaving;
        }

        double e_x = running_sum / static_cast<double>(window);
        double e_sq_x = running_sq_sum / static_cast<double>(window);
        output[global_idx] = static_cast<T>(max(0.0, e_sq_x - e_x * e_x));
    }
}

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void rolling_variance(const T* input, T* output, int n, int window) {
    const int TILE_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int NUM_BLOCKS = (n + TILE_SIZE - 1) / TILE_SIZE;

    const int SHARED_BYTES = (TILE_SIZE + window - 1) * sizeof(T);

    T* d_input;
    T* d_output;
    cudaMalloc(&d_input, n * sizeof(T));
    cudaMalloc(&d_output, n * sizeof(T));
    
    cudaMemcpy(d_input, input, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, n * sizeof(T));

    rolling_variance_kernel<T, ELEMENTS_PER_THREAD>
        <<<NUM_BLOCKS, BLOCK_SIZE, SHARED_BYTES>>>(d_input, d_output, n, window);

    cudaMemcpy(output, d_output, n * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return;
}

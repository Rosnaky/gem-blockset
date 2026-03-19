
template <typename T, int BLOCK_SIZE, int ELEMENTS_PER_THREAD>
__global__ void rolling_covariance_kernel(const T* __restrict__ input_x, const T* __restrict__ input_y, T* output, int n, int window) {
    
    extern __shared__ char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);
    
    const int tile_size = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int tile_start = blockIdx.x * tile_size;
    const int halo = window - 1;
    const int shared_len = tile_size + halo;
    T* smem_x = smem;
    T* smem_y = smem + shared_len;

    for (int i = threadIdx.x; i < shared_len; i += BLOCK_SIZE) {
        int gid = tile_start - halo + i;
        smem_x[i] = (gid >= 0 && gid < n) ? input_x[gid] : T(0);
        smem_y[i] = (gid >= 0 && gid < n) ? input_y[gid] : T(0);
    }

    __syncthreads();

    double running_x_sum = 0.0;
    double running_y_sum = 0.0;
    double running_xy_sum = 0.0;
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
            double x_sum = 0.0;
            double y_sum = 0.0;
            double xy_sum = 0.0;
            int smem_start = thread_idx;
            for (int j = 0; j < window; j++) {
                double val_x = static_cast<double>(smem_x[smem_start + j]);
                double val_y = static_cast<double>(smem_y[smem_start + j]);
                x_sum += val_x;
                y_sum += val_y;
                xy_sum += val_x * val_y;
            }
            running_x_sum = x_sum;
            running_y_sum = y_sum;
            running_xy_sum = xy_sum;
        }
        else {
            double entering_x = static_cast<double>(smem_x[smem_pos]);
            double leaving_x = static_cast<double>(smem_x[smem_pos - window]);
            double entering_y = static_cast<double>(smem_y[smem_pos]);
            double leaving_y = static_cast<double>(smem_y[smem_pos - window]);
            running_x_sum += entering_x - leaving_x;
            running_y_sum += entering_y - leaving_y;
            running_xy_sum += entering_x * entering_y - leaving_x * leaving_y;
        }

        double e_x = running_x_sum / static_cast<double>(window);
        double e_y = running_y_sum / static_cast<double>(window);
        double e_xy = running_xy_sum / static_cast<double>(window);
        output[global_idx] = static_cast<T>(e_xy - e_x * e_y);
    }
}

template <typename T, int BLOCK_SIZE, int ELEMENTS_PER_THREAD>
void rolling_covariance(const T* input_x, const T* input_y, T* output, int n, int window) {
    const int TILE_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int NUM_BLOCKS = (n + TILE_SIZE - 1) / TILE_SIZE;

    const int SHARED_BYTES = (TILE_SIZE + window - 1) * sizeof(T) * 2;

    T* d_input_x, *d_input_y;
    T* d_output;
    cudaMalloc(&d_input_x, n * sizeof(T));
    cudaMalloc(&d_input_y, n * sizeof(T));
    cudaMalloc(&d_output, n * sizeof(T));
    
    cudaMemcpy(d_input_x, input_x, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_y, input_y, n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, n * sizeof(T));

    rolling_covariance_kernel<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>
        <<<NUM_BLOCKS, BLOCK_SIZE, SHARED_BYTES>>>(d_input_x, d_input_y, d_output, n, window);

    cudaMemcpy(output, d_output, n * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_input_x);
    cudaFree(d_input_y);
    cudaFree(d_output);

    return;
}

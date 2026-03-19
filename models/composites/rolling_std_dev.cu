#include "rolling_variance.cu"
#include "elementwise_sqrt.cu"

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void rolling_std_dev(const T* input, T* output, int n, int window) {
    rolling_variance<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(input, output, n, window);
    elementwise_sqrt<T, BLOCK_SIZE>(output, output, n);
}

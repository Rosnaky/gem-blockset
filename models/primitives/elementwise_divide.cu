#include "elementwise_binary.cuh"

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void elementwise_divide(const T* a, const T* b, T* output, int n) {
    elementwise_binary<T, Div, BLOCK_SIZE, ELEMENTS_PER_THREAD>(a, b, output, n, Div{});
}

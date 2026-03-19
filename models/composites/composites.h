
template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void rolling_std_dev(const T* input, T* output, int n, int window);

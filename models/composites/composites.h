
template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void rolling_std_dev(const T* input, T* output, int n, int window);

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void rolling_z_score(const T* input, T* output, int n, int window);

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void rolling_ols(const T* input_x, const T* input_y, T* output, int n, int window);

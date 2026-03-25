/*
* Warning! Do not include this header. Include the .cu file directly
*/

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void difference(const T* input, T* output, int n);

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void elementwise_add(const T* a, const T* b, T* output, int n);

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void elementwise_divide(const T* a, const T* b, T* output, int n);

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void elementwise_ln(const T* input, T* output, int n);

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void elementwise_multiply(const T* a, const T* b, T* output, int n);

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void elementwise_sqrt(const T* input, T* output, int n);

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void elementwise_subtract(const T* a, const T* b, T* output, int n);

template <typename T, int BLOCK_SIZE = 256>
void exponential_moving_average(const T* input, T* output, int num_symbols, int n, T alpha);

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void log_return(const T* input, T* output, int n);

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void offset(const T* input, T* output, int n, double offset);

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void power(const T* input, T* output, int n, double exponent);

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void rolling_covariance(const T* input_x, const T* input_y, T* output, int n, int window);

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void rolling_variance(const T* input, T* output, int n, int window);

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void scale(const T* input, T* output, int n, double scalar);

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void simple_moving_average(const T* input, T* output, int n, int window);

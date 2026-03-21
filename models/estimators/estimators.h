
template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void ou_estimation(const T* input, int n, T* speed, T* equilibrium, T* volatility_sq, int window);

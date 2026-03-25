
/*
* Add
* Sub
* Mul
* Div
*/
template <typename T, typename Func, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void elementwise_binary(const T* a, const T* b, T* output, int n, Func f);

/*
* Sqrt
*/
template <typename T, typename Func, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void elementwise_unary(const T* input, T* output, int n, Func f);

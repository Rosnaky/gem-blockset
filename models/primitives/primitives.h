#pragma once

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void simple_moving_average(const T* input, T* output, int n, int window);

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void difference(const T* input, T* output, int n);

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void log_return(const T* input, T* output, int n);

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void rolling_variance(const T* input, T* output, int n, int window);


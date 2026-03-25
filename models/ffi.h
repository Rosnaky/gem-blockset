#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double tau;
    double gamma;
    double se_gamma;
    double residual_variance;
    int lags;
    bool reject_1pct;
    bool reject_5pct;
    bool reject_10pct;
} ADFResult_f64;

typedef struct {
    float tau;
    float gamma;
    float se_gamma;
    float residual_variance;
    int lags;
    bool reject_1pct;
    bool reject_5pct;
    bool reject_10pct;
} ADFResult_f32;

void rolling_zscore_f64(const double* input, double* output, int n, int window);
void rolling_zscore_f32(const float* input, float* output, int n, int window);

void ou_estimation_f64(const double* input, int n, double* speed,
                         double* equilibrium, double* volatility_sq, int window);
void ou_estimation_f32(const float* input, int n, float* speed,
                         float* equilibrium, float* volatility_sq, int window);

void adf_test_f64(const double* input, int n, int p, ADFResult_f64* result);
void adf_test_f32(const float* input, int n, int p, ADFResult_f32* result);

#ifdef __cplusplus
}
#endif
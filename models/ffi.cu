#include "primitives/primitives.h"
#include "composites/composites.h"
#include "estimators/estimators.h"
#include "ffi.h"

#ifdef __cplusplus
extern "C" {
#endif

void rolling_zscore_f64(const double* i, double* o, int n, int w) {
    rolling_z_score<double, 256, 4>(i, o, n, w);
}

void rolling_zscore_f32(const float* i, float* o, int n, int w) {
    rolling_z_score<float, 256, 4>(i, o, n, w);
}

void ou_estimation_f64(const double* i, int n, double* s, double* e, double* v, int w) {
    ou_estimation<double, 256, 4>(i, n, s, e, v, w);
}

void ou_estimation_f32(const float* i, int n, float* s, float* e, float* v, int w) {
    ou_estimation<float, 256, 4>(i, n, s, e, v, w);
}

void adf_test_f64(const double* i, int n, int p, ADFResult_f64* r) {
    adf_test<double, 256, 4>(i, n, p, reinterpret_cast<ADFResult*>(r));
}

void adf_test_f32(const float* i, int n, int p, ADFResult_f32* r) {
    adf_test<float, 256, 4>(i, n, p, reinterpret_cast<ADFResult*>(r));
}

#ifdef __cplusplus
}
#endif

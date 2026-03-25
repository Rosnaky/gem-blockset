/*
* Augmented Dickey-Fuller test
* Test if a time series is mean reverting
*/
#pragma once

#include "primitives.h"

typedef struct {
    double tau;
    double gamma;
    double se_gamma;
    double residual_variance;
    int lags;
    bool reject_1pct;
    bool reject_5pct;
    bool reject_10pct;
} ADFResult;

template <typename T, int BLOCK_SIZE = 256, int ELEMENTS_PER_THREAD = 4>
void adf_test(const T* input, int n, int p, ADFResult *result) {
    int m = n - p - 1; // Usable rows
    int k = p + 2; // Regressors + levels + p delta

    double *X = new double[m * k]; // Design matrix
    double *Y = new double[m]; // Response

    double *XtX = new double[k * k]();
    double *XtY = new double[k]();

    T* delta = new T[n];

    difference<T, BLOCK_SIZE, ELEMENTS_PER_THREAD>(input, delta, n);

    for (int i = 0; i < m; i++) {
        int t = i + p + 1;

        Y[i] = static_cast<double>(delta[t]);

        X[i * k + 0] = 1; // Constant from alpha
        X[i * k + 1] = static_cast<double>(input[t-1]); // Lagged level

        for (int j = 0; j < p; j++) {
            X[i * k + 2 + j] = static_cast<double>(delta[t - 1 - j]); // Lagged differences
        }
    }

    // Compute XtX and XtY
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            XtY[j] += X[i * k + j] * Y[i];
            
            for (int l = 0; l < k; l++) {
                XtX[j * k + l] += X[i * k + j] * X[i * k + l];
            }
        }
    }

    // Augmented matrix [XtX I XtY] for solve and inverse to eventually get B = (XtX)^(-1)XtY
    int w = 2 * k + 1;
    double *aug = new double[k * w];
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            aug[i * w + j] = XtX[i * k + j]; // XtX
            aug[i * w + j + k] = (i == j) ? 1 : 0; // I
        }
        aug[i * w + 2 * k] = XtY[i]; // XtY
    }

    // Forward elimination with partial pivoting
    for (int col = 0; col < k; col++) {
        int pivot = col;
        double max_val = fabs(aug[col * w + col]);

        // Find pivot
        for (int row = col + 1; row < k; row++) {
            if (fabs(aug[row * w + col]) > max_val) {
                if (fabs(aug[row * w + col]) > max_val) {
                    max_val = fabs(aug[row * w + col]);
                    pivot = row;
                }
            }
        }

        // Swap rows
        if (pivot != col) {
            for (int j = 0; j < w; j++) {
                double temp = aug[col * w + j];
                aug[col * w + j] = aug[pivot * w + j];
                aug[pivot * w + j] = temp;
            }
        }

        // Eliminate below
        double diag = aug[col * w + col];
        for (int j = 0; j < w; j++) aug[col * w + j] /= diag;

        for (int row = 0; row < k; row++) {
            if (row == col) continue;
            double factor = aug[row * w + col];
            for (int j = 0; j < w; j++) {
                aug[row * w + j] -= factor * aug[col * w + j];
            }
        }
    }

    double *beta = new double[k];
    double *XtX_inv = new double[k * k];

    for (int i = 0; i < k; i++) {
        beta[i] = aug[i * w + 2 * k];
        for (int j = 0; j < k; j++) {
            XtX_inv[i * k + j] = aug[i * w + k + j];
        }
    }

    // Residual variance: sigma^2 = (1/(m-k)) * sum(residuals^2)
    double sse = 0;
    for (int i = 0; i < m; i++) {
        double predicted = 0;
        for (int j = 0; j < k; j++) {
            predicted += X[i * k + j] * beta[j];
        }
        double residual = Y[i] - predicted;
        sse += residual * residual;
    }
    double sigma_sq = sse / (m - k);

    // Standard error of gamma is sqrt(sigma^2 * [(XtX)^(-1)]_{1, 1})
    // Gamma is beta[1]
    double se_gamma = sqrt(sigma_sq * XtX_inv[1 * k + 1]);

    // Test statistic with MacKinnon asymptotic critical values
    result->tau = beta[1] / se_gamma;
    result->gamma = beta[1];
    result->se_gamma = se_gamma;
    result->residual_variance = sigma_sq;
    result->lags = p;
    result->reject_1pct = (result->tau < -3.43);
    result->reject_5pct = (result->tau < -2.86);
    result->reject_10pct = (result->tau < -2.57);

    delete[] delta;
    delete[] X;
    delete[] Y;
    delete[] XtX;
    delete[] XtY;
    delete[] aug;
    delete[] beta;
    delete[] XtX_inv;
}
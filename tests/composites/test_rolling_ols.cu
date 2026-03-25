#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include "rolling_ols.cu"

template <typename T>
void cpu_rolling_ols(const T* input_x, const T* input_y, T* output_alpha, T* output_beta, int n, int window) {
    for (int i = 0; i < n; i++) {
        if (i < window - 1) {
            output_alpha[i] = nan("");
            output_beta[i] = nan("");
            continue;
        }
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
        for (int j = 0; j < window; j++) {
            double vx = static_cast<double>(input_x[i - j]);
            double vy = static_cast<double>(input_y[i - j]);
            sum_x += vx;
            sum_y += vy;
            sum_xy += vx * vy;
            sum_xx += vx * vx;
        }
        double mean_x = sum_x / window;
        double mean_y = sum_y / window;
        double var_x = sum_xx / window - mean_x * mean_x;
        double cov_xy = sum_xy / window - mean_x * mean_y;
        if (var_x == 0.0) {
            output_beta[i] = nan("");
            output_alpha[i] = nan("");
        } else {
            double beta = cov_xy / var_x;
            double alpha = mean_y - beta * mean_x;
            output_beta[i] = static_cast<T>(beta);
            output_alpha[i] = static_cast<T>(alpha);
        }
    }
}

class RollingOlsDouble : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<double>& x, const std::vector<double>& y, int window, double tol = 1e-4) {
        int n = static_cast<int>(x.size());
        std::vector<double> gpu_alpha(n), gpu_beta(n), cpu_alpha(n), cpu_beta(n);
        cpu_rolling_ols(x.data(), y.data(), cpu_alpha.data(), cpu_beta.data(), n, window);
        rolling_ols<double, BLOCK, EPT>(x.data(), y.data(), gpu_alpha.data(), gpu_beta.data(), n, window);
        for (int i = 0; i < n; i++) {
            if (i < window - 1) {
                EXPECT_TRUE(std::isnan(gpu_alpha[i])) << "expected NaN alpha at index " << i;
                EXPECT_TRUE(std::isnan(gpu_beta[i])) << "expected NaN beta at index " << i;
            } else if (std::isnan(cpu_beta[i])) {
                EXPECT_TRUE(std::isnan(gpu_beta[i]) || std::isinf(gpu_beta[i]))
                    << "expected NaN/inf beta at index " << i;
            } else {
                EXPECT_NEAR(gpu_beta[i], cpu_beta[i], tol) << "beta mismatch at index " << i;
                EXPECT_NEAR(gpu_alpha[i], cpu_alpha[i], tol) << "alpha mismatch at index " << i;
            }
        }
    }
};

class RollingOlsFloat : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<float>& x, const std::vector<float>& y, int window, float tol = 0.5f) {
        int n = static_cast<int>(x.size());
        std::vector<float> gpu_alpha(n), gpu_beta(n), cpu_alpha(n), cpu_beta(n);
        cpu_rolling_ols(x.data(), y.data(), cpu_alpha.data(), cpu_beta.data(), n, window);
        rolling_ols<float, BLOCK, EPT>(x.data(), y.data(), gpu_alpha.data(), gpu_beta.data(), n, window);
        for (int i = 0; i < n; i++) {
            if (i < window - 1) {
                EXPECT_TRUE(std::isnan(gpu_alpha[i])) << "expected NaN alpha at index " << i;
                EXPECT_TRUE(std::isnan(gpu_beta[i])) << "expected NaN beta at index " << i;
            } else if (std::isnan(cpu_beta[i])) {
                EXPECT_TRUE(std::isnan(gpu_beta[i]) || std::isinf(gpu_beta[i]))
                    << "expected NaN/inf beta at index " << i;
            } else {
                EXPECT_NEAR(gpu_beta[i], cpu_beta[i], tol) << "beta mismatch at index " << i;
                EXPECT_NEAR(gpu_alpha[i], cpu_alpha[i], tol) << "alpha mismatch at index " << i;
            }
        }
    }
};

TEST_F(RollingOlsDouble, PerfectLinearRelationship) {
    const int n = 100;
    const int window = 20;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; i++) {
        x[i] = static_cast<double>(i);
        y[i] = 3.0 * i + 7.0;
    }

    std::vector<double> alpha(n), beta(n);
    rolling_ols<double, BLOCK, EPT>(x.data(), y.data(), alpha.data(), beta.data(), n, window);

    for (int i = window - 1; i < n; i++) {
        EXPECT_NEAR(beta[i], 3.0, 1e-4) << "beta at index " << i;
        EXPECT_NEAR(alpha[i], 7.0, 1e-2) << "alpha at index " << i;
    }
}

TEST_F(RollingOlsDouble, NegativeSlope) {
    const int n = 100;
    const int window = 15;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; i++) {
        x[i] = static_cast<double>(i);
        y[i] = -2.5 * i + 100.0;
    }

    std::vector<double> alpha(n), beta(n);
    rolling_ols<double, BLOCK, EPT>(x.data(), y.data(), alpha.data(), beta.data(), n, window);

    for (int i = window - 1; i < n; i++) {
        EXPECT_NEAR(beta[i], -2.5, 1e-4) << "beta at index " << i;
        EXPECT_NEAR(alpha[i], 100.0, 1e-1) << "alpha at index " << i;
    }
}

TEST_F(RollingOlsDouble, ZeroSlope) {
    const int n = 100;
    const int window = 10;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; i++) {
        x[i] = static_cast<double>(i);
        y[i] = 42.0;
    }

    std::vector<double> alpha(n), beta(n);
    rolling_ols<double, BLOCK, EPT>(x.data(), y.data(), alpha.data(), beta.data(), n, window);

    for (int i = window - 1; i < n; i++) {
        EXPECT_NEAR(beta[i], 0.0, 1e-6) << "beta at index " << i;
        EXPECT_NEAR(alpha[i], 42.0, 1e-4) << "alpha at index " << i;
    }
}

TEST_F(RollingOlsDouble, HandVerifiedSmallArray) {
    double x[] = {1, 2, 3, 4, 5};
    double y[] = {2, 4, 5, 4, 5};
    double alpha[5], beta[5];
    rolling_ols<double, BLOCK, EPT>(x, y, alpha, beta, 5, 3);

    EXPECT_TRUE(std::isnan(alpha[0]));
    EXPECT_TRUE(std::isnan(alpha[1]));
    EXPECT_TRUE(std::isnan(beta[0]));
    EXPECT_TRUE(std::isnan(beta[1]));

    double cpu_alpha[5], cpu_beta[5];
    cpu_rolling_ols(x, y, cpu_alpha, cpu_beta, 5, 3);
    for (int i = 2; i < 5; i++) {
        EXPECT_NEAR(beta[i], cpu_beta[i], 1e-6) << "beta at index " << i;
        EXPECT_NEAR(alpha[i], cpu_alpha[i], 1e-6) << "alpha at index " << i;
    }
}

TEST_F(RollingOlsDouble, NanCountMatchesWindowMinusOne) {
    std::vector<double> x(150), y(150);
    std::iota(x.begin(), x.end(), 1.0);
    for (int i = 0; i < 150; i++) y[i] = 2.0 * x[i] + 1.0;

    for (int window : {2, 5, 20, 50}) {
        std::vector<double> alpha(150), beta(150);
        rolling_ols<double, BLOCK, EPT>(x.data(), y.data(), alpha.data(), beta.data(), 150, window);

        int nan_count = 0;
        for (int i = 0; i < window - 1; i++) {
            if (std::isnan(beta[i])) nan_count++;
        }
        EXPECT_EQ(nan_count, window - 1) << "window=" << window;
    }
}

TEST_F(RollingOlsDouble, ResidualsSumToZero) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(1.0, 100.0);
    std::normal_distribution<double> noise(0.0, 5.0);

    const int n = 500;
    const int window = 20;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; i++) {
        x[i] = dist(rng);
        y[i] = 2.0 * x[i] + 10.0 + noise(rng);
    }

    std::vector<double> alpha(n), beta(n);
    rolling_ols<double, BLOCK, EPT>(x.data(), y.data(), alpha.data(), beta.data(), n, window);

    for (int i = window - 1; i < n; i += 50) {
        double resid_sum = 0;
        for (int j = 0; j < window; j++) {
            double predicted = alpha[i] + beta[i] * x[i - j];
            resid_sum += y[i - j] - predicted;
        }
        EXPECT_NEAR(resid_sum, 0.0, 1e-3)
            << "residuals should sum to zero at index " << i;
    }
}

TEST_F(RollingOlsDouble, ConstantXProducesNan) {
    const int n = 50;
    const int window = 10;
    std::vector<double> x(n, 5.0);
    std::vector<double> y(n);
    for (int i = 0; i < n; i++) y[i] = static_cast<double>(i);

    std::vector<double> alpha(n), beta(n);
    rolling_ols<double, BLOCK, EPT>(x.data(), y.data(), alpha.data(), beta.data(), n, window);

    for (int i = window - 1; i < n; i++) {
        EXPECT_TRUE(std::isnan(beta[i]) || std::isinf(beta[i]))
            << "constant X has zero variance, beta should be NaN/inf at index " << i;
    }
}

TEST_F(RollingOlsDouble, NoisyLinearRelationship) {
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 1.0);

    const int n = 2000;
    const int window = 100;
    const double true_beta = 2.5;
    const double true_alpha = 10.0;

    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; i++) {
        x[i] = static_cast<double>(i) * 0.1;
        y[i] = true_alpha + true_beta * x[i] + noise(rng);
    }

    std::vector<double> alpha(n), beta(n);
    rolling_ols<double, BLOCK, EPT>(x.data(), y.data(), alpha.data(), beta.data(), n, window);

    double beta_sum = 0, alpha_sum = 0;
    int count = 0;
    for (int i = window - 1; i < n; i++) {
        beta_sum += beta[i];
        alpha_sum += alpha[i];
        count++;
    }
    EXPECT_NEAR(beta_sum / count, true_beta, 0.1)
        << "average beta should approximate true slope";
    EXPECT_NEAR(alpha_sum / count, true_alpha, 1.0)
        << "average alpha should approximate true intercept";
}

TEST_F(RollingOlsDouble, CpuReferenceRandom) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    const int n = 2048;
    const int window = 20;
    std::vector<double> x(n), y(n);
    for (auto& v : x) v = dist(rng);
    for (auto& v : y) v = dist(rng);

    run_and_compare(x, y, window);
}

TEST_F(RollingOlsDouble, TileBoundary) {
    const int tile_size = BLOCK * EPT;
    const int n = tile_size * 3 + 13;
    const int window = 25;

    std::mt19937 rng(77);
    std::uniform_real_distribution<double> dist(1.0, 500.0);
    std::vector<double> x(n), y(n);
    for (auto& v : x) v = dist(rng);
    for (auto& v : y) v = dist(rng);

    std::vector<double> gpu_alpha(n), gpu_beta(n), cpu_alpha(n), cpu_beta(n);
    cpu_rolling_ols(x.data(), y.data(), cpu_alpha.data(), cpu_beta.data(), n, window);
    rolling_ols<double, BLOCK, EPT>(x.data(), y.data(), gpu_alpha.data(), gpu_beta.data(), n, window);

    for (int boundary : {tile_size, tile_size * 2}) {
        for (int offset = -10; offset <= 10; offset++) {
            int i = boundary + offset;
            if (i < window - 1 || i >= n) continue;
            if (std::isnan(cpu_beta[i])) continue;
            EXPECT_NEAR(gpu_beta[i], cpu_beta[i], 1e-4)
                << "beta tile boundary divergence at index " << i;
            EXPECT_NEAR(gpu_alpha[i], cpu_alpha[i], 1e-4)
                << "alpha tile boundary divergence at index " << i;
        }
    }
}

TEST_F(RollingOlsDouble, NonDivisibleArrayLength) {
    std::mt19937 rng(256);
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    for (int n : {10, 33, 257, 1023}) {
        const int window = 5;
        std::vector<double> x(n), y(n);
        for (auto& v : x) v = dist(rng);
        for (auto& v : y) v = dist(rng);
        run_and_compare(x, y, window);
    }
}

TEST_F(RollingOlsFloat, PerfectLinearRelationship) {
    const int n = 100;
    const int window = 20;
    std::vector<float> x(n), y(n);
    for (int i = 0; i < n; i++) {
        x[i] = static_cast<float>(i);
        y[i] = 3.0f * i + 7.0f;
    }

    std::vector<float> alpha(n), beta(n);
    rolling_ols<float, BLOCK, EPT>(x.data(), y.data(), alpha.data(), beta.data(), n, window);

    for (int i = window - 1; i < n; i++) {
        EXPECT_NEAR(beta[i], 3.0f, 0.01f) << "beta at index " << i;
        EXPECT_NEAR(alpha[i], 7.0f, 0.5f) << "alpha at index " << i;
    }
}

TEST_F(RollingOlsFloat, CpuReferenceRandom) {
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(1.0f, 100.0f);

    const int n = 2048;
    const int window = 20;
    std::vector<float> x(n), y(n);
    for (auto& v : x) v = dist(rng);
    for (auto& v : y) v = dist(rng);

    run_and_compare(x, y, window);
}

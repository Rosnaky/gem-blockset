#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include "rolling_covariance.cu"

template <typename T>
void cpu_rolling_covariance(const T* input_x, const T* input_y, T* output, int n, int window) {
    for (int i = 0; i < n; i++) {
        if (i < window - 1) {
            output[i] = nan("");
            continue;
        }
        double x_sum = 0.0;
        double y_sum = 0.0;
        double xy_sum = 0.0;
        for (int j = 0; j < window; j++) {
            double vx = static_cast<double>(input_x[i - j]);
            double vy = static_cast<double>(input_y[i - j]);
            x_sum += vx;
            y_sum += vy;
            xy_sum += vx * vy;
        }
        double e_x = x_sum / window;
        double e_y = y_sum / window;
        double e_xy = xy_sum / window;
        output[i] = static_cast<T>(e_xy - e_x * e_y);
    }
}

class RollingCovarianceDouble : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<double>& x, const std::vector<double>& y, int window, double tol = 1e-6) {
        int n = static_cast<int>(x.size());
        std::vector<double> gpu_out(n), cpu_out(n);
        cpu_rolling_covariance(x.data(), y.data(), cpu_out.data(), n, window);
        rolling_covariance<double, BLOCK, EPT>(x.data(), y.data(), gpu_out.data(), n, window);
        for (int i = 0; i < n; i++) {
            if (i < window - 1) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else {
                EXPECT_NEAR(gpu_out[i], cpu_out[i], tol) << "mismatch at index " << i;
            }
        }
    }
};

class RollingCovarianceFloat : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<float>& x, const std::vector<float>& y, int window, float tol = 0.5f) {
        int n = static_cast<int>(x.size());
        std::vector<float> gpu_out(n), cpu_out(n);
        cpu_rolling_covariance(x.data(), y.data(), cpu_out.data(), n, window);
        rolling_covariance<float, BLOCK, EPT>(x.data(), y.data(), gpu_out.data(), n, window);
        for (int i = 0; i < n; i++) {
            if (i < window - 1) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else {
                EXPECT_NEAR(gpu_out[i], cpu_out[i], tol) << "mismatch at index " << i;
            }
        }
    }
};

TEST_F(RollingCovarianceDouble, HandVerifiedSmallArray) {
    double x[] = {1, 2, 3, 4, 5};
    double y[] = {2, 4, 6, 8, 10};
    double output[5];
    rolling_covariance<double, BLOCK, EPT>(x, y, output, 5, 3);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));

    double ex = (1 + 2 + 3) / 3.0;
    double ey = (2 + 4 + 6) / 3.0;
    double exy = (1*2 + 2*4 + 3*6) / 3.0;
    EXPECT_NEAR(output[2], exy - ex * ey, 1e-9);

    ex = (2 + 3 + 4) / 3.0;
    ey = (4 + 6 + 8) / 3.0;
    exy = (2*4 + 3*6 + 4*8) / 3.0;
    EXPECT_NEAR(output[3], exy - ex * ey, 1e-9);

    ex = (3 + 4 + 5) / 3.0;
    ey = (6 + 8 + 10) / 3.0;
    exy = (3*6 + 4*8 + 5*10) / 3.0;
    EXPECT_NEAR(output[4], exy - ex * ey, 1e-9);
}

TEST_F(RollingCovarianceDouble, PerfectPositiveCorrelation) {
    const int n = 200;
    const int window = 10;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; i++) {
        x[i] = static_cast<double>(i);
        y[i] = 3.0 * i + 7.0;
    }

    std::vector<double> output(n);
    rolling_covariance<double, BLOCK, EPT>(x.data(), y.data(), output.data(), n, window);

    for (int i = window - 1; i < n; i++) {
        EXPECT_GT(output[i], 0.0) << "positive covariance expected at index " << i;
    }
}

TEST_F(RollingCovarianceDouble, PerfectNegativeCorrelation) {
    const int n = 200;
    const int window = 10;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; i++) {
        x[i] = static_cast<double>(i);
        y[i] = -2.0 * i + 100.0;
    }

    std::vector<double> output(n);
    rolling_covariance<double, BLOCK, EPT>(x.data(), y.data(), output.data(), n, window);

    for (int i = window - 1; i < n; i++) {
        EXPECT_LT(output[i], 0.0) << "negative covariance expected at index " << i;
    }
}

TEST_F(RollingCovarianceDouble, CovarianceWithSelfEqualsVariance) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    const int n = 1000;
    const int window = 20;
    std::vector<double> x(n);
    for (auto& v : x) v = dist(rng);

    std::vector<double> cov_out(n), var_out(n);
    rolling_covariance<double, BLOCK, EPT>(x.data(), x.data(), cov_out.data(), n, window);

    double var_cpu[1000];
    for (int i = 0; i < n; i++) {
        if (i < window - 1) { var_cpu[i] = nan(""); continue; }
        double sum = 0, sq_sum = 0;
        for (int j = 0; j < window; j++) {
            sum += x[i - j];
            sq_sum += x[i - j] * x[i - j];
        }
        var_cpu[i] = sq_sum / window - (sum / window) * (sum / window);
    }

    for (int i = window - 1; i < n; i++) {
        EXPECT_NEAR(cov_out[i], var_cpu[i], 1e-6) << "Cov(X,X) != Var(X) at index " << i;
    }
}

TEST_F(RollingCovarianceDouble, IndependentSeriesNearZero) {
    std::mt19937 rng_x(42);
    std::mt19937 rng_y(999);
    std::normal_distribution<double> dist(0.0, 1.0);

    const int n = 5000;
    const int window = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; i++) {
        x[i] = dist(rng_x);
        y[i] = dist(rng_y);
    }

    std::vector<double> output(n);
    rolling_covariance<double, BLOCK, EPT>(x.data(), y.data(), output.data(), n, window);

    double max_abs_cov = 0.0;
    for (int i = window - 1; i < n; i++) {
        max_abs_cov = std::max(max_abs_cov, std::abs(output[i]));
    }
    EXPECT_LT(max_abs_cov, 0.5) << "independent series should have near-zero covariance";
}

TEST_F(RollingCovarianceDouble, NanCountMatchesWindowMinusOne) {
    std::vector<double> x(100), y(100);
    std::iota(x.begin(), x.end(), 1.0);
    std::iota(y.begin(), y.end(), 50.0);

    for (int window : {2, 5, 17, 50, 99}) {
        std::vector<double> output(100);
        rolling_covariance<double, BLOCK, EPT>(x.data(), y.data(), output.data(), 100, window);

        int nan_count = 0;
        for (int i = 0; i < 100; i++) {
            if (std::isnan(output[i])) nan_count++;
        }
        EXPECT_EQ(nan_count, window - 1) << "window=" << window;
    }
}

TEST_F(RollingCovarianceDouble, ConstantInputsZeroCovariance) {
    std::vector<double> x(500, 42.0);
    std::vector<double> y(500, 99.0);
    std::vector<double> output(500);
    rolling_covariance<double, BLOCK, EPT>(x.data(), y.data(), output.data(), 500, 20);

    for (int i = 19; i < 500; i++) {
        EXPECT_NEAR(output[i], 0.0, 1e-9) << "index " << i;
    }
}

TEST_F(RollingCovarianceDouble, WindowEqualsOne) {
    double x[] = {1.0, 2.0, 3.0, 4.0};
    double y[] = {5.0, 6.0, 7.0, 8.0};
    double output[4];
    rolling_covariance<double, BLOCK, EPT>(x, y, output, 4, 1);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(output[i], 0.0, 1e-12) << "index " << i;
    }
}

TEST_F(RollingCovarianceDouble, WindowEqualsN) {
    double x[] = {10, 20, 30, 40, 50};
    double y[] = {2, 4, 6, 8, 10};
    double output[5];
    rolling_covariance<double, BLOCK, EPT>(x, y, output, 5, 5);

    for (int i = 0; i < 4; i++) {
        EXPECT_TRUE(std::isnan(output[i])) << "index " << i;
    }

    double ex = 30.0, ey = 6.0;
    double exy = (10*2 + 20*4 + 30*6 + 40*8 + 50*10) / 5.0;
    EXPECT_NEAR(output[4], exy - ex * ey, 1e-9);
}

TEST_F(RollingCovarianceDouble, SingleElement) {
    double x[] = {42.0};
    double y[] = {99.0};
    double output[1];
    rolling_covariance<double, BLOCK, EPT>(x, y, output, 1, 1);

    EXPECT_NEAR(output[0], 0.0, 1e-12);
}

TEST_F(RollingCovarianceDouble, Symmetry) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    const int n = 500;
    const int window = 15;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; i++) {
        x[i] = dist(rng);
        y[i] = dist(rng);
    }

    std::vector<double> cov_xy(n), cov_yx(n);
    rolling_covariance<double, BLOCK, EPT>(x.data(), y.data(), cov_xy.data(), n, window);
    rolling_covariance<double, BLOCK, EPT>(y.data(), x.data(), cov_yx.data(), n, window);

    for (int i = window - 1; i < n; i++) {
        EXPECT_NEAR(cov_xy[i], cov_yx[i], 1e-9) << "Cov(X,Y) != Cov(Y,X) at index " << i;
    }
}

TEST_F(RollingCovarianceDouble, ScalingProperty) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(1.0, 50.0);

    const int n = 500;
    const int window = 15;
    const double a = 3.0, b = 2.0;

    std::vector<double> x(n), y(n), ax(n), by(n);
    for (int i = 0; i < n; i++) {
        x[i] = dist(rng);
        y[i] = dist(rng);
        ax[i] = a * x[i];
        by[i] = b * y[i];
    }

    std::vector<double> cov_xy(n), cov_axby(n);
    rolling_covariance<double, BLOCK, EPT>(x.data(), y.data(), cov_xy.data(), n, window);
    rolling_covariance<double, BLOCK, EPT>(ax.data(), by.data(), cov_axby.data(), n, window);

    for (int i = window - 1; i < n; i++) {
        EXPECT_NEAR(cov_axby[i], a * b * cov_xy[i], 1e-4)
            << "Cov(aX,bY) should equal ab*Cov(X,Y) at index " << i;
    }
}

TEST_F(RollingCovarianceDouble, CpuReferenceRandom) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(10.0, 200.0);

    const int n = 4096;
    const int window = 25;
    std::vector<double> x(n), y(n);
    for (auto& v : x) v = dist(rng);
    for (auto& v : y) v = dist(rng);

    run_and_compare(x, y, window);
}

TEST_F(RollingCovarianceDouble, TileBoundary) {
    const int tile_size = BLOCK * EPT;
    const int n = tile_size * 3 + 19;
    const int window = 30;

    std::mt19937 rng(77);
    std::uniform_real_distribution<double> dist(1.0, 500.0);
    std::vector<double> x(n), y(n);
    for (auto& v : x) v = dist(rng);
    for (auto& v : y) v = dist(rng);

    std::vector<double> gpu_out(n), cpu_out(n);
    cpu_rolling_covariance(x.data(), y.data(), cpu_out.data(), n, window);
    rolling_covariance<double, BLOCK, EPT>(x.data(), y.data(), gpu_out.data(), n, window);

    for (int boundary : {tile_size, tile_size * 2}) {
        for (int offset = -10; offset <= 10; offset++) {
            int i = boundary + offset;
            if (i < window - 1 || i >= n) continue;
            EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-6)
                << "tile boundary divergence at index " << i;
        }
    }
}

TEST_F(RollingCovarianceDouble, NonDivisibleArrayLength) {
    std::mt19937 rng(256);
    std::uniform_real_distribution<double> dist(0.0, 100.0);

    for (int n : {5, 33, 257, 1023, 4097}) {
        const int window = 7;
        std::vector<double> x(n), y(n);
        for (auto& v : x) v = dist(rng);
        for (auto& v : y) v = dist(rng);
        run_and_compare(x, y, window);
    }
}

TEST_F(RollingCovarianceDouble, LargeWindow) {
    const int n = 5000;
    const int window = 500;
    std::mt19937 rng(314);
    std::uniform_real_distribution<double> dist(10.0, 1000.0);

    std::vector<double> x(n), y(n);
    for (auto& v : x) v = dist(rng);
    for (auto& v : y) v = dist(rng);

    run_and_compare(x, y, window, 1e-3);
}

TEST_F(RollingCovarianceFloat, HandVerifiedSmallArray) {
    float x[] = {1, 2, 3, 4, 5};
    float y[] = {2, 4, 6, 8, 10};
    float output[5];
    rolling_covariance<float, BLOCK, EPT>(x, y, output, 5, 3);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));

    double ex = (1 + 2 + 3) / 3.0;
    double ey = (2 + 4 + 6) / 3.0;
    double exy = (1*2 + 2*4 + 3*6) / 3.0;
    EXPECT_NEAR(output[2], static_cast<float>(exy - ex * ey), 1e-4f);
}

TEST_F(RollingCovarianceFloat, CpuReferenceRandom) {
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    const int n = 4096;
    const int window = 20;
    std::vector<float> x(n), y(n);
    for (auto& v : x) v = dist(rng);
    for (auto& v : y) v = dist(rng);

    run_and_compare(x, y, window);
}

TEST_F(RollingCovarianceFloat, LargeWindowPrecisionDrift) {
    std::mt19937 rng(55);
    std::uniform_real_distribution<float> dist(100.0f, 10000.0f);

    const int n = 10000;
    const int window = 500;
    std::vector<float> x(n), y(n);
    for (auto& v : x) v = dist(rng);
    for (auto& v : y) v = dist(rng);

    run_and_compare(x, y, window, 100.0f);
}

TEST_F(RollingCovarianceFloat, ConstantInputsZeroCovariance) {
    std::vector<float> x(300, 7.0f);
    std::vector<float> y(300, 13.0f);
    std::vector<float> output(300);
    rolling_covariance<float, BLOCK, EPT>(x.data(), y.data(), output.data(), 300, 10);

    for (int i = 9; i < 300; i++) {
        EXPECT_NEAR(output[i], 0.0f, 1e-4f) << "index " << i;
    }
}

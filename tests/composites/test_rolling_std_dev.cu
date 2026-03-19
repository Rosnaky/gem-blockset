#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include "rolling_std_dev.cu"

template <typename T>
void cpu_rolling_std_dev(const T* input, T* output, int n, int window) {
    for (int i = 0; i < n; i++) {
        if (i < window - 1) {
            output[i] = nan("");
            continue;
        }
        double sum = 0.0;
        double sq_sum = 0.0;
        for (int j = 0; j < window; j++) {
            double val = static_cast<double>(input[i - j]);
            sum += val;
            sq_sum += val * val;
        }
        double e_x = sum / window;
        double e_sq_x = sq_sum / window;
        output[i] = static_cast<T>(std::sqrt(e_sq_x - e_x * e_x));
    }
}

class RollingStdDevDouble : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<double>& input, int window, double tol = 1e-6) {
        int n = static_cast<int>(input.size());
        std::vector<double> gpu_out(n), cpu_out(n);
        cpu_rolling_std_dev(input.data(), cpu_out.data(), n, window);
        rolling_std_dev<double, BLOCK, EPT>(input.data(), gpu_out.data(), n, window);
        for (int i = 0; i < n; i++) {
            if (i < window - 1) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else {
                EXPECT_NEAR(gpu_out[i], cpu_out[i], tol) << "mismatch at index " << i;
            }
        }
    }
};

class RollingStdDevFloat : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<float>& input, int window, float tol = 0.1f) {
        int n = static_cast<int>(input.size());
        std::vector<float> gpu_out(n), cpu_out(n);
        cpu_rolling_std_dev(input.data(), cpu_out.data(), n, window);
        rolling_std_dev<float, BLOCK, EPT>(input.data(), gpu_out.data(), n, window);
        for (int i = 0; i < n; i++) {
            if (i < window - 1) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else {
                EXPECT_NEAR(gpu_out[i], cpu_out[i], tol) << "mismatch at index " << i;
            }
        }
    }
};

TEST_F(RollingStdDevDouble, HandVerifiedSmallArray) {
    double input[] = {2, 4, 4, 4, 5, 5, 7, 9};
    double output[8];
    rolling_std_dev<double, BLOCK, EPT>(input, output, 8, 4);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));
    EXPECT_TRUE(std::isnan(output[2]));

    double vals[] = {2, 4, 4, 4};
    double mean = (2 + 4 + 4 + 4) / 4.0;
    double var = 0;
    for (double v : vals) var += (v - mean) * (v - mean);
    var /= 4.0;
    EXPECT_NEAR(output[3], std::sqrt(var), 1e-9);
}

TEST_F(RollingStdDevDouble, ResultIsNonNegative) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

    const int n = 4096;
    const int window = 30;
    std::vector<double> input(n), output(n);
    for (auto& v : input) v = dist(rng);

    rolling_std_dev<double, BLOCK, EPT>(input.data(), output.data(), n, window);

    for (int i = window - 1; i < n; i++) {
        EXPECT_GE(output[i], 0.0) << "negative std dev at index " << i;
    }
}

TEST_F(RollingStdDevDouble, ConstantInputZeroStd) {
    std::vector<double> input(500, 42.0);
    std::vector<double> output(500);
    rolling_std_dev<double, BLOCK, EPT>(input.data(), output.data(), 500, 20);

    for (int i = 19; i < 500; i++) {
        EXPECT_NEAR(output[i], 0.0, 1e-9) << "index " << i;
    }
}

TEST_F(RollingStdDevDouble, StdIsSquareRootOfVariance) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    const int n = 1000;
    const int window = 20;
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    std::vector<double> std_out(n), var_out(n);
    rolling_std_dev<double, BLOCK, EPT>(input.data(), std_out.data(), n, window);
    rolling_variance<double, BLOCK, EPT>(input.data(), var_out.data(), n, window);

    for (int i = window - 1; i < n; i++) {
        EXPECT_NEAR(std_out[i] * std_out[i], var_out[i], 1e-6)
            << "std^2 != variance at index " << i;
    }
}

TEST_F(RollingStdDevDouble, NanCountMatchesWindowMinusOne) {
    std::vector<double> input(150);
    std::iota(input.begin(), input.end(), 1.0);

    for (int window : {2, 7, 20, 50, 100}) {
        std::vector<double> output(150);
        rolling_std_dev<double, BLOCK, EPT>(input.data(), output.data(), 150, window);

        int nan_count = 0;
        for (int i = 0; i < 150; i++) {
            if (std::isnan(output[i])) nan_count++;
        }
        EXPECT_EQ(nan_count, window - 1) << "window=" << window;
    }
}

TEST_F(RollingStdDevDouble, WindowEqualsOne) {
    double input[] = {3.14, 2.71, 1.41, 1.73};
    double output[4];
    rolling_std_dev<double, BLOCK, EPT>(input, output, 4, 1);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(output[i], 0.0, 1e-12) << "index " << i;
    }
}

TEST_F(RollingStdDevDouble, WindowEqualsN) {
    double input[] = {10, 20, 30, 40, 50};
    double output[5];
    rolling_std_dev<double, BLOCK, EPT>(input, output, 5, 5);

    for (int i = 0; i < 4; i++) {
        EXPECT_TRUE(std::isnan(output[i])) << "index " << i;
    }

    double mean = 30.0;
    double var = (400 + 100 + 0 + 100 + 400) / 5.0;
    EXPECT_NEAR(output[4], std::sqrt(var), 1e-9);
}

TEST_F(RollingStdDevDouble, ScalingProperty) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    const int n = 500;
    const int window = 15;
    const double scale = 5.0;

    std::vector<double> input(n), scaled(n);
    for (int i = 0; i < n; i++) {
        input[i] = dist(rng);
        scaled[i] = input[i] * scale;
    }

    std::vector<double> out_orig(n), out_scaled(n);
    rolling_std_dev<double, BLOCK, EPT>(input.data(), out_orig.data(), n, window);
    rolling_std_dev<double, BLOCK, EPT>(scaled.data(), out_scaled.data(), n, window);

    for (int i = window - 1; i < n; i++) {
        EXPECT_NEAR(out_scaled[i], std::abs(scale) * out_orig[i], 1e-4)
            << "Std(cX) should equal |c|*Std(X) at index " << i;
    }
}

TEST_F(RollingStdDevDouble, ShiftInvariance) {
    std::mt19937 rng(456);
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    const int n = 500;
    const int window = 15;
    const double shift = 10000.0;

    std::vector<double> input(n), shifted(n);
    for (int i = 0; i < n; i++) {
        input[i] = dist(rng);
        shifted[i] = input[i] + shift;
    }

    std::vector<double> out_orig(n), out_shifted(n);
    rolling_std_dev<double, BLOCK, EPT>(input.data(), out_orig.data(), n, window);
    rolling_std_dev<double, BLOCK, EPT>(shifted.data(), out_shifted.data(), n, window);

    for (int i = window - 1; i < n; i++) {
        EXPECT_NEAR(out_shifted[i], out_orig[i], 1e-3)
            << "Std(X+c) should equal Std(X) at index " << i;
    }
}

TEST_F(RollingStdDevDouble, CpuReferenceRandom) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(50.0, 200.0);

    const int n = 4096;
    const int window = 20;
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, window);
}

TEST_F(RollingStdDevDouble, TileBoundary) {
    const int tile_size = BLOCK * EPT;
    const int n = tile_size * 3 + 11;
    const int window = 25;

    std::mt19937 rng(77);
    std::uniform_real_distribution<double> dist(1.0, 500.0);
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    std::vector<double> gpu_out(n), cpu_out(n);
    cpu_rolling_std_dev(input.data(), cpu_out.data(), n, window);
    rolling_std_dev<double, BLOCK, EPT>(input.data(), gpu_out.data(), n, window);

    for (int boundary : {tile_size, tile_size * 2}) {
        for (int offset = -10; offset <= 10; offset++) {
            int i = boundary + offset;
            if (i < window - 1 || i >= n) continue;
            EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-6)
                << "tile boundary divergence at index " << i;
        }
    }
}

TEST_F(RollingStdDevDouble, NonDivisibleArrayLength) {
    std::mt19937 rng(256);
    std::uniform_real_distribution<double> dist(0.0, 100.0);

    for (int n : {5, 33, 257, 1023, 4097}) {
        const int window = 7;
        std::vector<double> input(n);
        for (auto& v : input) v = dist(rng);
        run_and_compare(input, window);
    }
}

TEST_F(RollingStdDevDouble, AlternatingValues) {
    const int n = 100;
    const int window = 4;
    std::vector<double> input(n);
    for (int i = 0; i < n; i++) input[i] = (i % 2 == 0) ? 0.0 : 10.0;

    std::vector<double> output(n);
    rolling_std_dev<double, BLOCK, EPT>(input.data(), output.data(), n, window);

    double expected_var = 25.0;
    double expected_std = 5.0;
    for (int i = window - 1; i < n; i++) {
        EXPECT_NEAR(output[i], expected_std, 1e-9) << "index " << i;
    }
}

TEST_F(RollingStdDevFloat, HandVerifiedSmallArray) {
    float input[] = {2, 4, 4, 4, 5, 5, 7, 9};
    float output[8];
    rolling_std_dev<float, BLOCK, EPT>(input, output, 8, 4);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));
    EXPECT_TRUE(std::isnan(output[2]));

    float mean = (2 + 4 + 4 + 4) / 4.0f;
    float var = ((2 - mean) * (2 - mean) + 3 * (4 - mean) * (4 - mean)) / 4.0f;
    EXPECT_NEAR(output[3], std::sqrt(var), 1e-4f);
}

TEST_F(RollingStdDevFloat, CpuReferenceRandom) {
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    const int n = 4096;
    const int window = 20;
    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, window);
}

TEST_F(RollingStdDevFloat, LargeWindowPrecisionDrift) {
    std::mt19937 rng(55);
    std::uniform_real_distribution<float> dist(100.0f, 10000.0f);

    const int n = 10000;
    const int window = 500;
    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, window, 10.0f);
}

TEST_F(RollingStdDevFloat, ConstantInputZeroStd) {
    std::vector<float> input(300, 7.0f);
    std::vector<float> output(300);
    rolling_std_dev<float, BLOCK, EPT>(input.data(), output.data(), 300, 10);

    for (int i = 9; i < 300; i++) {
        EXPECT_NEAR(output[i], 0.0f, 1e-3f) << "index " << i;
    }
}

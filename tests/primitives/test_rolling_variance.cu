#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include "rolling_variance.cu"

template <typename T>
void cpu_rolling_variance(const T* input, T* output, int n, int window) {
    for (int i = 0; i < n; i++) {
        if (i < window - 1) {
            output[i] = nan("");
            continue;
        }
        T sum = T(0);
        T sq_sum = T(0);
        for (int j = 0; j < window; j++) {
            T val = input[i - j];
            sum += val;
            sq_sum += val * val;
        }
        T e_x = sum / static_cast<T>(window);
        T e_sq_x = sq_sum / static_cast<T>(window);
        output[i] = e_sq_x - e_x * e_x;
    }
}

class RollingVarianceDouble : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<double>& input, int window, double tol = 1e-6) {
        int n = static_cast<int>(input.size());
        std::vector<double> gpu_out(n), cpu_out(n);
        cpu_rolling_variance(input.data(), cpu_out.data(), n, window);
        rolling_variance<double, BLOCK, EPT>(input.data(), gpu_out.data(), n, window);
        for (int i = 0; i < n; i++) {
            if (i < window - 1) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else {
                EXPECT_NEAR(gpu_out[i], cpu_out[i], tol) << "mismatch at index " << i;
            }
        }
    }
};

class RollingVarianceFloat : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<float>& input, int window, float tol = 0.1f) {
        int n = static_cast<int>(input.size());
        std::vector<float> gpu_out(n), cpu_out(n);
        cpu_rolling_variance(input.data(), cpu_out.data(), n, window);
        rolling_variance<float, BLOCK, EPT>(input.data(), gpu_out.data(), n, window);
        for (int i = 0; i < n; i++) {
            if (i < window - 1) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else {
                EXPECT_NEAR(gpu_out[i], cpu_out[i], tol) << "mismatch at index " << i;
            }
        }
    }
};

TEST_F(RollingVarianceDouble, HandVerifiedSmallArray) {
    double input[] = {2, 4, 4, 4, 5, 5, 7, 9};
    double output[8];
    rolling_variance<double, BLOCK, EPT>(input, output, 8, 4);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));
    EXPECT_TRUE(std::isnan(output[2]));

    double vals0[] = {2, 4, 4, 4};
    double mean0 = (2 + 4 + 4 + 4) / 4.0;
    double var0 = 0;
    for (double v : vals0) var0 += (v - mean0) * (v - mean0);
    var0 /= 4.0;
    EXPECT_NEAR(output[3], var0, 1e-9);

    double vals1[] = {4, 4, 4, 5};
    double mean1 = (4 + 4 + 4 + 5) / 4.0;
    double var1 = 0;
    for (double v : vals1) var1 += (v - mean1) * (v - mean1);
    var1 /= 4.0;
    EXPECT_NEAR(output[4], var1, 1e-9);
}

TEST_F(RollingVarianceDouble, NanCountMatchesWindowMinusOne) {
    std::vector<double> input(150);
    std::iota(input.begin(), input.end(), 1.0);

    for (int window : {2, 7, 20, 50, 100}) {
        std::vector<double> output(150);
        rolling_variance<double, BLOCK, EPT>(input.data(), output.data(), 150, window);

        int nan_count = 0;
        for (int i = 0; i < 150; i++) {
            if (std::isnan(output[i])) nan_count++;
        }
        EXPECT_EQ(nan_count, window - 1) << "window=" << window;
    }
}

TEST_F(RollingVarianceDouble, ConstantInputZeroVariance) {
    std::vector<double> input(500, 42.0);
    std::vector<double> output(500);
    rolling_variance<double, BLOCK, EPT>(input.data(), output.data(), 500, 20);

    for (int i = 19; i < 500; i++) {
        EXPECT_NEAR(output[i], 0.0, 1e-9) << "index " << i;
    }
}

TEST_F(RollingVarianceDouble, VarianceIsNonNegative) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

    const int n = 4096;
    const int window = 30;
    std::vector<double> input(n), output(n);
    for (auto& v : input) v = dist(rng);

    rolling_variance<double, BLOCK, EPT>(input.data(), output.data(), n, window);

    for (int i = window - 1; i < n; i++) {
        EXPECT_GE(output[i], 0.0) << "negative variance at index " << i;
    }
}

TEST_F(RollingVarianceDouble, WindowEqualsOne) {
    double input[] = {3.14, 2.71, 1.41, 1.73, 0.577};
    double output[5];
    rolling_variance<double, BLOCK, EPT>(input, output, 5, 1);

    for (int i = 0; i < 5; i++) {
        EXPECT_NEAR(output[i], 0.0, 1e-12) << "index " << i;
    }
}

TEST_F(RollingVarianceDouble, WindowEqualsN) {
    double input[] = {10, 20, 30, 40, 50};
    double output[5];
    rolling_variance<double, BLOCK, EPT>(input, output, 5, 5);

    for (int i = 0; i < 4; i++) {
        EXPECT_TRUE(std::isnan(output[i])) << "index " << i;
    }

    double expected = (400 + 100 + 0 + 100 + 400) / 5.0;
    EXPECT_NEAR(output[4], expected, 1e-9);
}

TEST_F(RollingVarianceDouble, SingleElement) {
    double input[] = {99.0};
    double output[1];
    rolling_variance<double, BLOCK, EPT>(input, output, 1, 1);

    EXPECT_NEAR(output[0], 0.0, 1e-12);
}

TEST_F(RollingVarianceDouble, TwoElementWindow) {
    double input[] = {10, 20, 50, 50};
    double output[4];
    rolling_variance<double, BLOCK, EPT>(input, output, 4, 2);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_NEAR(output[1], 25.0, 1e-9);
    EXPECT_NEAR(output[2], 225.0, 1e-9);
    EXPECT_NEAR(output[3], 0.0, 1e-9);
}

TEST_F(RollingVarianceDouble, AlternatingValues) {
    const int n = 100;
    const int window = 4;
    std::vector<double> input(n);
    for (int i = 0; i < n; i++) input[i] = (i % 2 == 0) ? 0.0 : 10.0;

    std::vector<double> output(n);
    rolling_variance<double, BLOCK, EPT>(input.data(), output.data(), n, window);

    double expected = (25.0 + 25.0 + 25.0 + 25.0) / 4.0;
    for (int i = window - 1; i < n; i++) {
        EXPECT_NEAR(output[i], expected, 1e-9) << "index " << i;
    }
}

TEST_F(RollingVarianceDouble, CpuReferenceRandom) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(50.0, 200.0);

    const int n = 4096;
    const int window = 20;
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, window, 1e-6);
}

TEST_F(RollingVarianceDouble, TileBoundary) {
    const int tile_size = BLOCK * EPT;
    const int n = tile_size * 3 + 13;
    const int window = 25;

    std::mt19937 rng(77);
    std::uniform_real_distribution<double> dist(1.0, 500.0);
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    std::vector<double> gpu_out(n), cpu_out(n);
    cpu_rolling_variance(input.data(), cpu_out.data(), n, window);
    rolling_variance<double, BLOCK, EPT>(input.data(), gpu_out.data(), n, window);

    for (int boundary : {tile_size, tile_size * 2}) {
        for (int offset = -10; offset <= 10; offset++) {
            int i = boundary + offset;
            if (i < window - 1 || i >= n) continue;
            EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-6)
                << "tile boundary divergence at index " << i;
        }
    }
}

TEST_F(RollingVarianceDouble, NonDivisibleArrayLength) {
    std::mt19937 rng(256);
    std::uniform_real_distribution<double> dist(0.0, 100.0);

    for (int n : {5, 33, 257, 1023, 4097}) {
        const int window = 7;
        std::vector<double> input(n);
        for (auto& v : input) v = dist(rng);
        run_and_compare(input, window, 1e-6);
    }
}

TEST_F(RollingVarianceDouble, LargeWindow) {
    const int n = 5000;
    const int window = 500;
    std::mt19937 rng(314);
    std::uniform_real_distribution<double> dist(10.0, 1000.0);

    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, window, 1e-3);
}

TEST_F(RollingVarianceDouble, MonotonicallyIncreasing) {
    const int n = 1000;
    const int window = 10;
    std::vector<double> input(n);
    for (int i = 0; i < n; i++) input[i] = static_cast<double>(i);

    std::vector<double> output(n);
    rolling_variance<double, BLOCK, EPT>(input.data(), output.data(), n, window);

    double expected = 0;
    for (int j = 0; j < window; j++) {
        double diff = j - (window - 1) / 2.0;
        expected += diff * diff;
    }
    expected /= window;

    for (int i = window - 1; i < n; i++) {
        EXPECT_NEAR(output[i], expected, 1e-6) << "index " << i;
    }
}

TEST_F(RollingVarianceDouble, ScalingProperty) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    const int n = 500;
    const int window = 15;
    const double scale = 3.0;

    std::vector<double> input(n), scaled_input(n);
    for (int i = 0; i < n; i++) {
        input[i] = dist(rng);
        scaled_input[i] = input[i] * scale;
    }

    std::vector<double> out_orig(n), out_scaled(n);
    rolling_variance<double, BLOCK, EPT>(input.data(), out_orig.data(), n, window);
    rolling_variance<double, BLOCK, EPT>(scaled_input.data(), out_scaled.data(), n, window);

    for (int i = window - 1; i < n; i++) {
        EXPECT_NEAR(out_scaled[i], out_orig[i] * scale * scale, 1e-4)
            << "Var(cX) should equal c^2 * Var(X) at index " << i;
    }
}

TEST_F(RollingVarianceDouble, ShiftInvariance) {
    std::mt19937 rng(456);
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    const int n = 500;
    const int window = 15;
    const double shift = 10000.0;

    std::vector<double> input(n), shifted_input(n);
    for (int i = 0; i < n; i++) {
        input[i] = dist(rng);
        shifted_input[i] = input[i] + shift;
    }

    std::vector<double> out_orig(n), out_shifted(n);
    rolling_variance<double, BLOCK, EPT>(input.data(), out_orig.data(), n, window);
    rolling_variance<double, BLOCK, EPT>(shifted_input.data(), out_shifted.data(), n, window);

    for (int i = window - 1; i < n; i++) {
        EXPECT_NEAR(out_shifted[i], out_orig[i], 1e-3)
            << "Var(X+c) should equal Var(X) at index " << i;
    }
}

TEST_F(RollingVarianceFloat, HandVerifiedSmallArray) {
    float input[] = {2, 4, 4, 4, 5, 5, 7, 9};
    float output[8];
    rolling_variance<float, BLOCK, EPT>(input, output, 8, 4);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));
    EXPECT_TRUE(std::isnan(output[2]));

    float mean0 = (2 + 4 + 4 + 4) / 4.0f;
    float var0 = ((2 - mean0) * (2 - mean0) + (4 - mean0) * (4 - mean0)
                + (4 - mean0) * (4 - mean0) + (4 - mean0) * (4 - mean0)) / 4.0f;
    EXPECT_NEAR(output[3], var0, 1e-4f);
}

TEST_F(RollingVarianceFloat, CpuReferenceRandom) {
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    const int n = 4096;
    const int window = 20;
    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, window, 0.5f);
}

TEST_F(RollingVarianceFloat, ConstantInputZeroVariance) {
    std::vector<float> input(300, 7.0f);
    std::vector<float> output(300);
    rolling_variance<float, BLOCK, EPT>(input.data(), output.data(), 300, 10);

    for (int i = 9; i < 300; i++) {
        EXPECT_NEAR(output[i], 0.0f, 1e-4f) << "index " << i;
    }
}

TEST_F(RollingVarianceFloat, LargeWindowPrecisionDrift) {
    std::mt19937 rng(55);
    std::uniform_real_distribution<float> dist(100.0f, 10000.0f);

    const int n = 10000;
    const int window = 500;
    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, window, 100.0f);
}

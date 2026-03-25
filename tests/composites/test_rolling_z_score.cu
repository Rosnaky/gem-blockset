#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include "rolling_z_score.cu"

template <typename T>
void cpu_rolling_z_score(const T* input, T* output, int n, int window) {
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
        double mean = sum / window;
        double var = sq_sum / window - mean * mean;
        if (var < 0.0) var = 0.0;
        double std = std::sqrt(var);
        if (std == 0.0) {
            output[i] = nan("");
        } else {
            output[i] = static_cast<T>((static_cast<double>(input[i]) - mean) / std);
        }
    }
}

class ZScoreDouble : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<double>& input, int window, double tol = 1e-6) {
        int n = static_cast<int>(input.size());
        std::vector<double> gpu_out(n), cpu_out(n);
        cpu_rolling_z_score(input.data(), cpu_out.data(), n, window);
        rolling_z_score<double, BLOCK, EPT>(input.data(), gpu_out.data(), n, window);
        for (int i = 0; i < n; i++) {
            if (i < window - 1) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else if (std::isnan(cpu_out[i])) {
                EXPECT_TRUE(std::isnan(gpu_out[i]) || std::isinf(gpu_out[i]))
                    << "expected NaN or inf at index " << i << " (zero std)";
            } else {
                EXPECT_NEAR(gpu_out[i], cpu_out[i], tol) << "mismatch at index " << i;
            }
        }
    }
};

class ZScoreFloat : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<float>& input, int window, float tol = 0.1f) {
        int n = static_cast<int>(input.size());
        std::vector<float> gpu_out(n), cpu_out(n);
        cpu_rolling_z_score(input.data(), cpu_out.data(), n, window);
        rolling_z_score<float, BLOCK, EPT>(input.data(), gpu_out.data(), n, window);
        for (int i = 0; i < n; i++) {
            if (i < window - 1) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else if (std::isnan(cpu_out[i])) {
                EXPECT_TRUE(std::isnan(gpu_out[i]) || std::isinf(gpu_out[i]))
                    << "expected NaN or inf at index " << i << " (zero std)";
            } else {
                EXPECT_NEAR(gpu_out[i], cpu_out[i], tol) << "mismatch at index " << i;
            }
        }
    }
};

TEST_F(ZScoreDouble, HandVerified) {
    double input[] = {10, 20, 30, 40, 50};
    double output[5];
    rolling_z_score<double, BLOCK, EPT>(input, output, 5, 3);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));

    double mean2 = (10 + 20 + 30) / 3.0;
    double var2 = ((10 - mean2) * (10 - mean2) + (20 - mean2) * (20 - mean2) + (30 - mean2) * (30 - mean2)) / 3.0;
    double std2 = std::sqrt(var2);
    EXPECT_NEAR(output[2], (30 - mean2) / std2, 1e-6);

    double mean3 = (20 + 30 + 40) / 3.0;
    double var3 = ((20 - mean3) * (20 - mean3) + (30 - mean3) * (30 - mean3) + (40 - mean3) * (40 - mean3)) / 3.0;
    double std3 = std::sqrt(var3);
    EXPECT_NEAR(output[3], (40 - mean3) / std3, 1e-6);

    double mean4 = (30 + 40 + 50) / 3.0;
    double var4 = ((30 - mean4) * (30 - mean4) + (40 - mean4) * (40 - mean4) + (50 - mean4) * (50 - mean4)) / 3.0;
    double std4 = std::sqrt(var4);
    EXPECT_NEAR(output[4], (50 - mean4) / std4, 1e-6);
}

TEST_F(ZScoreDouble, NanCountMatchesWindowMinusOne) {
    std::vector<double> input(150);
    std::iota(input.begin(), input.end(), 1.0);

    for (int window : {2, 5, 20, 50}) {
        std::vector<double> output(150);
        rolling_z_score<double, BLOCK, EPT>(input.data(), output.data(), 150, window);

        int leading_nan = 0;
        for (int i = 0; i < window - 1; i++) {
            if (std::isnan(output[i])) leading_nan++;
        }
        EXPECT_EQ(leading_nan, window - 1) << "window=" << window;
    }
}

TEST_F(ZScoreDouble, ConstantInputProducesNanOrZero) {
    std::vector<double> input(500, 42.0);
    std::vector<double> output(500);
    rolling_z_score<double, BLOCK, EPT>(input.data(), output.data(), 500, 20);

    for (int i = 19; i < 500; i++) {
        EXPECT_TRUE(std::isnan(output[i]) || std::isinf(output[i]) || std::abs(output[i]) < 1e-9)
            << "constant input should give NaN/inf/0 at index " << i << ", got " << output[i];
    }
}

TEST_F(ZScoreDouble, LastElementInWindowIsPositive) {
    const int n = 200;
    const int window = 20;
    std::vector<double> input(n);
    for (int i = 0; i < n; i++) input[i] = static_cast<double>(i);

    std::vector<double> output(n);
    rolling_z_score<double, BLOCK, EPT>(input.data(), output.data(), n, window);

    for (int i = window - 1; i < n; i++) {
        EXPECT_GT(output[i], 0.0) << "newest element in monotonic series should be above mean at index " << i;
    }
}

TEST_F(ZScoreDouble, ZScoreOfStandardNormalNearOriginal) {
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    const int n = 10000;
    const int window = 1000;
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    std::vector<double> output(n);
    rolling_z_score<double, BLOCK, EPT>(input.data(), output.data(), n, window);

    double sum = 0, sq_sum = 0;
    int count = 0;
    for (int i = window - 1; i < n; i++) {
        if (!std::isnan(output[i]) && !std::isinf(output[i])) {
            sum += output[i];
            sq_sum += output[i] * output[i];
            count++;
        }
    }
    double mean = sum / count;
    double var = sq_sum / count - mean * mean;

    EXPECT_NEAR(mean, 0.0, 0.1) << "z-score mean should be near 0";
    EXPECT_NEAR(var, 1.0, 0.3) << "z-score variance should be near 1";
}

TEST_F(ZScoreDouble, ShiftInvariance) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    const int n = 500;
    const int window = 15;
    const double shift = 50000.0;

    std::vector<double> input(n), shifted(n);
    for (int i = 0; i < n; i++) {
        input[i] = dist(rng);
        shifted[i] = input[i] + shift;
    }

    std::vector<double> out_orig(n), out_shifted(n);
    rolling_z_score<double, BLOCK, EPT>(input.data(), out_orig.data(), n, window);
    rolling_z_score<double, BLOCK, EPT>(shifted.data(), out_shifted.data(), n, window);

    for (int i = window - 1; i < n; i++) {
        if (std::isnan(out_orig[i])) continue;
        EXPECT_NEAR(out_shifted[i], out_orig[i], 1e-3)
            << "z-score should be shift-invariant at index " << i;
    }
}

TEST_F(ZScoreDouble, ScaleInvariance) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    const int n = 500;
    const int window = 15;
    const double scale = 100.0;

    std::vector<double> input(n), scaled(n);
    for (int i = 0; i < n; i++) {
        input[i] = dist(rng);
        scaled[i] = input[i] * scale;
    }

    std::vector<double> out_orig(n), out_scaled(n);
    rolling_z_score<double, BLOCK, EPT>(input.data(), out_orig.data(), n, window);
    rolling_z_score<double, BLOCK, EPT>(scaled.data(), out_scaled.data(), n, window);

    for (int i = window - 1; i < n; i++) {
        if (std::isnan(out_orig[i])) continue;
        EXPECT_NEAR(out_scaled[i], out_orig[i], 1e-3)
            << "z-score should be scale-invariant at index " << i;
    }
}

TEST_F(ZScoreDouble, CpuReferenceRandom) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(50.0, 200.0);

    const int n = 4096;
    const int window = 20;
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, window);
}

TEST_F(ZScoreDouble, TileBoundary) {
    const int tile_size = BLOCK * EPT;
    const int n = tile_size * 3 + 17;
    const int window = 30;

    std::mt19937 rng(77);
    std::uniform_real_distribution<double> dist(1.0, 500.0);
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    std::vector<double> gpu_out(n), cpu_out(n);
    cpu_rolling_z_score(input.data(), cpu_out.data(), n, window);
    rolling_z_score<double, BLOCK, EPT>(input.data(), gpu_out.data(), n, window);

    for (int boundary : {tile_size, tile_size * 2}) {
        for (int offset = -10; offset <= 10; offset++) {
            int i = boundary + offset;
            if (i < window - 1 || i >= n) continue;
            if (std::isnan(cpu_out[i])) continue;
            EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-6)
                << "tile boundary divergence at index " << i;
        }
    }
}

TEST_F(ZScoreDouble, NonDivisibleArrayLength) {
    std::mt19937 rng(256);
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    for (int n : {10, 33, 257, 1023, 4097}) {
        const int window = 5;
        std::vector<double> input(n);
        for (auto& v : input) v = dist(rng);
        run_and_compare(input, window);
    }
}

TEST_F(ZScoreDouble, WindowEqualsTwo) {
    double input[] = {10, 20, 30};
    double output[3];
    rolling_z_score<double, BLOCK, EPT>(input, output, 3, 2);

    EXPECT_TRUE(std::isnan(output[0]));

    double mean1 = (10 + 20) / 2.0;
    double std1 = std::sqrt(((10 - mean1) * (10 - mean1) + (20 - mean1) * (20 - mean1)) / 2.0);
    EXPECT_NEAR(output[1], (20 - mean1) / std1, 1e-9);
}

TEST_F(ZScoreFloat, HandVerified) {
    float input[] = {10, 20, 30, 40, 50};
    float output[5];
    rolling_z_score<float, BLOCK, EPT>(input, output, 5, 3);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));

    double mean2 = (10 + 20 + 30) / 3.0;
    double var2 = ((10 - mean2) * (10 - mean2) + (20 - mean2) * (20 - mean2) + (30 - mean2) * (30 - mean2)) / 3.0;
    double std2 = std::sqrt(var2);
    float expected = static_cast<float>((30 - mean2) / std2);
    EXPECT_NEAR(output[2], expected, 0.01f);
}

TEST_F(ZScoreFloat, CpuReferenceRandom) {
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(10.0f, 200.0f);

    const int n = 4096;
    const int window = 20;
    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, window);
}

TEST_F(ZScoreFloat, LargeWindowPrecisionDrift) {
    std::mt19937 rng(55);
    std::uniform_real_distribution<float> dist(100.0f, 10000.0f);

    const int n = 10000;
    const int window = 500;
    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, window, 1.0f);
}

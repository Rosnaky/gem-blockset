#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include "simple_moving_average.cu"

template <typename T>
void cpu_rolling_mean(const T* input, T* output, int n, int window) {
    for (int i = 0; i < n; i++) {
        if (i < window - 1) {
            output[i] = nan("");
            continue;
        }
        T sum = T(0);
        for (int j = 0; j < window; j++) {
            sum += input[i - j];
        }
        output[i] = sum / static_cast<T>(window);
    }
}

class SimpleMovingAverageDouble : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<double>& input, int window, double tol = 1e-9) {
        int n = static_cast<int>(input.size());
        std::vector<double> gpu_out(n), cpu_out(n);
        cpu_rolling_mean(input.data(), cpu_out.data(), n, window);
        simple_moving_average<double, BLOCK, EPT>(input.data(), gpu_out.data(), n, window);
        for (int i = 0; i < n; i++) {
            if (i < window - 1) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else {
                EXPECT_NEAR(gpu_out[i], cpu_out[i], tol) << "mismatch at index " << i;
            }
        }
    }
};

class SimpleMovingAverageFloat : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<float>& input, int window, float tol = 1e-3f) {
        int n = static_cast<int>(input.size());
        std::vector<float> gpu_out(n), cpu_out(n);
        cpu_rolling_mean(input.data(), cpu_out.data(), n, window);
        simple_moving_average<float, BLOCK, EPT>(input.data(), gpu_out.data(), n, window);
        for (int i = 0; i < n; i++) {
            if (i < window - 1) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else {
                EXPECT_NEAR(gpu_out[i], cpu_out[i], tol) << "mismatch at index " << i;
            }
        }
    }
};

TEST_F(SimpleMovingAverageDouble, HandVerifiedSmallArray) {
    double input[] = {10, 20, 30, 40, 50, 60, 70, 80};
    double output[8];
    simple_moving_average<double, BLOCK, EPT>(input, output, 8, 3);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));
    EXPECT_NEAR(output[2], 20.0, 1e-9);
    EXPECT_NEAR(output[3], 30.0, 1e-9);
    EXPECT_NEAR(output[4], 40.0, 1e-9);
    EXPECT_NEAR(output[5], 50.0, 1e-9);
    EXPECT_NEAR(output[6], 60.0, 1e-9);
    EXPECT_NEAR(output[7], 70.0, 1e-9);
}

TEST_F(SimpleMovingAverageDouble, NanCountMatchesWindowMinusOne) {
    std::vector<double> input(100);
    std::iota(input.begin(), input.end(), 1.0);

    for (int window : {2, 5, 17, 50, 99}) {
        std::vector<double> output(100);
        simple_moving_average<double, BLOCK, EPT>(input.data(), output.data(), 100, window);

        int nan_count = 0;
        for (int i = 0; i < 100; i++) {
            if (std::isnan(output[i])) nan_count++;
        }
        EXPECT_EQ(nan_count, window - 1) << "window=" << window;
    }
}

TEST_F(SimpleMovingAverageDouble, WindowEqualsOne) {
    double input[] = {3.14, 2.71, 1.41, 1.73, 0.577};
    double output[5];
    simple_moving_average<double, BLOCK, EPT>(input, output, 5, 1);

    for (int i = 0; i < 5; i++) {
        EXPECT_NEAR(output[i], input[i], 1e-12) << "index " << i;
    }
}

TEST_F(SimpleMovingAverageDouble, WindowEqualsN) {
    double input[] = {10, 20, 30, 40, 50};
    double output[5];
    simple_moving_average<double, BLOCK, EPT>(input, output, 5, 5);

    for (int i = 0; i < 4; i++) {
        EXPECT_TRUE(std::isnan(output[i])) << "index " << i;
    }
    EXPECT_NEAR(output[4], 30.0, 1e-9);
}

TEST_F(SimpleMovingAverageDouble, SingleElement) {
    double input[] = {42.0};
    double output[1];
    simple_moving_average<double, BLOCK, EPT>(input, output, 1, 1);

    EXPECT_NEAR(output[0], 42.0, 1e-12);
}

TEST_F(SimpleMovingAverageDouble, ConstantInput) {
    std::vector<double> input(500, 7.77);
    std::vector<double> output(500);
    simple_moving_average<double, BLOCK, EPT>(input.data(), output.data(), 500, 25);

    for (int i = 24; i < 500; i++) {
        EXPECT_NEAR(output[i], 7.77, 1e-9) << "index " << i;
    }
}

TEST_F(SimpleMovingAverageDouble, CpuReferenceRandom) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(50.0, 200.0);

    const int n = 4096;
    const int window = 20;
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, window);
}

TEST_F(SimpleMovingAverageDouble, TileBoundaryExact) {
    const int tile_size = BLOCK * EPT;
    const int n = tile_size * 3;
    const int window = 30;

    std::mt19937 rng(77);
    std::uniform_real_distribution<double> dist(1.0, 500.0);
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    std::vector<double> gpu_out(n), cpu_out(n);
    cpu_rolling_mean(input.data(), cpu_out.data(), n, window);
    simple_moving_average<double, BLOCK, EPT>(input.data(), gpu_out.data(), n, window);

    for (int boundary : {tile_size, tile_size * 2}) {
        for (int offset = -10; offset <= 10; offset++) {
            int i = boundary + offset;
            if (i < window - 1 || i >= n) continue;
            EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-9)
                << "tile boundary divergence at index " << i;
        }
    }
}

TEST_F(SimpleMovingAverageDouble, NonDivisibleArrayLength) {
    std::mt19937 rng(256);
    std::uniform_real_distribution<double> dist(0.0, 100.0);

    for (int n : {7, 33, 257, 1023, 2049}) {
        const int window = 5;
        std::vector<double> input(n);
        for (auto& v : input) v = dist(rng);
        run_and_compare(input, window);
    }
}

TEST_F(SimpleMovingAverageDouble, LargeWindow) {
    const int n = 5000;
    const int window = 500;
    std::mt19937 rng(314);
    std::uniform_real_distribution<double> dist(10.0, 1000.0);

    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, window, 1e-6);
}

TEST_F(SimpleMovingAverageDouble, MonotonicallyIncreasingInput) {
    const int n = 1000;
    const int window = 10;
    std::vector<double> input(n);
    for (int i = 0; i < n; i++) input[i] = static_cast<double>(i);

    std::vector<double> output(n);
    simple_moving_average<double, BLOCK, EPT>(input.data(), output.data(), n, window);

    for (int i = window - 1; i < n; i++) {
        double expected = (i - window + 1 + i) / 2.0;
        EXPECT_NEAR(output[i], expected, 1e-9) << "index " << i;
    }
}

TEST_F(SimpleMovingAverageFloat, HandVerifiedSmallArray) {
    float input[] = {10, 20, 30, 40, 50, 60, 70, 80};
    float output[8];
    simple_moving_average<float, BLOCK, EPT>(input, output, 8, 3);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));
    EXPECT_NEAR(output[2], 20.0f, 1e-5f);
    EXPECT_NEAR(output[3], 30.0f, 1e-5f);
    EXPECT_NEAR(output[4], 40.0f, 1e-5f);
    EXPECT_NEAR(output[5], 50.0f, 1e-5f);
    EXPECT_NEAR(output[6], 60.0f, 1e-5f);
    EXPECT_NEAR(output[7], 70.0f, 1e-5f);
}

TEST_F(SimpleMovingAverageFloat, CpuReferenceRandom) {
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(0.0f, 1000.0f);

    const int n = 4096;
    const int window = 50;
    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, window, 0.1f);
}

TEST_F(SimpleMovingAverageFloat, LargeWindowPrecisionDrift) {
    std::mt19937 rng(55);
    std::uniform_real_distribution<float> dist(100.0f, 10000.0f);

    const int n = 10000;
    const int window = 500;
    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, window, 1.0f);
}

TEST_F(SimpleMovingAverageFloat, WindowEqualsOne) {
    float input[] = {1.5f, 2.5f, 3.5f};
    float output[3];
    simple_moving_average<float, BLOCK, EPT>(input, output, 3, 1);

    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(output[i], input[i], 1e-6f);
    }
}

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include "difference.cu"

template <typename T>
void cpu_difference(const T* input, T* output, int n) {
    for (int i = 0; i < n; i++) {
        if (i == 0) {
            output[i] = nan("");
            continue;
        }
        output[i] = input[i] - input[i - 1];
    }
}

class DifferenceDouble : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<double>& input, double tol = 1e-12) {
        int n = static_cast<int>(input.size());
        std::vector<double> gpu_out(n), cpu_out(n);
        cpu_difference(input.data(), cpu_out.data(), n);
        difference<double, BLOCK, EPT>(input.data(), gpu_out.data(), n);
        EXPECT_TRUE(std::isnan(gpu_out[0])) << "expected NaN at index 0";
        for (int i = 1; i < n; i++) {
            EXPECT_NEAR(gpu_out[i], cpu_out[i], tol) << "mismatch at index " << i;
        }
    }
};

class DifferenceFloat : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<float>& input, float tol = 1e-5f) {
        int n = static_cast<int>(input.size());
        std::vector<float> gpu_out(n), cpu_out(n);
        cpu_difference(input.data(), cpu_out.data(), n);
        difference<float, BLOCK, EPT>(input.data(), gpu_out.data(), n);
        EXPECT_TRUE(std::isnan(gpu_out[0])) << "expected NaN at index 0";
        for (int i = 1; i < n; i++) {
            EXPECT_NEAR(gpu_out[i], cpu_out[i], tol) << "mismatch at index " << i;
        }
    }
};

TEST_F(DifferenceDouble, HandVerifiedSmallArray) {
    double input[] = {10, 30, 25, 40, 38};
    double output[5];
    difference<double, BLOCK, EPT>(input, output, 5);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_NEAR(output[1], 20.0, 1e-12);
    EXPECT_NEAR(output[2], -5.0, 1e-12);
    EXPECT_NEAR(output[3], 15.0, 1e-12);
    EXPECT_NEAR(output[4], -2.0, 1e-12);
}

TEST_F(DifferenceDouble, SingleNanAtIndexZero) {
    std::vector<double> input(200);
    std::iota(input.begin(), input.end(), 1.0);
    std::vector<double> output(200);
    difference<double, BLOCK, EPT>(input.data(), output.data(), 200);

    int nan_count = 0;
    for (int i = 0; i < 200; i++) {
        if (std::isnan(output[i])) nan_count++;
    }
    EXPECT_EQ(nan_count, 1);
    EXPECT_TRUE(std::isnan(output[0]));
}

TEST_F(DifferenceDouble, SingleElement) {
    double input[] = {42.0};
    double output[1];
    difference<double, BLOCK, EPT>(input, output, 1);

    EXPECT_TRUE(std::isnan(output[0]));
}

TEST_F(DifferenceDouble, TwoElements) {
    double input[] = {100.0, 137.5};
    double output[2];
    difference<double, BLOCK, EPT>(input, output, 2);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_NEAR(output[1], 37.5, 1e-12);
}

TEST_F(DifferenceDouble, ConstantInput) {
    std::vector<double> input(500, 3.14);
    std::vector<double> output(500);
    difference<double, BLOCK, EPT>(input.data(), output.data(), 500);

    for (int i = 1; i < 500; i++) {
        EXPECT_NEAR(output[i], 0.0, 1e-12) << "index " << i;
    }
}

TEST_F(DifferenceDouble, MonotonicallyIncreasing) {
    const int n = 1000;
    std::vector<double> input(n);
    for (int i = 0; i < n; i++) input[i] = i * 2.5;

    std::vector<double> output(n);
    difference<double, BLOCK, EPT>(input.data(), output.data(), n);

    for (int i = 1; i < n; i++) {
        EXPECT_NEAR(output[i], 2.5, 1e-10) << "index " << i;
    }
}

TEST_F(DifferenceDouble, NegativeValues) {
    double input[] = {-10.0, -30.0, -5.0, -50.0};
    double output[4];
    difference<double, BLOCK, EPT>(input, output, 4);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_NEAR(output[1], -20.0, 1e-12);
    EXPECT_NEAR(output[2], 25.0, 1e-12);
    EXPECT_NEAR(output[3], -45.0, 1e-12);
}

TEST_F(DifferenceDouble, CpuReferenceRandom) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-500.0, 500.0);

    const int n = 4096;
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input);
}

TEST_F(DifferenceDouble, TileBoundary) {
    const int tile_size = BLOCK * EPT;
    const int n = tile_size * 3 + 17;

    std::mt19937 rng(88);
    std::uniform_real_distribution<double> dist(1.0, 1000.0);
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    std::vector<double> gpu_out(n), cpu_out(n);
    cpu_difference(input.data(), cpu_out.data(), n);
    difference<double, BLOCK, EPT>(input.data(), gpu_out.data(), n);

    for (int boundary : {tile_size, tile_size * 2}) {
        for (int offset = -5; offset <= 5; offset++) {
            int i = boundary + offset;
            if (i < 1 || i >= n) continue;
            EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-12)
                << "tile boundary divergence at index " << i;
        }
    }
}

TEST_F(DifferenceDouble, NonDivisibleArrayLength) {
    std::mt19937 rng(256);
    std::uniform_real_distribution<double> dist(0.0, 100.0);

    for (int n : {3, 17, 255, 1025, 4099}) {
        std::vector<double> input(n);
        for (auto& v : input) v = dist(rng);
        run_and_compare(input);
    }
}

TEST_F(DifferenceDouble, AlternatingValues) {
    const int n = 100;
    std::vector<double> input(n);
    for (int i = 0; i < n; i++) input[i] = (i % 2 == 0) ? 10.0 : 20.0;

    std::vector<double> output(n);
    difference<double, BLOCK, EPT>(input.data(), output.data(), n);

    for (int i = 1; i < n; i++) {
        double expected = (i % 2 == 0) ? -10.0 : 10.0;
        EXPECT_NEAR(output[i], expected, 1e-12) << "index " << i;
    }
}

TEST_F(DifferenceFloat, HandVerifiedSmallArray) {
    float input[] = {10, 30, 25, 40, 38};
    float output[5];
    difference<float, BLOCK, EPT>(input, output, 5);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_NEAR(output[1], 20.0f, 1e-5f);
    EXPECT_NEAR(output[2], -5.0f, 1e-5f);
    EXPECT_NEAR(output[3], 15.0f, 1e-5f);
    EXPECT_NEAR(output[4], -2.0f, 1e-5f);
}

TEST_F(DifferenceFloat, CpuReferenceRandom) {
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    const int n = 4096;
    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input);
}

TEST_F(DifferenceFloat, LargeArray) {
    std::mt19937 rng(33);
    std::uniform_real_distribution<float> dist(0.0f, 500.0f);

    const int n = 50000;
    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, 1e-3f);
}

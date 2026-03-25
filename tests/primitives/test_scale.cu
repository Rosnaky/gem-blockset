#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include "scale.cu"

class ScaleDouble : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<double>& input, double scalar, double tol = 1e-12) {
        int n = static_cast<int>(input.size());
        std::vector<double> gpu_out(n);
        scale<double, BLOCK, EPT>(input.data(), gpu_out.data(), n, scalar);
        for (int i = 0; i < n; i++) {
            if (std::isnan(input[i])) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else {
                double expected = input[i] * scalar;
                EXPECT_NEAR(gpu_out[i], expected, tol) << "mismatch at index " << i;
            }
        }
    }
};

class ScaleFloat : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<float>& input, double scalar, float tol = 1e-5f) {
        int n = static_cast<int>(input.size());
        std::vector<float> gpu_out(n);
        scale<float, BLOCK, EPT>(input.data(), gpu_out.data(), n, scalar);
        for (int i = 0; i < n; i++) {
            if (std::isnan(input[i])) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else {
                float expected = static_cast<float>(static_cast<double>(input[i]) * scalar);
                EXPECT_NEAR(gpu_out[i], expected, tol) << "mismatch at index " << i;
            }
        }
    }
};

TEST_F(ScaleDouble, HandVerifiedValues) {
    double input[] = {1, 2, 3, 4, 5};
    double output[5];
    scale<double, BLOCK, EPT>(input, output, 5, 10.0);

    EXPECT_NEAR(output[0], 10.0, 1e-12);
    EXPECT_NEAR(output[1], 20.0, 1e-12);
    EXPECT_NEAR(output[2], 30.0, 1e-12);
    EXPECT_NEAR(output[3], 40.0, 1e-12);
    EXPECT_NEAR(output[4], 50.0, 1e-12);
}

TEST_F(ScaleDouble, MultiplyByOne) {
    double input[] = {3.14, 2.71, 1.41, 42.0};
    double output[4];
    scale<double, BLOCK, EPT>(input, output, 4, 1.0);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(output[i], input[i], 1e-12) << "index " << i;
    }
}

TEST_F(ScaleDouble, MultiplyByZero) {
    double input[] = {3.14, 2.71, 1.41};
    double output[3];
    scale<double, BLOCK, EPT>(input, output, 3, 0.0);

    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(output[i], 0.0, 1e-12) << "index " << i;
    }
}

TEST_F(ScaleDouble, MultiplyByNegative) {
    double input[] = {10.0, -20.0, 30.0};
    double output[3];
    scale<double, BLOCK, EPT>(input, output, 3, -3.0);

    EXPECT_NEAR(output[0], -30.0, 1e-12);
    EXPECT_NEAR(output[1], 60.0, 1e-12);
    EXPECT_NEAR(output[2], -90.0, 1e-12);
}

TEST_F(ScaleDouble, MultiplyByFraction) {
    double input[] = {10.0, 20.0, 30.0};
    double output[3];
    scale<double, BLOCK, EPT>(input, output, 3, 0.5);

    EXPECT_NEAR(output[0], 5.0, 1e-12);
    EXPECT_NEAR(output[1], 10.0, 1e-12);
    EXPECT_NEAR(output[2], 15.0, 1e-12);
}

TEST_F(ScaleDouble, SingleElement) {
    double input[] = {7.0};
    double output[1];
    scale<double, BLOCK, EPT>(input, output, 1, 6.0);

    EXPECT_NEAR(output[0], 42.0, 1e-12);
}

TEST_F(ScaleDouble, NanPassthrough) {
    double input[] = {nan(""), 10.0, nan(""), 20.0};
    double output[4];
    scale<double, BLOCK, EPT>(input, output, 4, 5.0);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_NEAR(output[1], 50.0, 1e-12);
    EXPECT_TRUE(std::isnan(output[2]));
    EXPECT_NEAR(output[3], 100.0, 1e-12);
}

TEST_F(ScaleDouble, LargeScalar) {
    double input[] = {1.0, 2.0, 3.0};
    double output[3];
    scale<double, BLOCK, EPT>(input, output, 3, 1e10);

    EXPECT_NEAR(output[0], 1e10, 1.0);
    EXPECT_NEAR(output[1], 2e10, 1.0);
    EXPECT_NEAR(output[2], 3e10, 1.0);
}

TEST_F(ScaleDouble, SmallScalar) {
    double input[] = {1e10, 2e10, 3e10};
    double output[3];
    scale<double, BLOCK, EPT>(input, output, 3, 1e-10);

    EXPECT_NEAR(output[0], 1.0, 1e-12);
    EXPECT_NEAR(output[1], 2.0, 1e-12);
    EXPECT_NEAR(output[2], 3.0, 1e-12);
}

TEST_F(ScaleDouble, ZeroInput) {
    std::vector<double> input(100, 0.0);
    std::vector<double> output(100);
    scale<double, BLOCK, EPT>(input.data(), output.data(), 100, 42.0);

    for (int i = 0; i < 100; i++) {
        EXPECT_NEAR(output[i], 0.0, 1e-12) << "index " << i;
    }
}

TEST_F(ScaleDouble, CpuReferenceRandom) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-500.0, 500.0);

    const int n = 4096;
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, 7.77);
}

TEST_F(ScaleDouble, NonDivisibleArrayLength) {
    std::mt19937 rng(256);
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    for (int n : {1, 3, 7, 255, 257, 1023, 4097}) {
        std::vector<double> input(n);
        for (auto& v : input) v = dist(rng);
        run_and_compare(input, 3.14);
    }
}

TEST_F(ScaleDouble, LargeArray) {
    std::mt19937 rng(314);
    std::uniform_real_distribution<double> dist(0.001, 1e6);

    const int n = 100000;
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, 99.9, 1e-6);
}

TEST_F(ScaleFloat, HandVerifiedValues) {
    float input[] = {1, 2, 3};
    float output[3];
    scale<float, BLOCK, EPT>(input, output, 3, 4.0);

    EXPECT_NEAR(output[0], 4.0f, 1e-5f);
    EXPECT_NEAR(output[1], 8.0f, 1e-5f);
    EXPECT_NEAR(output[2], 12.0f, 1e-5f);
}

TEST_F(ScaleFloat, CpuReferenceRandom) {
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(1.0f, 500.0f);

    const int n = 4096;
    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, 13.0);
}

TEST_F(ScaleFloat, LargeArray) {
    std::mt19937 rng(33);
    std::uniform_real_distribution<float> dist(0.01f, 1e4f);

    const int n = 50000;
    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, 7.0, 1e-1f);
}

TEST_F(ScaleFloat, NanPassthrough) {
    float input[] = {nanf(""), 10.0f, nanf("")};
    float output[3];
    scale<float, BLOCK, EPT>(input, output, 3, 2.0);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_NEAR(output[1], 20.0f, 1e-5f);
    EXPECT_TRUE(std::isnan(output[2]));
}

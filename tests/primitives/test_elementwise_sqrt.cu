#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include "elementwise_sqrt.cu"

class ElementwiseSqrtDouble : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;

    void run_and_compare(const std::vector<double>& input, double tol = 1e-12) {
        int n = static_cast<int>(input.size());
        std::vector<double> gpu_out(n);
        elementwise_sqrt<double, BLOCK>(input.data(), gpu_out.data(), n);
        for (int i = 0; i < n; i++) {
            if (std::isnan(input[i])) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else {
                EXPECT_NEAR(gpu_out[i], std::sqrt(input[i]), tol) << "mismatch at index " << i;
            }
        }
    }
};

class ElementwiseSqrtFloat : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;

    void run_and_compare(const std::vector<float>& input, float tol = 1e-5f) {
        int n = static_cast<int>(input.size());
        std::vector<float> gpu_out(n);
        elementwise_sqrt<float, BLOCK>(input.data(), gpu_out.data(), n);
        for (int i = 0; i < n; i++) {
            if (std::isnan(input[i])) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else {
                EXPECT_NEAR(gpu_out[i], std::sqrt(input[i]), tol) << "mismatch at index " << i;
            }
        }
    }
};

TEST_F(ElementwiseSqrtDouble, HandVerifiedValues) {
    double input[] = {0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 100.0};
    double output[7];
    elementwise_sqrt<double, BLOCK>(input, output, 7);

    EXPECT_NEAR(output[0], 0.0, 1e-12);
    EXPECT_NEAR(output[1], 1.0, 1e-12);
    EXPECT_NEAR(output[2], 2.0, 1e-12);
    EXPECT_NEAR(output[3], 3.0, 1e-12);
    EXPECT_NEAR(output[4], 4.0, 1e-12);
    EXPECT_NEAR(output[5], 5.0, 1e-12);
    EXPECT_NEAR(output[6], 10.0, 1e-12);
}

TEST_F(ElementwiseSqrtDouble, NanPassthrough) {
    double input[] = {nan(""), 4.0, nan(""), 16.0, nan("")};
    double output[5];
    elementwise_sqrt<double, BLOCK>(input, output, 5);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_NEAR(output[1], 2.0, 1e-12);
    EXPECT_TRUE(std::isnan(output[2]));
    EXPECT_NEAR(output[3], 4.0, 1e-12);
    EXPECT_TRUE(std::isnan(output[4]));
}

TEST_F(ElementwiseSqrtDouble, SingleElement) {
    double input[] = {49.0};
    double output[1];
    elementwise_sqrt<double, BLOCK>(input, output, 1);

    EXPECT_NEAR(output[0], 7.0, 1e-12);
}

TEST_F(ElementwiseSqrtDouble, ZeroArray) {
    std::vector<double> input(100, 0.0);
    std::vector<double> output(100);
    elementwise_sqrt<double, BLOCK>(input.data(), output.data(), 100);

    for (int i = 0; i < 100; i++) {
        EXPECT_NEAR(output[i], 0.0, 1e-12) << "index " << i;
    }
}

TEST_F(ElementwiseSqrtDouble, SqrtOfSquaresRecoversOriginal) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1000.0);

    const int n = 2000;
    std::vector<double> original(n), squared(n), result(n);
    for (int i = 0; i < n; i++) {
        original[i] = dist(rng);
        squared[i] = original[i] * original[i];
    }

    elementwise_sqrt<double, BLOCK>(squared.data(), result.data(), n);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(result[i], original[i], 1e-6) << "index " << i;
    }
}

TEST_F(ElementwiseSqrtDouble, SmallValues) {
    double input[] = {1e-10, 1e-14, 1e-20};
    double output[3];
    elementwise_sqrt<double, BLOCK>(input, output, 3);

    EXPECT_NEAR(output[0], 1e-5, 1e-15);
    EXPECT_NEAR(output[1], 1e-7, 1e-17);
    EXPECT_NEAR(output[2], 1e-10, 1e-20);
}

TEST_F(ElementwiseSqrtDouble, LargeValues) {
    double input[] = {1e10, 1e16, 1e20};
    double output[3];
    elementwise_sqrt<double, BLOCK>(input, output, 3);

    EXPECT_NEAR(output[0], 1e5, 1e-5);
    EXPECT_NEAR(output[1], 1e8, 1e-2);
    EXPECT_NEAR(output[2], 1e10, 1.0);
}

TEST_F(ElementwiseSqrtDouble, CpuReferenceRandom) {
    std::mt19937 rng(77);
    std::uniform_real_distribution<double> dist(0.0, 10000.0);

    const int n = 4096;
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input);
}

TEST_F(ElementwiseSqrtDouble, NonDivisibleLength) {
    std::mt19937 rng(256);
    std::uniform_real_distribution<double> dist(0.0, 100.0);

    for (int n : {1, 7, 255, 257, 1023, 4097}) {
        std::vector<double> input(n);
        for (auto& v : input) v = dist(rng);
        run_and_compare(input);
    }
}

TEST_F(ElementwiseSqrtDouble, LargeArray) {
    std::mt19937 rng(314);
    std::uniform_real_distribution<double> dist(0.0, 1e6);

    const int n = 100000;
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, 1e-6);
}

TEST_F(ElementwiseSqrtFloat, HandVerifiedValues) {
    float input[] = {0.0f, 1.0f, 4.0f, 9.0f, 16.0f};
    float output[5];
    elementwise_sqrt<float, BLOCK>(input, output, 5);

    EXPECT_NEAR(output[0], 0.0f, 1e-6f);
    EXPECT_NEAR(output[1], 1.0f, 1e-6f);
    EXPECT_NEAR(output[2], 2.0f, 1e-6f);
    EXPECT_NEAR(output[3], 3.0f, 1e-6f);
    EXPECT_NEAR(output[4], 4.0f, 1e-6f);
}

TEST_F(ElementwiseSqrtFloat, NanPassthrough) {
    float input[] = {nan(""), 4.0f, nan("")};
    float output[3];
    elementwise_sqrt<float, BLOCK>(input, output, 3);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_NEAR(output[1], 2.0f, 1e-6f);
    EXPECT_TRUE(std::isnan(output[2]));
}

TEST_F(ElementwiseSqrtFloat, CpuReferenceRandom) {
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(0.0f, 10000.0f);

    const int n = 4096;
    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input);
}

TEST_F(ElementwiseSqrtFloat, LargeArray) {
    std::mt19937 rng(33);
    std::uniform_real_distribution<float> dist(0.0f, 1e6f);

    const int n = 50000;
    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, 1e-2f);
}

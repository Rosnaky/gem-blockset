#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include "offset.cu"

class OffsetDouble : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<double>& input, double constant, double tol = 1e-12) {
        int n = static_cast<int>(input.size());
        std::vector<double> gpu_out(n);
        offset<double, BLOCK, EPT>(input.data(), gpu_out.data(), n, constant);
        for (int i = 0; i < n; i++) {
            if (std::isnan(input[i])) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else {
                EXPECT_NEAR(gpu_out[i], input[i] + constant, tol) << "mismatch at index " << i;
            }
        }
    }
};

class OffsetFloat : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<float>& input, double constant, float tol = 1e-5f) {
        int n = static_cast<int>(input.size());
        std::vector<float> gpu_out(n);
        offset<float, BLOCK, EPT>(input.data(), gpu_out.data(), n, constant);
        for (int i = 0; i < n; i++) {
            if (std::isnan(input[i])) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else {
                float expected = static_cast<float>(static_cast<double>(input[i]) + constant);
                EXPECT_NEAR(gpu_out[i], expected, tol) << "mismatch at index " << i;
            }
        }
    }
};

TEST_F(OffsetDouble, HandVerified) {
    double input[] = {1, 2, 3, 4, 5};
    double output[5];
    offset<double, BLOCK, EPT>(input, output, 5, 10.0);

    EXPECT_NEAR(output[0], 11.0, 1e-12);
    EXPECT_NEAR(output[1], 12.0, 1e-12);
    EXPECT_NEAR(output[2], 13.0, 1e-12);
    EXPECT_NEAR(output[3], 14.0, 1e-12);
    EXPECT_NEAR(output[4], 15.0, 1e-12);
}

TEST_F(OffsetDouble, AddZero) {
    double input[] = {3.14, 2.71, 1.41};
    double output[3];
    offset<double, BLOCK, EPT>(input, output, 3, 0.0);

    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(output[i], input[i], 1e-12) << "index " << i;
    }
}

TEST_F(OffsetDouble, NegativeOffset) {
    double input[] = {10.0, 20.0, 30.0};
    double output[3];
    offset<double, BLOCK, EPT>(input, output, 3, -15.0);

    EXPECT_NEAR(output[0], -5.0, 1e-12);
    EXPECT_NEAR(output[1], 5.0, 1e-12);
    EXPECT_NEAR(output[2], 15.0, 1e-12);
}

TEST_F(OffsetDouble, SingleElement) {
    double input[] = {42.0};
    double output[1];
    offset<double, BLOCK, EPT>(input, output, 1, -42.0);

    EXPECT_NEAR(output[0], 0.0, 1e-12);
}

TEST_F(OffsetDouble, NanPassthrough) {
    double input[] = {nan(""), 10.0, nan(""), 20.0};
    double output[4];
    offset<double, BLOCK, EPT>(input, output, 4, 5.0);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_NEAR(output[1], 15.0, 1e-12);
    EXPECT_TRUE(std::isnan(output[2]));
    EXPECT_NEAR(output[3], 25.0, 1e-12);
}

TEST_F(OffsetDouble, LargeOffset) {
    double input[] = {1.0, 2.0, 3.0};
    double output[3];
    offset<double, BLOCK, EPT>(input, output, 3, 1e15);

    EXPECT_NEAR(output[0], 1e15 + 1.0, 1.0);
    EXPECT_NEAR(output[1], 1e15 + 2.0, 1.0);
    EXPECT_NEAR(output[2], 1e15 + 3.0, 1.0);
}

TEST_F(OffsetDouble, ZeroInput) {
    std::vector<double> input(100, 0.0);
    std::vector<double> output(100);
    offset<double, BLOCK, EPT>(input.data(), output.data(), 100, 7.77);

    for (int i = 0; i < 100; i++) {
        EXPECT_NEAR(output[i], 7.77, 1e-12) << "index " << i;
    }
}

TEST_F(OffsetDouble, InPlaceOperation) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> copy = data;
    offset<double, BLOCK, EPT>(data.data(), data.data(), 5, 10.0);

    for (int i = 0; i < 5; i++) {
        EXPECT_NEAR(data[i], copy[i] + 10.0, 1e-12) << "index " << i;
    }
}

TEST_F(OffsetDouble, CpuReferenceRandom) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-500.0, 500.0);

    const int n = 4096;
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, 123.456);
}

TEST_F(OffsetDouble, NonDivisibleLength) {
    std::mt19937 rng(256);
    std::uniform_real_distribution<double> dist(0.0, 100.0);

    for (int n : {1, 3, 7, 255, 257, 1023, 4097}) {
        std::vector<double> input(n);
        for (auto& v : input) v = dist(rng);
        run_and_compare(input, -99.9);
    }
}

TEST_F(OffsetDouble, LargeArray) {
    std::mt19937 rng(314);
    std::uniform_real_distribution<double> dist(-1e6, 1e6);

    const int n = 100000;
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, 42.0, 1e-6);
}

TEST_F(OffsetFloat, HandVerified) {
    float input[] = {1, 2, 3};
    float output[3];
    offset<float, BLOCK, EPT>(input, output, 3, 10.0);

    EXPECT_NEAR(output[0], 11.0f, 1e-5f);
    EXPECT_NEAR(output[1], 12.0f, 1e-5f);
    EXPECT_NEAR(output[2], 13.0f, 1e-5f);
}

TEST_F(OffsetFloat, CpuReferenceRandom) {
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-500.0f, 500.0f);

    const int n = 4096;
    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, 42.0);
}

TEST_F(OffsetFloat, NanPassthrough) {
    float input[] = {nanf(""), 10.0f, nanf("")};
    float output[3];
    offset<float, BLOCK, EPT>(input, output, 3, 5.0);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_NEAR(output[1], 15.0f, 1e-5f);
    EXPECT_TRUE(std::isnan(output[2]));
}

TEST_F(OffsetFloat, LargeArray) {
    std::mt19937 rng(33);
    std::uniform_real_distribution<float> dist(-1e4f, 1e4f);

    const int n = 50000;
    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, -777.0, 1e-2f);
}

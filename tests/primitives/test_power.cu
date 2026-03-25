#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include "power.cu"

class PowerDouble : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<double>& input, double exponent, double tol = 1e-9) {
        int n = static_cast<int>(input.size());
        std::vector<double> gpu_out(n);
        power<double, BLOCK, EPT>(input.data(), gpu_out.data(), n, exponent);
        for (int i = 0; i < n; i++) {
            if (std::isnan(input[i])) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else {
                double expected = std::pow(input[i], exponent);
                if (std::isnan(expected)) {
                    EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
                } else {
                    EXPECT_NEAR(gpu_out[i], expected, tol) << "mismatch at index " << i;
                }
            }
        }
    }
};

class PowerFloat : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<float>& input, double exponent, float tol = 1e-4f) {
        int n = static_cast<int>(input.size());
        std::vector<float> gpu_out(n);
        power<float, BLOCK, EPT>(input.data(), gpu_out.data(), n, exponent);
        for (int i = 0; i < n; i++) {
            if (std::isnan(input[i])) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else {
                float expected = static_cast<float>(std::pow(static_cast<double>(input[i]), exponent));
                if (std::isnan(expected)) {
                    EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
                } else {
                    EXPECT_NEAR(gpu_out[i], expected, tol) << "mismatch at index " << i;
                }
            }
        }
    }
};

TEST_F(PowerDouble, Square) {
    double input[] = {1, 2, 3, 4, 5};
    double output[5];
    power<double, BLOCK, EPT>(input, output, 5, 2.0);

    EXPECT_NEAR(output[0], 1.0, 1e-12);
    EXPECT_NEAR(output[1], 4.0, 1e-12);
    EXPECT_NEAR(output[2], 9.0, 1e-12);
    EXPECT_NEAR(output[3], 16.0, 1e-12);
    EXPECT_NEAR(output[4], 25.0, 1e-12);
}

TEST_F(PowerDouble, Cube) {
    double input[] = {2, 3, 4};
    double output[3];
    power<double, BLOCK, EPT>(input, output, 3, 3.0);

    EXPECT_NEAR(output[0], 8.0, 1e-12);
    EXPECT_NEAR(output[1], 27.0, 1e-12);
    EXPECT_NEAR(output[2], 64.0, 1e-12);
}

TEST_F(PowerDouble, SquareRoot) {
    double input[] = {4.0, 9.0, 16.0, 25.0};
    double output[4];
    power<double, BLOCK, EPT>(input, output, 4, 0.5);

    EXPECT_NEAR(output[0], 2.0, 1e-9);
    EXPECT_NEAR(output[1], 3.0, 1e-9);
    EXPECT_NEAR(output[2], 4.0, 1e-9);
    EXPECT_NEAR(output[3], 5.0, 1e-9);
}

TEST_F(PowerDouble, PowerOfZero) {
    double input[] = {1.0, 2.0, 3.0, 100.0};
    double output[4];
    power<double, BLOCK, EPT>(input, output, 4, 0.0);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(output[i], 1.0, 1e-12) << "x^0 should be 1 at index " << i;
    }
}

TEST_F(PowerDouble, PowerOfOne) {
    double input[] = {3.14, 2.71, 42.0};
    double output[3];
    power<double, BLOCK, EPT>(input, output, 3, 1.0);

    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(output[i], input[i], 1e-12) << "index " << i;
    }
}

TEST_F(PowerDouble, NegativeExponent) {
    double input[] = {2.0, 4.0, 5.0};
    double output[3];
    power<double, BLOCK, EPT>(input, output, 3, -1.0);

    EXPECT_NEAR(output[0], 0.5, 1e-12);
    EXPECT_NEAR(output[1], 0.25, 1e-12);
    EXPECT_NEAR(output[2], 0.2, 1e-12);
}

TEST_F(PowerDouble, NegativeBaseEvenExponent) {
    double input[] = {-2.0, -3.0, -4.0};
    double output[3];
    power<double, BLOCK, EPT>(input, output, 3, 2.0);

    EXPECT_NEAR(output[0], 4.0, 1e-12);
    EXPECT_NEAR(output[1], 9.0, 1e-12);
    EXPECT_NEAR(output[2], 16.0, 1e-12);
}

TEST_F(PowerDouble, ZeroBasePosExponent) {
    double input[] = {0.0, 0.0, 0.0};
    double output[3];
    power<double, BLOCK, EPT>(input, output, 3, 2.0);

    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(output[i], 0.0, 1e-12) << "index " << i;
    }
}

TEST_F(PowerDouble, SingleElement) {
    double input[] = {3.0};
    double output[1];
    power<double, BLOCK, EPT>(input, output, 1, 4.0);

    EXPECT_NEAR(output[0], 81.0, 1e-9);
}

TEST_F(PowerDouble, NanPassthrough) {
    double input[] = {nan(""), 4.0, nan("")};
    double output[3];
    power<double, BLOCK, EPT>(input, output, 3, 2.0);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_NEAR(output[1], 16.0, 1e-12);
    EXPECT_TRUE(std::isnan(output[2]));
}

TEST_F(PowerDouble, FractionalExponent) {
    double input[] = {8.0, 27.0, 64.0};
    double output[3];
    power<double, BLOCK, EPT>(input, output, 3, 1.0 / 3.0);

    EXPECT_NEAR(output[0], 2.0, 1e-9);
    EXPECT_NEAR(output[1], 3.0, 1e-9);
    EXPECT_NEAR(output[2], 4.0, 1e-9);
}

TEST_F(PowerDouble, SquareThenSqrtRecovery) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.01, 100.0);

    const int n = 1000;
    std::vector<double> input(n), squared(n), recovered(n);
    for (auto& v : input) v = dist(rng);

    power<double, BLOCK, EPT>(input.data(), squared.data(), n, 2.0);
    power<double, BLOCK, EPT>(squared.data(), recovered.data(), n, 0.5);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(recovered[i], input[i], 1e-6) << "index " << i;
    }
}

TEST_F(PowerDouble, CpuReferenceRandom) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.1, 100.0);

    const int n = 4096;
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, 2.5);
}

TEST_F(PowerDouble, NonDivisibleLength) {
    std::mt19937 rng(256);
    std::uniform_real_distribution<double> dist(0.1, 50.0);

    for (int n : {1, 3, 7, 255, 257, 1023, 4097}) {
        std::vector<double> input(n);
        for (auto& v : input) v = dist(rng);
        run_and_compare(input, 2.0);
    }
}

TEST_F(PowerDouble, LargeArray) {
    std::mt19937 rng(314);
    std::uniform_real_distribution<double> dist(0.01, 1000.0);

    const int n = 100000;
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, 1.5, 1e-4);
}

TEST_F(PowerFloat, Square) {
    float input[] = {2, 3, 4};
    float output[3];
    power<float, BLOCK, EPT>(input, output, 3, 2.0);

    EXPECT_NEAR(output[0], 4.0f, 1e-5f);
    EXPECT_NEAR(output[1], 9.0f, 1e-5f);
    EXPECT_NEAR(output[2], 16.0f, 1e-5f);
}

TEST_F(PowerFloat, CpuReferenceRandom) {
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(0.1f, 100.0f);

    const int n = 4096;
    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, 2.0, 1e-3f);
}

TEST_F(PowerFloat, NanPassthrough) {
    float input[] = {nanf(""), 4.0f, nanf("")};
    float output[3];
    power<float, BLOCK, EPT>(input, output, 3, 2.0);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_NEAR(output[1], 16.0f, 1e-5f);
    EXPECT_TRUE(std::isnan(output[2]));
}

TEST_F(PowerFloat, LargeArray) {
    std::mt19937 rng(33);
    std::uniform_real_distribution<float> dist(0.1f, 50.0f);

    const int n = 50000;
    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, 2.0, 1e-2f);
}

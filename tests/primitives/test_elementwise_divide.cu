#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include "elementwise_divide.cu"

class ElementwiseDivideDouble : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<double>& dividend, const std::vector<double>& divisor, double tol = 1e-12) {
        int n = static_cast<int>(dividend.size());
        std::vector<double> gpu_out(n);
        elementwise_divide<double, BLOCK, EPT>(dividend.data(), divisor.data(), gpu_out.data(), n);
        for (int i = 0; i < n; i++) {
            if (std::isnan(dividend[i]) || std::isnan(divisor[i])) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else if (divisor[i] == 0.0) {
                if (dividend[i] == 0.0) {
                    EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN for 0/0 at index " << i;
                } else {
                    EXPECT_TRUE(std::isinf(gpu_out[i])) << "expected inf at index " << i;
                }
            } else {
                double expected = dividend[i] / divisor[i];
                EXPECT_NEAR(gpu_out[i], expected, tol) << "mismatch at index " << i;
            }
        }
    }
};

class ElementwiseDivideFloat : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<float>& dividend, const std::vector<float>& divisor, float tol = 1e-5f) {
        int n = static_cast<int>(dividend.size());
        std::vector<float> gpu_out(n);
        elementwise_divide<float, BLOCK, EPT>(dividend.data(), divisor.data(), gpu_out.data(), n);
        for (int i = 0; i < n; i++) {
            if (std::isnan(dividend[i]) || std::isnan(divisor[i])) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else if (divisor[i] == 0.0f) {
                if (dividend[i] == 0.0f) {
                    EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN for 0/0 at index " << i;
                } else {
                    EXPECT_TRUE(std::isinf(gpu_out[i])) << "expected inf at index " << i;
                }
            } else {
                float expected = static_cast<float>(static_cast<double>(dividend[i]) / static_cast<double>(divisor[i]));
                EXPECT_NEAR(gpu_out[i], expected, tol) << "mismatch at index " << i;
            }
        }
    }
};

TEST_F(ElementwiseDivideDouble, HandVerifiedValues) {
    double dividend[] = {10, 20, 30, 100, 7};
    double divisor[]  = {2,  4,  5,  10,  7};
    double output[5];
    elementwise_divide<double, BLOCK, EPT>(dividend, divisor, output, 5);

    EXPECT_NEAR(output[0], 5.0, 1e-12);
    EXPECT_NEAR(output[1], 5.0, 1e-12);
    EXPECT_NEAR(output[2], 6.0, 1e-12);
    EXPECT_NEAR(output[3], 10.0, 1e-12);
    EXPECT_NEAR(output[4], 1.0, 1e-12);
}

TEST_F(ElementwiseDivideDouble, DivideByOnes) {
    double dividend[] = {3.14, 2.71, 1.41, 42.0};
    double divisor[]  = {1.0,  1.0,  1.0,  1.0};
    double output[4];
    elementwise_divide<double, BLOCK, EPT>(dividend, divisor, output, 4);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(output[i], dividend[i], 1e-12) << "index " << i;
    }
}

TEST_F(ElementwiseDivideDouble, DivideBySelf) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(1.0, 1000.0);

    const int n = 1000;
    std::vector<double> input(n), output(n);
    for (auto& v : input) v = dist(rng);

    elementwise_divide<double, BLOCK, EPT>(input.data(), input.data(), output.data(), n);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(output[i], 1.0, 1e-12) << "index " << i;
    }
}

TEST_F(ElementwiseDivideDouble, SingleElement) {
    double dividend[] = {42.0};
    double divisor[]  = {7.0};
    double output[1];
    elementwise_divide<double, BLOCK, EPT>(dividend, divisor, output, 1);

    EXPECT_NEAR(output[0], 6.0, 1e-12);
}

TEST_F(ElementwiseDivideDouble, NanPassthrough) {
    double dividend[] = {nan(""), 10.0, nan(""), 20.0};
    double divisor[]  = {5.0, nan(""), nan(""), 4.0};
    double output[4];
    elementwise_divide<double, BLOCK, EPT>(dividend, divisor, output, 4);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));
    EXPECT_TRUE(std::isnan(output[2]));
    EXPECT_NEAR(output[3], 5.0, 1e-12);
}

TEST_F(ElementwiseDivideDouble, DivisionByZero) {
    double dividend[] = {10.0, -10.0, 0.0};
    double divisor[]  = {0.0,   0.0,  0.0};
    double output[3];
    elementwise_divide<double, BLOCK, EPT>(dividend, divisor, output, 3);

    EXPECT_TRUE(std::isinf(output[0]) && output[0] > 0);
    EXPECT_TRUE(std::isinf(output[1]) && output[1] < 0);
    EXPECT_TRUE(std::isnan(output[2]));
}

TEST_F(ElementwiseDivideDouble, NegativeValues) {
    double dividend[] = {-10.0, 10.0, -10.0, -10.0};
    double divisor[]  = {2.0, -2.0, -2.0, 5.0};
    double output[4];
    elementwise_divide<double, BLOCK, EPT>(dividend, divisor, output, 4);

    EXPECT_NEAR(output[0], -5.0, 1e-12);
    EXPECT_NEAR(output[1], -5.0, 1e-12);
    EXPECT_NEAR(output[2], 5.0, 1e-12);
    EXPECT_NEAR(output[3], -2.0, 1e-12);
}

TEST_F(ElementwiseDivideDouble, MixedNanAndZeroDivisor) {
    double dividend[] = {nan(""), 10.0, 0.0, nan("")};
    double divisor[]  = {0.0,     0.0,  5.0, 3.0};
    double output[4];
    elementwise_divide<double, BLOCK, EPT>(dividend, divisor, output, 4);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isinf(output[1]) && output[1] > 0);
    EXPECT_NEAR(output[2], 0.0, 1e-12);
    EXPECT_TRUE(std::isnan(output[3]));
}

TEST_F(ElementwiseDivideDouble, SmallDivisors) {
    double dividend[] = {1.0, 1.0};
    double divisor[]  = {1e-15, 1e-300};
    double output[2];
    elementwise_divide<double, BLOCK, EPT>(dividend, divisor, output, 2);

    EXPECT_NEAR(output[0], 1e15, 1.0);
    EXPECT_NEAR(output[1], 1e300, 1e285);
}

TEST_F(ElementwiseDivideDouble, CpuReferenceRandom) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(1.0, 500.0);

    const int n = 4096;
    std::vector<double> dividend(n), divisor(n);
    for (auto& v : dividend) v = dist(rng);
    for (auto& v : divisor) v = dist(rng);

    run_and_compare(dividend, divisor);
}

TEST_F(ElementwiseDivideDouble, NonDivisibleArrayLength) {
    std::mt19937 rng(256);
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    for (int n : {1, 3, 7, 255, 257, 1023, 4097}) {
        std::vector<double> dividend(n), divisor(n);
        for (auto& v : dividend) v = dist(rng);
        for (auto& v : divisor) v = dist(rng);
        run_and_compare(dividend, divisor);
    }
}

TEST_F(ElementwiseDivideDouble, LargeArray) {
    std::mt19937 rng(314);
    std::uniform_real_distribution<double> dist(0.001, 1e6);

    const int n = 100000;
    std::vector<double> dividend(n), divisor(n);
    for (auto& v : dividend) v = dist(rng);
    for (auto& v : divisor) v = dist(rng);

    run_and_compare(dividend, divisor, 1e-6);
}

TEST_F(ElementwiseDivideFloat, HandVerifiedValues) {
    float dividend[] = {10, 20, 30};
    float divisor[]  = {2,  4,  5};
    float output[3];
    elementwise_divide<float, BLOCK, EPT>(dividend, divisor, output, 3);

    EXPECT_NEAR(output[0], 5.0f, 1e-5f);
    EXPECT_NEAR(output[1], 5.0f, 1e-5f);
    EXPECT_NEAR(output[2], 6.0f, 1e-5f);
}

TEST_F(ElementwiseDivideFloat, DivisionByZero) {
    float dividend[] = {10.0f, -10.0f, 0.0f};
    float divisor[]  = {0.0f,   0.0f,  0.0f};
    float output[3];
    elementwise_divide<float, BLOCK, EPT>(dividend, divisor, output, 3);

    EXPECT_TRUE(std::isinf(output[0]) && output[0] > 0);
    EXPECT_TRUE(std::isinf(output[1]) && output[1] < 0);
    EXPECT_TRUE(std::isnan(output[2]));
}

TEST_F(ElementwiseDivideFloat, CpuReferenceRandom) {
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(1.0f, 500.0f);

    const int n = 4096;
    std::vector<float> dividend(n), divisor(n);
    for (auto& v : dividend) v = dist(rng);
    for (auto& v : divisor) v = dist(rng);

    run_and_compare(dividend, divisor);
}

TEST_F(ElementwiseDivideFloat, LargeArray) {
    std::mt19937 rng(33);
    std::uniform_real_distribution<float> dist(0.01f, 1e4f);

    const int n = 50000;
    std::vector<float> dividend(n), divisor(n);
    for (auto& v : dividend) v = dist(rng);
    for (auto& v : divisor) v = dist(rng);

    run_and_compare(dividend, divisor, 1e-3f);
}

TEST_F(ElementwiseDivideFloat, NanPassthrough) {
    float dividend[] = {nan(""), 10.0f};
    float divisor[]  = {5.0f, nan("")};
    float output[2];
    elementwise_divide<float, BLOCK, EPT>(dividend, divisor, output, 2);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));
}

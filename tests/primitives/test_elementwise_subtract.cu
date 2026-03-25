#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include "elementwise_subtract.cu"

class ElementwiseSubtractDouble : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<double>& a, const std::vector<double>& b, double tol = 1e-12) {
        int n = static_cast<int>(a.size());
        std::vector<double> gpu_out(n);
        elementwise_subtract<double, BLOCK, EPT>(a.data(), b.data(), gpu_out.data(), n);
        for (int i = 0; i < n; i++) {
            if (std::isnan(a[i]) || std::isnan(b[i])) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else {
                EXPECT_NEAR(gpu_out[i], a[i] - b[i], tol) << "mismatch at index " << i;
            }
        }
    }
};

class ElementwiseSubtractFloat : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-5f) {
        int n = static_cast<int>(a.size());
        std::vector<float> gpu_out(n);
        elementwise_subtract<float, BLOCK, EPT>(a.data(), b.data(), gpu_out.data(), n);
        for (int i = 0; i < n; i++) {
            if (std::isnan(a[i]) || std::isnan(b[i])) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else {
                EXPECT_NEAR(gpu_out[i], a[i] - b[i], tol) << "mismatch at index " << i;
            }
        }
    }
};

TEST_F(ElementwiseSubtractDouble, HandVerified) {
    double a[] = {50, 30, 10, 5, 1};
    double b[] = {10, 20, 30, 40, 50};
    double output[5];
    elementwise_subtract<double, BLOCK, EPT>(a, b, output, 5);

    EXPECT_NEAR(output[0], 40.0, 1e-12);
    EXPECT_NEAR(output[1], 10.0, 1e-12);
    EXPECT_NEAR(output[2], -20.0, 1e-12);
    EXPECT_NEAR(output[3], -35.0, 1e-12);
    EXPECT_NEAR(output[4], -49.0, 1e-12);
}

TEST_F(ElementwiseSubtractDouble, SubtractZero) {
    double a[] = {3.14, 2.71, 1.41};
    double b[] = {0.0, 0.0, 0.0};
    double output[3];
    elementwise_subtract<double, BLOCK, EPT>(a, b, output, 3);

    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(output[i], a[i], 1e-12) << "index " << i;
    }
}

TEST_F(ElementwiseSubtractDouble, SubtractSelfIsZero) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

    const int n = 1000;
    std::vector<double> a(n), output(n);
    for (auto& v : a) v = dist(rng);

    elementwise_subtract<double, BLOCK, EPT>(a.data(), a.data(), output.data(), n);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(output[i], 0.0, 1e-12) << "index " << i;
    }
}

TEST_F(ElementwiseSubtractDouble, AntiCommutative) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-500.0, 500.0);

    const int n = 1000;
    std::vector<double> a(n), b(n), ab(n), ba(n);
    for (int i = 0; i < n; i++) { a[i] = dist(rng); b[i] = dist(rng); }

    elementwise_subtract<double, BLOCK, EPT>(a.data(), b.data(), ab.data(), n);
    elementwise_subtract<double, BLOCK, EPT>(b.data(), a.data(), ba.data(), n);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(ab[i], -ba[i], 1e-12) << "a-b != -(b-a) at index " << i;
    }
}

TEST_F(ElementwiseSubtractDouble, NanPassthrough) {
    double a[] = {nan(""), 10.0, nan("")};
    double b[] = {5.0, nan(""), nan("")};
    double output[3];
    elementwise_subtract<double, BLOCK, EPT>(a, b, output, 3);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));
    EXPECT_TRUE(std::isnan(output[2]));
}

TEST_F(ElementwiseSubtractDouble, SingleElement) {
    double a[] = {100.0};
    double b[] = {37.5};
    double output[1];
    elementwise_subtract<double, BLOCK, EPT>(a, b, output, 1);

    EXPECT_NEAR(output[0], 62.5, 1e-12);
}

TEST_F(ElementwiseSubtractDouble, NegativeResults) {
    double a[] = {1.0, 2.0, 3.0};
    double b[] = {10.0, 20.0, 30.0};
    double output[3];
    elementwise_subtract<double, BLOCK, EPT>(a, b, output, 3);

    EXPECT_NEAR(output[0], -9.0, 1e-12);
    EXPECT_NEAR(output[1], -18.0, 1e-12);
    EXPECT_NEAR(output[2], -27.0, 1e-12);
}

TEST_F(ElementwiseSubtractDouble, CpuReferenceRandom) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);

    const int n = 4096;
    std::vector<double> a(n), b(n);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    run_and_compare(a, b);
}

TEST_F(ElementwiseSubtractDouble, NonDivisibleLength) {
    std::mt19937 rng(256);
    std::uniform_real_distribution<double> dist(0.0, 100.0);

    for (int n : {1, 3, 7, 255, 257, 1023, 4097}) {
        std::vector<double> a(n), b(n);
        for (auto& v : a) v = dist(rng);
        for (auto& v : b) v = dist(rng);
        run_and_compare(a, b);
    }
}

TEST_F(ElementwiseSubtractDouble, LargeArray) {
    std::mt19937 rng(314);
    std::uniform_real_distribution<double> dist(-1e6, 1e6);

    const int n = 100000;
    std::vector<double> a(n), b(n);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    run_and_compare(a, b, 1e-6);
}

TEST_F(ElementwiseSubtractFloat, HandVerified) {
    float a[] = {50, 30, 10};
    float b[] = {10, 20, 30};
    float output[3];
    elementwise_subtract<float, BLOCK, EPT>(a, b, output, 3);

    EXPECT_NEAR(output[0], 40.0f, 1e-5f);
    EXPECT_NEAR(output[1], 10.0f, 1e-5f);
    EXPECT_NEAR(output[2], -20.0f, 1e-5f);
}

TEST_F(ElementwiseSubtractFloat, CpuReferenceRandom) {
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-500.0f, 500.0f);

    const int n = 4096;
    std::vector<float> a(n), b(n);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    run_and_compare(a, b);
}

TEST_F(ElementwiseSubtractFloat, LargeArray) {
    std::mt19937 rng(33);
    std::uniform_real_distribution<float> dist(-1e4f, 1e4f);

    const int n = 50000;
    std::vector<float> a(n), b(n);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    run_and_compare(a, b, 1e-2f);
}

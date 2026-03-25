#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include "elementwise_multiply.cu"

class ElementwiseMultiplyDouble : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<double>& a, const std::vector<double>& b, double tol = 1e-12) {
        int n = static_cast<int>(a.size());
        std::vector<double> gpu_out(n);
        elementwise_multiply<double, BLOCK, EPT>(a.data(), b.data(), gpu_out.data(), n);
        for (int i = 0; i < n; i++) {
            if (std::isnan(a[i]) || std::isnan(b[i])) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else {
                EXPECT_NEAR(gpu_out[i], a[i] * b[i], tol) << "mismatch at index " << i;
            }
        }
    }
};

class ElementwiseMultiplyFloat : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-5f) {
        int n = static_cast<int>(a.size());
        std::vector<float> gpu_out(n);
        elementwise_multiply<float, BLOCK, EPT>(a.data(), b.data(), gpu_out.data(), n);
        for (int i = 0; i < n; i++) {
            if (std::isnan(a[i]) || std::isnan(b[i])) {
                EXPECT_TRUE(std::isnan(gpu_out[i])) << "expected NaN at index " << i;
            } else {
                EXPECT_NEAR(gpu_out[i], a[i] * b[i], tol) << "mismatch at index " << i;
            }
        }
    }
};

TEST_F(ElementwiseMultiplyDouble, HandVerified) {
    double a[] = {2, 3, 4, 5, 6};
    double b[] = {10, 20, 30, 40, 50};
    double output[5];
    elementwise_multiply<double, BLOCK, EPT>(a, b, output, 5);

    EXPECT_NEAR(output[0], 20.0, 1e-12);
    EXPECT_NEAR(output[1], 60.0, 1e-12);
    EXPECT_NEAR(output[2], 120.0, 1e-12);
    EXPECT_NEAR(output[3], 200.0, 1e-12);
    EXPECT_NEAR(output[4], 300.0, 1e-12);
}

TEST_F(ElementwiseMultiplyDouble, MultiplyByOne) {
    double a[] = {3.14, 2.71, 1.41};
    double b[] = {1.0, 1.0, 1.0};
    double output[3];
    elementwise_multiply<double, BLOCK, EPT>(a, b, output, 3);

    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(output[i], a[i], 1e-12) << "index " << i;
    }
}

TEST_F(ElementwiseMultiplyDouble, MultiplyByZero) {
    double a[] = {3.14, 2.71, 1.41};
    double b[] = {0.0, 0.0, 0.0};
    double output[3];
    elementwise_multiply<double, BLOCK, EPT>(a, b, output, 3);

    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(output[i], 0.0, 1e-12) << "index " << i;
    }
}

TEST_F(ElementwiseMultiplyDouble, Commutative) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-500.0, 500.0);

    const int n = 1000;
    std::vector<double> a(n), b(n), ab(n), ba(n);
    for (int i = 0; i < n; i++) { a[i] = dist(rng); b[i] = dist(rng); }

    elementwise_multiply<double, BLOCK, EPT>(a.data(), b.data(), ab.data(), n);
    elementwise_multiply<double, BLOCK, EPT>(b.data(), a.data(), ba.data(), n);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(ab[i], ba[i], 1e-12) << "a*b != b*a at index " << i;
    }
}

TEST_F(ElementwiseMultiplyDouble, SquaringSelf) {
    double a[] = {3.0, -4.0, 5.0, -7.0};
    double output[4];
    elementwise_multiply<double, BLOCK, EPT>(a, a, output, 4);

    EXPECT_NEAR(output[0], 9.0, 1e-12);
    EXPECT_NEAR(output[1], 16.0, 1e-12);
    EXPECT_NEAR(output[2], 25.0, 1e-12);
    EXPECT_NEAR(output[3], 49.0, 1e-12);
}

TEST_F(ElementwiseMultiplyDouble, NegativeValues) {
    double a[] = {-2.0, 3.0, -4.0, -5.0};
    double b[] = {3.0, -4.0, -5.0, 6.0};
    double output[4];
    elementwise_multiply<double, BLOCK, EPT>(a, b, output, 4);

    EXPECT_NEAR(output[0], -6.0, 1e-12);
    EXPECT_NEAR(output[1], -12.0, 1e-12);
    EXPECT_NEAR(output[2], 20.0, 1e-12);
    EXPECT_NEAR(output[3], -30.0, 1e-12);
}

TEST_F(ElementwiseMultiplyDouble, NanPassthrough) {
    double a[] = {nan(""), 10.0, nan("")};
    double b[] = {5.0, nan(""), nan("")};
    double output[3];
    elementwise_multiply<double, BLOCK, EPT>(a, b, output, 3);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));
    EXPECT_TRUE(std::isnan(output[2]));
}

TEST_F(ElementwiseMultiplyDouble, SingleElement) {
    double a[] = {7.0};
    double b[] = {6.0};
    double output[1];
    elementwise_multiply<double, BLOCK, EPT>(a, b, output, 1);

    EXPECT_NEAR(output[0], 42.0, 1e-12);
}

TEST_F(ElementwiseMultiplyDouble, CpuReferenceRandom) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-500.0, 500.0);

    const int n = 4096;
    std::vector<double> a(n), b(n);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    run_and_compare(a, b);
}

TEST_F(ElementwiseMultiplyDouble, NonDivisibleLength) {
    std::mt19937 rng(256);
    std::uniform_real_distribution<double> dist(0.0, 100.0);

    for (int n : {1, 3, 7, 255, 257, 1023, 4097}) {
        std::vector<double> a(n), b(n);
        for (auto& v : a) v = dist(rng);
        for (auto& v : b) v = dist(rng);
        run_and_compare(a, b);
    }
}

TEST_F(ElementwiseMultiplyDouble, LargeArray) {
    std::mt19937 rng(314);
    std::uniform_real_distribution<double> dist(-1e3, 1e3);

    const int n = 100000;
    std::vector<double> a(n), b(n);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    run_and_compare(a, b, 1e-6);
}

TEST_F(ElementwiseMultiplyFloat, HandVerified) {
    float a[] = {2, 3, 4};
    float b[] = {10, 20, 30};
    float output[3];
    elementwise_multiply<float, BLOCK, EPT>(a, b, output, 3);

    EXPECT_NEAR(output[0], 20.0f, 1e-5f);
    EXPECT_NEAR(output[1], 60.0f, 1e-5f);
    EXPECT_NEAR(output[2], 120.0f, 1e-5f);
}

TEST_F(ElementwiseMultiplyFloat, CpuReferenceRandom) {
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-500.0f, 500.0f);

    const int n = 4096;
    std::vector<float> a(n), b(n);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    run_and_compare(a, b);
}

TEST_F(ElementwiseMultiplyFloat, LargeArray) {
    std::mt19937 rng(33);
    std::uniform_real_distribution<float> dist(-1e2f, 1e2f);

    const int n = 50000;
    std::vector<float> a(n), b(n);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    run_and_compare(a, b, 1e-2f);
}

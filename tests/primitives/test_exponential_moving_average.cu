#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include "exponential_moving_average.cu"

template <typename T>
void cpu_ema(const T* input, T* output, int num_symbols, int n, T alpha) {
    for (int s = 0; s < num_symbols; s++) {
        const T* in_row = input + s * n;
        T* out_row = output + s * n;
        out_row[0] = in_row[0];
        for (int i = 1; i < n; i++) {
            double a = static_cast<double>(alpha);
            out_row[i] = static_cast<T>(a * static_cast<double>(in_row[i]) + (1.0 - a) * static_cast<double>(out_row[i - 1]));
        }
    }
}

class EmaDouble : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;

    void run_and_compare(const std::vector<double>& input, int num_symbols, int n, double alpha, double tol = 1e-9) {
        std::vector<double> gpu_out(num_symbols * n), cpu_out(num_symbols * n);
        cpu_ema(input.data(), cpu_out.data(), num_symbols, n, alpha);
        exponential_moving_average<double, BLOCK>(input.data(), gpu_out.data(), num_symbols, n, alpha);
        for (int i = 0; i < num_symbols * n; i++) {
            EXPECT_NEAR(gpu_out[i], cpu_out[i], tol) << "mismatch at flat index " << i
                << " (symbol " << i / n << ", time " << i % n << ")";
        }
    }
};

class EmaFloat : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;

    void run_and_compare(const std::vector<float>& input, int num_symbols, int n, float alpha, float tol = 1e-3f) {
        std::vector<float> gpu_out(num_symbols * n), cpu_out(num_symbols * n);
        cpu_ema(input.data(), cpu_out.data(), num_symbols, n, alpha);
        exponential_moving_average<float, BLOCK>(input.data(), gpu_out.data(), num_symbols, n, alpha);
        for (int i = 0; i < num_symbols * n; i++) {
            EXPECT_NEAR(gpu_out[i], cpu_out[i], tol) << "mismatch at flat index " << i
                << " (symbol " << i / n << ", time " << i % n << ")";
        }
    }
};

TEST_F(EmaDouble, HandVerifiedSingleSymbol) {
    double input[] = {10, 20, 30, 40, 50};
    double output[5];
    double alpha = 0.5;
    exponential_moving_average<double, BLOCK>(input, output, 1, 5, alpha);

    EXPECT_NEAR(output[0], 10.0, 1e-12);
    double e1 = 0.5 * 20 + 0.5 * 10;
    EXPECT_NEAR(output[1], e1, 1e-12);
    double e2 = 0.5 * 30 + 0.5 * e1;
    EXPECT_NEAR(output[2], e2, 1e-12);
    double e3 = 0.5 * 40 + 0.5 * e2;
    EXPECT_NEAR(output[3], e3, 1e-12);
    double e4 = 0.5 * 50 + 0.5 * e3;
    EXPECT_NEAR(output[4], e4, 1e-12);
}

TEST_F(EmaDouble, AlphaOneEqualsInput) {
    double input[] = {3.14, 2.71, 1.41, 1.73, 0.577};
    double output[5];
    exponential_moving_average<double, BLOCK>(input, output, 1, 5, 1.0);

    for (int i = 0; i < 5; i++) {
        EXPECT_NEAR(output[i], input[i], 1e-12) << "index " << i;
    }
}

TEST_F(EmaDouble, AlphaZeroRepeatsFirst) {
    double input[] = {42.0, 100.0, 200.0, 300.0};
    double output[4];
    exponential_moving_average<double, BLOCK>(input, output, 1, 4, 0.0);

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(output[i], 42.0, 1e-12) << "index " << i;
    }
}

TEST_F(EmaDouble, ConstantInput) {
    std::vector<double> input(500, 7.77);
    std::vector<double> output(500);
    exponential_moving_average<double, BLOCK>(input.data(), output.data(), 1, 500, 0.3);

    for (int i = 0; i < 500; i++) {
        EXPECT_NEAR(output[i], 7.77, 1e-9) << "index " << i;
    }
}

TEST_F(EmaDouble, SingleElement) {
    double input[] = {99.0};
    double output[1];
    exponential_moving_average<double, BLOCK>(input, output, 1, 1, 0.5);

    EXPECT_NEAR(output[0], 99.0, 1e-12);
}

TEST_F(EmaDouble, TwoSymbols) {
    double input[] = {
        10, 20, 30, 40, 50,
        100, 200, 300, 400, 500
    };
    double output[10];
    double alpha = 0.3;
    exponential_moving_average<double, BLOCK>(input, output, 2, 5, alpha);

    double expected_s0[5], expected_s1[5];
    expected_s0[0] = 10;
    expected_s1[0] = 100;
    for (int i = 1; i < 5; i++) {
        expected_s0[i] = alpha * input[i] + (1 - alpha) * expected_s0[i - 1];
        expected_s1[i] = alpha * input[5 + i] + (1 - alpha) * expected_s1[i - 1];
    }

    for (int i = 0; i < 5; i++) {
        EXPECT_NEAR(output[i], expected_s0[i], 1e-9) << "symbol 0, time " << i;
        EXPECT_NEAR(output[5 + i], expected_s1[i], 1e-9) << "symbol 1, time " << i;
    }
}

TEST_F(EmaDouble, SymbolsAreIndependent) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    const int n = 200;
    const double alpha = 0.2;
    std::vector<double> single_input(n);
    for (auto& v : single_input) v = dist(rng);

    std::vector<double> single_output(n);
    exponential_moving_average<double, BLOCK>(single_input.data(), single_output.data(), 1, n, alpha);

    std::vector<double> noise(n);
    for (auto& v : noise) v = dist(rng);

    std::vector<double> packed_input(2 * n);
    std::copy(single_input.begin(), single_input.end(), packed_input.begin());
    std::copy(noise.begin(), noise.end(), packed_input.begin() + n);

    std::vector<double> packed_output(2 * n);
    exponential_moving_average<double, BLOCK>(packed_input.data(), packed_output.data(), 2, n, alpha);

    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(packed_output[i], single_output[i], 1e-12)
            << "adding a second symbol changed symbol 0 at time " << i;
    }
}

TEST_F(EmaDouble, HighAlphaTracks) {
    const int n = 100;
    const double alpha = 0.99;
    std::vector<double> input(n);
    for (int i = 0; i < n; i++) input[i] = static_cast<double>(i);

    std::vector<double> output(n);
    exponential_moving_average<double, BLOCK>(input.data(), output.data(), 1, n, alpha);

    for (int i = 10; i < n; i++) {
        EXPECT_NEAR(output[i], input[i], 2.0) << "high alpha should track closely at index " << i;
    }
}

TEST_F(EmaDouble, LowAlphaSmooths) {
    const int n = 100;
    const double alpha = 0.01;
    std::vector<double> input(n);
    for (int i = 0; i < n; i++) input[i] = (i % 2 == 0) ? 0.0 : 100.0;

    std::vector<double> output(n);
    exponential_moving_average<double, BLOCK>(input.data(), output.data(), 1, n, alpha);

    double range = *std::max_element(output.begin() + 50, output.end())
                 - *std::min_element(output.begin() + 50, output.end());
    EXPECT_LT(range, 15.0) << "low alpha should heavily smooth oscillations";
}

TEST_F(EmaDouble, ManySymbols) {
    std::mt19937 rng(77);
    std::uniform_real_distribution<double> dist(10.0, 500.0);

    const int num_symbols = 500;
    const int n = 200;
    const double alpha = 0.1;

    std::vector<double> input(num_symbols * n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, num_symbols, n, alpha);
}

TEST_F(EmaDouble, CpuReferenceRandom) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(50.0, 200.0);

    const int num_symbols = 10;
    const int n = 1000;
    const double alpha = 0.15;

    std::vector<double> input(num_symbols * n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, num_symbols, n, alpha);
}

TEST_F(EmaDouble, LongTimeSeries) {
    std::mt19937 rng(314);
    std::uniform_real_distribution<double> dist(1.0, 1000.0);

    const int n = 50000;
    const double alpha = 0.05;

    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, 1, n, alpha, 1e-6);
}

TEST_F(EmaDouble, VariousAlphas) {
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    const int n = 500;
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    for (double alpha : {0.01, 0.1, 0.2, 0.5, 0.8, 0.99}) {
        std::vector<double> gpu_out(n), cpu_out(n);
        cpu_ema(input.data(), cpu_out.data(), 1, n, alpha);
        exponential_moving_average<double, BLOCK>(input.data(), gpu_out.data(), 1, n, alpha);
        for (int i = 0; i < n; i++) {
            EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-9) << "alpha=" << alpha << " index " << i;
        }
    }
}

TEST_F(EmaDouble, MoreSymbolsThanBlockSize) {
    std::mt19937 rng(256);
    std::uniform_real_distribution<double> dist(1.0, 100.0);

    const int num_symbols = BLOCK * 3 + 37;
    const int n = 50;
    const double alpha = 0.2;

    std::vector<double> input(num_symbols * n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, num_symbols, n, alpha);
}

TEST_F(EmaFloat, HandVerifiedSingleSymbol) {
    float input[] = {10, 20, 30, 40, 50};
    float output[5];
    float alpha = 0.5f;
    exponential_moving_average<float, BLOCK>(input, output, 1, 5, alpha);

    EXPECT_NEAR(output[0], 10.0f, 1e-5f);
    float e1 = 0.5f * 20 + 0.5f * 10;
    EXPECT_NEAR(output[1], e1, 1e-4f);
}

TEST_F(EmaFloat, CpuReferenceRandom) {
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(0.0f, 500.0f);

    const int num_symbols = 10;
    const int n = 1000;
    const float alpha = 0.15f;

    std::vector<float> input(num_symbols * n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, num_symbols, n, alpha);
}

TEST_F(EmaFloat, ManySymbols) {
    std::mt19937 rng(33);
    std::uniform_real_distribution<float> dist(1.0f, 1000.0f);

    const int num_symbols = 500;
    const int n = 200;
    const float alpha = 0.1f;

    std::vector<float> input(num_symbols * n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, num_symbols, n, alpha, 0.01f);
}

TEST_F(EmaFloat, ConstantInput) {
    std::vector<float> input(300, 55.5f);
    std::vector<float> output(300);
    exponential_moving_average<float, BLOCK>(input.data(), output.data(), 1, 300, 0.3f);

    for (int i = 0; i < 300; i++) {
        EXPECT_NEAR(output[i], 55.5f, 1e-4f) << "index " << i;
    }
}

TEST_F(EmaFloat, LongSeriesPrecision) {
    std::mt19937 rng(55);
    std::uniform_real_distribution<float> dist(100.0f, 10000.0f);

    const int n = 50000;
    const float alpha = 0.05f;

    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, 1, n, alpha, 0.1f);
}

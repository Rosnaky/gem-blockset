#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include "log_return.cu"

template <typename T>
void cpu_log_return(const T* input, T* output, int n) {
    for (int i = 0; i < n; i++) {
        if (i == 0) {
            output[i] = nan("");
            continue;
        }
        output[i] = static_cast<T>(log(static_cast<double>(input[i]) / static_cast<double>(input[i - 1])));
    }
}

class LogReturnDouble : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<double>& input, double tol = 1e-12) {
        int n = static_cast<int>(input.size());
        std::vector<double> gpu_out(n), cpu_out(n);
        cpu_log_return(input.data(), cpu_out.data(), n);
        log_return<double, BLOCK, EPT>(input.data(), gpu_out.data(), n);
        EXPECT_TRUE(std::isnan(gpu_out[0])) << "expected NaN at index 0";
        for (int i = 1; i < n; i++) {
            EXPECT_NEAR(gpu_out[i], cpu_out[i], tol) << "mismatch at index " << i;
        }
    }
};

class LogReturnFloat : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;

    void run_and_compare(const std::vector<float>& input, float tol = 1e-5f) {
        int n = static_cast<int>(input.size());
        std::vector<float> gpu_out(n), cpu_out(n);
        cpu_log_return(input.data(), cpu_out.data(), n);
        log_return<float, BLOCK, EPT>(input.data(), gpu_out.data(), n);
        EXPECT_TRUE(std::isnan(gpu_out[0])) << "expected NaN at index 0";
        for (int i = 1; i < n; i++) {
            EXPECT_NEAR(gpu_out[i], cpu_out[i], tol) << "mismatch at index " << i;
        }
    }
};

TEST_F(LogReturnDouble, HandVerifiedSmallArray) {
    double input[] = {100.0, 110.0, 105.0, 120.0};
    double output[4];
    log_return<double, BLOCK, EPT>(input, output, 4);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_NEAR(output[1], log(110.0 / 100.0), 1e-12);
    EXPECT_NEAR(output[2], log(105.0 / 110.0), 1e-12);
    EXPECT_NEAR(output[3], log(120.0 / 105.0), 1e-12);
}

TEST_F(LogReturnDouble, SingleNanAtIndexZero) {
    std::vector<double> input(200);
    for (int i = 0; i < 200; i++) input[i] = 100.0 + i;
    std::vector<double> output(200);
    log_return<double, BLOCK, EPT>(input.data(), output.data(), 200);

    int nan_count = 0;
    for (int i = 0; i < 200; i++) {
        if (std::isnan(output[i])) nan_count++;
    }
    EXPECT_EQ(nan_count, 1);
    EXPECT_TRUE(std::isnan(output[0]));
}

TEST_F(LogReturnDouble, SingleElement) {
    double input[] = {50.0};
    double output[1];
    log_return<double, BLOCK, EPT>(input, output, 1);

    EXPECT_TRUE(std::isnan(output[0]));
}

TEST_F(LogReturnDouble, TwoElements) {
    double input[] = {80.0, 160.0};
    double output[2];
    log_return<double, BLOCK, EPT>(input, output, 2);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_NEAR(output[1], log(2.0), 1e-12);
}

TEST_F(LogReturnDouble, ConstantPrice) {
    std::vector<double> input(500, 42.0);
    std::vector<double> output(500);
    log_return<double, BLOCK, EPT>(input.data(), output.data(), 500);

    for (int i = 1; i < 500; i++) {
        EXPECT_NEAR(output[i], 0.0, 1e-12) << "index " << i;
    }
}

TEST_F(LogReturnDouble, DoublingPrices) {
    const int n = 20;
    std::vector<double> input(n);
    input[0] = 1.0;
    for (int i = 1; i < n; i++) input[i] = input[i - 1] * 2.0;

    std::vector<double> output(n);
    log_return<double, BLOCK, EPT>(input.data(), output.data(), n);

    for (int i = 1; i < n; i++) {
        EXPECT_NEAR(output[i], log(2.0), 1e-12) << "index " << i;
    }
}

TEST_F(LogReturnDouble, ReturnsAreAdditiveOverTime) {
    double input[] = {100.0, 120.0, 90.0, 150.0};
    double output[4];
    log_return<double, BLOCK, EPT>(input, output, 4);

    double cumulative = output[1] + output[2] + output[3];
    double total = log(150.0 / 100.0);
    EXPECT_NEAR(cumulative, total, 1e-12);
}

TEST_F(LogReturnDouble, SymmetricUpDown) {
    double input[] = {100.0, 200.0, 100.0};
    double output[3];
    log_return<double, BLOCK, EPT>(input, output, 3);

    EXPECT_NEAR(output[1], log(2.0), 1e-12);
    EXPECT_NEAR(output[2], -log(2.0), 1e-12);
    EXPECT_NEAR(output[1] + output[2], 0.0, 1e-12);
}

TEST_F(LogReturnDouble, SmallPriceChanges) {
    const int n = 100;
    std::vector<double> input(n);
    input[0] = 1000.0;
    for (int i = 1; i < n; i++) input[i] = input[i - 1] * 1.0001;

    std::vector<double> output(n);
    log_return<double, BLOCK, EPT>(input.data(), output.data(), n);

    for (int i = 1; i < n; i++) {
        EXPECT_NEAR(output[i], log(1.0001), 1e-12) << "index " << i;
    }
}

TEST_F(LogReturnDouble, CpuReferenceRandom) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(10.0, 500.0);

    const int n = 4096;
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input);
}

TEST_F(LogReturnDouble, TileBoundary) {
    const int tile_size = BLOCK * EPT;
    const int n = tile_size * 3 + 23;

    std::mt19937 rng(88);
    std::uniform_real_distribution<double> dist(50.0, 300.0);
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    std::vector<double> gpu_out(n), cpu_out(n);
    cpu_log_return(input.data(), cpu_out.data(), n);
    log_return<double, BLOCK, EPT>(input.data(), gpu_out.data(), n);

    for (int boundary : {tile_size, tile_size * 2}) {
        for (int offset = -5; offset <= 5; offset++) {
            int i = boundary + offset;
            if (i < 1 || i >= n) continue;
            EXPECT_NEAR(gpu_out[i], cpu_out[i], 1e-12)
                << "tile boundary divergence at index " << i;
        }
    }
}

TEST_F(LogReturnDouble, NonDivisibleArrayLength) {
    std::mt19937 rng(256);
    std::uniform_real_distribution<double> dist(1.0, 200.0);

    for (int n : {3, 17, 255, 1025, 4099}) {
        std::vector<double> input(n);
        for (auto& v : input) v = dist(rng);
        run_and_compare(input);
    }
}

TEST_F(LogReturnDouble, LargeArray) {
    std::mt19937 rng(314);
    std::uniform_real_distribution<double> dist(1.0, 1000.0);

    const int n = 50000;
    std::vector<double> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, 1e-10);
}

TEST_F(LogReturnFloat, HandVerifiedSmallArray) {
    float input[] = {100.0f, 110.0f, 105.0f, 120.0f};
    float output[4];
    log_return<float, BLOCK, EPT>(input, output, 4);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_NEAR(output[1], static_cast<float>(log(110.0 / 100.0)), 1e-5f);
    EXPECT_NEAR(output[2], static_cast<float>(log(105.0 / 110.0)), 1e-5f);
    EXPECT_NEAR(output[3], static_cast<float>(log(120.0 / 105.0)), 1e-5f);
}

TEST_F(LogReturnFloat, CpuReferenceRandom) {
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(10.0f, 500.0f);

    const int n = 4096;
    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input);
}

TEST_F(LogReturnFloat, LargeArray) {
    std::mt19937 rng(33);
    std::uniform_real_distribution<float> dist(1.0f, 1000.0f);

    const int n = 50000;
    std::vector<float> input(n);
    for (auto& v : input) v = dist(rng);

    run_and_compare(input, 1e-4f);
}

TEST_F(LogReturnFloat, ConstantPrice) {
    std::vector<float> input(300, 55.5f);
    std::vector<float> output(300);
    log_return<float, BLOCK, EPT>(input.data(), output.data(), 300);

    for (int i = 1; i < 300; i++) {
        EXPECT_NEAR(output[i], 0.0f, 1e-6f) << "index " << i;
    }
}

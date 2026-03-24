#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include "adf_test.cu"

std::vector<double> generate_random_walk(int n, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 1.0);
    std::vector<double> path(n);
    path[0] = 100.0;
    for (int i = 1; i < n; i++) {
        path[i] = path[i - 1] + noise(rng);
    }
    return path;
}

std::vector<double> generate_stationary_ar1(int n, double phi, double c, double sigma, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, sigma);
    double mu = c / (1.0 - phi);
    std::vector<double> path(n);
    path[0] = mu;
    for (int i = 1; i < n; i++) {
        path[i] = c + phi * path[i - 1] + noise(rng);
    }
    return path;
}

std::vector<double> generate_ou_process(int n, double theta, double mu, double sigma, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 1.0);
    std::vector<double> path(n);
    path[0] = mu;
    for (int i = 1; i < n; i++) {
        path[i] = path[i - 1] + theta * (mu - path[i - 1]) + sigma * noise(rng);
    }
    return path;
}

class AdfTestDouble : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;
};

TEST_F(AdfTestDouble, RejectsStationaryAR1) {
    auto path = generate_stationary_ar1(2000, 0.8, 2.0, 1.0, 42);
    ADFResult result;
    adf_test<double, BLOCK, EPT>(path.data(), path.size(), 4, &result);

    EXPECT_LT(result.tau, -2.86) << "should reject H0 for stationary AR(1)";
    EXPECT_TRUE(result.reject_5pct) << "should reject at 5%";
    EXPECT_LT(result.gamma, 0.0) << "gamma should be negative";
}

TEST_F(AdfTestDouble, FailsToRejectRandomWalk) {
    auto path = generate_random_walk(2000, 42);
    ADFResult result;
    adf_test<double, BLOCK, EPT>(path.data(), path.size(), 4, &result);

    EXPECT_GT(result.tau, -2.86) << "should not reject H0 for random walk";
    EXPECT_FALSE(result.reject_5pct) << "should not reject at 5%";
}

TEST_F(AdfTestDouble, RejectsStrongMeanReversion) {
    auto path = generate_ou_process(2000, 0.5, 100.0, 2.0, 77);
    ADFResult result;
    adf_test<double, BLOCK, EPT>(path.data(), path.size(), 4, &result);

    EXPECT_LT(result.tau, -3.43) << "strong mean reversion should reject at 1%";
    EXPECT_TRUE(result.reject_1pct);
    EXPECT_TRUE(result.reject_5pct);
    EXPECT_TRUE(result.reject_10pct);
}

TEST_F(AdfTestDouble, WeakMeanReversionMayNotReject) {
    auto path = generate_stationary_ar1(500, 0.99, 0.1, 1.0, 99);
    ADFResult result;
    adf_test<double, BLOCK, EPT>(path.data(), path.size(), 4, &result);

    EXPECT_GT(result.tau, -3.43) << "very weak mean reversion with small sample may not reject at 1%";
}

TEST_F(AdfTestDouble, GammaNegativeForStationary) {
    auto path = generate_stationary_ar1(5000, 0.7, 3.0, 1.0, 42);
    ADFResult result;
    adf_test<double, BLOCK, EPT>(path.data(), path.size(), 2, &result);

    EXPECT_LT(result.gamma, 0.0) << "gamma should be negative for stationary series";
    double estimated_phi = 1.0 + result.gamma;
    EXPECT_NEAR(estimated_phi, 0.7, 0.1) << "estimated phi should approximate true value";
}

TEST_F(AdfTestDouble, GammaNearZeroForRandomWalk) {
    auto path = generate_random_walk(5000, 123);
    ADFResult result;
    adf_test<double, BLOCK, EPT>(path.data(), path.size(), 4, &result);

    EXPECT_NEAR(result.gamma, 0.0, 0.01) << "gamma should be near zero for random walk";
}

TEST_F(AdfTestDouble, StandardErrorIsPositive) {
    auto path = generate_stationary_ar1(1000, 0.9, 1.0, 1.0, 42);
    ADFResult result;
    adf_test<double, BLOCK, EPT>(path.data(), path.size(), 4, &result);

    EXPECT_GT(result.se_gamma, 0.0) << "standard error must be positive";
    EXPECT_FALSE(std::isnan(result.se_gamma)) << "standard error must not be NaN";
    EXPECT_FALSE(std::isinf(result.se_gamma)) << "standard error must not be inf";
}

TEST_F(AdfTestDouble, ResidualVarianceIsPositive) {
    auto path = generate_random_walk(1000, 42);
    ADFResult result;
    adf_test<double, BLOCK, EPT>(path.data(), path.size(), 4, &result);

    EXPECT_GT(result.residual_variance, 0.0);
    EXPECT_FALSE(std::isnan(result.residual_variance));
}

TEST_F(AdfTestDouble, LagCountStoredCorrectly) {
    auto path = generate_random_walk(500, 42);
    for (int p : {1, 2, 4, 8}) {
        ADFResult result;
        adf_test<double, BLOCK, EPT>(path.data(), path.size(), p, &result);
        EXPECT_EQ(result.lags, p) << "lags should match input p=" << p;
    }
}

TEST_F(AdfTestDouble, MoreLagsReducesPower) {
    auto path = generate_stationary_ar1(1000, 0.9, 1.0, 1.0, 42);

    ADFResult result_few, result_many;
    adf_test<double, BLOCK, EPT>(path.data(), path.size(), 1, &result_few);
    adf_test<double, BLOCK, EPT>(path.data(), path.size(), 10, &result_many);

    EXPECT_LT(result_few.tau, result_many.tau)
        << "more lags should generally reduce test power (less negative tau)";
}

TEST_F(AdfTestDouble, ConsistencyAcrossSignificanceLevels) {
    auto path = generate_stationary_ar1(2000, 0.8, 2.0, 1.0, 42);
    ADFResult result;
    adf_test<double, BLOCK, EPT>(path.data(), path.size(), 4, &result);

    if (result.reject_1pct) {
        EXPECT_TRUE(result.reject_5pct) << "rejecting at 1% implies rejecting at 5%";
        EXPECT_TRUE(result.reject_10pct) << "rejecting at 1% implies rejecting at 10%";
    }
    if (result.reject_5pct) {
        EXPECT_TRUE(result.reject_10pct) << "rejecting at 5% implies rejecting at 10%";
    }
}

TEST_F(AdfTestDouble, TauEqualsGammaOverSE) {
    auto path = generate_stationary_ar1(1000, 0.85, 1.5, 1.0, 42);
    ADFResult result;
    adf_test<double, BLOCK, EPT>(path.data(), path.size(), 4, &result);

    EXPECT_NEAR(result.tau, result.gamma / result.se_gamma, 1e-10)
        << "tau should equal gamma / se_gamma exactly";
}

TEST_F(AdfTestDouble, LargerSampleMorePower) {
    auto path_small = generate_stationary_ar1(200, 0.9, 1.0, 1.0, 42);
    auto path_large = generate_stationary_ar1(5000, 0.9, 1.0, 1.0, 42);

    ADFResult result_small, result_large;
    adf_test<double, BLOCK, EPT>(path_small.data(), path_small.size(), 4, &result_small);
    adf_test<double, BLOCK, EPT>(path_large.data(), path_large.size(), 4, &result_large);

    EXPECT_LT(result_large.tau, result_small.tau)
        << "larger sample should give more negative tau for same process";
}

TEST_F(AdfTestDouble, StrongerMeanReversionMoreNegativeTau) {
    auto path_weak = generate_stationary_ar1(3000, 0.95, 0.5, 1.0, 42);
    auto path_strong = generate_stationary_ar1(3000, 0.7, 3.0, 1.0, 42);

    ADFResult result_weak, result_strong;
    adf_test<double, BLOCK, EPT>(path_weak.data(), path_weak.size(), 4, &result_weak);
    adf_test<double, BLOCK, EPT>(path_strong.data(), path_strong.size(), 4, &result_strong);

    EXPECT_LT(result_strong.tau, result_weak.tau)
        << "stronger mean reversion should produce more negative tau";
}

TEST_F(AdfTestDouble, ConstantSeriesRejects) {
    std::vector<double> constant(1000, 42.0);
    ADFResult result;
    adf_test<double, BLOCK, EPT>(constant.data(), constant.size(), 1, &result);

    EXPECT_TRUE(result.reject_5pct || std::isnan(result.tau))
        << "constant series is trivially stationary";
}

TEST_F(AdfTestDouble, DeterministicTrendDoesNotReject) {
    const int n = 2000;
    std::vector<double> trend(n);
    for (int i = 0; i < n; i++) {
        trend[i] = 100.0 + 0.1 * i;
    }

    ADFResult result;
    adf_test<double, BLOCK, EPT>(trend.data(), n, 4, &result);

    EXPECT_FALSE(result.reject_5pct)
        << "linear trend is non-stationary (constant-only ADF should not reject)";
}

TEST_F(AdfTestDouble, ZeroLags) {
    auto path = generate_stationary_ar1(2000, 0.8, 2.0, 1.0, 42);
    ADFResult result;
    adf_test<double, BLOCK, EPT>(path.data(), path.size(), 0, &result);

    EXPECT_EQ(result.lags, 0);
    EXPECT_FALSE(std::isnan(result.tau)) << "zero lags should still produce a valid result";
    EXPECT_LT(result.tau, 0.0) << "stationary series should still give negative tau";
}

TEST_F(AdfTestDouble, OneLag) {
    auto path = generate_stationary_ar1(2000, 0.8, 2.0, 1.0, 42);
    ADFResult result;
    adf_test<double, BLOCK, EPT>(path.data(), path.size(), 1, &result);

    EXPECT_EQ(result.lags, 1);
    EXPECT_LT(result.tau, -2.86) << "should reject for stationary series with 1 lag";
}

TEST_F(AdfTestDouble, RandomWalkWithDrift) {
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 1.0);
    const int n = 2000;
    std::vector<double> path(n);
    path[0] = 100.0;
    for (int i = 1; i < n; i++) {
        path[i] = path[i - 1] + 0.05 + noise(rng);
    }

    ADFResult result;
    adf_test<double, BLOCK, EPT>(path.data(), n, 4, &result);

    EXPECT_FALSE(result.reject_5pct)
        << "random walk with drift is still non-stationary";
}

TEST_F(AdfTestDouble, SineWaveRejects) {
    const int n = 2000;
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.5);
    std::vector<double> path(n);
    for (int i = 0; i < n; i++) {
        path[i] = 10.0 * sin(2.0 * M_PI * i / 50.0) + noise(rng);
    }

    ADFResult result;
    adf_test<double, BLOCK, EPT>(path.data(), n, 4, &result);

    EXPECT_TRUE(result.reject_5pct)
        << "sine wave is stationary and should reject";
}

TEST_F(AdfTestDouble, ReproducibleResults) {
    auto path = generate_stationary_ar1(1000, 0.85, 1.5, 1.0, 42);

    ADFResult result1, result2;
    adf_test<double, BLOCK, EPT>(path.data(), path.size(), 4, &result1);
    adf_test<double, BLOCK, EPT>(path.data(), path.size(), 4, &result2);

    EXPECT_DOUBLE_EQ(result1.tau, result2.tau) << "same input should produce identical tau";
    EXPECT_DOUBLE_EQ(result1.gamma, result2.gamma);
    EXPECT_DOUBLE_EQ(result1.se_gamma, result2.se_gamma);
}

TEST_F(AdfTestDouble, ManyRandomWalksFalsePositiveRate) {
    int rejections = 0;
    const int trials = 100;

    for (int seed = 0; seed < trials; seed++) {
        auto path = generate_random_walk(500, seed);
        ADFResult result;
        adf_test<double, BLOCK, EPT>(path.data(), path.size(), 4, &result);
        if (result.reject_5pct) rejections++;
    }

    double false_positive_rate = static_cast<double>(rejections) / trials;
    EXPECT_LT(false_positive_rate, 0.15)
        << "false positive rate should be near 5%, got " << false_positive_rate;
}

TEST_F(AdfTestDouble, ManyStationaryTruePositiveRate) {
    int rejections = 0;
    const int trials = 100;

    for (int seed = 0; seed < trials; seed++) {
        auto path = generate_stationary_ar1(1000, 0.8, 2.0, 1.0, seed);
        ADFResult result;
        adf_test<double, BLOCK, EPT>(path.data(), path.size(), 4, &result);
        if (result.reject_5pct) rejections++;
    }

    double true_positive_rate = static_cast<double>(rejections) / trials;
    EXPECT_GT(true_positive_rate, 0.5)
        << "should reject majority of stationary series, got " << true_positive_rate;
}

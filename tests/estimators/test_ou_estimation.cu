#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include "ou_estimation.cu"

std::vector<double> generate_ou_process(int n, double theta, double mu, double sigma, double x0, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 1.0);
    double dt = 1.0;
    std::vector<double> path(n);
    path[0] = x0;
    for (int i = 1; i < n; i++) {
        path[i] = path[i - 1] + theta * (mu - path[i - 1]) * dt + sigma * std::sqrt(dt) * noise(rng);
    }
    return path;
}

class OuEstimationDouble : public ::testing::Test {
protected:
    static constexpr int BLOCK = 256;
    static constexpr int EPT = 4;
};

TEST_F(OuEstimationDouble, RecoverParametersFromSyntheticData) {
    const double true_theta = 0.5;
    const double true_mu = 100.0;
    const double true_sigma = 2.0;
    const int n = 10000;
    const int window = 500;

    auto path = generate_ou_process(n, true_theta, true_mu, true_sigma, true_mu, 42);

    const int m = n - 1;
    std::vector<double> speed(m), equilibrium(m), vol_sq(m);
    ou_estimation<double, BLOCK, EPT>(path.data(), n, speed.data(), equilibrium.data(), vol_sq.data(), window);

    int start = window + 500;
    double theta_sum = 0, mu_sum = 0;
    int count = 0;
    for (int i = start; i < m; i++) {
        if (std::isnan(speed[i]) || std::isinf(speed[i])) continue;
        if (std::isnan(equilibrium[i]) || std::isinf(equilibrium[i])) continue;
        theta_sum += speed[i];
        mu_sum += equilibrium[i];
        count++;
    }

    if (count > 0) {
        double avg_theta = theta_sum / count;
        double avg_mu = mu_sum / count;
        EXPECT_NEAR(avg_theta, true_theta, 0.3)
            << "estimated theta should approximate true value";
        EXPECT_NEAR(avg_mu, true_mu, 10.0)
            << "estimated mu should approximate true value";
    }
}

TEST_F(OuEstimationDouble, StrongMeanReversionDetected) {
    const double theta = 2.0;
    const double mu = 50.0;
    const double sigma = 1.0;
    const int n = 5000;
    const int window = 200;

    auto path = generate_ou_process(n, theta, mu, sigma, mu, 77);

    const int m = n - 1;
    std::vector<double> speed(m), equilibrium(m), vol_sq(m);
    ou_estimation<double, BLOCK, EPT>(path.data(), n, speed.data(), equilibrium.data(), vol_sq.data(), window);

    int positive_theta = 0, total = 0;
    for (int i = window; i < m; i++) {
        if (std::isnan(speed[i]) || std::isinf(speed[i])) continue;
        if (speed[i] > 0) positive_theta++;
        total++;
    }

    if (total > 0) {
        double frac = static_cast<double>(positive_theta) / total;
        EXPECT_GT(frac, 0.8) << "strong mean reversion should give positive theta most of the time";
    }
}

TEST_F(OuEstimationDouble, WeakMeanReversionSmallTheta) {
    const double theta = 0.01;
    const double mu = 100.0;
    const double sigma = 5.0;
    const int n = 10000;
    const int window = 500;

    auto path = generate_ou_process(n, theta, mu, sigma, mu, 99);

    const int m = n - 1;
    std::vector<double> speed(m), equilibrium(m), vol_sq(m);
    ou_estimation<double, BLOCK, EPT>(path.data(), n, speed.data(), equilibrium.data(), vol_sq.data(), window);

    double theta_sum = 0;
    int count = 0;
    for (int i = window + 200; i < m; i++) {
        if (std::isnan(speed[i]) || std::isinf(speed[i])) continue;
        theta_sum += speed[i];
        count++;
    }

    if (count > 0) {
        double avg_theta = theta_sum / count;
        EXPECT_LT(std::abs(avg_theta), 0.5)
            << "weak mean reversion should give small theta";
    }
}

TEST_F(OuEstimationDouble, RandomWalkThetaNearZero) {
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 1.0);

    const int n = 5000;
    const int window = 200;
    std::vector<double> path(n);
    path[0] = 100.0;
    for (int i = 1; i < n; i++) {
        path[i] = path[i - 1] + noise(rng);
    }

    const int m = n - 1;
    std::vector<double> speed(m), equilibrium(m), vol_sq(m);
    ou_estimation<double, BLOCK, EPT>(path.data(), n, speed.data(), equilibrium.data(), vol_sq.data(), window);

    double theta_sum = 0;
    int count = 0;
    for (int i = window; i < m; i++) {
        if (std::isnan(speed[i]) || std::isinf(speed[i])) continue;
        theta_sum += std::abs(speed[i]);
        count++;
    }

    if (count > 0) {
        double avg_abs_theta = theta_sum / count;
        EXPECT_LT(avg_abs_theta, 0.5)
            << "random walk should give theta near zero";
    }
}

TEST_F(OuEstimationDouble, EquilibriumTracksProcessMean) {
    const double mu = 75.0;
    const double theta = 0.3;
    const double sigma = 1.0;
    const int n = 8000;
    const int window = 400;

    auto path = generate_ou_process(n, theta, mu, sigma, mu, 123);

    const int m = n - 1;
    std::vector<double> speed(m), equilibrium(m), vol_sq(m);
    ou_estimation<double, BLOCK, EPT>(path.data(), n, speed.data(), equilibrium.data(), vol_sq.data(), window);

    double mu_sum = 0;
    int count = 0;
    for (int i = window + 300; i < m; i++) {
        if (std::isnan(equilibrium[i]) || std::isinf(equilibrium[i])) continue;
        mu_sum += equilibrium[i];
        count++;
    }

    if (count > 0) {
        double avg_mu = mu_sum / count;
        EXPECT_NEAR(avg_mu, mu, 15.0)
            << "estimated equilibrium should be near true mu";
    }
}

TEST_F(OuEstimationDouble, VolatilityIsPositive) {
    const int n = 5000;
    const int window = 200;

    auto path = generate_ou_process(n, 0.5, 100.0, 3.0, 100.0, 42);

    const int m = n - 1;
    std::vector<double> speed(m), equilibrium(m), vol_sq(m);
    ou_estimation<double, BLOCK, EPT>(path.data(), n, speed.data(), equilibrium.data(), vol_sq.data(), window);

    for (int i = window; i < m; i++) {
        if (std::isnan(vol_sq[i])) continue;
        EXPECT_GE(vol_sq[i], 0.0)
            << "volatility squared should be non-negative at index " << i;
    }
}

TEST_F(OuEstimationDouble, NanForIncompleteWindows) {
    const int n = 100;
    const int window = 20;

    auto path = generate_ou_process(n, 0.5, 50.0, 1.0, 50.0, 42);

    const int m = n - 1;
    std::vector<double> speed(m), equilibrium(m), vol_sq(m);
    ou_estimation<double, BLOCK, EPT>(path.data(), n, speed.data(), equilibrium.data(), vol_sq.data(), window);

    for (int i = 0; i < window - 1; i++) {
        EXPECT_TRUE(std::isnan(speed[i]) || speed[i] == 0.0)
            << "expected NaN/zero for incomplete window at index " << i;
    }
}

TEST_F(OuEstimationDouble, HigherThetaFasterReversion) {
    const int n = 8000;
    const int window = 400;
    const double mu = 100.0;
    const double sigma = 2.0;

    auto path_fast = generate_ou_process(n, 1.0, mu, sigma, mu, 42);
    auto path_slow = generate_ou_process(n, 0.1, mu, sigma, mu, 42);

    const int m = n - 1;
    std::vector<double> speed_fast(m), eq_fast(m), vol_fast(m);
    std::vector<double> speed_slow(m), eq_slow(m), vol_slow(m);

    ou_estimation<double, BLOCK, EPT>(path_fast.data(), n, speed_fast.data(), eq_fast.data(), vol_fast.data(), window);
    ou_estimation<double, BLOCK, EPT>(path_slow.data(), n, speed_slow.data(), eq_slow.data(), vol_slow.data(), window);

    double avg_fast = 0, avg_slow = 0;
    int count_fast = 0, count_slow = 0;
    for (int i = window + 200; i < m; i++) {
        if (!std::isnan(speed_fast[i]) && !std::isinf(speed_fast[i])) {
            avg_fast += speed_fast[i];
            count_fast++;
        }
        if (!std::isnan(speed_slow[i]) && !std::isinf(speed_slow[i])) {
            avg_slow += speed_slow[i];
            count_slow++;
        }
    }

    if (count_fast > 0 && count_slow > 0) {
        avg_fast /= count_fast;
        avg_slow /= count_slow;
        EXPECT_GT(avg_fast, avg_slow)
            << "faster mean reversion should give higher estimated theta";
    }
}

TEST_F(OuEstimationDouble, DifferentStartingPoints) {
    const double theta = 0.5;
    const double mu = 100.0;
    const double sigma = 2.0;
    const int n = 8000;
    const int window = 400;

    auto path_at_mu = generate_ou_process(n, theta, mu, sigma, mu, 42);
    auto path_far = generate_ou_process(n, theta, mu, sigma, mu + 50, 42);

    const int m = n - 1;
    std::vector<double> speed1(m), eq1(m), vol1(m);
    std::vector<double> speed2(m), eq2(m), vol2(m);

    ou_estimation<double, BLOCK, EPT>(path_at_mu.data(), n, speed1.data(), eq1.data(), vol1.data(), window);
    ou_estimation<double, BLOCK, EPT>(path_far.data(), n, speed2.data(), eq2.data(), vol2.data(), window);

    double mu1_sum = 0, mu2_sum = 0;
    int c1 = 0, c2 = 0;
    for (int i = m - 500; i < m; i++) {
        if (!std::isnan(eq1[i]) && !std::isinf(eq1[i])) { mu1_sum += eq1[i]; c1++; }
        if (!std::isnan(eq2[i]) && !std::isinf(eq2[i])) { mu2_sum += eq2[i]; c2++; }
    }

    if (c1 > 0 && c2 > 0) {
        EXPECT_NEAR(mu1_sum / c1, mu2_sum / c2, 20.0)
            << "equilibrium estimates should converge regardless of starting point";
    }
}

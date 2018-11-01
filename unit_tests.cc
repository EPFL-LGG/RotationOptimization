#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "rotation_optimization.hh"
#include <cstdlib>

// random value in [0, 1]
double randDouble() {
    return double(random()) / RAND_MAX;
}

using namespace rotation_optimization;

// Test rotations around all three axes of vectors along all three axes;
TEST_CASE("Rotation test", "[rotation test]" ) {
    M3d I(M3d::Identity());
    for (size_t i = 0; i < 3; ++i) {
        V3d a = I.col(i);
        for (size_t j = 0; j < 3; ++j) {
            V3d v = I.col(j);
            for (size_t t = 0; t < 10000; ++t) {
                double theta = M_PI * (2 * randDouble() - 1.0); // try random angles in the interval [-pi, pi]
                auto vrot = rotated_vector(theta * a, v); 
                V3d rot_ground_truth = Eigen::AngleAxisd(theta, a) * v;
                REQUIRE((vrot - rot_ground_truth).norm() < 1e-15);
            }
        }
    }
}

M3d finite_diff_gradient(const V3d &w, const V3d &v, const double eps) {
    M3d I(M3d::Identity());
    M3d result;
    for (size_t j = 0; j < 3; ++j) result.col(j) = (0.5 / eps) * (rotated_vector(w + eps * I.col(j), v) - rotated_vector(w - eps * I.col(j), v));
    return result;
}

void finite_diff_hessian(const V3d &w, const V3d &v,
                         std::array<Eigen::Ref<M3d>, 3> hess_comp, const double eps) {
    M3d I(M3d::Identity());
    for (size_t j = 0; j < 3; ++j) {
        M3d  gdiff = (0.5 / eps) * (grad_rotated_vector(w + eps * I.col(j), v) -
                                    grad_rotated_vector(w - eps * I.col(j), v));
        for (size_t i = 0; i < 3; ++i) {
            for (size_t k = 0; k < 3; ++k)
                hess_comp[i](j, k) = gdiff(i, k);
        }
    }
}

// Exhaustively test gradient and Hessian using finite difference:
// By linearity, we can test on the three canonical basis vectors.
//
// We test the formulas exactly at the identity (w = 0)     (where a simpler formula is used)
// Near the identity with many random axes (||w|| << 1)     (where Taylor expansions are used to avoid approximating 0/0)
// Farther away from the identity with random axes/angles   (where the full formula with trig functions is evaluated)
TEST_CASE("gradient tests", "[gradients]" ) {
    M3d I(M3d::Identity());
    const double eps = 1e-7;

    SECTION("Variation around identity") {
        for (size_t i = 0; i < 3; ++i) {
            auto g_fd = finite_diff_gradient(V3d::Zero(), I.col(i), eps);
            auto g    = grad_rotated_vector(V3d::Zero(), I.col(i));

            // std::cout << g << std::endl << std::endl;
            // std::cout << g_fd << std::endl << std::endl;

            REQUIRE((g - g_fd).cwiseAbs().maxCoeff() / g_fd.norm() < 1e-14);
        }
    }
    SECTION("Variation around small rotations") {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t t = 0; t < 10000; ++t) {
                V3d a(V3d::Random()); // random vector in [-1, 1]^3
                double theta = 1e-6 * (2 * randDouble() - 1.0); // try random angles in the interval [-1e-6, 1e-6]
                V3d w = theta * (a / a.norm());
                auto g_fd = finite_diff_gradient(w, I.col(i), eps);
                auto g    =  grad_rotated_vector(w, I.col(i));
                REQUIRE((g - g_fd).cwiseAbs().maxCoeff() / g_fd.norm() < 1e-9);
            }
        }
    }
    SECTION("Variation around large rotations") {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t t = 0; t < 10000; ++t) {
                V3d a(V3d::Random()); // random vector in [-1, 1]^3
                double theta = 0.9 * M_PI * (2 * randDouble() - 1.0); // try random angles in the interval [-0.9 pi, 0.9 pi]
                V3d w = theta * (a / a.norm());
                auto g_fd = finite_diff_gradient(w, I.col(i), eps);
                auto g    =  grad_rotated_vector(w, I.col(i));
                REQUIRE((g - g_fd).cwiseAbs().maxCoeff() / g_fd.norm() < 1e-7);
            }
        }
    }
}

TEST_CASE("hessian tests", "[hessians]" ) {
    M3d I(M3d::Identity());
    Eigen::Matrix<double, 9, 3> H_fd, H;
    std::array<Eigen::Ref<M3d>, 3> fd_hess_comp{{ H_fd.block<3, 3>(0, 0), H_fd.block<3, 3>(3, 0), H_fd.block<3, 3>(6, 0) }};
    std::array<Eigen::Ref<M3d>, 3>    hess_comp{{ H   .block<3, 3>(0, 0), H   .block<3, 3>(3, 0), H   .block<3, 3>(6, 0) }};
    const double eps = 5e-6;

    SECTION("Variation around identity") {
        for (size_t i = 0; i < 3; ++i) {
            finite_diff_hessian(V3d::Zero(), I.col(i), fd_hess_comp, eps);
            hess_rotated_vector(V3d::Zero(), I.col(i),    hess_comp);

            REQUIRE((H - H_fd).cwiseAbs().maxCoeff() / H_fd.norm() < 1e-10);
        }
    }

    SECTION("Variation around small rotation") {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t t = 0; t < 10000; ++t) {
                V3d a(V3d::Random()); // random vector in [-1, 1]^3
                double theta = 1e-6 * (2 * randDouble() - 1.0); // try random angles in the interval [-1e-6, 1e-6]
                V3d w = theta * (a / a.norm());
                finite_diff_hessian(w, I.col(i), fd_hess_comp, eps);
                hess_rotated_vector(w, I.col(i),    hess_comp);

                REQUIRE((H - H_fd).cwiseAbs().maxCoeff() / H_fd.norm() < 1e-9);
            }
        }
    }

    SECTION("Variation around large rotation") {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t t = 0; t < 10000; ++t) {
                V3d a(V3d::Random()); // random vector in [-1, 1]^3
                double theta = 0.9 * M_PI * (2 * randDouble() - 1.0); // try random angles in the interval [-0.9 pi, 0.9 pi]
                V3d w = theta * (a / a.norm());
                finite_diff_hessian(w, I.col(i), fd_hess_comp, eps);
                hess_rotated_vector(w, I.col(i),    hess_comp);

                REQUIRE((H - H_fd).cwiseAbs().maxCoeff() / H_fd.norm() < 2e-8);
            }
        }
    }
}

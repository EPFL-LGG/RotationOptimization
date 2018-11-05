#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "rotation_optimization.hh"
#include <cstdlib>

// random value in [0, 1]
double randDouble() {
    return double(random()) / RAND_MAX;
}

using namespace rotation_optimization;

V3d randomAxisAngle(double magnitude) {
    V3d a(V3d::Random()); // random vector in [-1, 1]^3
    double theta = magnitude * (2 * randDouble() - 1.0); // try random angles in the interval [-magnitude, magnitude]
    return theta * (a / a.norm());
}

////////////////////////////////////////////////////////////////////////////////
// Test Rotated Vectors
////////////////////////////////////////////////////////////////////////////////
// Test rotations around all three axes of vectors along all three axes;
TEST_CASE("Rotated vector test", "[rotated vector]" ) {
    M3d I(M3d::Identity());
    for (size_t j = 0; j < 3; ++j) {
        V3d v = I.col(j);
        for (size_t t = 0; t < 10000; ++t) {
            auto w = randomAxisAngle(M_PI); // try random axes and rotation angles in the interval [-pi, pi]
            auto vrot = rotated_vector(w, v);
            V3d rot_ground_truth = Eigen::AngleAxisd(w.norm(), w.normalized()) * v;
            REQUIRE((vrot - rot_ground_truth).norm() < 1e-15);
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
TEST_CASE("Grad rotated vector tests", "[grad rotated vector]" ) {
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
                auto w = randomAxisAngle(1e-6);
                auto g_fd = finite_diff_gradient(w, I.col(i), eps);
                auto g    =  grad_rotated_vector(w, I.col(i));
                REQUIRE((g - g_fd).cwiseAbs().maxCoeff() / g_fd.norm() < 1e-9);
            }
        }
    }
    SECTION("Variation around large rotations") {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t t = 0; t < 10000; ++t) {
                auto w = randomAxisAngle(0.9 * M_PI);
                auto g_fd = finite_diff_gradient(w, I.col(i), eps);
                auto g    =  grad_rotated_vector(w, I.col(i));
                REQUIRE((g - g_fd).cwiseAbs().maxCoeff() / g_fd.norm() < 1e-7);
            }
        }
    }
}

TEST_CASE("Hessian rotated vector tests", "[Hessian rotated vector]" ) {
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
                auto w = randomAxisAngle(1e-6);
                finite_diff_hessian(w, I.col(i), fd_hess_comp, eps);
                hess_rotated_vector(w, I.col(i),    hess_comp);

                REQUIRE((H - H_fd).cwiseAbs().maxCoeff() / H_fd.norm() < 1e-9);
            }
        }
    }

    SECTION("Variation around large rotation") {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t t = 0; t < 10000; ++t) {
                auto w = randomAxisAngle(0.9 * M_PI);
                finite_diff_hessian(w, I.col(i), fd_hess_comp, 1e-5);
                hess_rotated_vector(w, I.col(i),    hess_comp);

                REQUIRE((H - H_fd).cwiseAbs().maxCoeff() / H_fd.norm() < 2e-8);
            }
        }
    }

    SECTION("Storage-backed interface") {
        std::array<M3d, 3> hess_comp_storage;
        for (size_t i = 0; i < 3; ++i) {
            for (size_t t = 0; t < 10000; ++t) {
                auto w = randomAxisAngle(0.9 * M_PI);
                hess_rotated_vector(w, I.col(i), hess_comp);
                hess_rotated_vector(w, I.col(i), hess_comp_storage);
                double diff = 0;
                for (size_t c = 0; c < 3; ++c)
                    diff += (hess_comp[c] - hess_comp_storage[c]).norm();
                REQUIRE(diff == 0);
            }
        }
    }
}

double norm(const Eigen::Tensor<double, 3> &a) {
    Eigen::array<Eigen::IndexPair<int>, 3> contraction_indices{{Eigen::IndexPair<int>(0, 0), Eigen::IndexPair<int>(1, 1), Eigen::IndexPair<int>(2, 2)}};
    Eigen::Tensor<double, 0> sumSquared = a.contract(a, contraction_indices);
    return std::sqrt(sumSquared(0));
}

double norm(const Eigen::Tensor<double, 4> &a) {
    Eigen::array<Eigen::IndexPair<int>, 4> contraction_indices{{Eigen::IndexPair<int>(0, 0), Eigen::IndexPair<int>(1, 1), Eigen::IndexPair<int>(2, 2), Eigen::IndexPair<int>(3, 3)}};
    Eigen::Tensor<double, 0> sumSquared = a.contract(a, contraction_indices);
    return std::sqrt(sumSquared(0));
}

template<int N>
double relError(const Eigen::Tensor<double, N> &a, const Eigen::Tensor<double, N> &b) {
    Eigen::Tensor<double, 0> absMax = (a - b).abs().maximum();
    return absMax(0) / norm(b);
}

////////////////////////////////////////////////////////////////////////////////
// Test Rotated Matrices
////////////////////////////////////////////////////////////////////////////////
template<class Mat>
Eigen::Tensor<double, 3> finite_diff_rotated_matrix_gradient(const V3d &w, const Mat &A, const double eps) {
    M3d I(M3d::Identity());
    const int ncols = A.cols();
    Eigen::Tensor<double, 3> result(3, ncols, 3);
    for (size_t k = 0; k < 3; ++k) {
        auto fdslice = ((0.5 / eps) * (rotated_matrix(w + eps * I.col(k), A) - rotated_matrix(w - eps * I.col(k), A))).eval();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < ncols; ++j)
                result(i, j, k) = fdslice(i, j);
    }
    return result;
}

template<class Mat>
Eigen::Tensor<double, 4> finite_diff_rotated_matrix_hessian(const V3d &w, const Mat &A, const double eps) {
    M3d I(M3d::Identity());
    const int ncols = A.cols();
    Eigen::Tensor<double, 4> result(3, ncols, 3, 3);
    for (size_t l = 0; l < 3; ++l) {
        Eigen::Tensor<double, 3>  fdslice = (0.5 / eps) * (grad_rotated_matrix(w + eps * I.col(l), A) -
                                                           grad_rotated_matrix(w - eps * I.col(l), A));
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < ncols; ++j) {
                for (int k = 0; k < 3; ++k)
                    result(i, j, k, l) = fdslice(i, j, k);
            }
        }
    }
    return result;
}

template<int N, class Test>
void run_rotated_matrix_test(const Test &the_test, double angleMag, int ncols = N) {
    Eigen::Matrix<double, 3, N> A(3, ncols);

    for (int mat_choice = 0; mat_choice < 100; ++mat_choice) {
        A.setRandom(); // Random components in [-1, 1]
        if (angleMag == 0)
            the_test(V3d::Zero(), A);
        else {
            for (int t = 0; t < 1000; ++t)
                the_test(randomAxisAngle(angleMag), A);
        }
    }
}

TEST_CASE("Rotated matrix test", "[rotated matrix]" ) {
    auto the_test = [](const V3d &w, const auto &A) {
        auto rot              = rotated_matrix(w, A);
        auto rot_ground_truth = Eigen::AngleAxisd(w.norm(), w.normalized()).matrix() * A;
        REQUIRE((rot - rot_ground_truth).norm() < 1e-9);
    };
    run_rotated_matrix_test<             1>(the_test, M_PI);
    run_rotated_matrix_test<             2>(the_test, M_PI);
    run_rotated_matrix_test<             3>(the_test, M_PI);
    run_rotated_matrix_test<             4>(the_test, M_PI);
    run_rotated_matrix_test<Eigen::Dynamic>(the_test, M_PI, 1);
    run_rotated_matrix_test<Eigen::Dynamic>(the_test, M_PI, 2);
    run_rotated_matrix_test<Eigen::Dynamic>(the_test, M_PI, 3);
    run_rotated_matrix_test<Eigen::Dynamic>(the_test, M_PI, 4);
}

TEST_CASE("Grad rotated matrix test", "[grad rotated matrix]" ) {
    const double eps = 5e-6;

    double tolerance = 0;
    bool verbose = false;
    auto the_test = [&](const V3d &w, const auto &A) {
        auto g    = grad_rotated_matrix(w, A);
        auto g_fd = finite_diff_rotated_matrix_gradient(w, A, eps);
        if (verbose) {
            std::cout << "analytic: \n" << g << std::endl << std::endl;
            std::cout << "fte diff: \n" << g_fd << std::endl << std::endl;
        }
        REQUIRE(relError(g, g_fd) < tolerance);
    };

    SECTION("Variation around identity") {
        tolerance = 1e-9;
        run_rotated_matrix_test<1>(the_test, 0);
        run_rotated_matrix_test<2>(the_test, 0);
        run_rotated_matrix_test<3>(the_test, 0);
        run_rotated_matrix_test<4>(the_test, 0);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 0, 1);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 0, 2);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 0, 3);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 0, 4);
    }
    SECTION("Variation around small rotations") {
        tolerance = 1e-8;
        run_rotated_matrix_test<             1>(the_test, 1e-6);
        run_rotated_matrix_test<             2>(the_test, 1e-6);
        run_rotated_matrix_test<             3>(the_test, 1e-6);
        run_rotated_matrix_test<             4>(the_test, 1e-6);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 1e-6, 1);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 1e-6, 2);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 1e-6, 3);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 1e-6, 4);
    }
    SECTION("Variation around large rotations") {
        tolerance = 2e-8;
        run_rotated_matrix_test<             1>(the_test, 0.9 * M_PI);
        run_rotated_matrix_test<             2>(the_test, 0.9 * M_PI);
        run_rotated_matrix_test<             3>(the_test, 0.9 * M_PI);
        run_rotated_matrix_test<             4>(the_test, 0.9 * M_PI);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 0.9 * M_PI, 1);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 0.9 * M_PI, 2);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 0.9 * M_PI, 3);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 0.9 * M_PI, 4);
    }
}

TEST_CASE("Hessian rotated matrix test", "[Hessian rotated matrix]" ) {
    double eps = 5e-6;

    double tolerance = 0;
    bool verbose = false;
    auto the_test = [&](const V3d &w, const auto &A) {
        auto H    = hess_rotated_matrix(w, A);
        auto H_fd = finite_diff_rotated_matrix_hessian(w, A, eps);
        if (verbose) {
            std::cout << "analytic: \n" << H << std::endl << std::endl;
            std::cout << "fte diff: \n" << H_fd << std::endl << std::endl;
        }
        REQUIRE(relError(H, H_fd) < tolerance);
    };

    SECTION("Variation around identity: static") {
        tolerance = 1e-9;
        run_rotated_matrix_test<1>(the_test, 0);
        run_rotated_matrix_test<2>(the_test, 0);
        run_rotated_matrix_test<3>(the_test, 0);
    }
    SECTION("Variation around identity: dynamic") {
        tolerance = 1e-9;
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 0, 1);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 0, 2);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 0, 3);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 0, 4);
    }
    SECTION("Variation around small rotations") {
        tolerance = 2e-8;
        run_rotated_matrix_test<             1>(the_test, 1e-6);
        run_rotated_matrix_test<             2>(the_test, 1e-6);
        run_rotated_matrix_test<             3>(the_test, 1e-6);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 1e-6, 1);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 1e-6, 2);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 1e-6, 3);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 1e-6, 4);
    }
    SECTION("Variation around large rotations") {
        eps = 1e-5;
        tolerance = 2e-8;
        run_rotated_matrix_test<             1>(the_test, 0.9 * M_PI);
        run_rotated_matrix_test<             2>(the_test, 0.9 * M_PI);
        run_rotated_matrix_test<             3>(the_test, 0.9 * M_PI);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 0.9 * M_PI, 1);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 0.9 * M_PI, 2);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 0.9 * M_PI, 3);
        run_rotated_matrix_test<Eigen::Dynamic>(the_test, 0.9 * M_PI, 4);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Test Rotation Matrices
////////////////////////////////////////////////////////////////////////////////
Eigen::Tensor<double, 3> finite_diff_rotation_gradient(const V3d &w, const double eps) {
    M3d I(M3d::Identity());
    Eigen::Tensor<double, 3> result(3, 3, 3);
    for (size_t k = 0; k < 3; ++k) {
        M3d fdslice = (0.5 / eps) * (rotation_matrix(w + eps * I.col(k)) - rotation_matrix(w - eps * I.col(k)));
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                result(i, j, k) = fdslice(i, j);
    }
    return result;
}

Eigen::Tensor<double, 4> finite_diff_rotation_hessian(const V3d &w, const double eps) {
    M3d I(M3d::Identity());
    Eigen::Tensor<double, 4> result(3, 3, 3, 3);
    for (size_t l = 0; l < 3; ++l) {
        Eigen::Tensor<double, 3>  fdslice = (0.5 / eps) * (grad_rotation_matrix(w + eps * I.col(l)) -
                                                           grad_rotation_matrix(w - eps * I.col(l)));
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                for (size_t k = 0; k < 3; ++k)
                    result(i, j, k, l) = fdslice(i, j, k);
            }
        }
    }
    return result;
}

TEST_CASE("Rotation matrix test", "[rotation matrix]" ) {
    for (size_t t = 0; t < 10000; ++t) {
        auto w = randomAxisAngle(M_PI); // try random axes and rotation angles in the interval [-pi, pi]
        M3d rot = rotation_matrix(w);
        M3d rot_ground_truth = Eigen::AngleAxisd(w.norm(), w.normalized()).matrix();
        REQUIRE((rot - rot_ground_truth).norm() < 1e-10);
    }
}

TEST_CASE("Grad rotation matrix test", "[grad rotation matrix]" ) {
    const double eps = 1e-7;

    SECTION("Variation around identity") {
        auto g_fd = finite_diff_rotation_gradient(V3d::Zero(), eps);
        auto g    = grad_rotation_matrix(V3d::Zero());

        REQUIRE(relError(g, g_fd) < 1e-14);
    }
    SECTION("Variation around small rotations") {
        for (size_t t = 0; t < 10000; ++t) {
            auto w = randomAxisAngle(1e-6);
            auto g_fd = finite_diff_rotation_gradient(w, eps);
            auto g    = grad_rotation_matrix(w);
            REQUIRE(relError(g, g_fd) < 1e-9);
        }
    }
    SECTION("Variation around large rotations") {
        for (size_t t = 0; t < 10000; ++t) {
            auto w = randomAxisAngle(0.9 * M_PI);
            auto g_fd = finite_diff_rotation_gradient(w, eps);
            auto g    = grad_rotation_matrix(w);

            REQUIRE(relError(g, g_fd) < 2e-8);
        }
    }
}

TEST_CASE("Hessian rotation matrix test", "[Hessian rotation matrix]" ) {
    const double eps = 5e-6;

    SECTION("Variation around identity") {
        auto H_fd = finite_diff_rotation_hessian(V3d::Zero(), eps);
        auto H    = hess_rotation_matrix(V3d::Zero());

        // std::cout << H    << std::endl << std::endl;
        // std::cout << H_fd << std::endl << std::endl;
        // std::cout << "relError: " << relError(H, H_fd) << std::endl;

        REQUIRE(relError(H, H_fd) < 1e-11);
    }
    SECTION("Variation around small rotations") {
        for (size_t t = 0; t < 10000; ++t) {
            auto w = randomAxisAngle(1e-6);
            auto H_fd = finite_diff_rotation_hessian(w, eps);
            auto H    = hess_rotation_matrix(w);

            // std::cout << H    << std::endl << std::endl;
            // std::cout << H_fd << std::endl << std::endl;
            // std::cout << "relError: " << relError(H, H_fd) << std::endl;

            REQUIRE(relError(H, H_fd) < 1e-9);
        }
    }

    SECTION("Variation around large rotations") {
        for (size_t t = 0; t < 10000; ++t) {
            auto w = randomAxisAngle(0.9 * M_PI);
            auto H_fd = finite_diff_rotation_hessian(w, eps);
            auto H    = hess_rotation_matrix(w);

            // std::cout << H    << std::endl << std::endl;
            // std::cout << H_fd << std::endl << std::endl;
            // std::cout << "w: " << w.transpose() << std::endl;
            // std::cout << "relError: " << relError(H, H_fd) << std::endl;

            REQUIRE(relError(H, H_fd) < 2e-8);
        }
    }
}

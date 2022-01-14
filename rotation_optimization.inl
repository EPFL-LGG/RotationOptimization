#include "rotation_optimization.hh"
#include <cmath>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
// Numerically robust formulas for the trig expressions in our formulas
////////////////////////////////////////////////////////////////////////////////
// Choose a good tradeoff between catastrophic cancellation in the direct
// calculation and truncation error in the Taylor series approximation.
constexpr double theta_sq_crossover_threshold = 2e-6;

// cos(theta); needed for robustness when Real_ is an automatic differentiation type
template<typename Real_>
Real_ cos(Real_ theta, Real_ theta_sq) {
    if (theta_sq < theta_sq_crossover_threshold) { return 1.0 - 0.5 * theta_sq + theta_sq * theta_sq / 24; }
    return cos(theta);
}

// sin(theta) / theta
template<typename Real_>
Real_ sinc(Real_ theta, Real_ theta_sq) {
    if (theta_sq < theta_sq_crossover_threshold) { return 1.0 - theta_sq / 6.0; }
    return sin(theta) / theta;
}

// (1 - cos(theta)) / theta^2
template<typename Real_>
Real_ one_minus_cos_div_theta_sq(Real_ theta, Real_ theta_sq) {
    if (theta_sq < theta_sq_crossover_threshold) { return 0.5 - theta_sq / 24.0; }
    return (1 - cos(theta)) / theta_sq;
}

// (theta cos(theta) - sin(theta)) / theta^3
template<typename Real_>
Real_ theta_cos_minus_sin_div_theta_cubed(Real_ theta, Real_ theta_sq) {
    if (theta_sq < theta_sq_crossover_threshold) { return -1.0 / 3.0 + theta_sq / 30.0; }
    return (theta * cos(theta) - sin(theta)) / (theta * theta_sq);
}

// (2 cos(theta) - 2 + theta sin(theta)) / theta^4
template<typename Real_>
Real_ two_cos_minus_2_plus_theta_sin_div_theta_pow_4(Real_ theta, Real_ theta_sq) {
    if (theta_sq < theta_sq_crossover_threshold) { return -1.0 / 12.0 + theta_sq / 180.0; }
    return (2 * cos(theta) - 2 + theta * sin(theta)) / (theta_sq * theta_sq);
}

// (8 + (theta^2 - 8) cos(theta) - 5 theta sin(theta)) / theta^6
template<typename Real_>
Real_ eight_plus_theta_sq_minus_eight_cos_minus_five_theta_sin_div_theta_pow_6(Real_ theta, Real_ theta_sq) {
    if (theta_sq < theta_sq_crossover_threshold) { return 1.0 / 90.0 - theta_sq / 1680.0; }
    return (8 + (theta_sq - 8) * cos(theta) - 5 * theta * sin(theta)) / (theta_sq * theta_sq * theta_sq);
}

// (3 theta cos(theta) + (theta^2 - 3) sin(theta)) / theta^5
template<typename Real_>
Real_ three_theta_cos_plus_theta_sq_minus_3_sin_div_theta_pow_5(Real_ theta, Real_ theta_sq) {
    if (theta_sq < theta_sq_crossover_threshold) { return -1.0 / 15.0 + theta_sq / 210.0; }
    return (3 * theta * cos(theta) + (theta_sq - 3) * sin(theta)) / (theta_sq * theta_sq * theta);
}

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
template<typename Real_>
auto rotation_optimization<Real_>::cross_product_matrix(const Vec3 &v) -> Mat3 {
    Mat3 result;
    result <<    0, -v[2],  v[1],
              v[2],     0, -v[0],
             -v[1],   v[0],    0;
    return result;
}

////////////////////////////////////////////////////////////////////////////////
// Rotation matrix, rotated vectors, and rotated matrices
////////////////////////////////////////////////////////////////////////////////
// Compute R(w)
template<typename Real_>
auto rotation_optimization<Real_>::rotation_matrix(const Vec3 &w) -> Mat3 {
    // When theta_sq = 0, sqrt(theta_sq) is non-differentiable, so the autodiff code will produce "NaN"s for theta's derivative.
    // However, using the taylor expansions of the output terms ignores theta (and its derivative) in the small angle case,
    // so this can safely be ignored.
    const Real_ theta_sq = w.squaredNorm();
    const Real_ theta    = sqrt(theta_sq);
    const Real_ cos_th   = cos(theta, theta_sq);

    return cos_th * Mat3::Identity() + (w * w.transpose()) * one_minus_cos_div_theta_sq(theta, theta_sq) + cross_product_matrix(w) * sinc(theta, theta_sq);
}

// Compute R(w) v
template<typename Real_>
auto rotation_optimization<Real_>::rotated_vector(const Vec3 &w, const Vec3 &v) -> Vec3 {
    const Real_ theta_sq = w.squaredNorm();
    const Real_ theta    = sqrt(theta_sq);
    const Real_ cos_th   = cos(theta, theta_sq);
    return v * cos_th + w * (w.dot(v) * one_minus_cos_div_theta_sq(theta, theta_sq)) + w.cross(v) * sinc(theta, theta_sq);
}

template<typename Real_>
template<int N>
auto rotation_optimization<Real_>::rotated_matrix(const Vec3 &w, const Eigen::Matrix<Real_, 3, N> &A) -> Eigen::Matrix<Real_, 3, N> {
    const Real_ theta_sq = w.squaredNorm();
    const Real_ theta    = sqrt(theta_sq);
    const Real_ cos_th   = cos(theta, theta_sq);

    return A * cos_th + w * (w.transpose() * A) * one_minus_cos_div_theta_sq(theta, theta_sq) - A.colwise().cross(w * sinc(theta, theta_sq));
}

////////////////////////////////////////////////////////////////////////////////
// Gradient of rotation matrix, rotated vectors, and rotated matrices
////////////////////////////////////////////////////////////////////////////////
// The gradient of R(w). This is a third order tensor:
//      g_ijk = D [R(w)]_ij / dw_k
template<typename Real_>
auto rotation_optimization<Real_>::grad_rotation_matrix(const Vec3 &w) -> Eigen::Tensor<Real_, 3> {
    Eigen::Tensor<Real_, 3> g(3, 3, 3);
    g.setZero();

    const Real_ theta_sq = w.squaredNorm();
    // Use simpler formula for variation around the identity
    // (But only if we're using a native floating point type;
    //  for autodiff types, we need the full formula).
    if ((theta_sq == 0) && (std::is_arithmetic<Real_>::value)) {
        // [e^i]_cross otimes e^i
        g(1, 2, 0) = -1; g(2, 1, 0) =  1;
        g(0, 2, 1) =  1; g(2, 0, 1) = -1;
        g(0, 1, 2) = -1; g(1, 0, 2) =  1;
        return g;
    }

    const Real_  theta = sqrt(theta_sq);
    const Real_ coeff0 = sinc(theta, theta_sq);
    const Real_ coeff1 = one_minus_cos_div_theta_sq(theta, theta_sq);
    const Real_ coeff2 = two_cos_minus_2_plus_theta_sin_div_theta_pow_4(theta, theta_sq);
    const Real_ coeff3 = theta_cos_minus_sin_div_theta_cubed(theta, theta_sq);

    // I otimes w * (-sinc(theta))
    for (int k = 0; k < 3; ++k) g(0, 0, k) = g(1, 1, k) = g(2, 2, k) = -coeff0 * w[k];
    // [e^i]_cross otimes e^i * (sinc(theta))
    g(1, 2, 0) = -coeff0; g(2, 1, 0) =  coeff0;
    g(0, 2, 1) =  coeff0; g(2, 0, 1) = -coeff0;
    g(0, 1, 2) = -coeff0; g(1, 0, 2) =  coeff0;

    // (e^i otimes w + w otimes e^i) otimes e^i * coeff1
    for (int j = 0; j < 3; ++j) {
        const Real_ val = w[j] * coeff1;
        for (int i = 0; i < 3; ++i) {
            g(i, j, i) += val;
            g(j, i, i) += val;
        }
    }

    Mat3 w_cross = cross_product_matrix(w);
    // w otimes w otimes w coeff2 + [w]_x otimes w coeff3
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                g(i, j, k) += w[i] * w[j] * w[k] * coeff2 + w_cross(i, j) * w[k] * coeff3;

    return g;
}

// Gradient of R(w) v. This is a second order tensor, returned as a 3x3 matrix:
//      g_ij = D [R(w) v]_i / dw_j
template<typename Real_>
auto rotation_optimization<Real_>::grad_rotated_vector(const Vec3 &w, const Vec3 &v) -> Mat3 {
    const Real_ theta_sq = w.squaredNorm();

    // Use simpler formula for variation around the identity
    // (But only if we're using a native floating point type;
    //  for autodiff types, we need the full formula).
    if ((theta_sq == 0) && (std::is_arithmetic<Real_>::value)) { return -cross_product_matrix(v); }

    const Real_ theta    = sqrt(theta_sq);
    const Real_ w_dot_v  = w.dot(v);

    // Mat3 result = (v * w.transpose() + cross_product_matrix(v)) * -sinc(theta, theta_sq);
    // result += (w_dot_v * Mat3::Identity() + w * v.transpose()) * one_minus_cos_div_theta_sq(theta, theta_sq);
    // result += (w * w.transpose()) * (w_dot_v * two_cos_minus_2_plus_theta_sin_div_theta_pow_4(theta, theta_sq));
    // result += (w.cross(v) * w.transpose()) * theta_cos_minus_sin_div_theta_cubed(theta, theta_sq);

    const Vec3 neg_v_sinc = v * (-sinc(theta, theta_sq));
    const Real_ coeff = one_minus_cos_div_theta_sq(theta, theta_sq);

    Mat3 result = (neg_v_sinc + w.cross(v) * theta_cos_minus_sin_div_theta_cubed(theta, theta_sq) + w * (w_dot_v * two_cos_minus_2_plus_theta_sin_div_theta_pow_4(theta, theta_sq))) * w.transpose()
                + (coeff * w) * v.transpose();
    result.diagonal().array() += w_dot_v * coeff;
    result(0, 1) += -neg_v_sinc[2];
    result(0, 2) +=  neg_v_sinc[1];
    result(1, 0) +=  neg_v_sinc[2];
    result(1, 2) += -neg_v_sinc[0];
    result(2, 0) += -neg_v_sinc[1];
    result(2, 1) +=  neg_v_sinc[0];

    return result;
}

// Gradient of R(w) A. This is a third order tensor, returned as a 3x3 matrix:
//      g_ijk = D [R(w) A]_ij / dw_k
// Could be optimized...
template<typename Real_>
template<int N>
auto rotation_optimization<Real_>::grad_rotated_matrix(const Vec3 &w, const Eigen::Matrix<Real_, 3, N> &A) -> Eigen::Tensor<Real_, 3> {
    int ncols = A.cols();
    Eigen::Tensor<Real_, 3> g(3, ncols, 3);

    // Use simpler formula for variation around the identity
    // (But only if we're using a native floating point type;
    //  for autodiff types, we need the full formula).
    const Real_ theta_sq = w.squaredNorm();
    if ((theta_sq == 0) && (std::is_arithmetic<Real_>::value)) {
        for (int j = 0; j < ncols; ++j) {
            Mat3 tmp = -cross_product_matrix(A.col(j));
            for (int i = 0; i < 3; ++i) {
                for (int k = 0; k < 3; ++k)
                    g(i, j, k) = tmp(i, k);
            }
        }
        return g;
    }

    const Real_ theta = sqrt(theta_sq);
    const Real_ coeff0 = -sinc(theta, theta_sq);
    const Real_ coeff1 = one_minus_cos_div_theta_sq(theta, theta_sq);
    const Real_ coeff2 = two_cos_minus_2_plus_theta_sin_div_theta_pow_4(theta, theta_sq);
    const Real_ coeff3 = theta_cos_minus_sin_div_theta_cubed(theta, theta_sq);

    for (int j = 0; j < ncols; ++j) {
        const auto &v = A.col(j);
        const Real_ w_dot_v  = w.dot(v);

        Mat3 tmp = (v * w.transpose() + cross_product_matrix(v)) * coeff0
                + (w_dot_v * Mat3::Identity() + w * v.transpose()) * coeff1
                + (w * w.transpose()) * (w_dot_v * coeff2)
                + (w.cross(v) * w.transpose()) * coeff3;
        for (int i = 0; i < 3; ++i) {
            for (int k = 0; k < 3; ++k)
                g(i, j, k) = tmp(i, k);
        }
    }

    return g;
}

////////////////////////////////////////////////////////////////////////////////
// Hessian of rotation matrix, rotated vectors, and rotated matrices
////////////////////////////////////////////////////////////////////////////////
// The Hessian of R(w). This is a fourth order tensor:
//      H_ijkl = d [R(w)]_ij / (dw_k dw_l)
template<typename Real_>
auto rotation_optimization<Real_>::hess_rotation_matrix(const Vec3 &w) -> Eigen::Tensor<Real_, 4> {
    Eigen::Tensor<Real_, 4> H(3, 3, 3, 3);
    H.setZero();

    const Real_ theta_sq = w.squaredNorm();
    // Use simpler formula for variation around the identity
    // (But only if we're using a native floating point type;
    //  for autodiff types, we need the full formula).
    if ((theta_sq == 0) && (std::is_arithmetic<Real_>::value)) {
        // - I otimes I
        for (int i = 0; i < 3; ++i)
            for (int k = 0; k < 3; ++k)
                H(i, i, k, k) = -1;
        // 0.5 (e^i otimes e^k otimes (e^i otimes e^k + e^k otimes e^i))
        for (int i = 0; i < 3; ++i) {
            for (int k = 0; k < 3; ++k) {
                H(i, k, i, k) += 0.5;
                H(i, k, k, i) += 0.5;
            }
        }
        return H;
    }

    const Real_ theta  = sqrt(theta_sq);
    const Real_ coeff0 = sinc                                                                     (theta, theta_sq),
                 coeff1 = theta_cos_minus_sin_div_theta_cubed                                     (theta, theta_sq),
                 coeff2 = one_minus_cos_div_theta_sq                                              (theta, theta_sq),
                 coeff3 = two_cos_minus_2_plus_theta_sin_div_theta_pow_4                          (theta, theta_sq),
                 coeff4 = eight_plus_theta_sq_minus_eight_cos_minus_five_theta_sin_div_theta_pow_6(theta, theta_sq),
                 coeff5 = three_theta_cos_plus_theta_sq_minus_3_sin_div_theta_pow_5               (theta, theta_sq);
    Mat3 w_cross = cross_product_matrix(w);

    // -(I otimes I) sinc(theta)
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k)
            H(i, i, k, k) = -coeff0;
    // (-I otimes w otimes w + w_cross otimes I) coeff1
    for (int i = 0; i < 3; ++i) {
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
                H(i, i, k, l) -= w[k] * w[l] * coeff1;
                H(i, k, l, l) += w_cross(i, k) * coeff1;
            }
        }
    }
    // ([e^i]_x otimes (e^i otimes w + w otimes e^i)) coeff1
    for (int k = 0; k < 3; ++k) {
        Real_ val = coeff1 * w[k];
        H(1, 2, 0, k) += -val; H(2, 1, 0, k) +=  val;
        H(0, 2, 1, k) +=  val; H(2, 0, 1, k) += -val;
        H(0, 1, 2, k) += -val; H(1, 0, 2, k) +=  val;

        H(1, 2, k, 0) += -val; H(2, 1, k, 0) +=  val;
        H(0, 2, k, 1) +=  val; H(2, 0, k, 1) += -val;
        H(0, 1, k, 2) += -val; H(1, 0, k, 2) +=  val;
    }

    // (e^i otimes e^k otimes (e^i otimes e^k + e^k otimes e^i)) coeff2
    for (int i = 0; i < 3; ++i) {
        for (int k = 0; k < 3; ++k) {
            H(i, k, i, k) += coeff2;
            H(i, k, k, i) += coeff2;
        }
    }

    // ((e^i otimes w + w otimes e^i) otimes (e^i otimes w + w otimes e^i) + w otimes w otimes e^i otimes e^i) coeff3
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                const Real_ val = coeff3 * w[j] * w[k];
                H(i, j, i, k) += val;
                H(i, j, k, i) += val;
                H(j, i, i, k) += val;
                H(j, i, k, i) += val;

                H(j, k, i, i) += val;
            }
        }
    }

    // (w otimes w otimes w otimes w) coeff4 - ([w]_x otimes w otimes w) coeff5
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                    H(i, j, k, l) += (w[i] * w[j] * coeff4 - w_cross(i, j) * coeff5) * w[k] * w[l];

    return H;
}

// The Hessian of R(w) v. This is a third order tensor:
//      H_ijk = d [R(w) v]_i / (dw_j dw_k)
// We output the i^th slice of this tensor
// (the Hessian of rotated vector component i) in hess_comp[i].
template<typename Real_>
void rotation_optimization<Real_>::
hess_rotated_vector(const Vec3 &w, const Vec3 &v,
                    std::array<Eigen::Ref<Mat3>, 3> hess_comp) {
    const Real_ theta_sq = w.squaredNorm();

    // Use simpler formula for variation around the identity
    // (But only if we're using a native floating point type;
    //  for autodiff types, we need the full formula).
    if ((theta_sq == 0) && (std::is_arithmetic<Real_>::value)) {
        const Vec3 half_v = 0.5 * v;
        for (int i = 0; i < 3; ++i) {
            // hess_comp[i] = -v[i] * I + 0.5 * (I.col(i) * v.transpose() + v * I.row(i));
            hess_comp[i].setZero();
            hess_comp[i].diagonal().array() = -v[i];
            hess_comp[i].row(i) += half_v.transpose();
            hess_comp[i].col(i) += half_v;
        }
        return;
    }

    const Real_ theta   = sqrt(theta_sq);
    const Real_ w_dot_v = w.dot(v);
    const Real_ coeff0 = sinc                                                                               (theta, theta_sq),
                 coeff1 = theta_cos_minus_sin_div_theta_cubed                                               (theta, theta_sq),
                 coeff2 = one_minus_cos_div_theta_sq                                                        (theta, theta_sq),
                 coeff3 = two_cos_minus_2_plus_theta_sin_div_theta_pow_4                                    (theta, theta_sq),
                 coeff4 = w_dot_v * eight_plus_theta_sq_minus_eight_cos_minus_five_theta_sin_div_theta_pow_6(theta, theta_sq),
                 coeff5 = three_theta_cos_plus_theta_sq_minus_3_sin_div_theta_pow_5                         (theta, theta_sq);
    const Mat3 coeff1_v_cross = cross_product_matrix(coeff1 * v);
    const Vec3 v_cross_w = v.cross(w);
    const Vec3 term1 = coeff2 * v + (coeff3 * w_dot_v) * w;
    const Vec3 coeff3_w = coeff3 * w;
    Mat3 coeff3_w_otimes_v = coeff3_w * v.transpose();
    for (int i = 0; i < 3; ++i) {
        // hess_comp[i] = (-coeff0 * v[i]) * I
        //              - coeff1 * (v[i] * w_otimes_w + w * v_cross.row(i) + v_cross.row(i).transpose() * w.transpose() + v_cross_w[i] * I)
        //              + coeff2 * (I.col(i) * v.transpose() + v * I.row(i))
        //              + coeff3 * (w_dot_v * (I.col(i) * w.transpose() + w * I.row(i) + w[i] * I) + w[i] * (w_otimes_v + w_otimes_v.transpose()))
        //              + (coeff4 * w[i] + coeff5 * v_cross_w[i]) * w_otimes_w;
        hess_comp[i] = ((coeff4 * w[i] + coeff5 * v_cross_w[i] - coeff1 * v[i]) * w + coeff1_v_cross.col(i) + coeff3_w[i] * v) * w.transpose()
                        - w * coeff1_v_cross.row(i)
                        + w[i] * coeff3_w_otimes_v;
        hess_comp[i].col(i) += term1;
        hess_comp[i].row(i) += term1.transpose();
        hess_comp[i].diagonal().array() += w_dot_v * coeff3_w[i] - coeff0 * v[i] - coeff1 * v_cross_w[i];
    }
}

// d_i H_ijk where H_ijk is the Hessian of R(w) v.
template<typename Real_>
Eigen::Matrix<Real_, 3, 3> rotation_optimization<Real_>::
d_contract_hess_rotated_vector(const Vec3 &w, const Vec3 &v, const Vec3 &d) {
    const Real_ theta_sq = w.squaredNorm();

    Mat3 result;
    result.setZero();

    // Use simpler formula for variation around the identity
    // (But only if we're using a native floating point type;
    //  for autodiff types, we need the full formula).
    if ((theta_sq == 0) && (std::is_arithmetic<Real_>::value)) {
        const Vec3 half_v = 0.5 * v;
        for (int i = 0; i < 3; ++i) {
            // hess_comp[i] = -v[i] * I + 0.5 * (I.col(i) * v.transpose() + v * I.row(i));
            result.diagonal().array() += -v[i] * d[i];
            result.row(i) += d[i] * half_v.transpose();
            result.col(i) += d[i] * half_v;
        }
        return result;
    }

    const Real_ theta   = sqrt(theta_sq);
    const Real_ w_dot_v = w.dot(v);
    const Real_ coeff0 = sinc                                                                               (theta, theta_sq),
                 coeff1 = theta_cos_minus_sin_div_theta_cubed                                               (theta, theta_sq),
                 coeff2 = one_minus_cos_div_theta_sq                                                        (theta, theta_sq),
                 coeff3 = two_cos_minus_2_plus_theta_sin_div_theta_pow_4                                    (theta, theta_sq),
                 coeff4 = w_dot_v * eight_plus_theta_sq_minus_eight_cos_minus_five_theta_sin_div_theta_pow_6(theta, theta_sq),
                 coeff5 = three_theta_cos_plus_theta_sq_minus_3_sin_div_theta_pow_5                         (theta, theta_sq);
    const Mat3 coeff1_v_cross = cross_product_matrix(coeff1 * v);
    const Vec3 v_cross_w = v.cross(w);
    const Vec3 term1 = coeff2 * v + (coeff3 * w_dot_v) * w;
    const Vec3 coeff3_w = coeff3 * w;
    Mat3 coeff3_w_otimes_v = coeff3_w * v.transpose();
    for (int i = 0; i < 3; ++i) {
        result += d[i] * (((coeff4 * w[i] + coeff5 * v_cross_w[i] - coeff1 * v[i]) * w + coeff1_v_cross.col(i) + coeff3_w[i] * v) * w.transpose()
                  - w * coeff1_v_cross.row(i)
                  + w[i] * coeff3_w_otimes_v);
        result.col(i) += d[i] * term1;
        result.row(i) += d[i] * term1.transpose();
        result.diagonal().array() += d[i] * (w_dot_v * coeff3_w[i] - coeff0 * v[i] - coeff1 * v_cross_w[i]);
    }
    return result;
}

// The Hessian of R(w) A for 3xN matrix A. This is a fourth order tensor:
//      H_ijkl = d [R(w) A]_ij / (dw_k dw_l)
template<typename Real_>
template<int N>
auto rotation_optimization<Real_>::hess_rotated_matrix(const Vec3 &w, const Eigen::Matrix<Real_, 3, N> &A) -> Eigen::Tensor<Real_, 4> {
    int numCols = A.cols();
    Eigen::Tensor<Real_, 4> H(3, numCols, 3, 3);
    const Real_ theta_sq = w.squaredNorm();

    // Use simpler formula for variation around the identity
    // (But only if we're using a native floating point type;
    //  for autodiff types, we need the full formula).
    Mat3 I(Mat3::Identity());
    if ((theta_sq == 0) && (std::is_arithmetic<Real_>::value)) {
        H.setZero();
        for (int j = 0; j < numCols; ++j) {
            for (int i = 0; i < 3; ++i) {
                H(i, j, 0, 0) = H(i, j, 1, 1) = H(i, j, 2, 2) = -A(i, j);
                for (int k = 0; k < 3; ++k) {
                    H(i, j, i, k) += 0.5 * A(k, j);
                    H(i, j, k, i) += 0.5 * A(k, j);
                }
            }
        }
        return H;
    }

    const Real_ theta  = sqrt(theta_sq);
    const Real_ coeff0 = sinc                                                                     (theta, theta_sq),
                 coeff1 = theta_cos_minus_sin_div_theta_cubed                                     (theta, theta_sq),
                 coeff2 = one_minus_cos_div_theta_sq                                              (theta, theta_sq),
                 coeff3 = two_cos_minus_2_plus_theta_sin_div_theta_pow_4                          (theta, theta_sq),
                 coeff4 = eight_plus_theta_sq_minus_eight_cos_minus_five_theta_sin_div_theta_pow_6(theta, theta_sq),
                 coeff5 = three_theta_cos_plus_theta_sq_minus_3_sin_div_theta_pow_5               (theta, theta_sq);
    H.setZero();
    for (int j = 0; j < numCols; ++j) { // Compute the Hessian of each rotated column vector of A
        const auto &v = A.col(j);
        const Real_ w_dot_v = w.dot(v);
        Mat3 v_cross = cross_product_matrix(v);
        Vec3 v_cross_w = v.cross(w);
        for (int i = 0; i < 3; ++i) { // Compute the Hessian of each component of the rotated column vector.
            H(i, j, 0, 0) = H(i, j, 1, 1) = H(i, j, 2, 2) = -coeff0 * v[i] - coeff1 * v_cross_w[i] + coeff3 * w_dot_v * w[i]; // Identity coefficients
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    const Real_ tmp = coeff3 * w[i] * w[k] * v[l] - coeff1 * (w[k] * v_cross(i, l));
                    H(i, j, k, l) += tmp + (w_dot_v * coeff4 * w[i] + coeff5 * v_cross_w[i] - coeff1 * v[i]) * w[k] * w[l];
                    H(i, j, l, k) += tmp;
                }
                const Real_ tmp = coeff2 * v[k] + coeff3 * w_dot_v * w[k];
                H(i, j, i, k) += tmp;
                H(i, j, k, i) += tmp;
            }
        }
    }
    return H;
}

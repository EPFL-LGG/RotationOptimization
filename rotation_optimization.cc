#include "rotation_optimization.hh"
#include <cmath>
#include <iostream>

namespace rotation_optimization {

////////////////////////////////////////////////////////////////////////////////
// Numerically robust formulas for the trig expressions in our formulas
////////////////////////////////////////////////////////////////////////////////
constexpr double theta_sq_crossover_threshold = 1e-8;
// sin(theta) / theta
inline double sinc(double theta, double theta_sq) {
    if (theta_sq < theta_sq_crossover_threshold) { return 1.0 - theta_sq / 6.0; }
    return sin(theta) / theta;
}

// (1 - cos(theta)) / theta^2
inline double one_minus_cos_div_theta_sq(double theta, double theta_sq) {
    if (theta_sq < theta_sq_crossover_threshold) { return 0.5 - theta_sq / 24.0; }
    return (1 - cos(theta)) / theta_sq;
}

// (theta cos(theta) - sin(theta)) / theta^3
inline double theta_cos_minus_sin_div_theta_cubed(double theta, double theta_sq) {
    if (theta_sq < theta_sq_crossover_threshold) { return -1.0 / 3.0 + theta_sq / 30.0; }
    return (theta * cos(theta) - sin(theta)) / (theta * theta_sq);
}

// (2 cos(theta) - 2 + theta sin(theta)) / theta^4
inline double two_cos_minus_2_plus_theta_sin_div_theta_pow_4(double theta, double theta_sq) {
    if (theta_sq < theta_sq_crossover_threshold) { return -1.0 / 12.0 + theta_sq / 180.0; }
    return (2 * cos(theta) - 2 + theta * sin(theta)) / (theta_sq * theta_sq);
}

// (8 + (theta^2 - 8) cos(theta) - 5 theta sin(theta)) / theta^6
inline double eight_plus_theta_sq_minus_eight_cos_minus_five_theta_sin_div_theta_pow_6(double theta, double theta_sq) {
    if (theta_sq < theta_sq_crossover_threshold) { return 1.0 / 90.0 - theta_sq / 1680.0; }
    return (8 + (theta_sq - 8) * cos(theta) - 5 * theta * sin(theta)) / (theta_sq * theta_sq * theta_sq);
}

// (3 theta cos(theta) + (theta^2 - 3) sin(theta)) / theta^5
inline double three_theta_cos_plus_theta_sq_minus_3_sin_div_theta_pow_5(double theta, double theta_sq) {
    if (theta_sq < theta_sq_crossover_threshold) { return -1.0 / 15.0 + theta_sq / 210.0; }
    return (3 * theta * cos(theta) + (theta_sq - 3) * sin(theta)) / (theta_sq * theta_sq * theta);
}

////////////////////////////////////////////////////////////////////////////////
// Rotation formula and its gradient/Hessian
////////////////////////////////////////////////////////////////////////////////
// Compute R(w) v
V3d rotated_vector(const V3d &w, const V3d &v) {
    const double theta_sq = w.squaredNorm();
    const double theta    = std::sqrt(theta_sq);
    const double cos_th = std::cos(theta);
    return v * cos_th + w * ((w.dot(v)) * one_minus_cos_div_theta_sq(theta, theta_sq)) + w.cross(v) * sinc(theta, theta_sq);
}

M3d cross_product_matrix(const V3d &v) {
    M3d result;
    result <<    0, -v[2],  v[1],
              v[2],     0, -v[0],
             -v[1],   v[0],    0;
    return result;
}

// Gradient of R(w) v; this is a second order tensor, returned as a 3x3 matrix:
//      G_ij = D [R(w) v]_i / dw_j
M3d grad_rotated_vector(const V3d &w, const V3d &v) {
    const double theta_sq = w.squaredNorm();

    // Use simpler formula for variation around the identity
    if (theta_sq == 0) { return -cross_product_matrix(v); }

    const double theta    = std::sqrt(theta_sq);
    const double w_dot_v  = w.dot(v);

    M3d result = (v * w.transpose() + cross_product_matrix(v)) * -sinc(theta, theta_sq);
    result += (w_dot_v * M3d::Identity() + w * v.transpose()) * one_minus_cos_div_theta_sq(theta, theta_sq);
    result += (w * w.transpose()) * (w_dot_v * two_cos_minus_2_plus_theta_sin_div_theta_pow_4(theta, theta_sq));
    result += (w.cross(v) * w.transpose()) * theta_cos_minus_sin_div_theta_cubed(theta, theta_sq);
    return result;
}

// The Hessian of R(w) v; this is a third order tensor:
//      H_ijk = d [R(w) v]_i / (dw_j dw_k)
// We output the i^th slice of this tensor (the Hessian of rotated vector
// component i) in hess_comp[i].
void hess_rotated_vector(const V3d &w, const V3d &v,
                         std::array<Eigen::Ref<M3d>, 3> hess_comp) {
    const double theta_sq = w.squaredNorm();

    // Use simpler formula for variation around the identity
    M3d I(M3d::Identity());
    if (theta_sq == 0) {
        for (size_t i = 0; i < 3; ++i)
            hess_comp[i] = -v[i] * I + 0.5 * (I.col(i) * v.transpose() + v * I.row(i));
        return;
    }

    const double theta   = std::sqrt(theta_sq);
    const double w_dot_v = w.dot(v);
    M3d w_otimes_w = w * w.transpose();
    M3d w_otimes_v = w * v.transpose();
    const double coeff0 = sinc                                                                               (theta, theta_sq),
                 coeff1 = theta_cos_minus_sin_div_theta_cubed                                                (theta, theta_sq),
                 coeff2 = one_minus_cos_div_theta_sq                                                         (theta, theta_sq),
                 coeff3 = two_cos_minus_2_plus_theta_sin_div_theta_pow_4                                     (theta, theta_sq),
                 coeff4 = w_dot_v * eight_plus_theta_sq_minus_eight_cos_minus_five_theta_sin_div_theta_pow_6 (theta, theta_sq),
                 coeff5 = three_theta_cos_plus_theta_sq_minus_3_sin_div_theta_pow_5                          (theta, theta_sq);
    M3d v_cross = cross_product_matrix(v);
    V3d v_cross_w = v.cross(w);
    for (size_t i = 0; i < 3; ++i) {
        hess_comp[i] = (-coeff0 * v[i]) * I
                     - coeff1 * (v[i] * w_otimes_w + w * v_cross.row(i) + v_cross.row(i).transpose() * w.transpose() + v_cross_w[i] * I)
                     + coeff2 * (I.col(i) * v.transpose() + v * I.row(i))
                     + coeff3 * (w_dot_v * (I.col(i) * w.transpose() + w * I.row(i) + w[i] * I) + w[i] * (w_otimes_v + w_otimes_v.transpose()))
                     + (coeff4 * w[i] + coeff5 * v_cross_w[i]) * w_otimes_w;
    }
}

}

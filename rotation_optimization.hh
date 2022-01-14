////////////////////////////////////////////////////////////////////////////////
// rotation_optimization.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Useful functions for optimizing over the space of rotations:
//  We use the tangent space to SO(3) at a reference rotation as the optimization
//  domain and provide functions to apply the represented rotation to a given
//  vector and compute gradients/Hessians of the rotation matrix/rotated vector.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  11/01/2018 11:38:57
////////////////////////////////////////////////////////////////////////////////
#ifndef ROTATION_OPTIMIZATION_HH
#define ROTATION_OPTIMIZATION_HH
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <array>

template<typename Real_>
struct rotation_optimization {
    using Vec3 = Eigen::Matrix<Real_, 3, 1>;
    using Mat3 = Eigen::Matrix<Real_, 3, 3>;

    ////////////////////////////////////////////////////////////////////////////////
    // Rotation matrix, rotated vectors, and rotated matrices
    ////////////////////////////////////////////////////////////////////////////////
    // Compute R(w)
    static Mat3 rotation_matrix(const Vec3 &w);

    // Compute R(w) v
    static Vec3 rotated_vector(const Vec3 &w, const Vec3 &v);

    // Compute R(w) A
    template<int N>
    static Eigen::Matrix<Real_, 3, N> rotated_matrix(const Vec3 &w, const Eigen::Matrix<Real_, 3, N> &A);

    ////////////////////////////////////////////////////////////////////////////////
    // Gradient of rotation matrix, rotated vectors, and rotated matrices
    ////////////////////////////////////////////////////////////////////////////////
    // Gradient of R(w). This is a third order tensor:
    //      g_ijk = D [R(w)]_ij / dw_k
    static Eigen::Tensor<Real_, 3> grad_rotation_matrix(const Vec3 &w);

    // Gradient of R(w) v. This is a second order tensor, returned as a 3x3 matrix:
    //      g_ij = D [R(w) v]_i / dw_j
    static Mat3 grad_rotated_vector(const Vec3 &w, const Vec3 &v);

    // Gradient of R(w) A. This is a third order tensor:
    //      g_ijk = D [R(w) A]_ij / dw_k
    template<int N>
    static Eigen::Tensor<Real_, 3> grad_rotated_matrix(const Vec3 &w, const Eigen::Matrix<Real_, 3, N> &A);

    ////////////////////////////////////////////////////////////////////////////////
    // Hessian of rotation matrix, rotated vectors, and rotated matrices
    ////////////////////////////////////////////////////////////////////////////////
    // The Hessian of R(w). this is a fourth order tensor:
    //      H_ijkl = d [R(w)]_ij / (dw_k dw_l)
    static Eigen::Tensor<Real_, 4> hess_rotation_matrix(const Vec3 &w);

    // Hessian of R(w) v. This is a third order tensor:
    //      H_ijk = d [R(w) v]_i / (dw_j dw_k)
    // We output the i^th slice of this tensor (the Hessian of rotated vector
    // component i) in hess_comp[i].
    static void hess_rotated_vector(const Vec3 &w, const Vec3 &v, std::array<Eigen::Ref<Mat3>, 3> hess_comp);

    // Hessian of R(w) v. This is a third order tensor:
    //      H_ijk = d [R(w) v]_i / (dw_j dw_k)
    // We output the i^th slice of this tensor (the Hessian of rotated vector
    // component i) in hess_comp[i].
    static void hess_rotated_vector(const Vec3 &w, const Vec3 &v, std::array<Mat3, 3> &hess_comp) {
        std::array<Eigen::Ref<Mat3>, 3> hc_refs{{hess_comp[0], hess_comp[1], hess_comp[2]}};
        hess_rotated_vector(w, v, hc_refs);
    }

    // d_i H_ijk where H_ijk is the Hessian of R(w) v.
    static Mat3 d_contract_hess_rotated_vector(const Vec3 &w, const Vec3 &v, const Vec3 &d);

    // The Hessian of R(w) A for 3xN matrix A; this is a fourth order tensor:
    //      H_ijkl = d [R(w) A]_ij / (dw_k dw_l)
    template<int N>
    static Eigen::Tensor<Real_, 4> hess_rotated_matrix(const Vec3 &w, const Eigen::Matrix<Real_, 3, N> &A);

    ////////////////////////////////////////////////////////////////////////////////
    // Helper functions
    ////////////////////////////////////////////////////////////////////////////////
    static Mat3 cross_product_matrix(const Vec3 &v);
};

#include "rotation_optimization.inl"

#endif /* end of include guard: ROTATION_OPTIMIZATION_HH */

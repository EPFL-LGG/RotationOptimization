////////////////////////////////////////////////////////////////////////////////
// rotation_optimization.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Useful functions for optimizing over the space of rotations:
//  We use the tangent space to SO(3) at a reference rotation as the optimization
//  domain and provide functions to apply the represented rotation to a given
//  vector and compute gradients/Hessians of this rotated vector.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  11/01/2018 11:38:57
////////////////////////////////////////////////////////////////////////////////
#ifndef ROTATION_OPTIMIZATION_HH
#define ROTATION_OPTIMIZATION_HH
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <array>

namespace rotation_optimization {

using V3d = Eigen::Vector3d;
using M3d = Eigen::Matrix3d;

////////////////////////////////////////////////////////////////////////////////
// Rotation matrix, rotated vectors, and rotated matrices
////////////////////////////////////////////////////////////////////////////////
// Compute R(w)
M3d rotation_matrix(const V3d &w);

// Compute R(w) v
V3d rotated_vector(const V3d &w, const V3d &v);

// Compute R(w) A
template<int N>
Eigen::Matrix<double, 3, N> rotated_matrix(const V3d &w, const Eigen::Matrix<double, 3, N> &A);

////////////////////////////////////////////////////////////////////////////////
// Gradient of rotation matrix, rotated vectors, and rotated matrices
////////////////////////////////////////////////////////////////////////////////
// Gradient of R(w). This is a third order tensor:
//      g_ijk = D [R(w)]_ij / dw_k
Eigen::Tensor<double, 3> grad_rotation_matrix(const V3d &w);

// Gradient of R(w) v. This is a second order tensor, returned as a 3x3 matrix:
//      g_ij = D [R(w) v]_i / dw_j
M3d grad_rotated_vector(const V3d &w, const V3d &v);

// Gradient of R(w) A. This is a third order tensor:
//      g_ijk = D [R(w) A]_ij / dw_k
template<int N>
Eigen::Tensor<double, 3> grad_rotated_matrix(const V3d &w, const Eigen::Matrix<double, 3, N> &A);

////////////////////////////////////////////////////////////////////////////////
// Hessian of rotation matrix, rotated vectors, and rotated matrices
////////////////////////////////////////////////////////////////////////////////
// The Hessian of R(w). this is a fourth order tensor:
//      H_ijkl = d [R(w)]_ij / (dw_k dw_l)
Eigen::Tensor<double, 4> hess_rotation_matrix(const V3d &w);

// Hessian of R(w) v. This is a third order tensor:
//      H_ijk = d [R(w) v]_i / (dw_j dw_k)
// We output the i^th slice of this tensor (the Hessian of rotated vector
// component i) in hess_comp[i].
void hess_rotated_vector(const V3d &w, const V3d &v, std::array<Eigen::Ref<M3d>, 3> hess_comp);

// The Hessian of R(w) A for 3xN matrix A; this is a fourth order tensor:
//      H_ijkl = d [R(w) A]_ij / (dw_k dw_l)
template<int N>
Eigen::Tensor<double, 4> hess_rotated_matrix(const V3d &w, const Eigen::Matrix<double, 3, N> &A);

}

#endif /* end of include guard: ROTATION_OPTIMIZATION_HH */

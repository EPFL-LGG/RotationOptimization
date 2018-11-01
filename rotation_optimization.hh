////////////////////////////////////////////////////////////////////////////////
// rotation_optimization.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Useful functions for optimizing over the space of rotations:
//  We use the tangent space to SO(3) at a reference rotation as the optimization
//  domain, and provide functions to apply the represented rotation to a given
//  vector and compute gradients/Hessians of this rotated vector.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  11/01/2018 11:38:57
////////////////////////////////////////////////////////////////////////////////
#ifndef ROTATION_OPTIMIZATION_HH
#define ROTATION_OPTIMIZATION_HH
#include <Eigen/Dense>
#include <array>

namespace rotation_optimization {

using V3d = Eigen::Vector3d;
using M3d = Eigen::Matrix3d;

// Compute R(w) v
V3d rotated_vector(const V3d &w, const V3d &v);

// Gradient of R(w) v: this is a second order tensor, returned as a 3x3 matrix:
//      G_ij = D [R(w) v]_i / dw_j
M3d grad_rotated_vector(const V3d &w, const V3d &v);

// Hessian of R(w) v: this is a third order tensor:
//      H_ijk = d [R(w) v]_i / (dw_j dw_k)
// We output the i^th slice of this tensor (the Hessian of rotated vector
// component i) in hess_comp[i].
void hess_rotated_vector(const V3d &w, const V3d &v, std::array<Eigen::Ref<M3d>, 3> hess_comp);

}

#endif /* end of include guard: ROTATION_OPTIMIZATION_HH */

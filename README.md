# Optimization over SO(3)
Gradients and Hessians needed to optimize over the space of rotations,
parametrized using the tangent space to SO(3)/the exponential map.
See `doc/OptimizingRotations.pdf` for more explanation and the derivations.

This code provides the gradient and Hessian of a rotated vector with respect to the
variables parametrizing the rotation. The gradient/Hessian of the rotation matrix itself
can be computed by differentiating the rotation of each of the canonical basis vectors.

#pragma once

#include <curand_kernel.h>

#include "math/vec3.cuh"

namespace Math {
__device__ Vec3 reflect(const Vec3 &v, const Vec3 &n);

__device__ Vec3 random_in_unit_sphere(curandState *local_rand_state);

}; // namespace Math

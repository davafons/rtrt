#pragma once

#include <curand_kernel.h>

#include "math/vec3.cuh"

namespace Math {
__device__ Vec3 reflect(const Vec3 &v, const Vec3 &n);

__device__ Vec3 random_in_unit_sphere(curandState *local_rand_state);

__device__ Vec3 random_in_unit_disk(curandState *local_rand_state);

__device__ bool refract(const Vec3 &v, const Vec3 &n, float ni_over_nt,
                        Vec3 &refracted);

__device__ float schlick(float cosine, float ref_idx);
}; // namespace Math

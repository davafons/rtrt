#include "math/math.cuh"

#define RANDVEC3(local_rand_state)                                             \
  Vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state),     \
       curand_uniform(local_rand_state))

namespace Math {
__device__ Vec3 reflect(const Vec3 &v, const Vec3 &n) {
  return v - 2 * dot(v, n) * n;
}

__device__ Vec3 random_in_unit_sphere(curandState *local_rand_state) {
  Vec3 p;
  do {
    p = 2.0f * RANDVEC3(local_rand_state) - Vec3(1, 1, 1);
  } while (p.squared_length() >= 1.0f);

  return p;
}

__device__ Vec3 random_in_unit_disk(curandState *local_rand_state) {
  Vec3 p;
  do {
    p = 2.0f * Vec3(curand_uniform(local_rand_state),
                    curand_uniform(local_rand_state), 0) -
        Vec3(1, 1, 0);
  } while (dot(p, p) >= 1.0f);

  return p;
}

__device__ bool refract(const Vec3 &v, const Vec3 &n, float ni_over_nt,
                        Vec3 &refracted) {
  Vec3 uv = unit_vector(v);
  float dt = dot(uv, n);
  float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);

  if (discriminant > 0) {
    refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
    return true;

  } else {
    return false;
  }
}

__device__ float schlick(float cosine, float ref_idx) {
  float r0 = (1 - ref_idx) / (1 + ref_idx);
  r0 = r0 * r0;

  return r0 + (1 - r0) * pow(1 - cosine, 5);
}

} // namespace Math

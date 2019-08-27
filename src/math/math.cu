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

} // namespace Math

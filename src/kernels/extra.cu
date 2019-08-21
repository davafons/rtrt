#include "extra.cuh"

#include "math/ray.cuh"
#include "math/vec3.cuh"

__device__ Vec3 color(const Ray &r) {
  Vec3 unit_direction = unit_vector(r.direction());
  float t = 0.5f * (unit_direction.y() + 1.0f);

  return (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
}

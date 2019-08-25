#include "kernels.cuh"

#include <cfloat>

#include "frontend/texturegpu.cuh"
#include "hitable/hitable_list.cuh"
#include "math/camera.cuh"
#include "math/ray.cuh"
#include "math/vec3.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/managed_ptr.cuh"
#include "utils/world.cuh"

__device__ float hit_sphere_6(const Vec3 &center, float radius, const Ray &r) {
  Vec3 oc = r.origin() - center;
  float a = dot(r.direction(), r.direction());
  float b = 2.0f * dot(oc, r.direction());
  float c = dot(oc, oc) - radius * radius;

  float discriminant = b * b - 4 * a * c;
  if (discriminant < 0) {
    return -1.0f;
  } else {
    return (-b - sqrt(discriminant)) / (2.0f * a);
  }
}

__device__ Vec3 color_6(const Ray &r, Hitable **hitable_objects) {
  HitRecord rec;

  if ((*hitable_objects)->hit(r, 0.0f, 10.0f, rec)) {

    return 0.5f *
           Vec3(rec.normal.x() + 1, rec.normal.y() + 1, rec.normal.z() + 1);

  } else {
    Vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);

    return (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
  }
}

__global__ void chapter_6_kernel(TextureGPU *tex, Camera camera,
                                 Hitable **hitable_objects,
                                 curandState *rand_state) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  size_t w = tex->get_width();
  size_t h = tex->get_height();

  if ((x >= w || (y >= h)))
    return;

  int pixel_index = y * w + x;
  curandState local_rand_state = rand_state[pixel_index];

  Vec3 col(0.0f, 0.0f, 0.0f);

  for (int s = 0; s < camera.get_ns(); ++s) {
    float u = float(x + curand_uniform(&local_rand_state)) / float(w);
    float v = float(h - y + curand_uniform(&local_rand_state)) / float(h);

    Ray ray = camera.get_ray(u, v);

    col += color_6(ray, hitable_objects);
  }

  col /= float(camera.get_ns());

  Uint8 r = col.r() * 255.99f;
  Uint8 g = col.g() * 255.99f;
  Uint8 b = col.b() * 255.99f;

  tex->set_rgb(x, y, r, g, b);
}

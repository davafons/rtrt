#include "kernels.cuh"

#include "frontend/texturegpu.cuh"
#include "math/ray.cuh"
#include "math/vec3.cuh"
#include "utils/world.cuh"

__device__ float hit_sphere_5(const Vec3 &center, float radius, const Ray &r) {
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

__device__ Vec3 color_5(const Ray &r) {
  float t = hit_sphere_5(Vec3(0, 0, -1), 0.5f, r);

  if (t > 0.0f) {

    Vec3 N = unit_vector(r.point_at_parameter(t) - Vec3(0, 0, -1));
    return 0.5f * Vec3(N.x() + 1, N.y() + 1, N.z() + 1);
  }

  Vec3 unit_direction = unit_vector(r.direction());
  t = 0.5f * (unit_direction.y() + 1.0f);

  return (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
}

__global__ void chapter_5_kernel(TextureGPU *tex, World world) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  size_t w = tex->get_width();
  size_t h = tex->get_height();

  if ((x >= w || (y >= h)))
    return;

  float u = float(x) / float(w);
  float v = float(h - y) / float(h);

  Ray ray(world.origin,
          world.lower_left_corner + u * world.horizontal + v * world.vertical);
  Vec3 col = color_5(ray);

  Uint8 r = col.r() * 255.99;
  Uint8 g = col.g() * 255.99;
  Uint8 b = col.b() * 255.99;

  tex->set_rgb(x, y, r, g, b);
}

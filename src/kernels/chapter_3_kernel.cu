#include "kernels.cuh"

#include "math/ray.cuh"
#include "math/vec3.cuh"

#include "frontend/texturegpu.cuh"
#include "utils/world.cuh"

__device__ Vec3 color(const Ray &r) {
  Vec3 unit_direction = unit_vector(r.direction());
  float t = 0.5f * (unit_direction.y() + 1.0f);

  return (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
}

__global__ void chapter_3_kernel(TextureGPU *tex, World world) {
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
  Vec3 col = color(ray);

  Uint8 r = col.r() * 255.99;
  Uint8 g = col.g() * 255.99;
  Uint8 b = col.b() * 255.99;

  tex->set_rgb(x, y, r, g, b);
}

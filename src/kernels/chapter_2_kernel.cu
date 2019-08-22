#include "kernels.cuh"

#include "frontend/texturegpu.cuh"
#include "math/vec3.cuh"

__global__ void chapter_2_kernel(TextureGPU *tex) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  size_t w = tex->get_width();
  size_t h = tex->get_height();

  if ((x >= w || (y >= h)))
    return;

  Vec3 col(float(x) / float(w), float(h - y) / float(h), 0.2f);

  Uint8 r = col.r() * 255.99f;
  Uint8 g = col.g() * 255.99f;
  Uint8 b = col.b() * 255.99f;

  tex->set_rgb(x, y, r, g, b);
}

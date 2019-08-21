#include <chrono>
#include <iostream>
#include <thread>

#include "SDL.h"

/* #include "ray.cuh" */
/* #include "vec3.cuh" */
/* #include "viewport.cuh" */
#include "cuda_utils.cuh"
#include "texturegpu.cuh"
#include "window.hpp"

int tx = 8;
int ty = 8;

/* Vec3 lower_left_corner(-2.0, -1.0, -1.0); */
/* Vec3 horizontal(4.0, 0.0, 0.0); */
/* Vec3 vertical(0.0, 2.0, 0.0); */
/* Vec3 origin(0.0, 0.0, 0.0); */
/*  */
/* __device__ Vec3 color(const Ray &r) { */
/*   Vec3 unit_direction = unit_vector(r.direction()); */
/*   float t = 0.5f * (unit_direction.y() + 1.0f); */
/*  */
/*   return (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f); */
/* } */
/*  */
/* __global__ void compute_shader(Viewport *viewport, Vec3 lower_left_corner, */
/*                                Vec3 horizontal, Vec3 vertical, Vec3 origin) {
 */
/*  */
/*   int x = threadIdx.x + blockIdx.x * blockDim.x; */
/*   int y = threadIdx.y + blockIdx.y * blockDim.y; */
/*  */
/*   size_t w = viewport->get_width(); */
/*   size_t h = viewport->get_height(); */
/*  */
/*   if ((x >= w || (y >= h))) */
/*     return; */
/*  */
/*   float u = float(x) / float(w); */
/*   float v = float(h - y) / float(h); */
/*  */
/*   Ray ray(origin, lower_left_corner + u * horizontal + v * vertical); */
/*   Vec3 col = color(ray); */
/*  */
/*   Uint8 r = col.r() * 255.99; */
/*   Uint8 g = col.g() * 255.99; */
/*   Uint8 b = col.b() * 255.99; */
/*  */
/*   viewport->set_rgb(x, y, r, g, b); */
/* } */
/*  */
/* void update_viewport(Viewport *viewport) { */
/*  */
/*   dim3 blocks(viewport->get_width() / tx + 1, viewport->get_height() / ty +
 * 1); */
/*   dim3 threads(tx, ty); */
/*  */
/*   viewport->lock_gpu(); */
/*  */
/*   compute_shader<<<blocks, threads>>>(viewport, lower_left_corner,
 * horizontal, */
/*                                       vertical, origin); */
/*  */
/*   cudaDeviceSynchronize(); */
/*  */
/*   viewport->unlock_gpu(); */
/* } */

__global__ void simple_kernel(TextureGPU *texture) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  size_t w = texture->get_width();
  size_t h = texture->get_height();

  if ((x >= w || (y >= h)))
    return;

  float u = float(x) / float(w);
  float v = float(h - y) / float(h);

  Uint8 r = 255 * u;
  Uint8 g = 255 * v;
  Uint8 b = 255 * u;

  texture->set_rgba(x, y, r, g, b, 255);
}

void launch_2D_texture_kernel(TextureGPU *texture,
                              void (*kernel)(TextureGPU *)) {
  dim3 blocks(texture->get_width() / tx + 1, texture->get_height() / ty + 1);
  dim3 threads(tx, ty);

  kernel<<<blocks, threads>>>(texture);

  texture->copy_to_cpu();
}

int main() {
  {
    Window window("Raytracer", 800, 600);

    managed_ptr<TextureGPU> viewport = make_managed<TextureGPU>(
        window.get_renderer(), window.get_width(), window.get_height());

    while (!window.should_quit()) {
      window.update_fps();

      SDL_Event e;
      while (SDL_PollEvent(&e) != 0) {
        if (e.type == SDL_QUIT) {
          window.close();
        }

        if (e.key.keysym.sym == SDLK_ESCAPE) {
          window.close();
        }
      }

      window.clear_renderer();

      launch_2D_texture_kernel(viewport.get(), simple_kernel);

      viewport->render(window.get_renderer());

      window.update_renderer();

      if (window.get_counted_frames() % 200 == 0) {
        std::cout << window.get_fps() << std::endl;
      }
    }
  }

  cudaErrchk(cudaDeviceReset());

  return 0;
}

#include <chrono>
#include <iostream>
#include <thread>

#include <SDL.h>
#include <curand_kernel.h>

#include "frontend/texturegpu.cuh"
#include "frontend/window.hpp"
#include "hitable/hitable_list.cuh"
#include "hitable/sphere.cuh"
#include "kernels/kernels.cuh"
#include "math/camera.cuh"
#include "math/ray.cuh"
#include "math/vec3.cuh"
#include "utils/config.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/managed_ptr.cuh"

template <typename... Args>
void launch_2D_texture_kernel(void (*kernel)(TextureGPU *, Args...),
                              const Config &config, TextureGPU *tex,
                              Args... args) {
  dim3 blocks = config.blocks(tex->get_width(), tex->get_height());

  kernel<<<blocks, config.threads>>>(tex, std::forward<Args>(args)...);
  cudaCheckErr(cudaGetLastError());

  tex->copy_to_cpu();
}

__global__ void create_world(HitableList **hitable_objects) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if ((x == 0) && (y == 0)) {
    *hitable_objects = new HitableList();
    (*hitable_objects)->push_back(new Sphere(Vec3(0, 0, -1), 0.5f));
    (*hitable_objects)->push_back(new Sphere(Vec3(0, -100.5f, -1), 100));
  }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if ((x >= max_x) || (y >= max_y)) {
    return;
  }

  int pixel_index = y * max_x + x;

  curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

template <class T, typename... Args>
__global__ void push_back(HitableList **hitable_objects, Args... args) {
  (*hitable_objects)->push_back(new T(args...));
}

int main() {
  Camera gCamera;
  Config gConfig;

  {
    Window window("Raytracer", 800, 400);

    managed_ptr<TextureGPU> viewport = make_managed<TextureGPU>(
        window.get_renderer(), window.get_width(), window.get_height(), 1.0f);

    HitableList **hitable_objects =
        cuda_malloc<HitableList *>(sizeof(HitableList *));
    create_world<<<1, 1>>>(hitable_objects);

    cudaCheckErr(cudaDeviceSynchronize());
    cudaCheckErr(cudaGetLastError());

    curandState *d_rand_state = cuda_malloc<curandState>(
        viewport->get_n_pixels() * sizeof(curandState));
    dim3 blocks = gConfig.blocks(viewport->get_width(), viewport->get_height());

    render_init<<<blocks, gConfig.threads>>>(
        viewport->get_width(), viewport->get_height(), d_rand_state);

    cudaCheckErr(cudaDeviceSynchronize());
    cudaCheckErr(cudaGetLastError());

    /* push_back<Sphere><<<1, 1>>>(hitable_objects, Vec3(-1, 0, -1), 0.5f); */
    /* cudaCheckErr(cudaDeviceSynchronize()); */
    /* cudaCheckErr(cudaGetLastError()); */

    int ns = 10;
    int MAX_NS = 200;
    gCamera.set_ns(ns);

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

      window.clear_render();

      if (ns < MAX_NS) {
        launch_2D_texture_kernel(chapter_7_kernel, gConfig, viewport.get(),
                                 gCamera, (Hitable **)hitable_objects,
                                 d_rand_state);
        ns += 20;
        gCamera.set_ns(ns);
      }

      viewport->copy_to_renderer(window.get_renderer());

      window.present_render();

      if (window.get_counted_frames() % 200 == 0) {
        std::cout << window.get_fps() << std::endl;
      }
    }
  }

  cudaCheckErr(cudaDeviceReset());

  return 0;
}

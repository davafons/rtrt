#include <chrono>
#include <iostream>
#include <thread>

#include <SDL.h>

#include "frontend/texturegpu.cuh"
#include "frontend/window.hpp"
#include "hitable/hitable_list.cuh"
#include "hitable/sphere.cuh"
#include "kernels/kernels.cuh"
#include "math/ray.cuh"
#include "math/vec3.cuh"
#include "utils/config.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/managed_ptr.cuh"
#include "utils/world.cuh"

template <typename... Args>
void launch_2D_texture_kernel(void (*kernel)(managed_ptr<TextureGPU>, Args...),
                              const Config &config, managed_ptr<TextureGPU> tex,
                              Args... args) {
  dim3 blocks(tex->get_width() / config.threads.x + 1,
              tex->get_height() / config.threads.y + 1);

  kernel<<<blocks, config.threads>>>(tex, std::forward<Args>(args)...);
  cudaCheckErr(cudaGetLastError());

  tex->copy_to_cpu();
}

__global__ void create_world(HitableList **hitable_objects) {
  *hitable_objects = new HitableList(2);
}

int main() {
  World gWorld;
  Config gConfig;

  {
    Window window("Raytracer", 800, 400);

    managed_ptr<TextureGPU> viewport = make_managed<TextureGPU>(
        window.get_renderer(), window.get_width(), window.get_height());

    HitableList **hitable_objects =
        cuda_malloc<HitableList *>(sizeof(HitableList *));
    create_world<<<1, 1>>>(hitable_objects);

    cudaCheckErr(cudaDeviceSynchronize());
    cudaCheckErr(cudaGetLastError());

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

      launch_2D_texture_kernel(chapter_5_kernel, gConfig, viewport, gWorld,
                               hitable_objects);

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

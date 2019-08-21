#include <chrono>
#include <iostream>
#include <thread>

#include <SDL.h>

#include "frontend/texturegpu.cuh"
#include "frontend/window.hpp"
#include "kernels/kernels.cuh"
#include "math/ray.cuh"
#include "math/vec3.cuh"
#include "utils/config.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/world.cuh"

template <typename... Args>
void launch_2D_texture_kernel(void (*kernel)(TextureGPU *, Args...),
                              const Config &config, TextureGPU *tex,
                              Args... args) {
  dim3 blocks(tex->get_width() / config.threads.x + 1,
              tex->get_height() / config.threads.y + 1);

  kernel<<<blocks, config.threads>>>(tex, std::forward<Args>(args)...);

  tex->copy_to_cpu();
}

int main() {
  World gWorld;
  Config gConfig;

  {
    Window window("Raytracer", 800, 600);

    managed_ptr<TextureGPU> viewport = make_managed<TextureGPU>(
        window.get_renderer(), window.get_width(), window.get_height(), 0.5f);

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

      launch_2D_texture_kernel(sky, gConfig, viewport.get(), gWorld);

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

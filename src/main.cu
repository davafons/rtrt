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
    float R = cos(M_PI / 4);
    *hitable_objects = new HitableList();
    (*hitable_objects)->push_back(new Sphere(Vec3(0, 0, -1), 0.5f));
    (*hitable_objects)->push_back(new Sphere(Vec3(0, -100.5f, -1), 100));
    (*hitable_objects)->push_back(new Sphere(Vec3(1, 0, -1), 0.5f));
    (*hitable_objects)->push_back(new Sphere(Vec3(-1, 0, -1), 0.5f));
    (*hitable_objects)->push_back(new Sphere(Vec3(-1, 0, -1), -0.45f));
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
  Config gConfig;

  {
    Window window("Raytracer", 800, 400);
    Camera gCamera(Vec3(-2, 2, 1), Vec3(0, 0, -1), Vec3(0, 1, 0), 90,
                   float(window.get_width()) / window.get_height());

    managed_ptr<TextureGPU> viewport = make_managed<TextureGPU>(
        window.get_renderer(), window.get_width(), window.get_height(), 0.75f);

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

    int ns = 20;
    int MAX_NS = 200;
    gCamera.set_ns(ns);

    while (!window.should_quit()) {
      SDL_Event e;
      while (SDL_PollEvent(&e) != 0) {
        if (e.type == SDL_QUIT) {
          window.close();
        }

        switch (e.key.keysym.sym) {

        case SDLK_ESCAPE:
          window.close();
          break;

        case SDLK_w:
          gCamera.move(Camera::Movement::FORWARD, window.get_delta_time());
          break;

        case SDLK_s:
          gCamera.move(Camera::Movement::BACKWARD, window.get_delta_time());
          break;

        case SDLK_d:
          gCamera.move(Camera::Movement::RIGHT, window.get_delta_time());
          break;

        case SDLK_a:
          gCamera.move(Camera::Movement::LEFT, window.get_delta_time());
          break;

        case SDLK_e:
          gCamera.move(Camera::Movement::UP, window.get_delta_time());
          break;

        case SDLK_q:
          gCamera.move(Camera::Movement::DOWN, window.get_delta_time());
          break;
        }
      }

      window.update_delta_time();

      window.clear_render();

      if (ns < MAX_NS) {
        launch_2D_texture_kernel(chapter_7_kernel, gConfig, viewport.get(),
                                 gCamera, (Hitable **)hitable_objects,
                                 d_rand_state);
        /* ns += 20; */
        /* gCamera.set_ns(ns); */
      }

      /* gCamera.move_right(0.01f): */

      viewport->copy_to_renderer(window.get_renderer());

      window.present_render();

      std::cout << window.get_fps() << std::endl;
    }
  }

  cudaCheckErr(cudaDeviceReset());

  return 0;
}

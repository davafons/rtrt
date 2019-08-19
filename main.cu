#include <chrono>
#include <iostream>
#include <thread>

#include "SDL.h"

#include "ray.cuh"
#include "vec3.cuh"
#include "viewport.cuh"

const int WIDTH = 800;
const int HEIGHT = 600;

bool cuda = true;

int tx = 8;
int ty = 8;
dim3 blocks(WIDTH / tx + 1, HEIGHT / ty + 1);
dim3 threads(tx, ty);

Vec3 lower_left_corner(-2.0, -1.0, -1.0);
Vec3 horizontal(4.0, 0.0, 0.0);
Vec3 vertical(0.0, 2.0, 0.0);
Vec3 origin(0.0, 0.0, 0.0);

Uint32 *d_pixels = NULL;

__device__ Vec3 color(const Ray &r) {
  Vec3 unit_direction = unit_vector(r.direction());
  float t = 0.5f * (unit_direction.y() + 1.0f);

  return (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
}

__global__ void compute_shader(Viewport *viewport, Vec3 lower_left_corner,
                               Vec3 horizontal, Vec3 vertical, Vec3 origin) {

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  size_t w = viewport->get_width();
  size_t h = viewport->get_height();

  if ((x >= w || (y >= h)))
    return;

  float u = float(x) / float(w);
  float v = float(h - y) / float(h);

  Ray ray(origin, lower_left_corner + u * horizontal + v * vertical);
  Vec3 col = color(ray);

  Uint8 r = col.r() * 255.99;
  Uint8 g = col.g() * 255.99;
  Uint8 b = col.b() * 255.99;

  viewport->set_rgb(x, y, r, g, b);
}

void update_viewport(Viewport *viewport) {
  viewport->lock_gpu();

  compute_shader<<<blocks, threads>>>(viewport, lower_left_corner, horizontal,
                                      vertical, origin);

  cudaDeviceSynchronize();

  viewport->unlock_gpu();
}

int main() {
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    throw std::runtime_error(SDL_GetError());
  }

  if (!SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1")) {
    std::cerr << "Warning: Linear texture filtering not enabled!" << std::endl;
  }

  SDL_Window *window = SDL_CreateWindow("Raytracing", SDL_WINDOWPOS_UNDEFINED,
                                        SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT,
                                        SDL_WINDOW_SHOWN);

  if (!window) {
    throw std::runtime_error(SDL_GetError());
  }

  SDL_Renderer *renderer =
      SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

  if (!renderer) {
    throw std::runtime_error(SDL_GetError());
  }

  SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);

  Viewport *viewport_mem = nullptr;
  cudaMallocManaged((void **)&viewport_mem, sizeof(Viewport));

  Viewport *viewport = new (viewport_mem) Viewport(renderer, WIDTH, HEIGHT);

  cudaMalloc((void **)&d_pixels,
             sizeof(Uint32) * viewport->get_width() * viewport->get_height());

  bool quit = false;
  long int iteration = 0;
  while (!quit) {

    SDL_Event e;
    while (SDL_PollEvent(&e) != 0) {
      if (e.type == SDL_QUIT) {
        quit = true;
      }

      if (e.key.keysym.sym == SDLK_ESCAPE) {
        quit = true;
      }
    }

    SDL_RenderClear(renderer);

    const auto before = std::chrono::system_clock::now();

    update_viewport(viewport);
    viewport->render();

    const std::chrono::duration<double> duration =
        std::chrono::system_clock::now() - before;

    ++iteration;

    if (iteration % 200 == 0) {
      std::cout << (1000 / duration.count()) / 1000 << std::endl;
    }

    SDL_RenderPresent(renderer);
  }

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);

  return 0;
}

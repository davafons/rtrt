#include <chrono>
#include <iostream>
#include <thread>

#include "SDL.h"

#include "ray.h"
#include "vec3.h"
#include "viewport.h"

const int WIDTH = 400;
const int HEIGHT = 200;

Vec3 lower_left_corner(-2.0, -1.0, -1.0);
Vec3 horizontal(4.0, 0.0, 0.0);
Vec3 vertical(0.0, 2.0, 0.0);
Vec3 origin(0.0, 0.0, 0.0);

Vec3 color(const Ray &r) {
  Vec3 unit_direction = unit_vector(r.direction());
  float t = 0.5f * (unit_direction.y() + 1.0f);

  return (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(1.0f, 0.7f, 0.5f);
}

void update_viewport(Viewport &viewport) {

  viewport.lock();

  for (int y = 0; y < viewport.get_height(); ++y) {
    for (int x = 0; x < viewport.get_width(); ++x) {

      float u = float(x) / float(viewport.get_width());
      float v = float(viewport.get_height() - y) / float(viewport.get_height());

      Ray r(origin, lower_left_corner + u * horizontal + v * vertical);
      Vec3 col = color(r);

      viewport.set_rgb(x, y, Uint8(col.b() * 255.99), Uint8(col.g() * 255.99),
                       Uint8(col.r() * 255.99));
    }
  }

  viewport.unlock();
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

  Viewport viewport(renderer, WIDTH, HEIGHT);

  bool quit = false;
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
    viewport.render();

    const std::chrono::duration<double> duration =
        std::chrono::system_clock::now() - before;

    std::cout << (1000 / duration.count()) / 1000 << std::endl;

    SDL_RenderPresent(renderer);
  }

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);

  return 0;
}

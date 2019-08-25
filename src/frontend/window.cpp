#include "window.hpp"

#include <iostream>
#include <stdexcept>

Window::Window(const std::string &title, size_t width, size_t height)
    : width_(width), height_(height) {
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    throw std::runtime_error(SDL_GetError());
  }

  if (!SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1")) {
    std::cout << "Warning: Linear texture filtering not enabled!" << std::endl;
  }

  window_ = SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_UNDEFINED,
                             SDL_WINDOWPOS_UNDEFINED, width_, height_,
                             SDL_WINDOW_SHOWN);

  if (!window_) {
    throw std::runtime_error(SDL_GetError());
  }

  renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_ACCELERATED);

  if (!renderer_) {
    throw std::runtime_error(SDL_GetError());
  }

  SDL_SetRenderDrawColor(renderer_, 0xFF, 0xFF, 0xFF, 0xFF);

  quit_ = false;
}

Window::~Window() {
  SDL_DestroyRenderer(renderer_);
  SDL_DestroyWindow(window_);
}

void Window::update_delta_time() {
  size_t current_frame = SDL_GetTicks();
  delta_time_ = float(current_frame - last_frame_) / 1000.0f;
  last_frame_ = current_frame;
}

#pragma once

#include "SDL.h"

class Viewport {
public:
  Viewport(SDL_Renderer *renderer, size_t w, size_t h,
           Uint32 pixel_format = SDL_PIXELFORMAT_RGBA8888);

  size_t get_width() const { return width_; }
  size_t get_height() const { return height_; }

  void lock();
  void unlock();
  bool is_locked() const { return locked_; }

  Uint32 &access(int x, int y);
  void set_rgb(int x, int y, Uint8 r, Uint8 g, Uint8 b);
  void set_rgba(int x, int y, Uint8 r, Uint8 g, Uint8 b, Uint8 a);

  void render() const { SDL_RenderCopy(renderer_, tex_, nullptr, nullptr); }

private:
  SDL_Renderer *renderer_{nullptr};
  SDL_Texture *tex_{nullptr};

  // Properties
  size_t width_{0};
  size_t height_{0};
  SDL_PixelFormat *fmt_{nullptr};

  // Bit manipulation
  void *pixels_{nullptr};
  int pitch_{0};
  bool locked_{false};
};

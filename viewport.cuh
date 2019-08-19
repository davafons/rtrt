#pragma once

#include "SDL.h"

class Viewport {
public:
  Viewport(SDL_Renderer *renderer, size_t w, size_t h,
           Uint32 pixel_format_enum = SDL_PIXELFORMAT_RGBA8888);

  __host__ __device__ size_t get_width() const { return width_; }
  __host__ __device__ size_t get_height() const { return height_; }
  __host__ __device__ SDL_PixelFormat *get_format() const { return fmt_; }

  void lock();
  void unlock();

  void lock_gpu();
  void unlock_gpu();

  bool is_locked() const { return locked_; }

  __host__ __device__ Uint32 &access(int x, int y);
  __host__ __device__ void set_rgb(int x, int y, Uint8 r, Uint8 g, Uint8 b);
  __host__ __device__ void set_rgba(int x, int y, Uint8 r, Uint8 g, Uint8 b,
                                    Uint8 a);

  Uint32 *get_pixels() { return static_cast<Uint32 *>(pixels_); }
  void set_pixels(Uint32 *pixels) {
    SDL_UpdateTexture(tex_, NULL, pixels, pitch_);
  }

  void render() const { SDL_RenderCopy(renderer_, tex_, NULL, NULL); }

private:
  SDL_Renderer *renderer_ = NULL;
  SDL_Texture *tex_ = NULL;

  // Properties
  size_t width_ = 0;
  size_t height_ = 0;
  SDL_PixelFormat *fmt_ = NULL;

  // Bit manipulation
  void *pixels_ = NULL;
  void *d_pixels_ = NULL;
  int pitch_ = 0;
  bool locked_ = false;
};

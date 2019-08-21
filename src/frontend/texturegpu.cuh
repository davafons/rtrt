#pragma once

#include "SDL.h"

#include "utils/cuda_utils.cuh"

class TextureGPU {
public:
  TextureGPU(SDL_Renderer *renderer, size_t width, size_t height,
             Uint32 pixel_format_enum = SDL_PIXELFORMAT_RGBA32);

  ~TextureGPU();

  __host__ __device__ size_t get_width() const { return width_; }
  __host__ __device__ size_t get_height() const { return height_; }

  void copy_to_cpu();

  __device__ Uint32 &access(int x, int y) {
    return static_cast<Uint32 *>(d_pixels_)[y * width_ + x];
  }

  __device__ void set_rgb(int x, int y, Uint8 r, Uint8 g, Uint8 b) {
    set_rgba(x, y, r, g, b, 0xFF);
  }

  __device__ void set_rgba(int x, int y, Uint8 r, Uint8 g, Uint8 b, Uint8 a) {
    access(x, y) = (r >> fmt_->Rloss) << fmt_->Rshift |
                   (g >> fmt_->Gloss) << fmt_->Gshift |
                   (b >> fmt_->Bloss) << fmt_->Bshift |
                   ((a >> fmt_->Aloss) << fmt_->Ashift & fmt_->Amask);
  }

  void copy_to_renderer(SDL_Renderer *renderer) {
    SDL_RenderCopy(renderer, tex_, NULL, NULL);
  }

private:
  SDL_Texture *tex_ = NULL;
  Uint32 *d_pixels_ = NULL;

  size_t width_ = 0;
  size_t height_ = 0;

  SDL_PixelFormat *fmt_ = NULL;
};

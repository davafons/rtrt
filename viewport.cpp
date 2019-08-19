#include "viewport.h"

Viewport::Viewport(SDL_Renderer *renderer, size_t w, size_t h,
                   Uint32 pixel_format)
    : renderer_(renderer), width_(w), height_(h) {
  tex_ = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
                           SDL_TEXTUREACCESS_STREAMING, w, h);

  fmt_ = SDL_AllocFormat(pixel_format);
}

void Viewport::lock() {
  SDL_LockTexture(tex_, nullptr, &pixels_, &pitch_);
  locked_ = true;
}

void Viewport::unlock() {
  SDL_UnlockTexture(tex_);
  pixels_ = nullptr;
  pitch_ = 0;
  locked_ = false;
}

Uint32 &Viewport::access(int x, int y) {
  return static_cast<Uint32 *>(pixels_)[y * width_ + x];
}

void Viewport::set_rgb(int x, int y, Uint8 r, Uint8 g, Uint8 b) {
  access(x, y) = SDL_MapRGB(fmt_, r, g, b);
}

void Viewport::set_rgba(int x, int y, Uint8 r, Uint8 g, Uint8 b, Uint8 a) {
  access(x, y) = SDL_MapRGBA(fmt_, r, g, b, a);
}

#include "viewport.cuh"

Viewport::Viewport(SDL_Renderer *renderer, size_t w, size_t h,
                   Uint32 pixel_format)
    : renderer_(renderer), width_(w), height_(h) {
  tex_ = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
                           SDL_TEXTUREACCESS_STREAMING, w, h);

  fmt_ = SDL_AllocFormat(pixel_format);
}

void Viewport::lock() {
  SDL_LockTexture(tex_, NULL, &pixels_, &pitch_);
  locked_ = true;
}

/* void Viewport::lock_gpu() { */
/*   lock(); */
/*  */
/*   cudaMalloc((void **)&d_pixels_, width_ * height_ * sizeof(Uint32)); */
/*   cudaMemcpy(d_pixels_, pixels_, width_ * height_ * sizeof(Uint32), */
/*              cudaMemcpyHostToDevice); */
/* } */

void Viewport::unlock() {
  SDL_UnlockTexture(tex_);
  pixels_ = NULL;
  pitch_ = 0;
  locked_ = false;
}

/* void Viewport::unlock_gpu() { */
/*   cudaMemcpy(pixels_, d_pixels_, width_ * height_ * sizeof(Uint32), */
/*              cudaMemcpyDeviceToHost); */
/*   d_pixels_ = NULL; */
/*  */
/*   unlock(); */
/* } */

Uint32 &Viewport::access(int x, int y) {
  return static_cast<Uint32 *>(pixels_)[y * width_ + x];
}

void Viewport::set_rgb(int x, int y, Uint8 r, Uint8 g, Uint8 b) {
  access(x, y) = (r >> fmt_->Rloss) << fmt_->Rshift |
                 (g >> fmt_->Gloss) << fmt_->Gshift |
                 (b >> fmt_->Bloss) << fmt_->Bshift | fmt_->Amask;
}

void Viewport::set_rgba(int x, int y, Uint8 r, Uint8 g, Uint8 b, Uint8 a) {
  access(x, y) = 0xFFFFFFFF;
  /* access(x, y) = (r >> fmt_->Rloss) << fmt_->Rshift | */
  /*                (g >> fmt_->Gloss) << fmt_->Gshift | */
  /*                (b >> fmt_->Bloss) << fmt_->Bshift | */
  /*                ((a >> fmt_->Aloss) << fmt_->Ashift & fmt_->Amask); */
}

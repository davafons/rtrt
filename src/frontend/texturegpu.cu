#include <type_traits>

#include "texturegpu.cuh"

TextureGPU::TextureGPU(SDL_Renderer *renderer, size_t width, size_t height,
                       float scale_factor)
    : width_(width * scale_factor), height_(height * scale_factor) {

  tex_ = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32,
                           SDL_TEXTUREACCESS_STREAMING, width_, height_);

  fmt_ = cuda_malloc_managed<SDL_PixelFormat>(sizeof(SDL_PixelFormat));
  cudaCheckErr(cudaMemcpy(fmt_, SDL_AllocFormat(SDL_PIXELFORMAT_RGBA32),
                          sizeof(SDL_PixelFormat), cudaMemcpyHostToDevice));

  size_in_bytes_ = width_ * height_ * fmt_->BytesPerPixel;

  d_pixels_ = cuda_malloc<Uint32>(size_in_bytes_);
}

TextureGPU::~TextureGPU() {
  cudaCheckErr(cudaFree(d_pixels_));
  cudaCheckErr(cudaFree(fmt_));
  SDL_DestroyTexture(tex_);
}

void TextureGPU::copy_to_cpu() {
  Uint32 *h_pixels;
  int pitch = 0;

  cudaDeviceSynchronize();

  SDL_LockTexture(tex_, NULL, (void **)&h_pixels, &pitch);

  cudaCheckErr(
      cudaMemcpy(h_pixels, d_pixels_, size_in_bytes_, cudaMemcpyDeviceToHost));

  SDL_UnlockTexture(tex_);
}

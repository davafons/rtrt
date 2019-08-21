#include "texturegpu.cuh"

TextureGPU::TextureGPU(SDL_Renderer *renderer, size_t width, size_t height,
                       Uint32 pixel_format_enum)
    : width_(width), height_(height) {

  tex_ = SDL_CreateTexture(renderer, pixel_format_enum,
                           SDL_TEXTUREACCESS_STREAMING, width, height);

  fmt_ = cuda_malloc_managed<SDL_PixelFormat>(sizeof(SDL_PixelFormat));
  cudaCheckErr(cudaMemcpy(fmt_, SDL_AllocFormat(pixel_format_enum),
                          sizeof(SDL_PixelFormat), cudaMemcpyHostToDevice));

  d_pixels_ = cuda_malloc<Uint32>(width_ * height_ * 4);
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

  cudaCheckErr(cudaMemcpy(h_pixels, d_pixels_,
                          width_ * height_ * sizeof(Uint32),
                          cudaMemcpyDeviceToHost));

  SDL_UnlockTexture(tex_);
}

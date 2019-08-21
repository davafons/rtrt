#include <chrono>
#include <iostream>
#include <thread>

#include "imgui.h"
#include "imgui_sdl.h"
#include <SDL.h>

#include "frontend/texturegpu.cuh"
#include "frontend/window.hpp"
#include "kernels/kernels.cuh"
#include "math/ray.cuh"
#include "math/vec3.cuh"
#include "utils/config.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/world.cuh"

template <typename... Args>
void launch_2D_texture_kernel(void (*kernel)(TextureGPU *, Args...),
                              const Config &config, TextureGPU *tex,
                              Args... args) {
  dim3 blocks(tex->get_width() / config.threads.x + 1,
              tex->get_height() / config.threads.y + 1);

  kernel<<<blocks, config.threads>>>(tex, std::forward<Args>(args)...);

  tex->copy_to_cpu();
}

int main() {
  World gWorld;
  Config gConfig;

  {
    Window window("Raytracer", 800, 400);

    ImGui::CreateContext();
    ImGuiSDL::Initialize(window.get_renderer(), window.get_width(),
                         window.get_height());

    SDL_Texture *texture =
        SDL_CreateTexture(window.get_renderer(), SDL_PIXELFORMAT_RGBA32,
                          SDL_TEXTUREACCESS_TARGET, 100, 100);
    {
      SDL_SetRenderTarget(window.get_renderer(), texture);
      SDL_SetRenderDrawColor(window.get_renderer(), 255, 0, 255, 255);
      SDL_RenderClear(window.get_renderer());
      SDL_SetRenderTarget(window.get_renderer(), nullptr);
    }

    managed_ptr<TextureGPU> viewport = make_managed<TextureGPU>(
        window.get_renderer(), window.get_width(), window.get_height(), 0.75f);

    while (!window.should_quit()) {
      ImGuiIO &io = ImGui::GetIO();

      window.update_fps();

      SDL_Event e;
      while (SDL_PollEvent(&e) != 0) {
        if (e.type == SDL_QUIT) {
          window.close();
        }

        if (e.key.keysym.sym == SDLK_ESCAPE) {
          window.close();
        }
      }

      int mouseX, mouseY;
      const int buttons = SDL_GetMouseState(&mouseX, &mouseY);

      io.DeltaTime = 1.0f / 60.0f;
      io.MousePos =
          ImVec2(static_cast<float>(mouseX), static_cast<float>(mouseY));
      io.MouseDown[0] = buttons & SDL_BUTTON(SDL_BUTTON_LEFT);
      io.MouseDown[1] = buttons & SDL_BUTTON(SDL_BUTTON_RIGHT);

      ImGui::NewFrame();

      ImGui::ShowDemoWindow();

      ImGui::Begin("Image");
      ImGui::Image(texture, ImVec2(100, 100));
      ImGui::End();

      window.clear_render();

      viewport->copy_to_renderer(window.get_renderer());

      ImGui::Render();
      ImGuiSDL::Render(ImGui::GetDrawData());

      launch_2D_texture_kernel(chapter_5_kernel, gConfig, viewport.get(),
                               gWorld);

      window.present_render();

      if (window.get_counted_frames() % 200 == 0) {
        std::cout << window.get_fps() << std::endl;
      }
    }
  }
  ImGuiSDL::Deinitialize();
  ImGui::DestroyContext();

  cudaCheckErr(cudaDeviceReset());

  return 0;
}

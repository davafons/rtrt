#pragma once

#include <string>

#include <SDL.h>

class Window {
public:
  friend class Viewport;

  Window(const std::string &title, size_t width, size_t height);
  ~Window();

  size_t get_width() const { return width_; }
  size_t get_height() const { return height_; }

  SDL_Renderer *get_renderer() const { return renderer_; }

  void clear_render() const { SDL_RenderClear(renderer_); }
  void present_render() const { SDL_RenderPresent(renderer_); }

  bool should_quit() const { return quit_; }
  void close() { quit_ = true; }

  float get_delta_time() const { return delta_time_; }
  float get_fps() const { return 1.0f / delta_time_; }
  void update_delta_time();
private:
  SDL_Window *window_;
  SDL_Renderer *renderer_;

  float delta_time_ = 0.0f;
  size_t last_frame_ = 0;

  size_t width_ = 0;
  size_t height_ = 0;

  bool quit_ = true;
};

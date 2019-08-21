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
  float get_fps() const { return fps_; }
  size_t get_counted_frames() const { return counted_frames_; }

  SDL_Renderer *get_renderer() const { return renderer_; }

  void clear_render() const { SDL_RenderClear(renderer_); }
  void present_render() const { SDL_RenderPresent(renderer_); }

  bool should_quit() const { return quit_; }
  void close() { quit_ = true; }

  void update_fps();

private:
  SDL_Window *window_;
  SDL_Renderer *renderer_;

  size_t width_ = 0;
  size_t height_ = 0;

  size_t counted_frames_ = 0;
  float fps_ = 0.0f;

  bool quit_ = true;
};

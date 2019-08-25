#pragma once

#include "math/ray.cuh"
#include "math/vec3.cuh"

class Camera {
public:
  enum class Movement { FORWARD, BACKWARD, LEFT, RIGHT, UP, DOWN };

  Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov, float aspect)
      : lookfrom_(lookfrom), lookat_(lookat), vup_(vup), ns_(10) {
    float theta = 90 * M_PI / 180;
    half_height_ = tan(theta / 2);
    half_width_ = aspect * half_height_;

    update_camera_vectors();
  }

  __device__ Ray get_ray(float s, float t) {
    return Ray(lookfrom_, lower_left_corner_ + s * horizontal_ + t * vertical_ -
                              lookfrom_);
  }

  __host__ __device__ int get_ns() const { return ns_; }
  void set_ns(int ns) { ns_ = ns; }

  void update_camera_vectors() {
    Vec3 u, v, w;
    w = unit_vector(lookfrom_ - lookat_);
    u = unit_vector(cross(vup_, w));
    v = cross(w, u);

    lower_left_corner_ = lookfrom_ - half_width_ * u - half_height_ * v - w;
    horizontal_ = 2 * half_width_ * u;
    vertical_ = 2 * half_height_ * v;
  }

  void move(Movement direction, float deltaTime) {
    float velocity = speed_ * deltaTime;

    switch (direction) {
    case Movement::FORWARD:
      lookfrom_ += Vec3(0, 0, -1) * velocity;
      break;

    case Movement::BACKWARD:
      lookfrom_ += Vec3(0, 0, 1) * velocity;
      break;

    case Movement::LEFT:
      lookfrom_ += Vec3(-1, 0, 0) * velocity;
      break;

    case Movement::RIGHT:
      lookfrom_ += Vec3(1, 0, 0) * velocity;
      break;

    case Movement::UP:
      lookfrom_ += Vec3(0, 1, 0) * velocity;
      break;

    case Movement::DOWN:
      lookfrom_ += Vec3(0, -1, 0) * velocity;
      break;
    }

    update_camera_vectors();
  }

private:
  Vec3 lookfrom_;
  Vec3 lookat_;
  Vec3 vup_;

  float half_width_;
  float half_height_;

  Vec3 lower_left_corner_;
  Vec3 horizontal_;
  Vec3 vertical_;
  int ns_;

  float speed_ = 2.5f;
};

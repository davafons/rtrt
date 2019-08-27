#pragma once

#include "math/math.cuh"
#include "math/ray.cuh"
#include "math/vec3.cuh"

class Camera {
public:
  enum class Movement { FORWARD, BACKWARD, LEFT, RIGHT, UP, DOWN };

  Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov, float aspect,
         float aperture)
      : lookfrom_(lookfrom), lookat_(lookat), vup_(vup), ns_(10) {

    float theta = vfov * (float(M_PI) / 180.0f);
    half_height_ = tan(theta / 2);
    half_width_ = aspect * half_height_;

    lens_radius_ = aperture / 2;

    update_camera_vectors();
  }

  __device__ Ray get_ray(float s, float t) {
    return Ray(lookfrom_, lower_left_corner_ + s * horizontal_ + t * vertical_ -
                              lookfrom_);
  }

  __device__ Ray get_ray(float s, float t, curandState *local_rand_state) {
    Vec3 rd = lens_radius_ * Math::random_in_unit_disk(local_rand_state);
    Vec3 offset = u_ * rd.x() + v_ * rd.y();

    return Ray(lookfrom_ + offset, lower_left_corner_ + s * horizontal_ +
                                       t * vertical_ - lookfrom_ - offset);
  }

  __host__ __device__ int get_ns() const { return ns_; }
  void set_ns(int ns) { ns_ = ns; }

  void update_camera_vectors() {
    focus_dist_ = (lookfrom_ - lookat_).length();

    w_ = unit_vector(lookfrom_ - lookat_);
    u_ = unit_vector(cross(vup_, w_));
    v_ = cross(w_, u_);

    lower_left_corner_ = lookfrom_ - half_width_ * focus_dist_ * u_ -
                         half_height_ * focus_dist_ * v_ - focus_dist_ * w_;
    horizontal_ = 2 * half_width_ * focus_dist_ * u_;
    vertical_ = 2 * half_height_ * focus_dist_ * v_;
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

  float lens_radius_;
  float focus_dist_;
  Vec3 u_, v_, w_;

  float speed_ = 0.0001f;
};

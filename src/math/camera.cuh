#pragma once

#include "math/ray.cuh"
#include "math/vec3.cuh"

class Camera {
public:
  Camera() {
    origin_ = Vec3(0.0f, 0.0f, 0.0f);
    lower_left_corner_ = Vec3(-2.0f, -1.0f, -1.0f);
    horizontal_ = Vec3(4.0f, 0.0f, 0.0f);
    vertical_ = Vec3(0.0f, 2.0f, 0.0f);

    ns_ = 10;
  }

  __device__ Ray get_ray(float u, float v) {
    return Ray(origin_,
               lower_left_corner_ + u * horizontal_ + v * vertical_ - origin_);
  }

  __host__ __device__ int get_ns() const { return ns_; }
  __host__ __device__ void set_ns(int ns) { ns_ = ns; }

private:
  Vec3 origin_;
  Vec3 lower_left_corner_;
  Vec3 horizontal_;
  Vec3 vertical_;
  int ns_;
};

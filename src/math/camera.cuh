#pragma once

#include "math/ray.cuh"
#include "math/vec3.cuh"

class Camera {
public:
  Camera(Vec3 lookfrom, Vec3 lookat, Vec3 vup, float vfov, float aspect) {
    Vec3 u, v, w;
    float theta = vfov * M_PI / 180;
    float half_height = tan(theta / 2);
    float half_width = aspect * half_height;

    origin_ = lookfrom;
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);

    lower_left_corner_ = origin_ - half_width * u - half_height * v - w;
    horizontal_ = 2 * half_width * u;
    vertical_ = 2 * half_height * v;

    ns_ = 10;
  }

  __device__ Ray get_ray(float s, float t) {
    return Ray(origin_,
               lower_left_corner_ + s * horizontal_ + t * vertical_ - origin_);
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

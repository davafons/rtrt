#pragma once

#include "vec3.cuh"

class Ray {
public:
  __host__ __device__ Ray(const Vec3 &a, const Vec3 &b) : A_(a), B_(b){};

  __host__ __device__ Vec3 origin() const { return A_; }
  __host__ __device__ Vec3 direction() const { return B_; }
  __host__ __device__ Vec3 point_at_parameter(float t) const {
    return A_ + t * B_;
  }

private:
  Vec3 A_;
  Vec3 B_;
};

#pragma once

#include "vec3.h"

class Ray {
public:
  Ray() = default;
  Ray(const Vec3 &a, const Vec3 &b) : A_(a), B_(b){};

  Vec3 origin() const { return A_; }
  Vec3 direction() const { return B_; }
  Vec3 point_at_parameter(float t) const { return A_ + t * B_; }

private:
  Vec3 A_;
  Vec3 B_;
};

#pragma once

#include "hitable/hitable.cuh"
#include "material/material.cuh"
#include "math/vec3.cuh"

class Sphere : public Hitable {
public:
  __device__ Sphere(Vec3 cen, float r, Material *mat_ptr)
      : center_(cen), radius_(r), mat_ptr_(mat_ptr) {}

  __device__ virtual bool hit(const Ray &r, float tmin, float tmax,
                              HitRecord &rec) const;

private:
  Vec3 center_;
  float radius_;
  Material *mat_ptr_;
};

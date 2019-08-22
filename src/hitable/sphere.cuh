#pragma once

#include "hitable/hitable.cuh"
#include "math/vec3.cuh"

class Sphere : public Hitable {
public:
  __host__ __device__ Sphere(Vec3 cen, float r) : center_(cen), radius_(r) {}
  __device__ virtual bool hit(const Ray &r, float tmin, float tmax,
                              HitRecord &rec) const;

private:
  Vec3 center_;
  float radius_;
};

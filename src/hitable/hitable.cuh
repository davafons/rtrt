#pragma once

#include "math/ray.cuh"
#include "math/vec3.cuh"

struct HitRecord {
  float t;
  Vec3 p;
  Vec3 normal;
};
typedef struct HitRecord HitRecord;

class Hitable {
public:
  __device__ virtual bool hit(const Ray &r, float t_min, float t_max,
                              HitRecord &rec) const = 0;
};

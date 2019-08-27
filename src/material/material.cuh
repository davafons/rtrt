#pragma once

#include <curand_kernel.h>

class Ray;
class HitRecord;

class Material {
public:
  __device__ virtual bool scatter(const Ray &r_in, const HitRecord &rec,
                                  Vec3 &attenuation, Ray &scattered,
                                  curandState *local_rand_state) = 0;
};
